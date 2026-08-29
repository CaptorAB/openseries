"""Test that tool and dependency versions stay aligned across config files."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
PYPROJECT_PATH = ROOT / "pyproject.toml"
LOCK_PATH = ROOT / "uv.lock"
PRE_COMMIT_PATH = ROOT / ".pre-commit-config.yaml"
MAKEFILE_PATH = ROOT / "Makefile"
MAKE_PS1_PATH = ROOT / "make.ps1"
INSTALLATION_RST_PATH = ROOT / "docs" / "source" / "user_guide" / "installation.rst"
CONTRIBUTING_RST_PATH = ROOT / "docs" / "source" / "development" / "contributing.rst"
PYTHON_VERSION_PATH = ROOT / ".python-version"
ZIZMOR_SCRIPT_PATH = ROOT / "scripts" / "run-zizmor.sh"
WORKFLOW_DIR = ROOT / ".github" / "workflows"

UV_WORKFLOW_FILES = (
    "test.yml",
    "build.yml",
    "docs.yml",
    "deploy.yml",
    "codeql.yml",
    "zizmor.yml",
    "supply-chain.yml",
)

MYPY_ADDITIONAL_DEPENDENCIES = (
    "pandas-stubs",
    "pydantic",
    "scipy-stubs",
    "types-openpyxl",
    "types-python-dateutil",
    "types-requests",
)


class VersionAlignmentError(Exception):
    """Raised when a pinned version is not aligned across project files."""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _requirement_parts(requirement: str) -> tuple[str, str]:
    compact = requirement.strip().replace(" ", "")
    for separator in ("==", ">="):
        if separator in compact:
            name, spec = compact.split(separator, 1)
            return name, f"{separator}{spec}"
    msg = f"Unsupported requirement specifier: {requirement}"
    raise VersionAlignmentError(msg)


def _requirement_map(requirements: list[str]) -> dict[str, str]:
    return dict(_requirement_parts(item) for item in requirements)


def _makefile_value(text: str, name: str) -> str:
    match = re.search(rf"^{name} \?= (.+)$", text, flags=re.MULTILINE)
    if match is None:
        msg = f"{MAKEFILE_PATH.name} is missing {name}"
        raise VersionAlignmentError(msg)
    return match.group(1).strip()


def _ps1_value(text: str, name: str) -> str:
    match = re.search(rf'\${name} = "([^"]+)"', text)
    if match is None:
        msg = f"{MAKE_PS1_PATH.name} is missing {name}"
        raise VersionAlignmentError(msg)
    return match.group(1)


def _workflow_env_value(text: str, name: str) -> str:
    match = re.search(rf"^\s*{name}: \"([^\"]+)\"", text, flags=re.MULTILINE)
    if match is None:
        msg = f"Workflow is missing {name}"
        raise VersionAlignmentError(msg)
    return match.group(1)


def _pre_commit_rev(text: str, repo_url: str) -> str:
    pattern = rf"repo: {re.escape(repo_url)}\n\s+rev: (.+)"
    match = re.search(pattern, text)
    if match is None:
        msg = f"{PRE_COMMIT_PATH.name} is missing rev for {repo_url}"
        raise VersionAlignmentError(msg)
    return match.group(1).strip()


def _pre_commit_additional_dependencies(text: str) -> dict[str, str]:
    match = re.search(
        r"additional_dependencies:\n((?:          - .+\n)+)",
        text,
    )
    if match is None:
        msg = f"{PRE_COMMIT_PATH.name} is missing mypy additional_dependencies"
        raise VersionAlignmentError(msg)
    requirements = [
        line.strip()[2:].strip()
        for line in match.group(1).splitlines()
        if line.strip().startswith("- ")
    ]
    return _requirement_map(requirements)


def _lock_requires_dist(
    lock_data: dict[str, Any],
) -> dict[tuple[str, str | None], str]:
    packages = lock_data.get("package", [])
    for package in packages:
        if package.get("name") != "openseries":
            continue
        metadata = package.get("metadata", {})
        requires_dist = metadata.get("requires-dist", [])
        mapped: dict[tuple[str, str | None], str] = {}
        for item in requires_dist:
            extra = None
            marker = item.get("marker")
            if marker is not None:
                extra_match = re.search(r"extra == '([^']+)'", marker)
                if extra_match is not None:
                    extra = extra_match.group(1)
            mapped[(item["name"], extra)] = item["specifier"]
        return mapped
    msg = "uv.lock is missing the openseries package metadata"
    raise VersionAlignmentError(msg)


def _python_minor_versions(requires_python: str) -> list[str]:
    match = re.fullmatch(r">=(\d+\.\d+),<(\d+\.\d+)", requires_python.replace(" ", ""))
    if match is None:
        msg = f"Unexpected requires-python: {requires_python}"
        raise VersionAlignmentError(msg)
    start_major, start_minor = (int(part) for part in match.group(1).split("."))
    end_major, end_minor = (int(part) for part in match.group(2).split("."))
    if start_major != end_major:
        msg = f"requires-python spans multiple majors: {requires_python}"
        raise VersionAlignmentError(msg)
    return [f"{start_major}.{minor}" for minor in range(start_minor, end_minor)]


def _raise_mismatch(label: str, expected: str, actual: str) -> None:
    msg = f"{label} is {actual!r}, expected {expected!r}"
    raise VersionAlignmentError(msg)


def _required_search(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        msg = f"{label} is missing"
        raise VersionAlignmentError(msg)
    return match


def _check_python_classifiers(classifiers: list[str], versions: list[str]) -> None:
    for version in versions:
        expected = f"Programming Language :: Python :: {version}"
        if expected not in classifiers:
            msg = f"pyproject.toml classifiers are missing {expected}"
            raise VersionAlignmentError(msg)


def _check_default_python_pins(python_version: str) -> None:
    actual_default = _read_text(PYTHON_VERSION_PATH).strip()
    if actual_default != python_version:
        _raise_mismatch(".python-version", python_version, actual_default)
    language_match = _required_search(
        r"^  python: python(.+)$",
        _read_text(PRE_COMMIT_PATH),
        "pre-commit default_language_version",
    )
    if language_match.group(1) != python_version:
        _raise_mismatch(
            "pre-commit python",
            python_version,
            language_match.group(1),
        )


def _check_python_matrix_and_docs(versions: list[str]) -> None:
    matrix_match = _required_search(
        r"python-version: \[ ([^\]]+) \]",
        _read_text(WORKFLOW_DIR / "build.yml"),
        "build.yml python-version matrix",
    )
    matrix_versions = [
        item.strip().strip("'\"") for item in matrix_match.group(1).split(",")
    ]
    if matrix_versions != versions:
        _raise_mismatch(
            "build.yml python-version matrix",
            ", ".join(versions),
            ", ".join(matrix_versions),
        )
    listed = ", ".join(versions)
    if listed not in _read_text(INSTALLATION_RST_PATH):
        msg = f"{INSTALLATION_RST_PATH.name} is missing Python versions {listed}"
        raise VersionAlignmentError(msg)


def _check_type_checker_targets(
    pyproject: dict[str, Any],
    versions: list[str],
) -> None:
    mypy_version = pyproject["tool"]["mypy"]["python_version"]
    if mypy_version not in versions:
        msg = (
            "tool.mypy python_version must be a supported Python version, "
            f"got {mypy_version!r}"
        )
        raise VersionAlignmentError(msg)
    ruff_target = pyproject["tool"]["ruff"]["target-version"]
    expected_ruff_target = f"py{versions[0].replace('.', '')}"
    if ruff_target != expected_ruff_target:
        _raise_mismatch("tool.ruff target-version", expected_ruff_target, ruff_target)


class TestVersionAlignment:
    """class to verify dependency and tool versions stay aligned."""

    def test_lockfile_matches_pyproject(self: TestVersionAlignment) -> None:
        """Test uv.lock metadata matches pyproject.toml specifiers."""
        pyproject = _load_toml(PYPROJECT_PATH)
        lock_requires = _lock_requires_dist(_load_toml(LOCK_PATH))
        expected: dict[tuple[str, str | None], str] = {}
        for requirement in pyproject["project"]["dependencies"]:
            name, specifier = _requirement_parts(requirement)
            expected[(name, None)] = specifier
        extras = pyproject["project"]["optional-dependencies"]
        for extra, requirements in extras.items():
            for requirement in requirements:
                name, specifier = _requirement_parts(requirement)
                expected[(name, extra)] = specifier
        if lock_requires != expected:
            msg = (
                "uv.lock package metadata does not match pyproject.toml: "
                f"{lock_requires} != {expected}"
            )
            raise VersionAlignmentError(msg)

    def test_tool_versions_match(self: TestVersionAlignment) -> None:
        """Test uv, ruff, and mypy versions match across tooling files."""
        pyproject = _load_toml(PYPROJECT_PATH)
        dev_requirements = _requirement_map(
            pyproject["project"]["optional-dependencies"]["dev"],
        )
        pre_commit = _read_text(PRE_COMMIT_PATH)
        makefile = _read_text(MAKEFILE_PATH)
        make_ps1 = _read_text(MAKE_PS1_PATH)

        uv_version = _makefile_value(makefile, "UV_VERSION")
        if _ps1_value(make_ps1, "UV_VERSION") != uv_version:
            _raise_mismatch(
                "make.ps1 UV_VERSION",
                uv_version,
                _ps1_value(make_ps1, "UV_VERSION"),
            )
        uv_rev = _pre_commit_rev(
            pre_commit,
            "https://github.com/astral-sh/uv-pre-commit",
        )
        if uv_rev != uv_version:
            _raise_mismatch("pre-commit uv rev", uv_version, uv_rev)

        ruff_spec = dev_requirements["ruff"]
        if not ruff_spec.startswith("=="):
            msg = f"ruff must be pinned exactly in pyproject.toml, got {ruff_spec}"
            raise VersionAlignmentError(msg)
        ruff_version = ruff_spec[2:]
        ruff_rev = _pre_commit_rev(
            pre_commit,
            "https://github.com/astral-sh/ruff-pre-commit",
        )
        if ruff_rev != f"v{ruff_version}":
            _raise_mismatch("pre-commit ruff rev", f"v{ruff_version}", ruff_rev)

        mypy_spec = dev_requirements["mypy"]
        if not mypy_spec.startswith("=="):
            msg = f"mypy must be pinned exactly in pyproject.toml, got {mypy_spec}"
            raise VersionAlignmentError(msg)
        mypy_version = mypy_spec[2:]
        mypy_rev = _pre_commit_rev(
            pre_commit,
            "https://github.com/pre-commit/mirrors-mypy",
        )
        if mypy_rev != f"v{mypy_version}":
            _raise_mismatch("pre-commit mypy rev", f"v{mypy_version}", mypy_rev)

        for workflow_name in UV_WORKFLOW_FILES:
            workflow_text = _read_text(WORKFLOW_DIR / workflow_name)
            actual = _workflow_env_value(workflow_text, "UV_VERSION")
            if actual != uv_version:
                _raise_mismatch(
                    f"{workflow_name} UV_VERSION",
                    uv_version,
                    actual,
                )

    def test_audit_and_zizmor_versions_match(self: TestVersionAlignment) -> None:
        """Test pip-audit and zizmor versions match across scripts and CI."""
        makefile = _read_text(MAKEFILE_PATH)
        make_ps1 = _read_text(MAKE_PS1_PATH)
        pip_audit_version = _makefile_value(makefile, "PIP_AUDIT_VERSION")
        if _ps1_value(make_ps1, "PIP_AUDIT_VERSION") != pip_audit_version:
            _raise_mismatch(
                "make.ps1 PIP_AUDIT_VERSION",
                pip_audit_version,
                _ps1_value(make_ps1, "PIP_AUDIT_VERSION"),
            )
        supply_chain = _read_text(WORKFLOW_DIR / "supply-chain.yml")
        actual_pip_audit = _workflow_env_value(supply_chain, "PIP_AUDIT_VERSION")
        if actual_pip_audit != pip_audit_version:
            _raise_mismatch(
                "supply-chain.yml PIP_AUDIT_VERSION",
                pip_audit_version,
                actual_pip_audit,
            )

        zizmor_workflow = _read_text(WORKFLOW_DIR / "zizmor.yml")
        zizmor_version = _workflow_env_value(zizmor_workflow, "ZIZMOR_VERSION")
        script_match = re.search(
            r"^readonly ZIZMOR_VERSION=(.+)$",
            _read_text(ZIZMOR_SCRIPT_PATH),
            flags=re.MULTILINE,
        )
        if script_match is None:
            msg = "scripts/run-zizmor.sh is missing ZIZMOR_VERSION"
            raise VersionAlignmentError(msg)
        if script_match.group(1) != zizmor_version:
            _raise_mismatch(
                "scripts/run-zizmor.sh ZIZMOR_VERSION",
                zizmor_version,
                script_match.group(1),
            )

    def test_mypy_additional_dependencies_match_pyproject(
        self: TestVersionAlignment,
    ) -> None:
        """Test pre-commit mypy extra deps match pyproject specifiers."""
        pyproject = _load_toml(PYPROJECT_PATH)
        declared = _requirement_map(pyproject["project"]["dependencies"])
        declared.update(
            _requirement_map(pyproject["project"]["optional-dependencies"]["dev"]),
        )
        actual = _pre_commit_additional_dependencies(_read_text(PRE_COMMIT_PATH))
        expected = {
            name: declared[name]
            for name in MYPY_ADDITIONAL_DEPENDENCIES
            if name in declared
        }
        if actual != expected:
            msg = (
                "pre-commit mypy additional_dependencies do not match "
                f"pyproject.toml: {actual} != {expected}"
            )
            raise VersionAlignmentError(msg)

    def test_docs_list_pyproject_specifiers(self: TestVersionAlignment) -> None:
        """Test installation and contributing docs list current specifiers."""
        pyproject = _load_toml(PYPROJECT_PATH)
        installation = _read_text(INSTALLATION_RST_PATH)
        contributing = _read_text(CONTRIBUTING_RST_PATH)
        uv_version = _makefile_value(_read_text(MAKEFILE_PATH), "UV_VERSION")

        runtime = _requirement_map(pyproject["project"]["dependencies"])
        for name, specifier in runtime.items():
            expected = f"**{name}** ({specifier})"
            if expected not in installation:
                msg = f"{INSTALLATION_RST_PATH.name} is missing {expected}"
                raise VersionAlignmentError(msg)

        dev = _requirement_map(pyproject["project"]["optional-dependencies"]["dev"])
        documented_dev = (
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "mypy",
            "ruff",
            "pre-commit",
        )
        for name in documented_dev:
            expected = f"**{name}** ({dev[name]})"
            if expected not in installation:
                msg = f"{INSTALLATION_RST_PATH.name} is missing {expected}"
                raise VersionAlignmentError(msg)

        uv_pin = f"uv=={uv_version}"
        if uv_pin not in installation:
            msg = f"{INSTALLATION_RST_PATH.name} is missing {uv_pin}"
            raise VersionAlignmentError(msg)
        if uv_pin not in contributing:
            msg = f"{CONTRIBUTING_RST_PATH.name} is missing {uv_pin}"
            raise VersionAlignmentError(msg)

    def test_python_versions_match(self: TestVersionAlignment) -> None:
        """Test declared Python versions match CI, docs, and tool targets."""
        pyproject = _load_toml(PYPROJECT_PATH)
        requires_python = pyproject["project"]["requires-python"].replace(" ", "")
        versions = _python_minor_versions(requires_python)
        default_version = versions[-1]
        _check_python_classifiers(pyproject["project"]["classifiers"], versions)
        _check_default_python_pins(default_version)
        _check_python_matrix_and_docs(versions)
        _check_type_checker_targets(pyproject, versions)
