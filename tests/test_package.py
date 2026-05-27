"""Test suite for the openseries package."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import metadata
from pathlib import Path
from re import match

import pytest


class PackageTestError(Exception):
    """Custom exception used for signaling test failures."""


_PACKAGE_DATA_FILES = (
    "openseries/plotly_layouts.json",
    "openseries/plotly_captor_logo.json",
    "openseries/py.typed",
)


def _venv_executable(venv_dir: Path, name: str) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _prepare_build_tree(build_dir: Path, project_root: Path) -> None:
    shutil.copytree(project_root / "openseries", build_dir / "openseries")
    for filename in ("pyproject.toml", "README.md", "LICENSE.md"):
        shutil.copy2(project_root / filename, build_dir / filename)


def _uv_executable() -> str:
    uv_path = shutil.which("uv")
    if uv_path is None:
        msg = "uv executable not found on PATH"
        raise PackageTestError(msg)
    return uv_path


def _run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


class TestPackage:
    """class to test openseries packaging."""

    def test_metadata(self: TestPackage) -> None:
        """Test package metadata."""
        package_metadata = metadata("openseries")

        directory = Path(__file__).parent.parent
        pyproject_file = directory.joinpath("pyproject.toml")
        with pyproject_file.open(mode="r", encoding="utf-8") as pfile:
            lines = pfile.readlines()

        toml_version = lines[2].strip()[lines[2].strip().find('"') :].replace('"', "")

        attribute_names = [
            "Name",
            "Version",
            "Summary",
            "Author",
            "Requires-Python",
            "Project-URL",
        ]

        expected_values = [
            "^(openseries)$",
            f"^({toml_version})$",
            "^(Tools for analyzing financial timeseries.)$",
            "^(Martin Karrin)$",
            "^(>=3.11,<3.15)$",
            "^(Documentation, https://openseries.readthedocs.io/)$",
        ]

        for name, value in zip(attribute_names, expected_values, strict=True):
            if name == "Requires-Python":
                actual_specifiers = {
                    part.strip() for part in package_metadata[name].split(",")
                }
                expected_specifiers = {">=3.11", "<3.15"}
                if actual_specifiers != expected_specifiers:
                    msg = (
                        f"Package metadata {name} not as "
                        f"expected: {package_metadata[name]}"
                    )
                    raise PackageTestError(msg)
                continue
            if name == "Project-URL":
                project_urls = package_metadata.get_all("Project-URL") or []
                expected_url = "Documentation, https://openseries.readthedocs.io/"
                if expected_url not in project_urls:
                    msg = f"Package metadata {name} not as expected: {project_urls}"
                    raise PackageTestError(msg)
                continue
            if match(value, package_metadata[name]) is None:
                msg = (
                    f"Package metadata {name} not as "
                    f"expected: {package_metadata[name]}"
                )
                raise PackageTestError(msg)

    @pytest.mark.xdist_group(name="packaging")
    def test_wheel_includes_package_data(self: TestPackage, tmp_path: Path) -> None:
        """Test wheel includes package data files required at runtime."""
        project_root = Path(__file__).parent.parent
        build_dir = tmp_path / "project"
        dist_dir = tmp_path / "dist"
        build_dir.mkdir()
        dist_dir.mkdir()
        _prepare_build_tree(build_dir, project_root)

        _run_checked(
            [_uv_executable(), "build", "--out-dir", str(dist_dir)],
            cwd=build_dir,
        )

        wheel_files = list(dist_dir.glob("*.whl"))
        if len(wheel_files) != 1:
            msg = f"Expected one wheel file, found: {wheel_files}"
            raise PackageTestError(msg)

        with zipfile.ZipFile(wheel_files[0]) as wheel:
            wheel_names = set(wheel.namelist())
            missing_files = [
                filename
                for filename in _PACKAGE_DATA_FILES
                if filename not in wheel_names
            ]
            if missing_files:
                msg = f"Wheel missing package data files: {missing_files}"
                raise PackageTestError(msg)

    @pytest.mark.xdist_group(name="packaging")
    def test_load_plotly_dict_from_installed_wheel(
        self: TestPackage,
        tmp_path: Path,
    ) -> None:
        """Test load_plotly_dict works from an installed wheel."""
        project_root = Path(__file__).parent.parent
        build_dir = tmp_path / "project"
        dist_dir = tmp_path / "dist"
        venv_dir = tmp_path / "venv"
        build_dir.mkdir()
        dist_dir.mkdir()
        _prepare_build_tree(build_dir, project_root)

        _run_checked(
            [_uv_executable(), "build", "--out-dir", str(dist_dir)],
            cwd=build_dir,
        )

        wheel_files = list(dist_dir.glob("*.whl"))
        if len(wheel_files) != 1:
            msg = f"Expected one wheel file, found: {wheel_files}"
            raise PackageTestError(msg)

        _run_checked([sys.executable, "-m", "venv", str(venv_dir)])

        pip = _venv_executable(venv_dir, "pip")
        python = _venv_executable(venv_dir, "python")
        _run_checked([str(pip), "install", str(wheel_files[0])])

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(  # noqa: S603
            [
                str(python),
                "-c",
                (
                    "from openseries.load_plotly import load_plotly_dict; "
                    "fig, _ = load_plotly_dict(); "
                    "assert 'config' in fig and 'layout' in fig"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            msg = (
                "load_plotly_dict failed from installed wheel: "
                f"{result.stderr or result.stdout}"
            )
            raise PackageTestError(msg)
