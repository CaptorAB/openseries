"""Test pull-request path filtering used by GitHub Actions."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "ci-pr-paths.sh"


class CiPrPathsError(Exception):
    """Raised when ci-pr-paths.sh does not behave as expected."""


def _run(
    *patterns: str,
    env: dict[str, str],
) -> str:
    bash = shutil.which("bash")
    if bash is None:
        msg = "bash executable not found"
        raise CiPrPathsError(msg)
    merged = os.environ.copy()
    merged.pop("GITHUB_OUTPUT", None)
    merged.update(env)
    completed = subprocess.run(  # noqa: S603
        [bash, str(SCRIPT), *patterns],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env=merged,
        text=True,
    )
    return completed.stdout.strip()


class TestCiPrPaths:
    """class to verify scripts/ci-pr-paths.sh output."""

    def test_non_pull_request_always_runs(self: TestCiPrPaths) -> None:
        """Test workflow_dispatch runs regardless of patterns."""
        output = _run("openseries/*", env={"GITHUB_EVENT_NAME": "workflow_dispatch"})
        if output != "run=true":
            msg = f"expected run=true, got {output!r}"
            raise CiPrPathsError(msg)

    def test_matching_python_path_runs(self: TestCiPrPaths) -> None:
        """Test a Python source change matches openseries/*."""
        output = _run(
            "openseries/*",
            "tests/*",
            env={
                "GITHUB_EVENT_NAME": "pull_request",
                "CI_PR_PATHS_FILES": "README.md\nopenseries/series.py",
            },
        )
        if output != "run=true":
            msg = f"expected run=true, got {output!r}"
            raise CiPrPathsError(msg)

    def test_docs_only_change_skips(self: TestCiPrPaths) -> None:
        """Test a docs-only PR does not match Python test paths."""
        output = _run(
            "openseries/*",
            "tests/*",
            "pyproject.toml",
            env={
                "GITHUB_EVENT_NAME": "pull_request",
                "CI_PR_PATHS_FILES": "docs/source/index.rst\ndocs/README.md",
            },
        )
        if output != "run=false":
            msg = f"expected run=false, got {output!r}"
            raise CiPrPathsError(msg)

    def test_missing_pr_context_runs(self: TestCiPrPaths) -> None:
        """Test a pull_request without API context fails open and runs."""
        output = _run("openseries/*", env={"GITHUB_EVENT_NAME": "pull_request"})
        if output != "run=true":
            msg = f"expected run=true, got {output!r}"
            raise CiPrPathsError(msg)
