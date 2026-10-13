"""Test pull-request path filtering used by GitHub Actions."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


class CiPrPathsError(Exception):
    """Raised when ci-pr-paths.sh does not behave as expected."""


def _run_script(patterns: list[str], extra: dict[str, str]) -> str:
    bash = shutil.which("bash")
    if bash is None:
        msg = "bash was not found on PATH"
        raise CiPrPathsError(msg)
    env = os.environ.copy()
    env.pop("GITHUB_OUTPUT", None)
    env.update(extra)
    completed = subprocess.run(  # noqa: S603
        [bash, "scripts/ci-pr-paths.sh", *patterns],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        text=True,
    )
    return completed.stdout.strip()


class TestCiPrPaths:
    """class to verify scripts/ci-pr-paths.sh output."""

    def test_non_pull_request_always_runs(self: TestCiPrPaths) -> None:
        """Test workflow_dispatch runs regardless of patterns."""
        output = _run_script(
            ["openseries/*"],
            {"GITHUB_EVENT_NAME": "workflow_dispatch"},
        )
        if output != "run=true":
            msg = f"expected run=true, got {output!r}"
            raise CiPrPathsError(msg)

    def test_matching_python_path_runs(self: TestCiPrPaths) -> None:
        """Test a Python source change matches openseries/*."""
        output = _run_script(
            ["openseries/*", "tests/*"],
            {
                "GITHUB_EVENT_NAME": "pull_request",
                "CI_PR_PATHS_FILES": "README.md\nopenseries/series.py",
            },
        )
        if output != "run=true":
            msg = f"expected run=true, got {output!r}"
            raise CiPrPathsError(msg)

    def test_docs_only_change_skips(self: TestCiPrPaths) -> None:
        """Test a docs-only PR does not match Python test paths."""
        output = _run_script(
            ["openseries/*", "tests/*", "pyproject.toml"],
            {
                "GITHUB_EVENT_NAME": "pull_request",
                "CI_PR_PATHS_FILES": "docs/source/index.rst\ndocs/README.md",
            },
        )
        if output != "run=false":
            msg = f"expected run=false, got {output!r}"
            raise CiPrPathsError(msg)

    def test_missing_pr_context_runs(self: TestCiPrPaths) -> None:
        """Test a pull_request without API context fails open and runs."""
        output = _run_script(
            ["openseries/*"],
            {"GITHUB_EVENT_NAME": "pull_request"},
        )
        if output != "run=true":
            msg = f"expected run=true, got {output!r}"
            raise CiPrPathsError(msg)
