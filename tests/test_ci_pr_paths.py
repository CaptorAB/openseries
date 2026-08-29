"""Test pull-request path filtering used by GitHub Actions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


class CiPrPathsError(Exception):
    """Raised when ci-pr-paths.sh does not behave as expected."""


def _env(extra: dict[str, str]) -> dict[str, str]:
    merged = os.environ.copy()
    merged.pop("GITHUB_OUTPUT", None)
    merged.update(extra)
    return merged


class TestCiPrPaths:
    """class to verify scripts/ci-pr-paths.sh output."""

    def test_non_pull_request_always_runs(self: TestCiPrPaths) -> None:
        """Test workflow_dispatch runs regardless of patterns."""
        completed = subprocess.run(
            ["/usr/bin/env", "bash", "scripts/ci-pr-paths.sh", "openseries/*"],
            check=True,
            capture_output=True,
            cwd=ROOT,
            env=_env({"GITHUB_EVENT_NAME": "workflow_dispatch"}),
            text=True,
        )
        if completed.stdout.strip() != "run=true":
            msg = f"expected run=true, got {completed.stdout.strip()!r}"
            raise CiPrPathsError(msg)

    def test_matching_python_path_runs(self: TestCiPrPaths) -> None:
        """Test a Python source change matches openseries/*."""
        completed = subprocess.run(
            [
                "/usr/bin/env",
                "bash",
                "scripts/ci-pr-paths.sh",
                "openseries/*",
                "tests/*",
            ],
            check=True,
            capture_output=True,
            cwd=ROOT,
            env=_env(
                {
                    "GITHUB_EVENT_NAME": "pull_request",
                    "CI_PR_PATHS_FILES": "README.md\nopenseries/series.py",
                }
            ),
            text=True,
        )
        if completed.stdout.strip() != "run=true":
            msg = f"expected run=true, got {completed.stdout.strip()!r}"
            raise CiPrPathsError(msg)

    def test_docs_only_change_skips(self: TestCiPrPaths) -> None:
        """Test a docs-only PR does not match Python test paths."""
        completed = subprocess.run(
            [
                "/usr/bin/env",
                "bash",
                "scripts/ci-pr-paths.sh",
                "openseries/*",
                "tests/*",
                "pyproject.toml",
            ],
            check=True,
            capture_output=True,
            cwd=ROOT,
            env=_env(
                {
                    "GITHUB_EVENT_NAME": "pull_request",
                    "CI_PR_PATHS_FILES": "docs/source/index.rst\ndocs/README.md",
                }
            ),
            text=True,
        )
        if completed.stdout.strip() != "run=false":
            msg = f"expected run=false, got {completed.stdout.strip()!r}"
            raise CiPrPathsError(msg)

    def test_missing_pr_context_runs(self: TestCiPrPaths) -> None:
        """Test a pull_request without API context fails open and runs."""
        completed = subprocess.run(
            ["/usr/bin/env", "bash", "scripts/ci-pr-paths.sh", "openseries/*"],
            check=True,
            capture_output=True,
            cwd=ROOT,
            env=_env({"GITHUB_EVENT_NAME": "pull_request"}),
            text=True,
        )
        if completed.stdout.strip() != "run=true":
            msg = f"expected run=true, got {completed.stdout.strip()!r}"
            raise CiPrPathsError(msg)
