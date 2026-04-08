"""Test suite for the openseries package."""

from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path
from re import match


class PackageTestError(Exception):
    """Custom exception used for signaling test failures."""


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
