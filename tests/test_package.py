"""Test suite for the openseries package."""

from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path
from re import match

from tomllib import load as toml_load


class PackageTestError(Exception):
    """Custom exception used for signaling test failures."""


class TestPackage:
    """class to test openseries packaging."""

    def test_metadata(self: TestPackage) -> None:
        """Test package metadata."""
        package_metadata = metadata("openseries")

        directory = Path(__file__).parent.parent
        pyproject_file = directory.joinpath("pyproject.toml")
        with pyproject_file.open(mode="rb") as pfile:
            data = toml_load(pfile)

        toml_version = data["project"]["version"]

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
            "^(>=3.10,<3.14)$",
            "^(Homepage, https://github.com/CaptorAB/openseries)$",
        ]

        for name, value in zip(attribute_names, expected_values, strict=True):
            if match(value, package_metadata[name]) is None:
                msg = (
                    f"Package metadata {name} not as "
                    f"expected: {package_metadata[name]}"
                )
                raise PackageTestError(msg)
