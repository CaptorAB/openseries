"""Test suite for the openseries package."""

from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path
from re import match


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
            "Summary",
            "Version",
            "Home-page",
            "License",
            "Requires-Python",
        ]

        expected_values = [
            "^(openseries)$",
            "^(Tools for analyzing financial timeseries.)$",
            f"^({toml_version})$",
            "^(https://github.com/CaptorAB/openseries)$",
            "^(BSD-3-Clause)$",
            "^(>=3.10,<3.14)$",
        ]

        for name, value in zip(attribute_names, expected_values, strict=False):
            if match(value, package_metadata[name]) is None:
                msg = (
                    f"Package metadata {name} not as "
                    f"expected: {package_metadata[name]}"
                )
                raise ValueError(msg)
