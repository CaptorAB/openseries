"""Test suite for the openseries package."""
from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path
from re import match
from unittest import TestCase


class TestPackage(TestCase):

    """class to test openseries packaging."""

    def test_metadata(self: TestPackage) -> None:
        """Test package metadata."""
        package_metadata = metadata("openseries")

        directory = Path(__file__).resolve().parent.parent
        pyproject_file = directory.joinpath("pyproject.toml")
        with Path.open(pyproject_file, "r") as pfile:
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
            "^(Package for analyzing financial timeseries.)$",
            f"^({toml_version})$",
            "^(https://github.com/CaptorAB/openseries)$",
            "^(BSD-3-Clause)$",
            "^(>=3.9,<3.13)$",
        ]

        for name, value in zip(attribute_names, expected_values):
            if match(value, package_metadata[name]) is None:
                msg = (
                    f"Package metadata {name} not as "
                    f"expected: {package_metadata[name]}"
                )
                raise ValueError(msg)
