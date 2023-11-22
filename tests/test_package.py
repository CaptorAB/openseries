"""Test suite for the openseries package."""
from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path
from re import match
from unittest import TestCase

from toml import load as tomlload


class TestPackage(TestCase):

    """class to test openseries packaging."""

    def test_metadata(self: TestPackage) -> None:
        """Test package metadata."""
        package_metadata = metadata("openseries")

        directory = Path(__file__).resolve().parent.parent
        pyproject_file = directory.joinpath("pyproject.toml")
        toml_version = tomlload(pyproject_file)["tool"]["poetry"]["version"]

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
            (
                "^(Package for analyzing financial timeseries.|"
                "Package for simple financial time series analysis.)$"
            ),
            f"^({toml_version}|1.3.6)$",
            "https://github.com/CaptorAB/OpenSeries",
            "BSD-3-Clause",
            ">=3.9,<3.12",
        ]
        for name, value in zip(attribute_names, expected_values):
            if match(value, package_metadata[name]) is None:
                msg = (
                    f"Package metadata {name} not as "
                    f"expected: {package_metadata[name]}"
                )
                raise ValueError(msg)
