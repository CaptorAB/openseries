"""Test suite for the openseries package.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

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
