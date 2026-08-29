"""Test suite for the openseries package."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import zipfile
from contextlib import chdir
from importlib.metadata import metadata
from pathlib import Path
from re import match
from unittest.mock import patch

import pytest


class PackageTestError(Exception):
    """Custom exception used for signaling test failures."""


_PACKAGE_DATA_FILES = (
    "openseries/plotly_layouts.json",
    "openseries/plotly_captor_logo.json",
    "openseries/py.typed",
)


def _prepare_build_tree(build_dir: Path, project_root: Path) -> None:
    shutil.copytree(project_root / "openseries", build_dir / "openseries")
    for filename in ("pyproject.toml", "README.md", "LICENSE.md"):
        shutil.copy2(project_root / filename, build_dir / filename)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a wheel once for packaging tests."""
    project_root = Path(__file__).parent.parent
    tmp_path = tmp_path_factory.mktemp("packaging")
    build_dir = tmp_path / "project"
    dist_dir = tmp_path / "dist"
    build_dir.mkdir()
    dist_dir.mkdir()
    _prepare_build_tree(build_dir, project_root)

    with chdir(build_dir):
        build_meta = importlib.import_module("setuptools.build_meta")
        wheel_name = build_meta.build_wheel(str(dist_dir))
    if not isinstance(wheel_name, str):
        msg = f"Expected wheel filename string, got: {wheel_name!r}"
        raise PackageTestError(msg)
    return dist_dir / wheel_name


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
    def test_wheel_includes_package_data(
        self: TestPackage,
        built_wheel: Path,
    ) -> None:
        """Test wheel includes package data files required at runtime."""
        with zipfile.ZipFile(built_wheel) as wheel:
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
        built_wheel: Path,
        tmp_path: Path,
    ) -> None:
        """Test load_plotly_dict works from packaged wheel contents."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(built_wheel) as wheel:
            wheel.extractall(path=extract_dir)

        module_path = extract_dir / "openseries" / "load_plotly.py"
        spec = importlib.util.spec_from_file_location(
            "openseries_wheel_load_plotly",
            module_path,
        )
        if spec is None or spec.loader is None:
            msg = f"Failed to load module from wheel path: {module_path}"
            raise PackageTestError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, "_check_remote_file_existence", return_value=True):
            fig, _ = module.load_plotly_dict()
        if "config" not in fig or "layout" not in fig:
            msg = "load_plotly_dict failed from installed wheel: missing config/layout"
            raise PackageTestError(msg)
