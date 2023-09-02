"""Test suite for the openseries/common_model.py module."""
from __future__ import annotations

from typing import TypeVar
from unittest import TestCase

from pandas import DataFrame

from openseries.common_model import CommonModel

TypeTestCommonModel = TypeVar("TypeTestCommonModel", bound="TestCommonModel")


class TestCommonModel(TestCase):

    """class to run unittests on the module frame.py."""

    def test_valid_tsdf(self: TestCommonModel) -> None:
        """Test valid pandas.DataFrame property."""
        cm = CommonModel()
        self.assertIsInstance(cm.tsdf, DataFrame)
        self.assertTrue(cm.tsdf.empty)
