"""Test suite for the openseries/fra_common_model.py module.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import datetime as dt
from typing import cast

import pytest
from pandas import Series

from openseries._common_model import _calculate_time_factor, _get_base_column_data

from .test_common_sim import CommonTestCase


class OpenFrameTestError(Exception):
    """Custom exception used for signaling test failures."""


class TestOpenFrame(CommonTestCase):
    """class to run tests on the module frame.py."""

    def test_get_base_column_data(self: TestOpenFrame) -> None:
        """Test _get_base_column_data function."""
        frame = self.randomframe.from_deepcopy()
        earlier = dt.date(2010, 1, 1)
        later = dt.date(2019, 6, 30)

        first_column = frame.tsdf.columns[0]
        data_tuple, item_tuple, label_tuple = _get_base_column_data(
            self=frame,
            base_column=first_column,  # type: ignore[arg-type]
            earlier=earlier,
            later=later,
        )

        msg = "_get_base_column_data did not return Series for tuple column reference"
        if not isinstance(data_tuple, Series):
            raise OpenFrameTestError(msg)

        if cast("str", item_tuple) != first_column:
            msg = (
                "_get_base_column_data item mismatch: "
                f"expected {first_column}, got {item_tuple}"
            )
            raise OpenFrameTestError(msg)

        msg = (
            "_get_base_column_data did not return "
            "string label for tuple column reference"
        )
        if not isinstance(label_tuple, str):
            raise OpenFrameTestError(msg)

        if label_tuple != first_column[0]:
            msg = (
                "_get_base_column_data label mismatch: "
                f"expected {first_column[0]}, got {label_tuple}"
            )
            raise OpenFrameTestError(msg)

        # Test with different date ranges
        earlier_narrow = dt.date(2015, 6, 1)
        later_narrow = dt.date(2015, 6, 30)

        data_narrow, _, _ = _get_base_column_data(
            self=frame,
            base_column=first_column,  # type: ignore[arg-type]
            earlier=earlier_narrow,
            later=later_narrow,
        )

        # Verify narrower date range returns fewer data points
        data_full, _, _ = _get_base_column_data(
            self=frame,
            base_column=first_column,  # type: ignore[arg-type]
            earlier=earlier,
            later=later,
        )
        if len(data_narrow) > len(data_full):
            msg = (
                "Narrower date range returned more "
                f"data: {len(data_narrow)} > {len(data_full)}"
            )
            raise OpenFrameTestError(msg)

        # Test with last column
        last_col_idx = len(frame.tsdf.columns) - 1
        data_last, item_last, label_last = _get_base_column_data(
            self=frame,
            base_column=frame.tsdf.columns[last_col_idx],  # type: ignore[arg-type]
            earlier=earlier,
            later=later,
        )

        msg = "_get_base_column_data did not return Series for last column reference"
        if not isinstance(data_last, Series):
            raise OpenFrameTestError(msg)

        if cast("str", item_last) != frame.tsdf.columns[last_col_idx]:
            msg = (
                "_get_base_column_data last column item mismatch: "
                f"expected {frame.tsdf.columns[last_col_idx]}, got {item_last}"
            )
            raise OpenFrameTestError(msg)

        if label_last != frame.tsdf.columns[last_col_idx][0]:
            msg = (
                "_get_base_column_data last column label mismatch: "
                f"expected {frame.tsdf.columns[last_col_idx][0]}, got {label_last}"
            )
            raise OpenFrameTestError(msg)

    def test_get_base_column_data_int(self: TestOpenFrame) -> None:
        """Test _get_base_column_data function with integer column reference."""
        frame = self.randomframe.from_deepcopy()
        earlier = dt.date(2010, 1, 1)
        later = dt.date(2019, 6, 30)

        data_int, item_int, label_int = _get_base_column_data(
            self=frame,
            base_column=0,
            earlier=earlier,
            later=later,
        )

        msg = (
            "_get_base_column_data did not return Series for integer column reference"
        )
        if not isinstance(data_int, Series):
            raise OpenFrameTestError(msg)

        msg = (
            "_get_base_column_data did not return "
            "tuple item for integer column reference"
        )
        if not isinstance(item_int, tuple):
            raise OpenFrameTestError(msg)

        two = 2
        if len(item_int) != two:
            msg = (
                "_get_base_column_data item tuple "
                f"length mismatch: expected {two}, got {len(item_int)}"
            )
            raise OpenFrameTestError(msg)

        msg = (
            "_get_base_column_data did not return "
            "string label for integer column reference"
        )
        if not isinstance(label_int, str):
            raise OpenFrameTestError(msg)

        if label_int != frame.tsdf.columns[0][0]:
            msg = (
                "_get_base_column_data label mismatch: "
                f"expected {frame.tsdf.columns[0][0]}, got {label_int}"
            )
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match=r"base_column should be a tuple\[str, ValueType\] or an integer\.",
        ):
            _get_base_column_data(
                self=frame,
                base_column="invalid_column",  # type: ignore[arg-type]
                earlier=earlier,
                later=later,
            )

    def test_calculate_time_factor(self: TestOpenFrame) -> None:
        """Test _calculate_time_factor function."""
        frame = self.randomframe
        earlier = frame.first_idx
        later = frame.last_idx
        expected_fixed_periods = 252.0

        time_factor_fixed = _calculate_time_factor(
            data=frame.tsdf.iloc[:, 0],
            earlier=earlier,
            later=later,
            periods_in_a_year_fixed=252,
        )
        if time_factor_fixed != expected_fixed_periods:
            msg = (
                f"Fixed periods should return the fixed value: "
                f"expected {expected_fixed_periods}, got {time_factor_fixed}"
            )
            raise OpenFrameTestError(msg)

        time_factor_calc = _calculate_time_factor(
            data=frame.tsdf.iloc[:, 0],
            earlier=earlier,
            later=later,
            periods_in_a_year_fixed=None,
        )

        msg = f"Time factor should be a float, got {type(time_factor_calc)}"
        if not isinstance(time_factor_calc, float):
            raise OpenFrameTestError(msg)
        if time_factor_calc <= 0:
            msg = f"Time factor should be positive, got {time_factor_calc}"
            raise OpenFrameTestError(msg)

        mid_date = frame.tsdf.index[len(frame.tsdf.index) // 2]
        time_factor_half = _calculate_time_factor(
            data=frame.tsdf.iloc[:, 0],
            earlier=earlier,
            later=mid_date,
            periods_in_a_year_fixed=None,
        )
        msg = f"Time factor should be a float, got {type(time_factor_half)}"
        if not isinstance(time_factor_half, float):
            raise OpenFrameTestError(msg)
        if time_factor_half <= 0:
            msg = f"Time factor should be positive, got {time_factor_half}"
            raise OpenFrameTestError(msg)
