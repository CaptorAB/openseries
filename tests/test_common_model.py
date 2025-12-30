"""Test suite for the openseries/fra_common_model.py module."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from pandas import DataFrame, Series

from openseries import OpenTimeSeries
from openseries._common_model import _calculate_time_factor, _get_base_column_data

if TYPE_CHECKING:  # pragma: no cover
    from openseries.frame import OpenFrame
    from openseries.owntypes import ValueType
    from openseries.simulation import ReturnSimulation


class CommonModelTestError(Exception):
    """Custom exception used for signaling test failures."""


class TestCommonModel:
    """class to run tests on the module _common_model.py."""

    seed: int
    seriesim: ReturnSimulation
    randomframe: OpenFrame
    randomseries: OpenTimeSeries
    random_properties: dict[str, dt.date | int | float]

    def test_get_base_column_data(self: TestCommonModel) -> None:
        """Test _get_base_column_data function."""
        frame = self.randomframe.from_deepcopy()
        earlier = dt.date(2010, 1, 1)
        later = dt.date(2019, 6, 30)

        first_column = cast("tuple[str, ValueType]", frame.tsdf.columns[0])
        data_tuple, item_tuple, label_tuple = _get_base_column_data(
            self=frame,
            base_column=first_column,
            earlier=earlier,
            later=later,
        )

        msg = "_get_base_column_data did not return Series for tuple column reference"
        if not isinstance(data_tuple, Series):
            raise CommonModelTestError(msg)

        if item_tuple != first_column:
            msg = (
                "_get_base_column_data item mismatch: "
                f"expected {first_column}, got {item_tuple}"
            )
            raise CommonModelTestError(msg)

        msg = (
            "_get_base_column_data did not return "
            "string label for tuple column reference"
        )
        if not isinstance(label_tuple, str):
            raise CommonModelTestError(msg)

        if label_tuple != first_column[0]:
            msg = (
                "_get_base_column_data label mismatch: "
                f"expected {first_column[0]}, got {label_tuple}"
            )
            raise CommonModelTestError(msg)

        # Test with different date ranges
        earlier_narrow = dt.date(2015, 6, 1)
        later_narrow = dt.date(2015, 6, 30)

        data_narrow, _, _ = _get_base_column_data(
            self=frame,
            base_column=first_column,
            earlier=earlier_narrow,
            later=later_narrow,
        )

        # Verify narrower date range returns fewer data points
        data_full, _, _ = _get_base_column_data(
            self=frame,
            base_column=first_column,
            earlier=earlier,
            later=later,
        )
        if len(data_narrow) > len(data_full):
            msg = (
                "Narrower date range returned more "
                f"data: {len(data_narrow)} > {len(data_full)}"
            )
            raise CommonModelTestError(msg)

        # Test with last column
        last_col_idx = len(frame.tsdf.columns) - 1
        data_last, item_last, label_last = _get_base_column_data(
            self=frame,
            base_column=cast(
                "tuple[str, ValueType]", frame.tsdf.columns[last_col_idx]
            ),
            earlier=earlier,
            later=later,
        )

        msg = "_get_base_column_data did not return Series for last column reference"
        if not isinstance(data_last, Series):
            raise CommonModelTestError(msg)

        last_column = cast("tuple[str, ValueType]", frame.tsdf.columns[last_col_idx])
        if item_last != last_column:
            msg = (
                "_get_base_column_data last column item mismatch: "
                f"expected {frame.tsdf.columns[last_col_idx]}, got {item_last}"
            )
            raise CommonModelTestError(msg)

        if label_last != frame.tsdf.columns[last_col_idx][0]:
            msg = (
                "_get_base_column_data last column label mismatch: "
                f"expected {frame.tsdf.columns[last_col_idx][0]}, got {label_last}"
            )
            raise CommonModelTestError(msg)

    def test_get_base_column_data_int(self: TestCommonModel) -> None:
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
            raise CommonModelTestError(msg)

        msg = (
            "_get_base_column_data did not return "
            "tuple item for integer column reference"
        )
        if not isinstance(item_int, tuple):
            raise CommonModelTestError(msg)

        two = 2
        if len(item_int) != two:
            msg = (
                "_get_base_column_data item tuple "
                f"length mismatch: expected {two}, got {len(item_int)}"
            )
            raise CommonModelTestError(msg)

        msg = (
            "_get_base_column_data did not return "
            "string label for integer column reference"
        )
        if not isinstance(label_int, str):
            raise CommonModelTestError(msg)

        if label_int != frame.tsdf.columns[0][0]:
            msg = (
                "_get_base_column_data label mismatch: "
                f"expected {frame.tsdf.columns[0][0]}, got {label_int}"
            )
            raise CommonModelTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match=r"base_column should be a tuple\[str, ValueType\] or an integer\.",
        ):
            _get_base_column_data(
                self=frame,
                base_column=cast("tuple[str, ValueType] | int", "invalid_column"),
                earlier=earlier,
                later=later,
            )

    def test_calculate_time_factor(self: TestCommonModel) -> None:
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
            raise CommonModelTestError(msg)

        time_factor_calc = _calculate_time_factor(
            data=frame.tsdf.iloc[:, 0],
            earlier=earlier,
            later=later,
            periods_in_a_year_fixed=None,
        )

        msg = f"Time factor should be a float, got {type(time_factor_calc)}"
        if not isinstance(time_factor_calc, float):
            raise CommonModelTestError(msg)
        if time_factor_calc <= 0:
            msg = f"Time factor should be positive, got {time_factor_calc}"
            raise CommonModelTestError(msg)

        mid_date = frame.tsdf.index[len(frame.tsdf.index) // 2]
        time_factor_half = _calculate_time_factor(
            data=frame.tsdf.iloc[:, 0],
            earlier=earlier,
            later=mid_date,
            periods_in_a_year_fixed=None,
        )
        msg = f"Time factor should be a float, got {type(time_factor_half)}"
        if not isinstance(time_factor_half, float):
            raise CommonModelTestError(msg)
        if time_factor_half <= 0:
            msg = f"Time factor should be positive, got {time_factor_half}"
            raise CommonModelTestError(msg)

    def test_outliers_opentimeseries(self: TestCommonModel) -> None:
        """Test outliers method with OpenTimeSeries."""
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        values = [100.0, 101.0, 150.0, 99.0, 50.0]

        series = OpenTimeSeries.from_arrays(
            dates=dates, values=values, name="Test Series"
        )

        outliers = series.outliers()
        msg = "outliers() should return a Series for OpenTimeSeries"
        if not isinstance(outliers, Series):
            raise CommonModelTestError(msg)

        outliers_low_threshold = series.outliers(threshold=1.0)
        msg = "Lower threshold should catch more outliers"
        if len(outliers_low_threshold) <= len(outliers):
            raise CommonModelTestError(msg)

        outliers_subset = series.outliers(
            threshold=1.0, from_date=dt.date(2023, 1, 1), to_date=dt.date(2023, 1, 3)
        )
        msg = "Date range should limit outliers to specified period"
        max_expected_outliers = 3
        if len(outliers_subset) > max_expected_outliers:
            raise CommonModelTestError(msg)

        outliers_recent = series.outliers(
            threshold=1.0,
            months_from_last=0,
        )
        msg = "months_from_last should limit outliers to recent period"
        if not isinstance(outliers_recent, Series):
            raise CommonModelTestError(msg)

    def test_outliers_openframe(self: TestCommonModel) -> None:
        """Test outliers method with OpenFrame."""
        frame = self.randomframe.from_deepcopy()

        outliers = frame.outliers()
        msg = "outliers() should return a DataFrame for OpenFrame"
        if not isinstance(outliers, DataFrame):
            raise CommonModelTestError(msg)

        outliers_low_threshold = frame.outliers(threshold=1.0)
        msg = "Lower threshold should potentially catch more outliers"
        if not isinstance(outliers_low_threshold, DataFrame):
            raise CommonModelTestError(msg)

        outliers_subset = frame.outliers(
            threshold=1.0, from_date=dt.date(2015, 1, 1), to_date=dt.date(2015, 12, 31)
        )
        msg = "Date range should limit outliers to specified period"
        if not isinstance(outliers_subset, DataFrame):
            raise CommonModelTestError(msg)

        outliers_recent = frame.outliers(threshold=1.0, months_from_last=12)
        msg = "months_from_last should limit outliers to recent period"
        if not isinstance(outliers_recent, DataFrame):
            raise CommonModelTestError(msg)

    def test_outliers_edge_cases(self: TestCommonModel) -> None:
        """Test outliers method edge cases."""
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
        values = [100.0, 100.1, 99.9, 100.05]

        series_no_outliers = OpenTimeSeries.from_arrays(
            dates=dates, values=values, name="No Outliers"
        )

        outliers = series_no_outliers.outliers(threshold=2.0)
        msg = "Series with no outliers should return empty Series"
        if not isinstance(outliers, Series) or len(outliers) != 0:
            raise CommonModelTestError(msg)

        outliers_high_threshold = series_no_outliers.outliers(threshold=10.0)
        msg = "Very high threshold should return empty Series"
        if (
            not isinstance(outliers_high_threshold, Series)
            or len(outliers_high_threshold) != 0
        ):
            raise CommonModelTestError(msg)

        series_single = OpenTimeSeries.from_arrays(
            dates=["2023-01-01"], values=[100.0], name="Single Point"
        )

        outliers_single = series_single.outliers()
        msg = "Single data point should return empty Series"
        if not isinstance(outliers_single, Series) or len(outliers_single) != 0:
            raise CommonModelTestError(msg)

        values_with_nan = [100.0, np.nan, 150.0, 99.0]
        series_with_nan = OpenTimeSeries.from_arrays(
            dates=dates, values=values_with_nan, name="With NaN"
        )

        outliers_with_nan = series_with_nan.outliers(threshold=1.0)
        msg = "Series with NaN should handle outliers correctly"
        if not isinstance(outliers_with_nan, Series):
            raise CommonModelTestError(msg)
