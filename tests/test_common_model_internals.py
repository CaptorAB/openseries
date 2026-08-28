"""Tests for internal _CommonModel helpers (coverage for shared base logic)."""

# ruff: noqa: SLF001 — exercises private base helpers intentionally.

from __future__ import annotations

import datetime as dt

import pytest
from numpy import float64
from pandas import DataFrame, MultiIndex, Series

from openseries._common_model import _CommonModel, _create_distplot
from openseries.owntypes import ValueType

_EXPECTED_SCALAR = 0.25


class _CoerceSubject(_CommonModel[float]):
    """Subclass using base ``_coerce_result`` (no override)."""


def test_base_coerce_result_scalar_for_single_column() -> None:
    """Base ``_coerce_result`` squeezes to float when ``tsdf`` has one column."""
    cols = MultiIndex.from_tuples([("A", ValueType.RTRN)])
    tsdf = DataFrame(
        [[1.0]],
        columns=cols,
        index=[dt.date(2024, 1, 1)],
        dtype=float64,
    )
    obj = _CoerceSubject(tsdf=tsdf)
    res = obj._coerce_result(
        Series([_EXPECTED_SCALAR], index=obj.tsdf.columns, dtype=float64),
        "metric",
    )
    if res != _EXPECTED_SCALAR:
        msg = "Expected scalar float from base _coerce_result"
        raise AssertionError(msg)


def test_base_coerce_result_series_for_multi_column() -> None:
    """Base ``_coerce_result`` returns a Series for a multi-column ``tsdf``."""
    cols = MultiIndex.from_tuples(
        [("A", ValueType.RTRN), ("B", ValueType.RTRN)],
    )
    tsdf = DataFrame(
        [[1.0, 2.0]],
        columns=cols,
        index=[dt.date(2024, 1, 1)],
        dtype=float64,
    )
    obj = _CoerceSubject(tsdf=tsdf)
    res = obj._coerce_result(
        Series([0.1, 0.2], index=obj.tsdf.columns, dtype=float64),
        "metric",
    )
    if not isinstance(res, Series):
        msg = "Expected Series from base _coerce_result for multi-column frame"
        raise TypeError(msg)
    if list(res) != [0.1, 0.2]:
        msg = "Series values from _coerce_result do not match input"
        raise AssertionError(msg)


def test_get_or_set_countries_set_raises_without_constituents() -> None:
    """Set path raises when there is no ``countries`` field nor constituents."""
    raw = _CommonModel[float].model_construct(tsdf=DataFrame(dtype="float64"))
    with pytest.raises(TypeError, match="Cannot set countries without constituents"):
        raw._get_or_set_countries("SE")


def test_get_or_set_countries_get_raises_without_constituents() -> None:
    """Get path raises when there is no ``countries`` field nor constituents."""
    raw = _CommonModel[float].model_construct(tsdf=DataFrame(dtype="float64"))
    with pytest.raises(TypeError, match="Cannot get countries without constituents"):
        raw._get_or_set_countries(None)


def _distplot_sample_series() -> Series[float]:
    return Series([1.0, 1.2, 1.1, 2.0, 2.1, 1.9, 3.0, 2.8], dtype=float64)


def test_create_distplot_kde_without_rug() -> None:
    """KDE lines plot emits one legend scatter trace and no rug axis."""
    fig = _create_distplot(
        hist_data=[_distplot_sample_series()],
        group_labels=["alpha"],
        curve_type="kde",
        histnorm="probability",
        show_rug=False,
    )
    if len(fig.data) != 1:
        msg = f"Expected a single curve trace, got {len(fig.data)}"
        raise AssertionError(msg)
    trace = fig.data[0].to_plotly_json()
    if trace["type"] != "scatter" or trace["mode"] != "lines":
        msg = f"Unexpected curve trace: type={trace['type']} mode={trace['mode']}"
        raise AssertionError(msg)
    if trace["name"] != "alpha":
        msg = f"Unexpected trace name: {trace['name']}"
        raise AssertionError(msg)
    if "yaxis2" in fig.to_dict()["layout"]:
        msg = "Rug y-axis should be absent when show_rug is False"
        raise AssertionError(msg)


def test_create_distplot_normal_with_rug() -> None:
    """Normal curve plus rug uses a secondary axis and hides the rug legend."""
    labels = ["alpha", "beta"]
    fig = _create_distplot(
        hist_data=[_distplot_sample_series(), _distplot_sample_series() + 1.0],
        group_labels=labels,
        curve_type="normal",
        histnorm="probability density",
        show_rug=True,
    )
    expected_traces = len(labels) * 2
    if len(fig.data) != expected_traces:
        msg = f"Expected two curve traces and two rug traces, got {len(fig.data)}"
        raise AssertionError(msg)
    curve_names = [trace["name"] for trace in fig.data[:2]]
    if curve_names != labels:
        msg = f"Curve labels mismatch: {curve_names}"
        raise AssertionError(msg)
    rug = fig.data[2].to_plotly_json()
    if rug["mode"] != "markers" or rug["showlegend"] is not False:
        msg = "Rug traces must be unmarked in the legend"
        raise AssertionError(msg)
    if "yaxis2" not in fig.to_dict()["layout"]:
        msg = "Rug y-axis should be present when show_rug is True"
        raise AssertionError(msg)
