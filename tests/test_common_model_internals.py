"""Tests for internal _CommonModel helpers (coverage for shared base logic)."""

# ruff: noqa: SLF001 — exercises private base helpers intentionally.

from __future__ import annotations

import datetime as dt

import pytest
from numpy import float64
from pandas import DataFrame, MultiIndex, Series

from openseries._common_model import _CommonModel
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
