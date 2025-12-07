"""Functions calculating risk measures."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, cast

from numpy import (
    mean,
    nan_to_num,
    quantile,
    sort,
)
from pandas import DataFrame, Series

if TYPE_CHECKING:
    from .owntypes import LiteralQuantileInterp  # pragma: no cover


def _cvar_down_calc(
    data: DataFrame | Series[float] | list[float],
    level: float = 0.95,
) -> float:
    """Calculate downside Conditional Value at Risk (CVaR).

    Reference: https://www.investopedia.com/terms/c/conditional_value_at_risk.asp.

    Args:
        data: The data to perform the calculation over.
        level: The sought CVaR level. Defaults to 0.95.

    Returns:
        Downside Conditional Value At Risk "CVaR".
    """
    if isinstance(data, DataFrame):
        clean = nan_to_num(data.iloc[:, 0])
    else:
        clean = nan_to_num(data)
    ret = clean[1:] / clean[:-1] - 1
    array = sort(ret)
    return cast("float", mean(array[: ceil(len(array) * (1 - level))]))


def _var_down_calc(
    data: DataFrame | Series[float] | list[float],
    level: float = 0.95,
    interpolation: LiteralQuantileInterp = "lower",
) -> float:
    """Calculate downside Value At Risk (VaR).

    The equivalent of percentile.inc([...], 1-level) over returns in MS Excel.

    Reference: https://www.investopedia.com/terms/v/var.asp.

    Args:
        data: The data to perform the calculation over.
        level: The sought VaR level. Defaults to 0.95.
        interpolation: Type of interpolation in Pandas.DataFrame.quantile() function.
            Defaults to "lower".

    Returns:
        Downside Value At Risk.
    """
    if isinstance(data, DataFrame):
        clean = nan_to_num(data.iloc[:, 0])
    else:
        clean = nan_to_num(data)
    ret = clean[1:] / clean[:-1] - 1
    return cast("float", quantile(ret, 1 - level, method=interpolation))
