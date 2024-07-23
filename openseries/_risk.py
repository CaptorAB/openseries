"""Various risk related functions."""

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
    from .types import LiteralQuantileInterp  # pragma: no cover


def _cvar_down_calc(
    data: DataFrame | Series[float] | list[float],
    level: float = 0.95,
) -> float:
    """Calculate downside Conditional Value at Risk (CVaR).

    https://www.investopedia.com/terms/c/conditional_value_at_risk.asp.

    Parameters
    ----------
    data: DataFrame | Series[float] | list[float]
        The data to perform the calculation over
    level: float, default: 0.95
        The sought CVaR level

    Returns
    -------
    float
        Downside Conditional Value At Risk "CVaR"

    """
    if isinstance(data, DataFrame):
        clean = nan_to_num(data.iloc[:, 0])
    else:
        clean = nan_to_num(data)
    ret = clean[1:] / clean[:-1] - 1
    array = sort(ret)
    return cast(float, mean(array[: int(ceil(len(array) * (1 - level)))]))


def _var_down_calc(
    data: DataFrame | Series[float] | list[float],
    level: float = 0.95,
    interpolation: LiteralQuantileInterp = "lower",
) -> float:
    """Calculate downside Value At Risk (VaR).

    The equivalent of percentile.inc([...], 1-level) over returns in MS Excel
    https://www.investopedia.com/terms/v/var.asp.

    Parameters
    ----------
    data: DataFrame | Series[float] | list[float]
        The data to perform the calculation over
    level: float, default: 0.95
        The sought VaR level
    interpolation: LiteralQuantileInterp, default: "lower"
        type of interpolation in Pandas.DataFrame.quantile() function.

    Returns
    -------
    float
        Downside Value At Risk

    """
    if isinstance(data, DataFrame):
        clean = nan_to_num(data.iloc[:, 0])
    else:
        clean = nan_to_num(data)
    ret = clean[1:] / clean[:-1] - 1
    return cast(float, quantile(ret, 1 - level, method=interpolation))
