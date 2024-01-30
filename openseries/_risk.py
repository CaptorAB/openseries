"""Various risk related functions."""
from __future__ import annotations

from math import ceil
from typing import Union, cast

from numpy import (
    NaN,
    divide,
    float64,
    isinf,
    mean,
    nan_to_num,
    quantile,
    sort,
    sqrt,
    square,
    std,
)
from numpy.typing import NDArray
from pandas import DataFrame, Series

from openseries.types import LiteralQuantileInterp


def _cvar_down_calc(
    data: Union[DataFrame, Series[float], list[float]],
    level: float = 0.95,
) -> float:
    """
    Calculate downside Conditional Value at Risk (CVaR).

    https://www.investopedia.com/terms/c/conditional_value_at_risk.asp.

    Parameters
    ----------
    data: Union[DataFrame, Series[float], list[float]]
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
    data: Union[DataFrame, Series[float], list[float]],
    level: float = 0.95,
    interpolation: LiteralQuantileInterp = "lower",
) -> float:
    """
    Calculate downside Value At Risk (VaR).

    The equivalent of percentile.inc([...], 1-level) over returns in MS Excel
    https://www.investopedia.com/terms/v/var.asp.

    Parameters
    ----------
    data: Union[DataFrame, Series[float], list[float]]
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


def _ewma_calc(
    reeturn: float,
    prev_ewma: float,
    time_factor: float,
    lmbda: float = 0.94,
) -> float:
    """
    Calculate Exponentially Weighted Moving Average volatility.

    Parameters
    ----------
    reeturn : float
        Return value
    prev_ewma : float
        Previous EWMA volatility value
    time_factor : float
        Scaling factor to annualize
    lmbda: float, default: 0.94
        Scaling factor to determine weighting.

    Returns
    -------
    float
        EWMA volatility value

    """
    return cast(
        float,
        sqrt(square(reeturn) * time_factor * (1 - lmbda) + square(prev_ewma) * lmbda),
    )


def _calc_inv_vol_weights(returns: DataFrame) -> NDArray[float64]:
    """
    Calculate weights proportional to inverse volatility.

    Source: https://github.com/pmorissette/ffn.
    Function copied here because of FutureWarning from pandas ^2.1.0

    Parameters
    ----------
    returns: pandas.DataFrame
        returns data

    Returns
    -------
    NDArray[float64]
        Calculated weights

    """
    vol = divide(1.0, std(returns, axis=0, ddof=1))
    vol[isinf(vol)] = NaN
    volsum = vol.sum()
    return cast(NDArray[float64], divide(vol, volsum))
