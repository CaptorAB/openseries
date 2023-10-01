"""
Value-at-Risk, Conditional-Value-at-Risk and drawdown functions.

Source:
https://github.com/pmorissette/ffn/blob/master/ffn/core.py
"""
from __future__ import annotations

import datetime as dt
from math import ceil
from typing import Union, cast

from numpy import (
    Inf,
    NaN,
    divide,
    float64,
    isinf,
    isnan,
    maximum,
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


def cvar_down_calc(
    data: Union[DataFrame, Series[type[float]], list[float]],
    level: float = 0.95,
) -> float:
    """
    Calculate downside Conditional Value at Risk (CVaR).

    https://www.investopedia.com/terms/c/conditional_value_at_risk.asp.

    Parameters
    ----------
    data: Union[DataFrame, Series[type[float]], list[float]]
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


def var_down_calc(
    data: Union[DataFrame, Series[type[float]], list[float]],
    level: float = 0.95,
    interpolation: LiteralQuantileInterp = "lower",
) -> float:
    """
    Calculate downside Value At Risk (VaR).

    The equivalent of percentile.inc([...], 1-level) over returns in MS Excel
    https://www.investopedia.com/terms/v/var.asp.

    Parameters
    ----------
    data: Union[DataFrame, Series[type[float]], list[float]]
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


def drawdown_series(
    prices: Union[DataFrame, Series[type[float]]],
) -> DataFrame:
    """
    Convert series into a maximum drawdown series.

    Calculates https://www.investopedia.com/terms/d/drawdown.asp
    This returns a series representing a drawdown. When the price is at all-time
    highs, the drawdown is 0. However, when prices are below high watermarks,
    the drawdown series = current / hwm - 1 The max drawdown can be obtained by
    simply calling .min() on the result (since the drawdown series is negative)
    Method ignores all gaps of NaN's in the price series.

    Parameters
    ----------
    prices: Union[DataFrame, Series[type[float]]]
        A timeserie of dates and values

    Returns
    -------
    DataFrame
        A drawdown timeserie
    """
    drawdown = prices.copy()
    drawdown = drawdown.ffill()
    drawdown[isnan(drawdown)] = -Inf
    roll_max = maximum.accumulate(drawdown)
    return DataFrame(drawdown / roll_max - 1.0)


def drawdown_details(
    prices: Union[DataFrame, Series[type[float]]],
    min_periods: int = 1,
) -> Series[type[float]]:
    """
    Details of the maximum drawdown.

    Parameters
    ----------
    prices: Union[DataFrame, Series[type[float]]]
        A timeserie of dates and values
    min_periods: int, default: 1
        Smallest number of observations to use to find the maximum drawdown

    Returns
    -------
    Series[type[float]]
        Max Drawdown
        Start of drawdown
        Date of bottom
        Days from start to bottom
        Average fall per day
    """
    zero: float = 0.0
    mdd_date = cast(
        Series,  # type: ignore[type-arg]
        (prices / prices.expanding(min_periods=min_periods).max()).idxmin(),
    ).to_numpy()[0]
    mdate = (
        dt.datetime.strptime(str(mdd_date)[:10], "%Y-%m-%d")
        .replace(tzinfo=dt.timezone.utc)
        .date()
    )
    maxdown = (
        (prices / prices.expanding(min_periods=min_periods).max()).min() - 1
    ).iloc[0]
    ddata = prices.copy()
    drwdwn = drawdown_series(ddata).loc[: cast(int, mdate)]
    drwdwn = drwdwn.sort_index(ascending=False)
    sdate = Series(drwdwn[drwdwn == zero].idxmax()).to_numpy()[0]
    sdate = (
        dt.datetime.strptime(str(sdate)[:10], "%Y-%m-%d")
        .replace(tzinfo=dt.timezone.utc)
        .date()
    )
    duration = (mdate - sdate).days
    ret_per_day = maxdown / duration

    return Series(
        data=[maxdown, sdate, mdate, duration, ret_per_day],
        index=[
            "Max Drawdown",
            "Start of drawdown",
            "Date of bottom",
            "Days from start to bottom",
            "Average fall per day",
        ],
        name="Drawdown details",
    )


def ewma_calc(
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


def calc_inv_vol_weights(returns: DataFrame) -> NDArray[float64]:
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
