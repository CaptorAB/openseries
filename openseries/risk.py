"""
Value-at-Risk, Conditional-Value-at-Risk and drawdown functions.

Source:
https://github.com/pmorissette/ffn/blob/master/ffn/core.py
"""
import datetime as dt
from math import ceil
from typing import cast, List
from numpy import (
    Inf,
    isnan,
    maximum,
    mean,
    nan_to_num,
    quantile,
    sort,
)
from pandas import DataFrame, Series

from openseries.types import LiteralQuantileInterp


def cvar_down(data: DataFrame | Series | List[float], level: float = 0.95) -> float:
    """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

    Parameters
    ----------
    data: DataFrame | Series | List[float]
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


def var_down(
    data: DataFrame | Series | List[float],
    level: float = 0.95,
    interpolation: LiteralQuantileInterp = "lower",
) -> float:
    """Downside Value At Risk, "VaR". The equivalent of
    percentile.inc([...], 1-level) over returns in MS Excel \n
    https://www.investopedia.com/terms/v/var.asp

    Parameters
    ----------
    data: DataFrame | Series | List[float]
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


def drawdown_series(prices: DataFrame | Series) -> DataFrame | Series:
    """Calculates https://www.investopedia.com/terms/d/drawdown.asp
    This returns a series representing a drawdown. When the price is at all-time
    highs, the drawdown is 0. However, when prices are below high watermarks,
    the drawdown series = current / hwm - 1 The max drawdown can be obtained by
    simply calling .min() on the result (since the drawdown series is negative)
    Method ignores all gaps of NaN's in the price series.

    Parameters
    ----------
    prices: DataFrame | Series
        A timeserie of dates and values

    Returns
    -------
    DataFrame | Series
        A drawdown timeserie
    """
    drawdown = prices.copy()
    drawdown = drawdown.fillna(method="ffill")
    drawdown[isnan(drawdown)] = -Inf
    roll_max = maximum.accumulate(drawdown)
    drawdown = drawdown / roll_max - 1.0
    return drawdown


def drawdown_details(prices: DataFrame | Series) -> Series:
    """Details of the maximum drawdown

    Parameters
    ----------
    prices: DataFrame | Series
        A timeserie of dates and values

    Returns
    -------
    Series
        Max Drawdown
        Start of drawdown
        Date of bottom
        Days from start to bottom
        Average fall per day
    """

    mdd_date = (prices / prices.expanding(min_periods=1).max()).idxmin().values[0]
    mdate = dt.datetime.strptime(str(mdd_date)[:10], "%Y-%m-%d").date()
    maxdown = ((prices / prices.expanding(min_periods=1).max()).min() - 1).iloc[0]
    ddata = prices.copy()
    drwdwn = drawdown_series(ddata).loc[: cast(int, mdate)]
    drwdwn.sort_index(ascending=False, inplace=True)
    sdate = drwdwn[drwdwn == 0.0].idxmax().values[0]
    sdate = dt.datetime.strptime(str(sdate)[:10], "%Y-%m-%d").date()
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
