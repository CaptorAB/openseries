"""
Source:
https://github.com/pmorissette/ffn/blob/master/ffn/core.py
"""
from datetime import date, datetime
from math import ceil
from numpy import Inf, isnan, maximum, mean, nan_to_num, quantile, sort
from pandas import DataFrame, Series
from typing import List, Literal


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
    return float(mean(array[: int(ceil(len(array) * (1 - level)))]))


def var_down(
    data: DataFrame | Series | List[float],
    level: float = 0.95,
    interpolation: Literal[
        "linear", "lower", "higher", "midpoint", "nearest"
    ] = "lower",
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
    interpolation: Literal["linear", "lower", "higher",
                           "midpoint", "nearest"], default: "lower"
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
    result = quantile(ret, 1 - level, method=interpolation)
    return result


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

    # make a copy so that we don't modify original data
    drawdown = prices.copy()

    # Fill NaN's with previous values
    drawdown = drawdown.fillna(method="ffill")

    # Ignore problems with NaN's in the beginning
    drawdown[isnan(drawdown)] = -Inf

    # Rolling maximum
    roll_max = maximum.accumulate(drawdown)
    drawdown = drawdown / roll_max - 1.0
    return drawdown


def max_drawdown_date(prices: DataFrame | Series) -> date:
    """Date when maximum drawdown occurred

    Parameters
    ----------
    prices: DataFrame | Series
        A timeserie of dates and values

    Returns
    -------
    datetime.date
        Maximum drawdown date
    """

    mdd_date = (
        (prices / prices.expanding(min_periods=1).max())
        .idxmin()
        .values[0]
        .astype(datetime)
    )
    return datetime.fromtimestamp(mdd_date / 1e9).date()


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

    mdate = max_drawdown_date(prices)
    md = float((prices / prices.expanding(min_periods=1).max()).min() - 1)
    dd = prices.copy()
    drwdwn = drawdown_series(dd).loc[:mdate]
    drwdwn.sort_index(ascending=False, inplace=True)
    sdate = drwdwn[drwdwn == 0.0].idxmax().values[0].astype(datetime)
    sdate = datetime.fromtimestamp(sdate / 1e9).date()
    duration = (mdate - sdate).days
    ret_per_day = md / duration
    df = Series(
        data=[md, sdate, mdate, duration, ret_per_day],
        index=[
            "Max Drawdown",
            "Start of drawdown",
            "Date of bottom",
            "Days from start to bottom",
            "Average fall per day",
        ],
        name="Drawdown details",
    )
    return df
