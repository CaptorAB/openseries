"""
Value-at-Risk, Conditional-Value-at-Risk and drawdown functions.

Source:
https://github.com/pmorissette/ffn/blob/master/ffn/core.py
"""
from __future__ import annotations

import datetime as dt
from math import ceil
from typing import Optional, Union, cast

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
from scipy.stats import norm  # type: ignore[import-untyped,unused-ignore]

from openseries.datefixer import _get_calc_range
from openseries.types import DaysInYearType, LiteralQuantileInterp


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


def _var_implied_vol_and_target_func(
    data: DataFrame,
    level: float,
    target_vol: Optional[float] = None,
    min_leverage_local: float = 0.0,
    max_leverage_local: float = 99999.0,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    interpolation: LiteralQuantileInterp = "lower",
    periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    *,
    drift_adjust: bool = False,
) -> Union[float, Series[float]]:
    """
    Volatility implied from VaR or Target Weight.

    The function returns a position weight multiplier from the ratio between
    a VaR implied volatility and a given target volatility if the argument
    target_vol is provided. Otherwise the function returns the VaR implied
    volatility. Multiplier = 1.0 -> target met.

    Parameters
    ----------
    data: DataFrame
        Timeseries data
    level: float
        The sought VaR level
    target_vol: Optional[float]
        Target Volatility
    min_leverage_local: float, default: 0.0
        A minimum adjustment factor
    max_leverage_local: float, default: 99999.0
        A maximum adjustment factor
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date
    interpolation: LiteralQuantileInterp, default: "lower"
        type of interpolation in Pandas.DataFrame.quantile() function.
    periods_in_a_year_fixed : DaysInYearType, optional
        Allows locking the periods-in-a-year to simplify test cases and
        comparisons
    drift_adjust: bool, default: False
        An adjustment to remove the bias implied by the average return

    Returns
    -------
    Union[float, Pandas.Series[float]]
        Target volatility if target_vol is provided otherwise the VaR
        implied volatility.
    """
    earlier, later = _get_calc_range(
        data=data,
        months_offset=months_from_last,
        from_dt=from_date,
        to_dt=to_date,
    )
    if periods_in_a_year_fixed:
        time_factor = float(periods_in_a_year_fixed)
    else:
        fraction = (later - earlier).days / 365.25
        how_many = data.loc[cast(int, earlier) : cast(int, later)].count().iloc[0]
        time_factor = how_many / fraction
    if drift_adjust:
        imp_vol = (-sqrt(time_factor) / norm.ppf(level)) * (
            data.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))
            .quantile(1 - level, interpolation=interpolation)
            - data.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))
            .sum()
            / len(
                data.loc[cast(int, earlier) : cast(int, later)].pct_change(
                    fill_method=cast(str, None),
                ),
            )
        )
    else:
        imp_vol = (
            -sqrt(time_factor)
            * data.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))
            .quantile(1 - level, interpolation=interpolation)
            / norm.ppf(level)
        )

    if target_vol:
        result = imp_vol.apply(
            lambda x: max(min_leverage_local, min(target_vol / x, max_leverage_local)),
        )
        label = "Weight from target vol"
    else:
        result = imp_vol
        label = f"Imp vol from VaR {level:.0%}"

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        index=data.columns,
        name=label,
        dtype="float64",
    )
