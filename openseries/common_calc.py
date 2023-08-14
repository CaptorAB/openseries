"""
Defining common calculation functions
"""
import datetime as dt
from typing import cast, Optional, Union
from math import ceil
from numpy import cumprod, sqrt
from pandas import DataFrame, DatetimeIndex, Series
from scipy.stats import kurtosis, norm, skew

from openseries.common_tools import get_calc_range
from openseries.types import LiteralQuantileInterp


def calc_arithmetic_ret(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    periods_in_a_year_fixed: Optional[int] = None,
) -> Union[float, Series]:
    """https://www.investopedia.com/terms/a/arithmeticmean.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date
    periods_in_a_year_fixed : int, optional
        Allows locking the periods-in-a-year to simplify test cases and
        comparisons

    Returns
    -------
    Union[float, Pandas.Series]
        Annualized arithmetic mean of returns
    """

    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    if periods_in_a_year_fixed:
        time_factor = float(periods_in_a_year_fixed)
    else:
        fraction = (later - earlier).days / 365.25
        how_many = data.loc[
            cast(int, earlier) : cast(int, later), data.columns.values[0]
        ].count()
        time_factor = how_many / fraction

    result = (
        data.loc[cast(int, earlier) : cast(int, later)].pct_change().mean()
        * time_factor
    )

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        name="Arithmetic return",
        dtype="float64",
    )


def calc_cvar_down(
    data: DataFrame,
    level: float = 0.95,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
) -> Union[float, Series]:
    """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    level: float, default: 0.95
        The sought CVaR level
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date

    Returns
    -------
    Union[float, Pandas.Series]
        Downside Conditional Value At Risk "CVaR"
    """

    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    cvar_df = data.loc[cast(int, earlier) : cast(int, later)].copy(deep=True)
    result = [
        cvar_df.loc[:, x]
        .pct_change()
        .sort_values()
        .iloc[: int(ceil((1 - level) * cvar_df.loc[:, x].pct_change().count()))]
        .mean()
        for x in data
    ]
    if data.shape[1] == 1:
        return float(result[0])
    return Series(
        data=result,
        index=data.columns,
        name=f"CVaR {level:.1%}",
        dtype="float64",
    )


def calc_downside_deviation(
    data: DataFrame,
    min_accepted_return: float = 0.0,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    periods_in_a_year_fixed: Optional[int] = None,
) -> Union[float, Series]:
    """The standard deviation of returns that are below a Minimum Accepted
    Return of zero.
    It is used to calculate the Sortino Ratio \n
    https://www.investopedia.com/terms/d/downside-deviation.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    min_accepted_return : float, optional
        The annualized Minimum Accepted Return (MAR)
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date
    periods_in_a_year_fixed : int, optional
        Allows locking the periods-in-a-year to simplify test cases and
        comparisons

    Returns
    -------
    Union[float, Pandas.Series]
        Downside deviation
    """

    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    how_many = (
        data.loc[cast(int, earlier) : cast(int, later)]
        .pct_change()
        .count(numeric_only=True)
    )
    if periods_in_a_year_fixed:
        time_factor = periods_in_a_year_fixed
    else:
        fraction = (later - earlier).days / 365.25
        time_factor = how_many / fraction

    dddf = (
        data.loc[cast(int, earlier) : cast(int, later)]
        .pct_change()
        .sub(min_accepted_return / time_factor)
    )

    result = sqrt((dddf[dddf < 0.0] ** 2).sum() / how_many) * sqrt(time_factor)

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        index=data.columns,
        name="Downside deviation",
        dtype="float64",
    )


def calc_geo_ret(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
) -> Union[float, Series]:
    """https://www.investopedia.com/terms/c/cagr.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date

    Returns
    -------
    Union[float, Pandas.Series]
        Compounded Annual Growth Rate (CAGR)
    """
    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    fraction = (later - earlier).days / 365.25

    if (
        0.0 in data.loc[earlier].tolist()
        or data.loc[[earlier, later]].lt(0.0).any().any()
    ):
        raise ValueError(
            "Geometric return cannot be calculated due to an initial "
            "value being zero or a negative value."
        )

    result = (data.iloc[-1] / data.iloc[0]) ** (1 / fraction) - 1

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        index=data.columns,
        name="Geometric return",
        dtype="float64",
    )


def calc_kurtosis(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
) -> Union[float, Series]:
    """https://www.investopedia.com/terms/k/kurtosis.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date

    Returns
    -------
    Union[float, Pandas.Series]
        Kurtosis of the return distribution
    """

    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    result = kurtosis(
        data.loc[cast(int, earlier) : cast(int, later)].pct_change(),
        fisher=True,
        bias=True,
        nan_policy="omit",
    )

    if data.shape[1] == 1:
        return float(result[0])
    return Series(
        data=result,
        index=data.columns,
        name="Kurtosis",
        dtype="float64",
    )


def calc_max_drawdown(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    min_periods: int = 1,
) -> Union[float, Series]:
    """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date
    min_periods: int, default: 1
        Smallest number of observations to use to find the maximum drawdown

    Returns
    -------
    Union[float, Pandas.Series]
        Maximum drawdown without any limit on date range
    """
    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    result = (
        data.loc[cast(int, earlier) : cast(int, later)]
        / data.loc[cast(int, earlier) : cast(int, later)]
        .expanding(min_periods=min_periods)
        .max()
    ).min() - 1
    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        index=data.columns,
        name="Max Drawdown",
        dtype="float64",
    )


def calc_max_drawdown_cal_year(data: DataFrame) -> Union[float, Series]:
    """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

    Returns
    -------
    Union[float, Pandas.Series]
        Maximum drawdown in a single calendar year.
    """
    years = [d.year for d in data.index]
    result = (
        data.groupby(years)
        .apply(
            lambda prices: (prices / prices.expanding(min_periods=1).max()).min() - 1
        )
        .min()
    )
    result.name = "Max Drawdown in cal yr"
    result = result.astype("float64")
    if data.shape[1] == 1:
        return float(result.iloc[0])
    return result


def calc_positive_share(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
) -> Union[float, Series]:
    """The share of percentage changes that are greater than zero

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date

    Returns
    -------
    Union[float, Pandas.Series]
        The share of percentage changes that are greater than zero
    """
    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    pos = (
        data.loc[cast(int, earlier) : cast(int, later)]
        .pct_change()[1:][
            data.loc[cast(int, earlier) : cast(int, later)].pct_change()[1:] > 0.0
        ]
        .count()
    )
    tot = data.loc[cast(int, earlier) : cast(int, later)].pct_change()[1:].count()
    result = pos / tot
    result.name = "Positive Share"
    result = result.astype("float64")
    if data.shape[1] == 1:
        return float(result.iloc[0])
    return result


def calc_ret_vol_ratio(
    data: DataFrame,
    riskfree_rate: Optional[float] = None,
    riskfree_column: int = -1,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    periods_in_a_year_fixed: Optional[int] = None,
) -> Union[float, Series]:
    """The ratio of annualized arithmetic mean of returns and annualized
    volatility or, if riskfree return provided, Sharpe ratio calculated
    as ( geometric return - risk-free return ) / volatility. The latter ratio
    implies that the riskfree asset has zero volatility. \n
    https://www.investopedia.com/terms/s/sharperatio.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    riskfree_rate : float, optional
        The return of the zero volatility asset used to calculate Sharpe ratio
    riskfree_column : int, default: -1
        The return of the zero volatility asset used to calculate Sharpe ratio
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date
    periods_in_a_year_fixed : int, optional
        Allows locking the periods-in-a-year to simplify test cases and
        comparisons

    Returns
    -------
    Union[float, Pandas.Series]
        Ratio of the annualized arithmetic mean of returns and annualized
        volatility or,
        if risk-free return provided, Sharpe ratio
    """

    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    how_many = data.loc[cast(int, earlier) : cast(int, later)].iloc[:, 0].count()
    fraction = (later - earlier).days / 365.25

    if periods_in_a_year_fixed:
        time_factor = periods_in_a_year_fixed
    else:
        time_factor = how_many / fraction

    ratios = []
    if riskfree_rate is None:
        if isinstance(riskfree_column, int):
            riskfree = data.loc[cast(int, earlier) : cast(int, later)].iloc[
                :, riskfree_column
            ]
            riskfree_item = data.iloc[:, riskfree_column].name
        else:
            raise ValueError("base_column argument should be an integer.")

        for item in data:
            if item == riskfree_item:
                ratios.append(0.0)
            else:
                longdf = data.loc[cast(int, earlier) : cast(int, later)].loc[:, item]
                ret = float(longdf.pct_change().mean() * time_factor)
                riskfree_ret = float(riskfree.pct_change().mean() * time_factor)
                vol = float(longdf.pct_change().std() * sqrt(time_factor))
                ratios.append((ret - riskfree_ret) / vol)
    else:
        for item in data:
            longdf = data.loc[cast(int, earlier) : cast(int, later)].loc[:, item]
            ret = float(longdf.pct_change().mean() * time_factor)
            vol = float(longdf.pct_change().std() * sqrt(time_factor))
            ratios.append((ret - riskfree_rate) / vol)

    if data.shape[1] == 1:
        return ratios[0]
    return Series(
        data=ratios,
        index=data.columns,
        name="Return vol ratio",
        dtype="float64",
    )


def calc_skew(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
) -> Union[float, Series]:
    """https://www.investopedia.com/terms/s/skewness.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date

    Returns
    -------
    Union[float, Pandas.Series]
        Skew of the return distribution
    """
    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    result = skew(
        a=data.loc[cast(int, earlier) : cast(int, later)].pct_change().values,
        bias=True,
        nan_policy="omit",
    )

    if data.shape[1] == 1:
        return float(result[0])
    return Series(
        data=result,
        index=data.columns,
        name="Skew",
        dtype="float64",
    )


def calc_sortino_ratio(
    data: DataFrame,
    riskfree_rate: Optional[float] = None,
    riskfree_column: int = -1,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    periods_in_a_year_fixed: Optional[int] = None,
) -> Union[float, Series]:
    """The Sortino ratio calculated as ( return - risk free return )
    / downside deviation. The ratio implies that the riskfree asset has zero
    volatility, and a minimum acceptable return of zero. The ratio is
    calculated using the annualized arithmetic mean of returns. \n
    https://www.investopedia.com/terms/s/sortinoratio.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    riskfree_rate : float, optional
        The return of the zero volatility asset
    riskfree_column : int, default: -1
        The return of the zero volatility asset used to calculate Sharpe ratio
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date
    periods_in_a_year_fixed : int, optional
        Allows locking the periods-in-a-year to simplify test cases and
        comparisons

    Returns
    -------
    Union[float, Pandas.Series]
        Sortino ratio calculated as ( return - riskfree return ) /
        downside deviation
    """
    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    how_many = data.loc[cast(int, earlier) : cast(int, later)].iloc[:, 0].count()
    fraction = (later - earlier).days / 365.25

    if periods_in_a_year_fixed:
        time_factor = periods_in_a_year_fixed
    else:
        time_factor = how_many / fraction

    ratios = []
    if riskfree_rate is None:
        if isinstance(riskfree_column, int):
            riskfree = data.loc[cast(int, earlier) : cast(int, later)].iloc[
                :, riskfree_column
            ]
            riskfree_item = data.iloc[:, riskfree_column].name
        else:
            raise ValueError("base_column argument should be an integer.")

        for item in data:
            if item == riskfree_item:
                ratios.append(0.0)
            else:
                longdf = data.loc[cast(int, earlier) : cast(int, later)].loc[:, item]
                ret = float(longdf.pct_change().mean() * time_factor)
                riskfree_ret = float(riskfree.pct_change().mean() * time_factor)
                dddf = longdf.pct_change()
                downdev = float(
                    sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / how_many)
                    * sqrt(time_factor)
                )
                ratios.append((ret - riskfree_ret) / downdev)

    else:
        for item in data:
            longdf = data.loc[cast(int, earlier) : cast(int, later)].loc[:, item]
            ret = float(longdf.pct_change().mean() * time_factor)
            dddf = longdf.pct_change()
            downdev = float(
                sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / how_many)
                * sqrt(time_factor)
            )
            ratios.append((ret - riskfree_rate) / downdev)

    if data.shape[1] == 1:
        return ratios[0]
    return Series(
        data=ratios,
        index=data.columns,
        name="Sortino ratio",
        dtype="float64",
    )


def calc_var_implied_vol_and_target(
    data: DataFrame,
    level: float,
    target_vol: Optional[float] = None,
    min_leverage_local: float = 0.0,
    max_leverage_local: float = 99999.0,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    interpolation: LiteralQuantileInterp = "lower",
    drift_adjust: bool = False,
    periods_in_a_year_fixed: Optional[int] = None,
) -> Union[float, Series]:
    """A position weight multiplier from the ratio between a VaR implied
    volatility and a given target volatility. Multiplier = 1.0 -> target met

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
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
    drift_adjust: bool, default: False
        An adjustment to remove the bias implied by the average return
    periods_in_a_year_fixed : int, optional
        Allows locking the periods-in-a-year to simplify test cases and
        comparisons

    Returns
    -------
    Union[float, Pandas.Series]
        A position weight multiplier from the ratio between a VaR implied
        volatility and a given target volatility. Multiplier = 1.0 -> target met
    """
    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
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
            .pct_change()
            .quantile(1 - level, interpolation=interpolation)
            - data.loc[cast(int, earlier) : cast(int, later)].pct_change().sum()
            / len(data.loc[cast(int, earlier) : cast(int, later)].pct_change())
        )
    else:
        imp_vol = (
            -sqrt(time_factor)
            * data.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()
            .quantile(1 - level, interpolation=interpolation)
            / norm.ppf(level)
        )

    if target_vol:
        result = imp_vol.apply(
            lambda x: max(min_leverage_local, min(target_vol / x, max_leverage_local))
        )
        label = "Weight from target vol"
    else:
        result = imp_vol
        label = "Imp vol from VaR"

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        index=data.columns,
        name=label,
        dtype="float64",
    )


def calc_value_ret(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
) -> Union[float, Series]:
    """
    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date

    Returns
    -------
    Union[float, Pandas.Series]
        Simple return
    """

    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    if 0.0 in data.iloc[0].tolist():
        raise ValueError(
            f"Simple return cannot be calculated due to an "
            f"initial value being zero. ({data.head(3)})"
        )

    result = data.loc[later] / data.loc[earlier] - 1

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        index=data.columns,
        name="Simple return",
        dtype="float64",
    )


def calc_value_ret_calendar_period(
    data: DataFrame, year: int, month: Optional[int] = None
) -> Union[float, Series]:
    """
    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    year : int
        Calendar year of the period to calculate.
    month : int, optional
        Calendar month of the period to calculate.

    Returns
    -------
    Pandas.Series
        Simple return for a specific calendar period
    """

    if month is None:
        period = str(year)
    else:
        period = "-".join([str(year), str(month).zfill(2)])
    vrdf = data.copy()
    vrdf.index = DatetimeIndex(vrdf.index)
    result = vrdf.pct_change().copy()
    result = result.loc[period] + 1
    result = result.apply(cumprod, axis="index").iloc[-1] - 1
    result.name = period
    result = result.astype("float64")
    if data.shape[1] == 1:
        return float(result.iloc[0])
    return result
