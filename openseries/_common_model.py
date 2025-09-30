"""Defining the _CommonModel class.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import datetime as dt
from inspect import stack
from json import dump
from math import ceil
from pathlib import Path
from secrets import choice
from string import ascii_letters
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

from numpy import asarray, float64, inf, isnan, log, maximum, sqrt

from .owntypes import (
    CaptorLogoType,
    DateAlignmentError,
    InitialValueZeroError,
    NumberOfItemsAndLabelsNotSameError,
    PlotlyConfigType,
    ResampleDataLossError,
    Self,
    SeriesOrFloat_co,
    ValueType,
)

if TYPE_CHECKING:  # pragma: no cover
    from openpyxl.worksheet.worksheet import Worksheet
    from pandas import Timestamp

    from .owntypes import (
        CountriesType,
        DaysInYearType,
        LiteralBarPlotMode,
        LiteralJsonOutput,
        LiteralLinePlotMode,
        LiteralNanMethod,
        LiteralPandasReindexMethod,
        LiteralPlotlyHistogramBarMode,
        LiteralPlotlyHistogramCurveType,
        LiteralPlotlyHistogramHistNorm,
        LiteralPlotlyHistogramPlotType,
        LiteralPlotlyJSlib,
        LiteralPlotlyOutput,
        LiteralQuantileInterp,
    )
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.workbook.workbook import Workbook
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    date_range,
    to_datetime,
)
from pandas.tseries.offsets import CustomBusinessDay
from plotly.figure_factory import create_distplot  # type: ignore[import-untyped]
from plotly.graph_objs import Figure  # type: ignore[import-untyped]
from plotly.io import to_html  # type: ignore[import-untyped]
from plotly.offline import plot  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, DirectoryPath, ValidationError
from scipy.stats import (
    kurtosis,
    norm,
    skew,
)

from ._risk import (
    _cvar_down_calc,
    _var_down_calc,
)
from .datefixer import (
    _do_resample_to_business_period_ends,
    date_offset_foll,
    holiday_calendar,
)
from .load_plotly import load_plotly_dict


def _get_date_range_and_factor(
    self: _CommonModel[SeriesOrFloat_co],
    months_from_last: int | None = None,
    from_date: dt.date | None = None,
    to_date: dt.date | None = None,
    periods_in_a_year_fixed: DaysInYearType | None = None,
) -> tuple[dt.date, dt.date, float, DataFrame]:
    """Common logic for date range and time factor calculation.

    Parameters
    ----------
    months_from_last : int, optional
        Number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date
    periods_in_a_year_fixed : DaysInYearType, optional
        Allows locking the periods-in-a-year to simplify test cases and
        comparisons

    Returns:
    -------
    tuple[dt.date, dt.date, float, DataFrame]
        earlier, later, time_factor, data
    """
    earlier, later = self.calc_range(
        months_offset=months_from_last,
        from_dt=from_date,
        to_dt=to_date,
    )

    if periods_in_a_year_fixed:
        time_factor = float(periods_in_a_year_fixed)
    else:
        how_many = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .count()
            .iloc[0]
        )
        fraction = (later - earlier).days / 365.25
        time_factor = how_many / fraction

    data = self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
    return earlier, later, time_factor, data


def _get_base_column_data(
    self: _CommonModel[SeriesOrFloat_co],
    base_column: tuple[str, ValueType] | int,
    earlier: dt.date,
    later: dt.date,
) -> tuple[Series[float], tuple[str, ValueType], str]:
    """Common logic for base column data extraction.

    Parameters
    ----------
    base_column : tuple[str, ValueType] | int
        Column reference
    earlier : dt.date
        Start date
    later : dt.date
        End date

    Returns:
    -------
    tuple[Series[float], tuple[str, ValueType], str]
        data, item, label
    """
    if isinstance(base_column, tuple):
        data = self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)][
            base_column
        ]
        item = base_column
        label = cast("tuple[str, str]", self.tsdf[base_column].name)[0]
    elif isinstance(base_column, int):
        data = self.tsdf.loc[
            cast("Timestamp", earlier) : cast("Timestamp", later)
        ].iloc[:, base_column]
        item = cast("tuple[str, ValueType]", self.tsdf.iloc[:, base_column].name)
        label = cast("tuple[str, str]", self.tsdf.iloc[:, base_column].name)[0]
    else:
        msg = "base_column should be a tuple[str, ValueType] or an integer."
        raise TypeError(msg)

    return data, item, label


def _calculate_time_factor(
    data: Series[float],
    earlier: dt.date,
    later: dt.date,
    periods_in_a_year_fixed: DaysInYearType | None = None,
) -> float:
    """Calculate time factor for annualization.

    Parameters
    ----------
    data : Series[float]
        Data series for counting observations
    earlier : dt.date
        Start date
    later : dt.date
        End date
    periods_in_a_year_fixed : DaysInYearType, optional
        Fixed periods in year

    Returns:
    -------
    float
        Time factor
    """
    if periods_in_a_year_fixed:
        return float(periods_in_a_year_fixed)

    fraction = (later - earlier).days / 365.25
    return data.count() / fraction


class _CommonModel(BaseModel, Generic[SeriesOrFloat_co]):
    """Declare _CommonModel."""

    tsdf: DataFrame = DataFrame(dtype="float64")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="always",
    )

    def _coerce_result(
        self: Self, result: Series[float], name: str
    ) -> SeriesOrFloat_co:
        if self.tsdf.shape[1] == 1:
            arr = float(asarray(a=result, dtype=float64).squeeze())
            return cast("SeriesOrFloat_co", arr)  # type: ignore[redundant-cast]
        return cast(
            "SeriesOrFloat_co",
            Series(
                data=result,
                index=self.tsdf.columns,
                name=name,
                dtype="float64",
            ),
        )

    @property
    def length(self: Self) -> int:
        """Number of observations.

        Returns:
        -------
        int
            Number of observations

        """
        return len(self.tsdf.index)

    @property
    def first_idx(self: Self) -> dt.date:
        """The first date in the timeseries.

        Returns:
        -------
        datetime.date
            The first date in the timeseries

        """
        return cast("dt.date", self.tsdf.index[0])

    @property
    def last_idx(self: Self) -> dt.date:
        """The last date in the timeseries.

        Returns:
        -------
        datetime.date
            The last date in the timeseries

        """
        return cast("dt.date", self.tsdf.index[-1])

    @property
    def span_of_days(self: Self) -> int:
        """Number of days from the first date to the last.

        Returns:
        -------
        int
            Number of days from the first date to the last

        """
        return (self.last_idx - self.first_idx).days

    @property
    def yearfrac(self: Self) -> float:
        """Length of series expressed in years assuming all years have 365.25 days.

        Returns:
        -------
        float
            Length of the timeseries expressed in years assuming all years
            have 365.25 days

        """
        return self.span_of_days / 365.25

    @property
    def periods_in_a_year(self: Self) -> float:
        """The average number of observations per year.

        Returns:
        -------
        float
            The average number of observations per year

        """
        return self.length / self.yearfrac

    @property
    def max_drawdown_cal_year(self: Self) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Maximum drawdown in a single calendar year.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        years = Index(d.year for d in self.tsdf.index)
        result = (
            self.tsdf.groupby(years)
            .apply(
                lambda prices: (prices / prices.expanding(min_periods=1).max()).min()
                - 1,
            )
            .min()
        )
        return self._coerce_result(result=result, name="Max drawdown in cal yr")

    @property
    def geo_ret(self: Self) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/c/cagr.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Compounded Annual Growth Rate (CAGR).
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.geo_ret_func()

    @property
    def arithmetic_ret(self: Self) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/a/arithmeticmean.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Annualized arithmetic mean of returns.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.arithmetic_ret_func()

    @property
    def value_ret(self: Self) -> SeriesOrFloat_co:
        """Simple return.

        Returns:
        -------
        SeriesOrFloat_co
            Simple return.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.value_ret_func()

    @property
    def vol(self: Self) -> SeriesOrFloat_co:
        """Annualized volatility.

        Based on Pandas .std() which is the equivalent of stdev.s([...]) in MS Excel.
        https://www.investopedia.com/terms/v/volatility.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Annualized volatility.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.vol_func()

    @property
    def downside_deviation(self: Self) -> SeriesOrFloat_co:
        """Downside Deviation.

        Standard deviation of returns that are below a Minimum Accepted Return
        of zero. It is used to calculate the Sortino Ratio.
        https://www.investopedia.com/terms/d/downside-deviation.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Downside deviation.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        min_accepted_return: float = 0.0
        order: Literal[2, 3] = 2
        return self.lower_partial_moment_func(
            min_accepted_return=min_accepted_return,
            order=order,
        )

    @property
    def ret_vol_ratio(self: Self) -> SeriesOrFloat_co:
        """Ratio of annualized arithmetic mean of returns and annualized volatility.

        Returns:
        -------
        SeriesOrFloat_co
            Ratio of the annualized arithmetic mean of returns and annualized.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.
            volatility.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        riskfree_rate: float = 0.0
        return self.ret_vol_ratio_func(riskfree_rate=riskfree_rate)

    @property
    def sortino_ratio(self: Self) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/s/sortinoratio.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Sortino ratio calculated as the annualized arithmetic mean of returns.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame
            / downside deviation. The ratio implies that the riskfree asset has zero
            volatility, and a minimum acceptable return of zero.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        riskfree_rate: float = 0.0
        minimum_accepted_return: float = 0.0
        return self.sortino_ratio_func(
            riskfree_rate=riskfree_rate,
            min_accepted_return=minimum_accepted_return,
        )

    @property
    def kappa3_ratio(self: Self) -> SeriesOrFloat_co:
        """Kappa-3 ratio.

        The Kappa-3 ratio is a generalized downside-risk ratio defined as
        annualized arithmetic return divided by the cubic-root of the
        lower partial moment of order 3 (with respect to a minimum acceptable
        return, MAR). It penalizes larger downside outcomes more heavily than
        the Sortino ratio (which uses order 2).

        Returns:
        -------
        SeriesOrFloat_co
            Kappa-3 ratio calculation with the riskfree rate and.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame
            Minimum Acceptable Return (MAR) both set to zero.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        riskfree_rate: float = 0.0
        minimum_accepted_return: float = 0.0
        order: Literal[2, 3] = 3
        return self.sortino_ratio_func(
            riskfree_rate=riskfree_rate,
            min_accepted_return=minimum_accepted_return,
            order=order,
        )

    @property
    def omega_ratio(self: Self) -> SeriesOrFloat_co:
        """https://en.wikipedia.org/wiki/Omega_ratio.

        Returns:
        -------
        SeriesOrFloat_co
            Omega ratio calculation.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        minimum_accepted_return: float = 0.0
        return self.omega_ratio_func(min_accepted_return=minimum_accepted_return)

    @property
    def z_score(self: Self) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/z/zscore.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Z-score as (last return - mean return) / standard deviation of returns.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.z_score_func()

    @property
    def max_drawdown(self: Self) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Maximum drawdown without any limit on date range.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.max_drawdown_func()

    @property
    def max_drawdown_date(self: Self) -> dt.date | Series[dt.date]:
        """Date when the maximum drawdown occurred.

        https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Returns:
        -------
        datetime.date | pandas.Series[dt.date]
            Date when the maximum drawdown occurred

        """
        mdddf = self.tsdf.copy()
        mdddf.index = DatetimeIndex(mdddf.index)
        result = (mdddf / mdddf.expanding(min_periods=1).max()).idxmin().dt.date

        if self.tsdf.shape[1] == 1:
            return result.iloc[0]
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Max drawdown date",
            dtype="datetime64[ns]",
        ).dt.date

    @property
    def worst(self: Self) -> SeriesOrFloat_co:
        """Most negative percentage change.

        Returns:
        -------
        SeriesOrFloat_co
            Most negative percentage change.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        observations: int = 1
        return self.worst_func(observations=observations)

    @property
    def worst_month(self: Self) -> SeriesOrFloat_co:
        """Most negative month.

        Returns:
        -------
        SeriesOrFloat_co
            Most negative month.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        method: LiteralPandasReindexMethod = "nearest"

        try:
            countries = self.countries
            markets = self.markets
        except AttributeError:
            countries = self.constituents[0].countries  # type: ignore[attr-defined]
            markets = self.constituents[0].markets  # type: ignore[attr-defined]

        wmdf = self.tsdf.copy()

        dates = _do_resample_to_business_period_ends(
            data=wmdf,
            freq="BME",
            countries=countries,
            markets=markets,
        )

        wmdf = wmdf.reindex(index=[deyt.date() for deyt in dates], method=method)
        wmdf.index = DatetimeIndex(wmdf.index)

        vtypes = [x == ValueType.RTRN for x in wmdf.columns.get_level_values(1)]
        if any(vtypes):
            msg = (
                "Do not run worst_month on return series. The operation will "
                "pick the last data point in the sparser series. It will not sum "
                "returns and therefore data will be lost and result will be wrong."
            )
            raise ResampleDataLossError(msg)

        result = wmdf.ffill().pct_change().min()

        return self._coerce_result(result=result, name="Worst month")

    @property
    def positive_share(self: Self) -> SeriesOrFloat_co:
        """The share of percentage changes that are greater than zero.

        Returns:
        -------
        SeriesOrFloat_co
            The share of percentage changes that are greater than zero.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.positive_share_func()

    @property
    def skew(self: Self) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/s/skewness.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Skew of the return distribution.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.skew_func()

    @property
    def kurtosis(self: Self) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/k/kurtosis.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Kurtosis of the return distribution.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.kurtosis_func()

    @property
    def cvar_down(self: Self) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Downside 95% Conditional Value At Risk "CVaR".
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        level: float = 0.95
        return self.cvar_down_func(level=level)

    @property
    def var_down(self: Self) -> SeriesOrFloat_co:
        """Downside 95% Value At Risk (VaR).

        The equivalent of percentile.inc([...], 1-level) over returns in MS Excel.
        https://www.investopedia.com/terms/v/var.asp.

        Returns:
        -------
        SeriesOrFloat_co
            Downside 95% Value At Risk (VaR).
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        level: float = 0.95
        interpolation: LiteralQuantileInterp = "lower"
        return self.var_down_func(level=level, interpolation=interpolation)

    @property
    def vol_from_var(self: Self) -> SeriesOrFloat_co:
        """Implied annualized volatility from Downside 95% Value at Risk.

        Assumes that returns are normally distributed.

        Returns:
        -------
        SeriesOrFloat_co
            Implied annualized volatility from the Downside 95% VaR using the.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.
            assumption that returns are normally distributed.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        level: float = 0.95
        interpolation: LiteralQuantileInterp = "lower"
        return self.vol_from_var_func(level=level, interpolation=interpolation)

    def calc_range(
        self: Self,
        months_offset: int | None = None,
        from_dt: dt.date | None = None,
        to_dt: dt.date | None = None,
    ) -> tuple[dt.date, dt.date]:
        """Create user defined date range.

        Parameters
        ----------
        months_offset: int, optional
            Number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_dt: datetime.date, optional
            Specific from date
        to_dt: datetime.date, optional
            Specific from date

        Returns:
        -------
        tuple[datetime.date, datetime.date]
            Start and end date of the chosen date range

        """
        earlier, later = self.first_idx, self.last_idx
        if months_offset is not None:
            earlier = date_offset_foll(
                raw_date=self.last_idx,
                months_offset=-months_offset,
                adjust=False,
                following=True,
            )
            if earlier < self.first_idx:
                msg = (
                    "Argument months_offset implies start"
                    "date before first date in series."
                )
                raise DateAlignmentError(msg)
            later = self.last_idx
        else:
            if from_dt is not None:
                if from_dt < self.first_idx:
                    msg = "Given from_dt date < series start"
                    raise DateAlignmentError(msg)
                earlier = from_dt
            if to_dt is not None:
                if to_dt > self.last_idx:
                    msg = "Given to_dt date > series end"
                    raise DateAlignmentError(msg)
                later = to_dt
        while earlier not in self.tsdf.index:
            earlier -= dt.timedelta(days=1)
        while later not in self.tsdf.index:
            later += dt.timedelta(days=1)

        return earlier, later

    def align_index_to_local_cdays(
        self: Self,
        countries: CountriesType | None = None,
        markets: list[str] | str | None = None,
        custom_holidays: list[str] | str | None = None,
        method: LiteralPandasReindexMethod = "nearest",
    ) -> Self:
        """Align the index of .tsdf with local calendar business days.

        Parameters
        ----------
        countries: CountriesType, optional
            (List of) country code(s) according to ISO 3166-1 alpha-2
        markets: list[str] | str, optional
            (List of) markets code(s) according to pandas-market-calendars
        custom_holidays: list[str] | str, optional
            Argument where missing holidays can be added
        method: LiteralPandasReindexMethod, default: "nearest"

        Returns:
        -------
        OpenFrame
            An OpenFrame object

        """
        startyear = cast("int", to_datetime(self.tsdf.index[0]).year)
        endyear = cast("int", to_datetime(self.tsdf.index[-1]).year)

        if countries:
            try:
                self.countries = countries
            except ValidationError:
                for serie in self.constituents:  # type: ignore[attr-defined]
                    serie.countries = countries
        else:
            try:
                countries = self.countries
            except AttributeError:
                countries = self.constituents[0].countries  # type: ignore[attr-defined]

        if markets:
            try:
                self.markets = markets
            except ValidationError:
                for serie in self.constituents:  # type: ignore[attr-defined]
                    serie.markets = markets
        else:
            try:
                markets = self.markets
            except AttributeError:
                markets = self.constituents[0].markets  # type: ignore[attr-defined]

        calendar = holiday_calendar(
            startyear=startyear,
            endyear=endyear,
            countries=countries or "SE",
            markets=markets,
            custom_holidays=custom_holidays,
        )

        d_range = [
            d.date()
            for d in date_range(
                start=cast("dt.date", self.tsdf.first_valid_index()),
                end=cast("dt.date", self.tsdf.last_valid_index()),
                freq=CustomBusinessDay(calendar=calendar)
                if any([countries, markets, custom_holidays])
                else None,
            )
        ]
        self.tsdf = self.tsdf.reindex(labels=d_range, method=method, copy=False)

        return self

    def value_to_log(self: Self) -> Self:
        """Series of values converted into logarithmic weighted series.

        Equivalent to LN(value[t] / value[t=0]) in Excel.

        Returns:
        -------
        self
            An object of the same class

        """
        self.tsdf = DataFrame(
            data=log(self.tsdf / self.tsdf.iloc[0]),
            index=self.tsdf.index,
            columns=self.tsdf.columns,
        )
        return self

    def value_nan_handle(self: Self, method: LiteralNanMethod = "fill") -> Self:
        """Handle missing values in a valueseries.

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with last known or drop

        Returns:
        -------
        self
            An object of the same class

        """
        if method == "fill":
            self.tsdf = self.tsdf.ffill()
        else:
            self.tsdf = self.tsdf.dropna()
        return self

    def return_nan_handle(self: Self, method: LiteralNanMethod = "fill") -> Self:
        """Handle missing values in a returnseries.

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with zero or drop

        Returns:
        -------
        self
            An object of the same class

        """
        if method == "fill":
            self.tsdf = self.tsdf.fillna(value=0.0)
        else:
            self.tsdf = self.tsdf.dropna()
        return self

    def to_drawdown_series(self: Self) -> Self:
        """Convert timeseries into a drawdown series.

        Returns:
        -------
        self
            An object of the same class

        """
        drawdown = self.tsdf.copy()
        drawdown[isnan(drawdown)] = -inf
        roll_max = maximum.accumulate(drawdown, axis=0)
        self.tsdf = DataFrame(drawdown / roll_max - 1.0)
        return self

    def to_json(
        self: Self,
        what_output: LiteralJsonOutput,
        filename: str,
        directory: DirectoryPath | None = None,
    ) -> list[dict[str, str | bool | ValueType | list[str] | list[float]]]:
        """Dump timeseries data into a json file.

        Parameters
        ----------
        what_output: LiteralJsonOutput
            Choice on whether the raw values or the tsdf Dataframe values are
            returned as json and exported as json file.
        filename: str
            Filename including filetype
        directory: DirectoryPath, optional
            File folder location

        Returns:
        -------
        list[dict[str, str | bool | ValueType | list[str] | list[float]]]
            A list of dictionaries with the data of the series

        """
        if directory:
            dirpath = Path(directory).resolve()
        elif Path.home().joinpath("Documents").exists():
            dirpath = Path.home().joinpath("Documents")
        else:
            dirpath = Path(stack()[1].filename).parent

        cleaner_list = ["label", "tsdf"]
        data = dict(self.__dict__)
        output = []
        if "label" in data:
            if what_output == "tsdf":
                values = Series(self.tsdf.iloc[:, 0]).tolist()
            else:
                values = list(cast("list[float]", data.get("values")))
            for item in cleaner_list:
                data.pop(item)
            valuetype = cast("ValueType", data.get("valuetype")).value
            data.update({"valuetype": valuetype})
            data.update({"values": values})
            output.append(dict(data))
        else:
            for serie in cast("list[Any]", data.get("constituents")):
                if what_output == "tsdf":
                    values = serie.tsdf.iloc[:, 0].tolist()
                else:
                    values = list(serie.values)
                itemdata = dict(serie.__dict__)
                for item in cleaner_list:
                    itemdata.pop(item)
                valuetype = cast("ValueType", itemdata["valuetype"]).value
                itemdata.update({"valuetype": valuetype})
                itemdata.update({"values": values})
                output.append(dict(itemdata))

        with dirpath.joinpath(filename).open(mode="w", encoding="utf-8") as jsonfile:
            dump(obj=output, fp=jsonfile, indent=2, sort_keys=False)

        return output

    def to_xlsx(
        self: Self,
        filename: str,
        sheet_title: str | None = None,
        directory: DirectoryPath | None = None,
        *,
        overwrite: bool = True,
    ) -> str:
        """Save .tsdf DataFrame to an Excel spreadsheet file.

        Parameters
        ----------
        filename: str
            Filename that should include .xlsx
        sheet_title: str, optional
            Name of the sheet in the Excel file
        directory: DirectoryPath, optional
            The file directory where the Excel file is saved.
        overwrite: bool, default: True
            Flag whether to overwrite an existing file

        Returns:
        -------
        str
            The Excel file path

        """
        if filename[-5:].lower() != ".xlsx":
            msg = "Filename must end with .xlsx"
            raise NameError(msg)

        if directory:
            dirpath = Path(directory).resolve()
        elif Path.home().joinpath("Documents").exists():
            dirpath = Path.home().joinpath("Documents")
        else:
            dirpath = Path(stack()[1].filename).parent

        sheetfile = dirpath.joinpath(filename)

        wrkbook = Workbook()
        wrksheet = wrkbook.active

        if sheet_title:
            cast("Worksheet", wrksheet).title = sheet_title

        for row in dataframe_to_rows(df=self.tsdf, index=True, header=True):
            cast("Worksheet", wrksheet).append(row)

        if not overwrite and Path(sheetfile).exists():
            msg = f"{sheetfile!s} already exists."
            raise FileExistsError(msg)

        wrkbook.save(sheetfile)

        return str(sheetfile)

    @staticmethod
    def _ensure_labels(
        ncols: int,
        labels: list[str] | None,
        default_labels: list[str],
    ) -> list[str]:
        """Validate or infer labels for plotting."""
        if labels:
            if len(labels) != ncols:
                msg = "Must provide same number of labels as items in frame."
                raise NumberOfItemsAndLabelsNotSameError(msg)
            return labels
        return default_labels

    @staticmethod
    def _resolve_dir(directory: DirectoryPath | None) -> Path:
        """Resolve output directory for plot files."""
        if directory:
            return Path(directory).resolve()
        if (Path.home() / "Documents").exists():
            return Path.home() / "Documents"
        return Path(stack()[2].filename).parent

    @staticmethod
    def _hover_xy(tick_fmt: str | None) -> str:
        """Create hovertemplate for y-value and date x-axis."""
        return (
            f"%{{y:{tick_fmt}}}<br>%{{x|{'%Y-%m-%d'}}}"
            if tick_fmt
            else "%{y}<br>%{x|%Y-%m-%d}"
        )

    @staticmethod
    def _hover_hist(x_fmt: str | None, y_fmt: str | None) -> str:
        """Create hovertemplate for histogram plots."""
        y = f"%{{y:{y_fmt}}}" if y_fmt else "%{y}"
        x = f"%{{x:{x_fmt}}}" if x_fmt else "%{x}"
        return f"Count: {y}<br>{x}"

    @staticmethod
    def _apply_title_logo(
        figure: Figure,
        logo: CaptorLogoType,
        title: str | None,
        *,
        add_logo: bool,
    ) -> None:
        """Apply optional title and logo to a Plotly Figure."""
        if add_logo:
            figure.add_layout_image(logo)
        if title:
            figure.update_layout(
                {"title": {"text": f"<b>{title}</b><br>", "font": {"size": 36}}},
            )

    @staticmethod
    def _emit_output(
        figure: Figure,
        fig_config: PlotlyConfigType,
        output_type: LiteralPlotlyOutput,
        plotfile: Path,
        filename: str,
        *,
        include_plotlyjs_bool: bool,
        auto_open: bool,
    ) -> str:
        """Write a file or return inline HTML string from a Plotly Figure."""
        if output_type == "file":
            plot(
                figure_or_data=figure,
                filename=str(plotfile),
                auto_open=auto_open,
                auto_play=False,
                link_text="",
                include_plotlyjs=include_plotlyjs_bool,
                config=fig_config,
                output_type=output_type,
            )
            return str(plotfile)

        div_id = filename.rsplit(".", 1)[0]
        return cast(
            "str",
            to_html(
                fig=figure,
                config=fig_config,
                auto_play=False,
                include_plotlyjs=include_plotlyjs_bool,
                full_html=False,
                div_id=div_id,
            ),
        )

    def plot_bars(
        self: Self,
        mode: LiteralBarPlotMode = "group",
        title: str | None = None,
        tick_fmt: str | None = None,
        filename: str | None = None,
        directory: DirectoryPath | None = None,
        labels: list[str] | None = None,
        output_type: LiteralPlotlyOutput = "file",
        include_plotlyjs: LiteralPlotlyJSlib = "cdn",
        *,
        auto_open: bool = True,
        add_logo: bool = True,
    ) -> tuple[Figure, str]:
        """Create a Plotly Bar Figure.

        Parameters
        ----------
        self.tsdf: pandas.DataFrame
            The timeseries self.tsdf
        mode: LiteralBarPlotMode
            The type of bar to use
        title: str, optional
            A title above the plot
        tick_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show
        filename: str, optional
            Name of the Plotly html file
        directory: DirectoryPath, optional
            Directory where Plotly html file is saved
        labels: list[str], optional
            A list of labels to manually override using the names of
            the input self.tsdf
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type
        include_plotlyjs: LiteralPlotlyJSlib, default: "cdn"
            Determines how the plotly.js library is included in the output
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot

        Returns:
        -------
        tuple[plotly.go.Figure, str]
            Plotly Figure and a div section or a html filename with location

        """
        labels = self._ensure_labels(
            ncols=self.tsdf.shape[1],
            labels=labels,
            default_labels=list(self.tsdf.columns.get_level_values(0)),
        )

        dirpath = self._resolve_dir(directory=directory)
        if not filename:
            filename = f"{''.join(choice(ascii_letters) for _ in range(6))}.html"
        plotfile = dirpath / filename

        fig, logo = load_plotly_dict()
        figure = Figure(fig)

        opacity = 0.7 if mode == "overlay" else None
        hovertemplate = self._hover_xy(tick_fmt=tick_fmt)

        for item in range(self.tsdf.shape[1]):
            figure.add_bar(
                x=self.tsdf.index,
                y=self.tsdf.iloc[:, item],
                hovertemplate=hovertemplate,
                name=labels[item],
                opacity=opacity,
            )
        figure.update_layout(barmode=mode, yaxis={"tickformat": tick_fmt})

        self._apply_title_logo(
            figure=figure,
            title=title,
            add_logo=add_logo,
            logo=logo,
        )

        string_output = self._emit_output(
            figure=figure,
            fig_config=fig["config"],
            include_plotlyjs_bool=cast("bool", include_plotlyjs),
            output_type=output_type,
            auto_open=auto_open,
            plotfile=plotfile,
            filename=filename,
        )

        return figure, string_output

    def plot_series(
        self: Self,
        mode: LiteralLinePlotMode = "lines",
        title: str | None = None,
        tick_fmt: str | None = None,
        filename: str | None = None,
        directory: DirectoryPath | None = None,
        labels: list[str] | None = None,
        output_type: LiteralPlotlyOutput = "file",
        include_plotlyjs: LiteralPlotlyJSlib = "cdn",
        *,
        auto_open: bool = True,
        add_logo: bool = True,
        show_last: bool = False,
    ) -> tuple[Figure, str]:
        """Create a Plotly Scatter Figure.

        Parameters
        ----------
        self.tsdf: pandas.DataFrame
            The timeseries self.tsdf
        mode: LiteralLinePlotMode, default: "lines"
            The type of scatter to use
        title: str, optional
            A title above the plot
        tick_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show
        filename: str, optional
            Name of the Plotly html file
        directory: DirectoryPath, optional
            Directory where Plotly html file is saved
        labels: list[str], optional
            A list of labels to manually override using the names of
            the input self.tsdf
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type
        include_plotlyjs: LiteralPlotlyJSlib, default: "cdn"
            Determines how the plotly.js library is included in the output
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot
        show_last: bool, default: False
            If True the last self.tsdf point is highlighted as red dot with a label

        Returns:
        -------
        tuple[plotly.go.Figure, str]
            Plotly Figure and a div section or a html filename with location

        """
        labels = self._ensure_labels(
            ncols=self.tsdf.shape[1],
            labels=labels,
            default_labels=list(self.tsdf.columns.get_level_values(0)),
        )

        dirpath = self._resolve_dir(directory=directory)
        if not filename:
            filename = f"{''.join(choice(ascii_letters) for _ in range(6))}.html"
        plotfile = dirpath / filename

        fig, logo = load_plotly_dict()
        figure = Figure(fig)

        hovertemplate = self._hover_xy(tick_fmt=tick_fmt)

        for item in range(self.tsdf.shape[1]):
            figure.add_scatter(
                x=self.tsdf.index,
                y=self.tsdf.iloc[:, item],
                hovertemplate=hovertemplate,
                line={"width": 2.5, "dash": "solid"},
                mode=mode,
                name=labels[item],
            )
        figure.update_layout(yaxis={"tickformat": tick_fmt})

        if show_last:
            txt = f"Last {{:{tick_fmt}}}" if tick_fmt else "Last {}"
            for item in range(self.tsdf.shape[1]):
                figure.add_scatter(
                    x=[Series(self.tsdf.iloc[:, item]).index[-1]],
                    y=[self.tsdf.iloc[-1, item]],
                    mode="markers + text",
                    marker={"color": "red", "size": 12},
                    hovertemplate=hovertemplate,
                    showlegend=False,
                    name=labels[item],
                    text=[txt.format(self.tsdf.iloc[-1, item])],
                    textposition="top center",
                )

        self._apply_title_logo(
            figure=figure,
            title=title,
            add_logo=add_logo,
            logo=logo,
        )

        string_output = self._emit_output(
            figure=figure,
            fig_config=fig["config"],
            include_plotlyjs_bool=cast("bool", include_plotlyjs),
            output_type=output_type,
            auto_open=auto_open,
            plotfile=plotfile,
            filename=filename,
        )

        return figure, string_output

    def plot_histogram(
        self: Self,
        plot_type: LiteralPlotlyHistogramPlotType = "bars",
        histnorm: LiteralPlotlyHistogramHistNorm = "probability",
        barmode: LiteralPlotlyHistogramBarMode = "overlay",
        xbins_size: float | None = None,
        opacity: float = 0.75,
        bargap: float = 0.0,
        bargroupgap: float = 0.0,
        curve_type: LiteralPlotlyHistogramCurveType = "kde",
        title: str | None = None,
        x_fmt: str | None = None,
        y_fmt: str | None = None,
        filename: str | None = None,
        directory: DirectoryPath | None = None,
        labels: list[str] | None = None,
        output_type: LiteralPlotlyOutput = "file",
        include_plotlyjs: LiteralPlotlyJSlib = "cdn",
        *,
        cumulative: bool = False,
        show_rug: bool = False,
        auto_open: bool = True,
        add_logo: bool = True,
    ) -> tuple[Figure, str]:
        """Create a Plotly Histogram Figure.

        Parameters
        ----------
        plot_type: LiteralPlotlyHistogramPlotType, default: bars
            Type of plot
        histnorm: LiteralPlotlyHistogramHistNorm, default: percent
            Sets the normalization mode
        barmode: LiteralPlotlyHistogramBarMode, default: overlay
            Specifies how bar traces are displayed relative to one another
        xbins_size: float, optional
            Explicitly sets the width of each bin along the x-axis in data units
        opacity: float, default: 0.75
            Sets the trace opacity, must be between 0 (fully transparent) and 1
        bargap: float, default: 0.0
            Sets the gap between bars of adjacent location coordinates
        bargroupgap: float, default: 0.0
            Sets the gap between bar “groups” at the same location coordinate
        curve_type: LiteralPlotlyHistogramCurveType, default: kde
            Specifies the type of distribution curve to overlay on the histogram
        title: str, optional
            A title above the plot
        y_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show on the y-axis
        x_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show on the x-axis
        filename: str, optional
            Name of the Plotly html file
        directory: DirectoryPath, optional
            Directory where Plotly html file is saved
        labels: list[str], optional
            A list of labels to manually override using the names of
            the input self.tsdf
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type
        include_plotlyjs: LiteralPlotlyJSlib, default: "cdn"
            Determines how the plotly.js library is included in the output
        cumulative: bool, default: False
            Determines whether to compute a cumulative histogram
        show_rug: bool, default: False
            Determines whether to draw a rug plot alongside the distribution
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot

        Returns:
        -------
        tuple[plotly.go.Figure, str]
            Plotly Figure and a div section or a html filename with location

        """
        labels = self._ensure_labels(
            ncols=self.tsdf.shape[1],
            labels=labels,
            default_labels=list(self.tsdf.columns.get_level_values(0)),
        )

        dirpath = self._resolve_dir(directory=directory)
        if not filename:
            filename = f"{''.join(choice(ascii_letters) for _ in range(6))}.html"
        plotfile = dirpath / filename

        fig_dict, logo = load_plotly_dict()
        hovertemplate = self._hover_hist(x_fmt=x_fmt, y_fmt=y_fmt)

        msg = "plot_type must be 'bars' or 'lines'."
        if plot_type == "bars":
            figure = Figure(fig_dict)
            for item in range(self.tsdf.shape[1]):
                figure.add_histogram(
                    x=self.tsdf.iloc[:, item],
                    cumulative={"enabled": cumulative},
                    histfunc="count",
                    histnorm=histnorm,
                    name=labels[item],
                    xbins={"size": xbins_size},
                    opacity=opacity,
                    hovertemplate=hovertemplate,
                )
            figure.update_layout(
                barmode=barmode,
                bargap=bargap,
                bargroupgap=bargroupgap,
            )
        elif plot_type == "lines":
            hist_data = [self.tsdf[col] for col in self.tsdf.columns]
            figure = create_distplot(
                hist_data=hist_data,
                curve_type=curve_type,
                group_labels=labels,
                show_hist=False,
                show_rug=show_rug,
                histnorm=histnorm,
            )
            figure.update_layout(dict1=fig_dict["layout"])
        else:
            raise TypeError(msg)

        figure.update_layout(xaxis={"tickformat": x_fmt}, yaxis={"tickformat": y_fmt})
        figure.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="lightgrey")
        figure.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="lightgrey")

        self._apply_title_logo(
            figure=figure,
            title=title,
            add_logo=add_logo,
            logo=logo,
        )

        string_output = self._emit_output(
            figure=figure,
            fig_config=fig_dict["config"],
            include_plotlyjs_bool=cast("bool", include_plotlyjs),
            output_type=output_type,
            auto_open=auto_open,
            plotfile=plotfile,
            filename=filename,
        )

        return figure, string_output

    def arithmetic_ret_func(
        self: Self,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> SeriesOrFloat_co:
        """https://www.investopedia.com/terms/a/arithmeticmean.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns:
        -------
        SeriesOrFloat_co
            Annualized arithmetic mean of returns.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        _earlier, _later, time_factor, data = _get_date_range_and_factor(
            self=self,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

        result = data.ffill().pct_change().mean() * time_factor

        return self._coerce_result(result=result, name="Arithmetic return")

    def vol_func(
        self: Self,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> SeriesOrFloat_co:
        """Annualized volatility.

        Based on Pandas .std() which is the equivalent of stdev.s([...]) in MS Excel.
        https://www.investopedia.com/terms/v/volatility.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns:
        -------
        SeriesOrFloat_co
            Annualized volatility.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        _earlier, _later, time_factor, data = _get_date_range_and_factor(
            self=self,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

        result = data.ffill().pct_change().std().mul(sqrt(time_factor))

        return self._coerce_result(result=result, name="Volatility")

    def vol_from_var_func(
        self: Self,
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        interpolation: LiteralQuantileInterp = "lower",
        periods_in_a_year_fixed: DaysInYearType | None = None,
        *,
        drift_adjust: bool = False,
    ) -> SeriesOrFloat_co:
        """Implied annualized volatility.

        Implied annualized volatility from the Downside VaR using the assumption
        that returns are normally distributed.

        Parameters
        ----------
        level: float, default: 0.95
            The sought VaR level
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

        Returns:
        -------
        SeriesOrFloat_co
            Implied annualized volatility from the Downside VaR using the.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.
            assumption that returns are normally distributed.

        """
        return self._var_implied_vol_and_target_func(
            level=level,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            interpolation=interpolation,
            drift_adjust=drift_adjust,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

    def target_weight_from_var(
        self: Self,
        target_vol: float = 0.175,
        level: float = 0.95,
        min_leverage_local: float = 0.0,
        max_leverage_local: float = 99999.0,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        interpolation: LiteralQuantileInterp = "lower",
        periods_in_a_year_fixed: DaysInYearType | None = None,
        *,
        drift_adjust: bool = False,
    ) -> SeriesOrFloat_co:
        """Target weight from VaR.

        A position weight multiplier from the ratio between a VaR implied
        volatility and a given target volatility. Multiplier = 1.0 -> target met.

        Parameters
        ----------
        target_vol: float, default: 0.175
            Target Volatility
        level: float, default: 0.95
            The sought VaR level
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

        Returns:
        -------
        SeriesOrFloat_co
            A position weight multiplier from the ratio between a VaR implied.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.
            volatility and a given target volatility. Multiplier = 1.0 -> target met

        """
        return self._var_implied_vol_and_target_func(
            target_vol=target_vol,
            level=level,
            min_leverage_local=min_leverage_local,
            max_leverage_local=max_leverage_local,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            interpolation=interpolation,
            drift_adjust=drift_adjust,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

    def _var_implied_vol_and_target_func(
        self: Self,
        level: float,
        target_vol: float | None = None,
        min_leverage_local: float = 0.0,
        max_leverage_local: float = 99999.0,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        interpolation: LiteralQuantileInterp = "lower",
        periods_in_a_year_fixed: DaysInYearType | None = None,
        *,
        drift_adjust: bool = False,
    ) -> SeriesOrFloat_co:
        """Volatility implied from VaR or Target Weight.

        The function returns a position weight multiplier from the ratio between
        a VaR implied volatility and a given target volatility if the argument
        target_vol is provided. Otherwise the function returns the VaR implied
        volatility. Multiplier = 1.0 -> target met.

        Parameters
        ----------
        level: float
            The sought VaR level
        target_vol: float | None
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

        Returns:
        -------
        SeriesOrFloat_co
            Target volatility if target_vol is provided otherwise the VaR.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.
            implied volatility.

        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            fraction = (later - earlier).days / 365.25
            how_many = (
                self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
                .count()
                .iloc[0]
            )
            time_factor = how_many / fraction
        if drift_adjust:
            imp_vol = (-sqrt(time_factor) / norm.ppf(level)) * (
                self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
                .ffill()
                .pct_change()
                .quantile(1 - level, interpolation=interpolation)
                - self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
                .ffill()
                .pct_change()
                .sum()
                / len(
                    self.tsdf.loc[
                        cast("Timestamp", earlier) : cast("Timestamp", later)
                    ]
                    .ffill()
                    .pct_change(),
                )
            )
        else:
            imp_vol = (
                -sqrt(time_factor)
                * self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
                .ffill()
                .pct_change()
                .quantile(1 - level, interpolation=interpolation)
                / norm.ppf(level)
            )

        if target_vol:
            result = imp_vol.apply(
                lambda x: max(
                    min_leverage_local,
                    min(target_vol / x, max_leverage_local),
                ),
            )
            label = "Weight from target vol"
        else:
            result = imp_vol
            label = f"Imp vol from VaR {level:.0%}"

        return self._coerce_result(result=result, name=label)

    def cvar_down_func(
        self: Self,
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> SeriesOrFloat_co:
        """Downside Conditional Value At Risk "CVaR".

        https://www.investopedia.com/terms/c/conditional_value_at_risk.asp.

        Parameters
        ----------
        level: float, default: 0.95
            The sought CVaR level
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns:
        -------
        SeriesOrFloat_co
            Downside Conditional Value At Risk "CVaR".
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        cvar_df = self.tsdf.loc[
            cast("Timestamp", earlier) : cast("Timestamp", later)
        ].copy(deep=True)
        result = [
            (r := cvar_df[col].ffill().pct_change().sort_values())[
                : ceil((1 - level) * r.count())
            ].mean()
            for col in cvar_df.columns
        ]

        return self._coerce_result(
            result=cast("Series[float]", result),
            name=f"CVaR {level:.1%}",
        )

    def lower_partial_moment_func(
        self: Self,
        min_accepted_return: float = 0.0,
        order: Literal[2, 3] = 2,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> SeriesOrFloat_co:
        """Downside Deviation if order set to 2.

        If order is set to 2 the function calculates the standard
        deviation of returns that are below a Minimum Accepted
        Return of zero. For general order p, it returns LPM_p^(1/p),
        i.e., the rooted lower partial moment of order p.

        Parameters
        ----------
        min_accepted_return : float, optional
            The annualized Minimum Accepted Return (MAR)
        order: int, default: 2
            Order of partial moment
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns:
        -------
        SeriesOrFloat_co
            Downside deviation if order set to 2.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        msg = f"'order' must be 2 or 3, got {order!r}."
        if order not in (2, 3):
            raise ValueError(msg)

        zero: float = 0.0
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )

        how_many = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .ffill()
            .pct_change()
            .count(numeric_only=True)
        )
        if periods_in_a_year_fixed:
            time_factor = Series(
                data=[float(periods_in_a_year_fixed)] * how_many.shape[0],
                index=self.tsdf.columns,
                dtype="float64",
            )
        else:
            fraction = (later - earlier).days / 365.25
            time_factor = how_many.div(fraction)

        per_period_mar = min_accepted_return / time_factor
        diff = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .ffill()
            .pct_change()
            .sub(per_period_mar)
        )

        shortfall = (-diff).clip(lower=zero)
        base = shortfall.pow(order).sum() / how_many
        result = base.pow(1.0 / float(order))
        result *= sqrt(time_factor)

        dd_order = 2

        return self._coerce_result(
            result=result,
            name="Downside deviation" if order == dd_order else f"LPM{order}",
        )

    def geo_ret_func(
        self: Self,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> SeriesOrFloat_co:
        """Compounded Annual Growth Rate (CAGR).

        https://www.investopedia.com/terms/c/cagr.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns:
        -------
        SeriesOrFloat_co
            Compounded Annual Growth Rate (CAGR).
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        zero = 0.0
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        fraction = (later - earlier).days / 365.25

        any_below_zero = any(self.tsdf.loc[[earlier, later]].lt(0.0).any().to_numpy())
        if zero in self.tsdf.loc[earlier].to_numpy() or any_below_zero:
            msg = (
                "Geometric return cannot be calculated due to "
                "an initial value being zero or a negative value."
            )
            raise InitialValueZeroError(msg)

        result = (self.tsdf.loc[later] / self.tsdf.loc[earlier]) ** (1 / fraction) - 1

        return self._coerce_result(
            result=cast("Series[float]", result),
            name="Geometric return",
        )

    def skew_func(
        self: Self,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> SeriesOrFloat_co:
        """Skew of the return distribution.

        https://www.investopedia.com/terms/s/skewness.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns:
        -------
        SeriesOrFloat_co
            Skew of the return distribution.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = skew(
            a=(
                self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
                .ffill()
                .pct_change()
            ),
            bias=True,
            nan_policy="omit",
        )

        return self._coerce_result(result=cast("Series[float]", result), name="Skew")

    def kurtosis_func(
        self: Self,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> SeriesOrFloat_co:
        """Kurtosis of the return distribution.

        https://www.investopedia.com/terms/k/kurtosis.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns:
        -------
        SeriesOrFloat_co
            Kurtosis of the return distribution.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = kurtosis(
            a=(
                self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
                .ffill()
                .pct_change()
            ),
            fisher=True,
            bias=True,
            nan_policy="omit",
        )

        return self._coerce_result(
            result=cast("Series[float]", result),
            name="Kurtosis",
        )

    def max_drawdown_func(
        self: Self,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        min_periods: int = 1,
    ) -> SeriesOrFloat_co:
        """Maximum drawdown without any limit on date range.

        https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        min_periods: int, default: 1
            Smallest number of observations to use to find the maximum drawdown

        Returns:
        -------
        SeriesOrFloat_co
            Maximum drawdown without any limit on date range.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            / self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .expanding(min_periods=min_periods)
            .max()
        ).min() - 1

        return self._coerce_result(result=result, name="Max drawdown")

    def positive_share_func(
        self: Self,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> SeriesOrFloat_co:
        """Calculate share of percentage changes that are greater than zero.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns:
        -------
        SeriesOrFloat_co
            Calculate share of percentage changes that are greater than zero.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        zero: float = 0.0
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        pos = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .ffill()
            .pct_change()[1:][
                self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
                .ffill()
                .pct_change()[1:]
                > zero
            ]
            .count()
        )
        tot = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .ffill()
            .pct_change()
            .count()
        )
        result = pos / tot

        return self._coerce_result(result=result, name="Positive share")

    def ret_vol_ratio_func(
        self: Self,
        riskfree_rate: float = 0.0,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> SeriesOrFloat_co:
        """Ratio between arithmetic mean of returns and annualized volatility.

        The ratio of annualized arithmetic mean of returns and annualized
        volatility or, if riskfree return provided, Sharpe ratio calculated
        as ( geometric return - risk-free return ) / volatility. The latter ratio
        implies that the riskfree asset has zero volatility.
        https://www.investopedia.com/terms/s/sharperatio.asp.

        Parameters
        ----------
        riskfree_rate : float
            The return of the zero volatility asset used to calculate Sharpe ratio
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns:
        -------
        SeriesOrFloat_co
            Ratio of the annualized arithmetic mean of returns and annualized.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.
            volatility or, if risk-free return provided, Sharpe ratio

        """
        result = Series(
            self.arithmetic_ret_func(
                months_from_last=months_from_last,
                from_date=from_date,
                to_date=to_date,
                periods_in_a_year_fixed=periods_in_a_year_fixed,
            )
            - riskfree_rate,
        ) / self.vol_func(
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

        return self._coerce_result(result=result, name="Return vol ratio")

    def sortino_ratio_func(
        self: Self,
        riskfree_rate: float = 0.0,
        min_accepted_return: float = 0.0,
        order: Literal[2, 3] = 2,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> SeriesOrFloat_co:
        """Sortino Ratio or Kappa3 Ratio.

        The Sortino ratio calculated as ( return - risk free return )
        / downside deviation. The ratio implies that the riskfree asset has zero
        volatility, and a minimum acceptable return of zero. The ratio is
        calculated using the annualized arithmetic mean of returns.
        https://www.investopedia.com/terms/s/sortinoratio.asp.
        If order is set to 3 the ratio calculated becomes Kappa3 which
        penalizes larger downside outcomes more heavily than the Sortino
        ratio (which uses order 2).

        Parameters
        ----------
        riskfree_rate : float
            The return of the zero volatility asset
        min_accepted_return : float, optional
            The annualized Minimum Accepted Return (MAR)
        order: int, default: 2
            Order of partial moment
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns:
        -------
        SeriesOrFloat_co
            Sortino ratio calculated as ( return - riskfree return ) /.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.
            downside deviation (std dev of returns below MAR)

        """
        result = Series(
            self.arithmetic_ret_func(
                months_from_last=months_from_last,
                from_date=from_date,
                to_date=to_date,
                periods_in_a_year_fixed=periods_in_a_year_fixed,
            )
            - riskfree_rate,
        ) / self.lower_partial_moment_func(
            min_accepted_return=min_accepted_return,
            order=order,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

        sortino_order = 2
        name = "Sortino ratio" if order == sortino_order else "Kappa-3 ratio"

        return self._coerce_result(result=result, name=name)

    def omega_ratio_func(
        self: Self,
        min_accepted_return: float = 0.0,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> SeriesOrFloat_co:
        """Omega Ratio.

        The Omega Ratio compares returns above a certain target level
        (often referred to as the “minimum acceptable return” or “MAR”)
        to the total downside risk below that same threshold.
        https://en.wikipedia.org/wiki/Omega_ratio.

        Parameters
        ----------
        min_accepted_return : float, optional
            The annualized Minimum Accepted Return (MAR)
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns:
        -------
        SeriesOrFloat_co
            Omega ratio calculation.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        retdf = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .ffill()
            .pct_change()
        )
        pos = retdf[retdf > min_accepted_return].sub(min_accepted_return).sum()
        neg = retdf[retdf < min_accepted_return].sub(min_accepted_return).sum()
        result = pos / -neg

        return self._coerce_result(result=result, name="Omega ratio")

    def value_ret_func(
        self: Self,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> SeriesOrFloat_co:
        """Calculate simple return.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns:
        -------
        SeriesOrFloat_co
            Calculate simple return.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        zero: float = 0.0
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        if zero in self.tsdf.iloc[0].tolist():
            msg = (
                "Simple return cannot be calculated due to "
                f"an initial value being zero. ({self.tsdf.head(3)})"
            )
            raise InitialValueZeroError(msg)

        result = cast(
            "Series[float]",
            self.tsdf.loc[later] / self.tsdf.loc[earlier] - 1,
        )

        return self._coerce_result(result=result, name="Simple return")

    def value_ret_calendar_period(
        self: Self,
        year: int,
        month: int | None = None,
    ) -> SeriesOrFloat_co:
        """Calculate simple return for a specific calendar period.

        Parameters
        ----------
        year : int
            Calendar year of the period to calculate.
        month : int, optional
            Calendar month of the period to calculate.

        Returns:
        -------
        SeriesOrFloat_co
            Calculate simple return for a specific calendar period.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        if month is None:
            period = str(year)
        else:
            period = "-".join([str(year), str(month).zfill(2)])
        vrdf = self.tsdf.copy()
        vrdf.index = DatetimeIndex(vrdf.index)
        resultdf = DataFrame(vrdf.ffill().pct_change())
        plus_one = resultdf.loc[period] + 1
        result = plus_one.cumprod(axis="index").iloc[-1] - 1

        return self._coerce_result(result=result, name=period)

    def var_down_func(
        self: Self,
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        interpolation: LiteralQuantileInterp = "lower",
    ) -> SeriesOrFloat_co:
        """Downside Value At Risk, "VaR".

        The equivalent of percentile.inc([...], 1-level) over returns in MS Excel.
        https://www.investopedia.com/terms/v/var.asp.

        Parameters
        ----------
        level: float, default: 0.95
            The sought VaR level
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        interpolation: LiteralQuantileInterp, default: "lower"
            Type of interpolation in Pandas.DataFrame.quantile() function.

        Returns:
        -------
        SeriesOrFloat_co
            Downside Value At Risk.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .ffill()
            .pct_change()
            .quantile(1 - level, interpolation=interpolation)
        )

        return self._coerce_result(result=result, name=f"VaR {level:.1%}")

    def worst_func(
        self: Self,
        observations: int = 1,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> SeriesOrFloat_co:
        """Most negative percentage change over a rolling number of observations.

        Parameters
        ----------
        observations: int, default: 1
            Number of observations over which to measure the worst outcome
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns:
        -------
        SeriesOrFloat_co
            Most negative percentage change over a rolling number of observations.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.
            within a chosen date range

        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .ffill()
            .pct_change()
            .rolling(observations, min_periods=observations)
            .sum()
            .min()
        )

        return self._coerce_result(result=result, name="Worst")

    def z_score_func(
        self: Self,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> SeriesOrFloat_co:
        """Z-score as (last return - mean return) / standard deviation of returns.

        https://www.investopedia.com/terms/z/zscore.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns:
        -------
        SeriesOrFloat_co
            Z-score as (last return - mean return) / standard deviation of returns.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        zscframe = (
            self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
            .ffill()
            .pct_change()
        )
        result = (zscframe.iloc[-1] - zscframe.mean()) / zscframe.std()

        return self._coerce_result(result=result, name="Z-score")

    def rolling_cvar_down(
        self: Self,
        column: int = 0,
        level: float = 0.95,
        observations: int = 252,
    ) -> DataFrame:
        """Calculate rolling annualized downside CVaR.

        Parameters
        ----------
        column: int, default: 0
            Position as integer of column to calculate
        level: float, default: 0.95
            The sought Conditional Value At Risk level
        observations: int, default: 252
            Number of observations in the overlapping window.

        Returns:
        -------
        Pandas.DataFrame
            Calculate rolling annualized downside CVaR

        """
        cvar_label = cast("tuple[str]", self.tsdf.iloc[:, column].name)[0]
        cvarseries = (
            Series(self.tsdf.iloc[:, column])
            .rolling(observations, min_periods=observations)
            .apply(lambda x: _cvar_down_calc(x, level=level))
        )
        cvardf = cvarseries.dropna().to_frame()
        cvardf.columns = MultiIndex.from_arrays([[cvar_label], ["Rolling CVaR"]])

        return cvardf

    def rolling_return(
        self: Self,
        column: int = 0,
        observations: int = 21,
    ) -> DataFrame:
        """Calculate rolling returns.

        Parameters
        ----------
        column: int, default: 0
            Position as integer of column to calculate
        observations: int, default: 21
            Number of observations in the overlapping window.

        Returns:
        -------
        Pandas.DataFrame
            Calculate rolling returns

        """
        ret_label = cast("tuple[str]", self.tsdf.iloc[:, column].name)[0]
        retseries = (
            Series(self.tsdf.iloc[:, column])
            .ffill()
            .pct_change()
            .rolling(observations, min_periods=observations)
            .sum()
        )
        retdf = retseries.dropna().to_frame()
        retdf.columns = MultiIndex.from_arrays([[ret_label], ["Rolling returns"]])

        return retdf

    def rolling_var_down(
        self: Self,
        column: int = 0,
        level: float = 0.95,
        observations: int = 252,
        interpolation: LiteralQuantileInterp = "lower",
    ) -> DataFrame:
        """Calculate rolling annualized downside Value At Risk "VaR".

        Parameters
        ----------
        column: int, default: 0
            Position as integer of column to calculate
        level: float, default: 0.95
            The sought Value At Risk level
        observations: int, default: 252
            Number of observations in the overlapping window.
        interpolation: LiteralQuantileInterp, default: "lower"
            Type of interpolation in Pandas.DataFrame.quantile() function.

        Returns:
        -------
        Pandas.DataFrame
           Calculate rolling annualized downside Value At Risk "VaR"

        """
        var_label = cast("tuple[str]", self.tsdf.iloc[:, column].name)[0]
        varseries = (
            Series(self.tsdf.iloc[:, column])
            .rolling(observations, min_periods=observations)
            .apply(
                lambda x: _var_down_calc(x, level=level, interpolation=interpolation),
            )
        )
        vardf = varseries.dropna().to_frame()
        vardf.columns = MultiIndex.from_arrays([[var_label], ["Rolling VaR"]])

        return vardf

    def rolling_vol(
        self: Self,
        column: int = 0,
        observations: int = 21,
        periods_in_a_year_fixed: DaysInYearType | None = None,
        dlta_degr_freedms: int = 1,
    ) -> DataFrame:
        """Calculate rolling annualised volatilities.

        Parameters
        ----------
        column: int, default: 0
            Position as integer of column to calculate
        observations: int, default: 21
            Number of observations in the overlapping window.
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons
        dlta_degr_freedms: int, default: 1
            Variance bias factor taking the value 0 or 1.

        Returns:
        -------
        Pandas.DataFrame
            Calculate rolling annualised volatilities

        """
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = self.periods_in_a_year

        vol_label = cast("tuple[str, ValueType]", self.tsdf.iloc[:, column].name)[0]

        s = log(self.tsdf.iloc[:, column]).diff()
        volseries = s.rolling(window=observations, min_periods=observations).std(
            ddof=dlta_degr_freedms,
        ) * sqrt(time_factor)

        voldf = volseries.dropna().to_frame()

        voldf.columns = MultiIndex.from_arrays(
            [
                [vol_label],
                ["Rolling volatility"],
            ],
        )

        return DataFrame(voldf)
