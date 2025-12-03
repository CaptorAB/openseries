"""The _CommonModel class.

_CommonModel class which is the base class for OpenFrame and OpenTimeSeries.
"""

from __future__ import annotations

import datetime as dt
from inspect import stack
from json import dump
from math import ceil
from pathlib import Path
from secrets import choice
from string import ascii_letters
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, cast

from numpy import asarray, float64, inf, isnan, log, maximum, sqrt

from .owntypes import (
    CaptorLogoType,
    DateAlignmentError,
    InitialValueZeroError,
    NumberOfItemsAndLabelsNotSameError,
    PlotlyConfigType,
    ResampleDataLossError,
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

    Args:
        self: The instance.
        months_from_last: Number of months offset as a positive integer. Overrides
            use of ``from_date`` and ``to_date``.
        from_date: Specific from date.
        to_date: Specific to date.
        periods_in_a_year_fixed: Allows locking the periods-in-a-year to simplify
            test cases and comparisons.

    Returns:
        A tuple of ``(earlier, later, time_factor, data)`` where ``earlier`` and
        ``later`` are the selected dates, ``time_factor`` is the inferred periods
        per year, and ``data`` is the sliced ``DataFrame``.
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

    Args:
        self: The instance.
        base_column: Column reference as a ``(label, ValueType)`` tuple or
            integer position.
        earlier: Start date.
        later: End date.

    Returns:
        A tuple ``(data, item, label)`` where ``data`` is the selected series,
        ``item`` is the resolved column key and ``label`` its first-level label.
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

    Args:
        data: Data series for counting observations.
        earlier: Start date.
        later: End date.
        periods_in_a_year_fixed: Fixed periods in year.

    Returns:
        Time factor expressed as observations per year.
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
        series_result: SeriesOrFloat_co = Series(  # type: ignore[assignment]
            data=result,
            index=self.tsdf.columns,
            name=name,
            dtype="float64",
        )
        return series_result

    @property
    def length(self: Self) -> int:
        """Number of observations.

        Returns:
            Number of observations.
        """
        return len(self.tsdf.index)

    @property
    def first_idx(self: Self) -> dt.date:
        """The first date in the timeseries.

        Returns:
            The first date in the timeseries.
        """
        return cast("dt.date", self.tsdf.index[0])

    @property
    def last_idx(self: Self) -> dt.date:
        """The last date in the timeseries.

        Returns:
            The last date in the timeseries.
        """
        return cast("dt.date", self.tsdf.index[-1])

    @property
    def span_of_days(self: Self) -> int:
        """Number of days from the first date to the last.

        Returns:
            Number of days from the first date to the last.
        """
        return (self.last_idx - self.first_idx).days

    @property
    def yearfrac(self: Self) -> float:
        """Length of series in years assuming 365.25 days per year.

        Returns:
            Length of the timeseries in years assuming 365.25 days per year.
        """
        return self.span_of_days / 365.25

    @property
    def periods_in_a_year(self: Self) -> float:
        """The average number of observations per year.

        Returns:
            The average number of observations per year.
        """
        return self.length / self.yearfrac

    @property
    def max_drawdown_cal_year(self: Self) -> SeriesOrFloat_co:
        """Maximum drawdown in a single calendar year.

        Reference: https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Returns:
        --------
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
        """Compounded Annual Growth Rate (CAGR).

        Reference: https://www.investopedia.com/terms/c/cagr.asp.

        Returns:
        --------
        SeriesOrFloat_co
            Compounded Annual Growth Rate (CAGR).
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.geo_ret_func()

    @property
    def arithmetic_ret(self: Self) -> SeriesOrFloat_co:
        """Annualized arithmetic mean of returns.

        Reference: https://www.investopedia.com/terms/a/arithmeticmean.asp.

        Returns:
        --------
        SeriesOrFloat_co
            Annualized arithmetic mean of returns.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.arithmetic_ret_func()

    @property
    def value_ret(self: Self) -> SeriesOrFloat_co:
        """Simple return.

        Returns:
        --------
        SeriesOrFloat_co
            Simple return.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.value_ret_func()

    @property
    def vol(self: Self) -> SeriesOrFloat_co:
        """Annualized volatility.

        Based on Pandas .std() which is the equivalent of stdev.s([...]) in MS Excel.

        Reference: https://www.investopedia.com/terms/v/volatility.asp.

        Returns:
        --------
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

        Reference: https://www.investopedia.com/terms/d/downside-deviation.asp.

        Returns:
        --------
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
        --------
        SeriesOrFloat_co
            Ratio of the annualized arithmetic mean of returns and annualized
            volatility.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.

        """
        riskfree_rate: float = 0.0
        return self.ret_vol_ratio_func(riskfree_rate=riskfree_rate)

    @property
    def sortino_ratio(self: Self) -> SeriesOrFloat_co:
        """Sortino ratio.

        Reference: https://www.investopedia.com/terms/s/sortinoratio.asp.

        Returns:
        --------
        SeriesOrFloat_co
            Sortino ratio calculated as the annualized arithmetic mean of returns
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
        --------
        SeriesOrFloat_co
            Kappa-3 ratio calculation with the riskfree rate and.
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
        """Omega ratio.

        Reference: https://en.wikipedia.org/wiki/Omega_ratio.

        Returns:
        --------
        SeriesOrFloat_co
            Omega ratio calculation.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        minimum_accepted_return: float = 0.0
        return self.omega_ratio_func(min_accepted_return=minimum_accepted_return)

    @property
    def z_score(self: Self) -> SeriesOrFloat_co:
        """Z-score.

        Reference: https://www.investopedia.com/terms/z/zscore.asp.

        Returns:
        --------
        SeriesOrFloat_co
            Z-score as (last return - mean return) / standard deviation of returns.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.z_score_func()

    @property
    def max_drawdown(self: Self) -> SeriesOrFloat_co:
        """Maximum drawdown without any limit on date range.

        Reference: https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Returns:
        --------
        SeriesOrFloat_co
            Maximum drawdown without any limit on date range.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.max_drawdown_func()

    @property
    def max_drawdown_date(self: Self) -> dt.date | Series[dt.date]:
        """Date when the maximum drawdown occurred.

        Reference: https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Returns:
        --------
        datetime.date | pandas.Series[dt.date]
            Date when the maximum drawdown occurred

        """
        mdddf = self.tsdf.copy()
        mdddf.index = DatetimeIndex(mdddf.index)
        result = (mdddf / mdddf.expanding(min_periods=1).max()).idxmin().dt.date  # type: ignore[attr-defined,arg-type]

        if self.tsdf.shape[1] == 1:
            return cast("dt.date", result.iloc[0])
        date_series = Series(
            data=result,
            index=self.tsdf.columns,
            name="Max drawdown date",
            dtype="datetime64[ns]",
        ).dt.date  # type: ignore[attr-defined]
        return cast("Series[dt.date]", date_series)

    @property
    def worst(self: Self) -> SeriesOrFloat_co:
        """Most negative percentage change.

        Returns:
        --------
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
        --------
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
        --------
        SeriesOrFloat_co
            The share of percentage changes that are greater than zero.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.positive_share_func()

    @property
    def skew(self: Self) -> SeriesOrFloat_co:
        """Skew of the return distribution.

        Reference: https://www.investopedia.com/terms/s/skewness.asp.

        Returns:
        --------
        SeriesOrFloat_co
            Skew of the return distribution.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.skew_func()

    @property
    def kurtosis(self: Self) -> SeriesOrFloat_co:
        """Kurtosis of the return distribution.

        Reference: https://www.investopedia.com/terms/k/kurtosis.asp.

        Returns:
        --------
        SeriesOrFloat_co
            Kurtosis of the return distribution.
            Returns float for OpenTimeSeries, Series[float] for OpenFrame.


        """
        return self.kurtosis_func()

    @property
    def cvar_down(self: Self) -> SeriesOrFloat_co:
        """Downside 95% Conditional Value At Risk "CVaR".

        Reference: https://www.investopedia.com/terms/c/conditional_value_at_risk.asp.

        Returns:
        --------
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
        --------
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
        --------
        SeriesOrFloat_co
            Implied annualized volatility from the Downside 95% VaR using the
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
        """Create a user-defined date range aligned to index.

        Args:
            months_offset: Number of months offset as a positive integer. Overrides
                use of ``from_dt`` and ``to_dt``.
            from_dt: Specific from date.
            to_dt: Specific to date.

        Returns:
            A tuple ``(earlier, later)`` representing the start and end date of the
            chosen date range aligned to existing index values.

        Raises:
            DateAlignmentError: If the implied range is outside series bounds.
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
        """Align the index of ``.tsdf`` with local calendar business days.

        Args:
            countries: Country code(s) (ISO 3166-1 alpha-2).
            markets: Market code(s) supported by ``exchange_calendars``.
            custom_holidays: Missing holidays that should be added.
            method: Method for reindexing when aligning to business days.

        Returns:
            The modified object.
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
        """Convert value series to log-weighted series.

        Equivalent to ``LN(value[t] / value[t=0])`` in Excel.

        Returns:
            The modified object.
        """
        self.tsdf = DataFrame(
            data=log(self.tsdf / self.tsdf.iloc[0]),
            index=self.tsdf.index,
            columns=self.tsdf.columns,
        )
        return self

    def value_nan_handle(self: Self, method: LiteralNanMethod = "fill") -> Self:
        """Handle missing values in a value series.

        Args:
            method: Method used to handle NaN. Either ``"fill"`` (last known) or
                ``"drop"``.

        Returns:
            The modified object.
        """
        if method == "fill":
            self.tsdf = self.tsdf.ffill()
        else:
            self.tsdf = self.tsdf.dropna()
        return self

    def return_nan_handle(self: Self, method: LiteralNanMethod = "fill") -> Self:
        """Handle missing values in a return series.

        Args:
            method: Method used to handle NaN. Either ``"fill"`` (zero) or
                ``"drop"``.

        Returns:
            The modified object.
        """
        if method == "fill":
            self.tsdf = self.tsdf.fillna(value=0.0)
        else:
            self.tsdf = self.tsdf.dropna()
        return self

    def to_drawdown_series(self: Self) -> Self:
        """Convert timeseries into a drawdown series.

        Returns:
            The modified object.
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
        """Dump timeseries data into a JSON file.

        Args:
            what_output: Whether to export raw values or ``tsdf`` values.
            filename: Filename including extension.
            directory: Folder where the file will be written.

        Returns:
            A list of dictionaries with the data of the series.
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
        """Save ``.tsdf`` DataFrame to an Excel spreadsheet file.

        Args:
            filename: Filename that should include ``.xlsx``.
            sheet_title: Name of the sheet in the Excel file.
            directory: Directory where the Excel file is saved.
            overwrite: Whether to overwrite an existing file.

        Returns:
            The Excel file path.

        Raises:
            NameError: If ``filename`` does not end with ``.xlsx``.
            FileExistsError: If the file exists and ``overwrite`` is False.
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
        """Validate or infer labels for plotting.

        Args:
            ncols: Number of columns expected.
            labels: Provided labels, if any.
            default_labels: Labels to use if ``labels`` is ``None``.

        Returns:
            A list of labels with length ``ncols``.

        Raises:
            NumberOfItemsAndLabelsNotSameError: If ``labels`` length does not match
                ``ncols``.
        """
        if labels:
            if len(labels) != ncols:
                msg = "Must provide same number of labels as items in frame."
                raise NumberOfItemsAndLabelsNotSameError(msg)
            return labels
        return default_labels

    @staticmethod
    def _resolve_dir(directory: DirectoryPath | None) -> Path:
        """Resolve output directory for plot files.

        Args:
            directory: Optional directory override.

        Returns:
            Resolved directory path.
        """
        if directory:
            return Path(directory).resolve()
        if (Path.home() / "Documents").exists():
            return Path.home() / "Documents"
        return Path(stack()[2].filename).parent

    @staticmethod
    def _hover_xy(tick_fmt: str | None) -> str:
        """Create hovertemplate for y-value and date x-axis.

        Args:
            tick_fmt: Plotly tick format string for the y-axis.

        Returns:
            Plotly hovertemplate string.
        """
        return (
            f"%{{y:{tick_fmt}}}<br>%{{x|{'%Y-%m-%d'}}}"
            if tick_fmt
            else "%{y}<br>%{x|%Y-%m-%d}"
        )

    @staticmethod
    def _hover_hist(x_fmt: str | None, y_fmt: str | None) -> str:
        """Create hovertemplate for histogram plots.

        Args:
            x_fmt: Plotly tick format string for the x-axis.
            y_fmt: Plotly tick format string for the y-axis.

        Returns:
            Plotly hovertemplate string.
        """
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
        """Apply optional title and logo to a Plotly Figure.

        Args:
            figure: Plotly figure to update.
            logo: Plotly layout image dict.
            title: Optional plot title.
            add_logo: Whether to add the logo to the figure.
        """
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
        include_plotlyjs_bool: LiteralPlotlyJSlib,
        auto_open: bool,
    ) -> str:
        """Write a file or return inline HTML string from a Plotly Figure.

        Args:
            figure: Plotly figure to render.
            fig_config: Plotly config dict.
            output_type: Output type: ``"file"`` or ``"div"``.
            plotfile: Full path to the output html file.
            filename: Output filename used for the ``div_id`` when inline.
            include_plotlyjs_bool: How plotly.js is included.
            auto_open: Whether to auto-open the file in a browser.

        Returns:
            If ``output_type`` is ``"file"``, the path to the file; otherwise an
            inline HTML string (div).
        """
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

        Args:
            mode: The type of bar to use.
            title: A title above the plot.
            tick_fmt: Tick format for the y-axis, e.g. ``'%'`` or ``'.1%'``.
            filename: Name of the Plotly HTML file.
            directory: Directory where the Plotly HTML file is saved.
            labels: Labels to override the column names of ``self.tsdf``.
            output_type: Determines output type.
            include_plotlyjs: How the plotly.js library is included.
            auto_open: Whether to open a browser window with the plot.
            add_logo: If True, a Captor logo is added to the plot.

        Returns:
            A tuple ``(figure, output)`` where ``output`` is either a div string or
            a file path.
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
            include_plotlyjs_bool=include_plotlyjs,
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

        Args:
            mode: The type of scatter to use.
            title: A title above the plot.
            tick_fmt: Tick format for the y-axis, e.g. ``'%'`` or ``'.1%'``.
            filename: Name of the Plotly HTML file.
            directory: Directory where the Plotly HTML file is saved.
            labels: Labels to override the column names of ``self.tsdf``.
            output_type: Determines output type.
            include_plotlyjs: How the plotly.js library is included.
            auto_open: Whether to open a browser window with the plot.
            add_logo: If True, a Captor logo is added to the plot.
            show_last: If True, highlight the last point in red with a label.

        Returns:
            A tuple ``(figure, output)`` where ``output`` is either a div string or
            a file path.
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
            include_plotlyjs_bool=include_plotlyjs,
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

        Args:
            plot_type: Type of plot, ``"bars"`` or ``"lines"``.
            histnorm: Normalization mode.
            barmode: How bar traces are displayed relative to one another.
            xbins_size: Width of each bin along the x-axis in data units.
            opacity: Trace opacity between 0 and 1.
            bargap: Gap between bars of adjacent location coordinates.
            bargroupgap: Gap between bar groups at the same location coordinate.
            curve_type: Type of distribution curve to overlay on the histogram.
            title: A title above the plot.
            x_fmt: Tick format for the x-axis.
            y_fmt: Tick format for the y-axis.
            filename: Name of the Plotly HTML file.
            directory: Directory where the Plotly HTML file is saved.
            labels: Labels to override the column names of ``self.tsdf``.
            output_type: Determines output type.
            include_plotlyjs: How the plotly.js library is included.
            cumulative: Whether to compute a cumulative histogram.
            show_rug: Whether to draw a rug plot alongside the distribution.
            auto_open: Whether to open a browser window with the plot.
            add_logo: If True, a Captor logo is added to the plot.

        Returns:
            A tuple ``(figure, output)`` where ``output`` is either a div string or
            a file path.
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
            include_plotlyjs_bool=include_plotlyjs,
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
        """Annualized arithmetic mean of returns.

        Reference: ``https://www.investopedia.com/terms/a/arithmeticmean.asp``.

        Args:
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            periods_in_a_year_fixed: Lock periods-in-a-year to simplify tests and
                comparisons.

        Returns:
            Annualized arithmetic mean of returns. Float for OpenTimeSeries,
            ``Series[float]`` for OpenFrame.
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

        Based on ``pandas.Series.std()`` (Excel ``STDEV.S`` equivalent).
        Reference: ``https://www.investopedia.com/terms/v/volatility.asp``.

        Args:
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            periods_in_a_year_fixed: Lock periods-in-a-year to simplify tests and
                comparisons.

        Returns:
            Annualized volatility. Float for OpenTimeSeries, ``Series[float]`` for
            OpenFrame.
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
        """Implied annualized volatility from downside VaR.

        Assumes normally distributed returns.

        Args:
            level: The sought VaR level.
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            interpolation: Interpolation type used by ``DataFrame.quantile``.
            periods_in_a_year_fixed: Lock periods-in-a-year to simplify tests and
                comparisons.
            drift_adjust: Adjustment to remove the bias implied by the average
                return.

        Returns:
            Implied annualized volatility. Float for OpenTimeSeries,
            ``Series[float]`` for OpenFrame.
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

        Computes a position weight multiplier from the ratio between a VaR implied
        volatility and a given target volatility. Multiplier = 1.0 → target met.

        Args:
            target_vol: Target volatility.
            level: The sought VaR level.
            min_leverage_local: Minimum adjustment factor.
            max_leverage_local: Maximum adjustment factor.
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            interpolation: Interpolation type used by ``DataFrame.quantile``.
            periods_in_a_year_fixed: Lock periods-in-a-year to simplify tests and
                comparisons.
            drift_adjust: Adjustment to remove the bias implied by the average
                return.

        Returns:
            Weight multiplier (or implied volatility if used downstream). Float for
            OpenTimeSeries, ``Series[float]`` for OpenFrame.
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

        If ``target_vol`` is provided, returns a weight multiplier from the ratio
        between a VaR implied volatility and ``target_vol``; otherwise returns the
        VaR-implied volatility. Multiplier = 1.0 → target met.

        Args:
            level: The sought VaR level.
            target_vol: Target volatility.
            min_leverage_local: Minimum adjustment factor.
            max_leverage_local: Maximum adjustment factor.
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            interpolation: Interpolation type used by ``DataFrame.quantile``.
            periods_in_a_year_fixed: Lock periods-in-a-year to simplify tests and
                comparisons.
            drift_adjust: Adjustment to remove the bias implied by the average
                return.

        Returns:
            Target weight multiplier if ``target_vol`` is provided; otherwise the
            VaR-implied volatility. Float for OpenTimeSeries, ``Series[float]`` for
            OpenFrame.
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
        """Downside Conditional Value At Risk (CVaR).

        Reference: ``https://www.investopedia.com/terms/c/conditional_value_at_risk.asp``.

        Args:
            level: The sought CVaR level.
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            Downside CVaR. Float for OpenTimeSeries, ``Series[float]`` for
            OpenFrame.
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
        """Lower partial moment and downside deviation (order=2).

        If ``order`` is 2 calculates standard deviation of returns below MAR=0.
        For general order ``p``, returns ``(LPM_p)^(1/p)``.

        Args:
            min_accepted_return: Annualized Minimum Accepted Return (MAR).
            order: Order of partial moment (2 or 3).
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            periods_in_a_year_fixed: Lock periods-in-a-year to simplify tests and
                comparisons.

        Returns:
            Downside deviation if ``order`` is 2; otherwise rooted lower partial
            moment. Float for OpenTimeSeries, ``Series[float]`` for OpenFrame.

        Raises:
            ValueError: If ``order`` is not 2 or 3.
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

        Reference: ``https://www.investopedia.com/terms/c/cagr.asp``.

        Args:
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            CAGR. Float for OpenTimeSeries, ``Series[float]`` for OpenFrame.

        Raises:
            InitialValueZeroError: If initial value is zero or there are negative
                values.
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

        Reference: ``https://www.investopedia.com/terms/s/skewness.asp``.

        Args:
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            Skewness. Float for OpenTimeSeries, ``Series[float]`` for OpenFrame.
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

        Reference: ``https://www.investopedia.com/terms/k/kurtosis.asp``.

        Args:
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            Kurtosis. Float for OpenTimeSeries, ``Series[float]`` for OpenFrame.
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

        Reference: ``https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp``.

        Args:
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            min_periods: Smallest number of observations for rolling max.

        Returns:
            Maximum drawdown. Float for OpenTimeSeries, ``Series[float]`` for
            OpenFrame.
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
        """Share of percentage changes greater than zero.

        Args:
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            Share of positive returns. Float for OpenTimeSeries, ``Series[float]``
            for OpenFrame.
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

        If ``riskfree_rate`` provided, computes the Sharpe ratio as
        ``(arithmetic return - risk-free) / volatility``. Assumes zero volatility
        for the risk-free asset. Reference:
        ``https://www.investopedia.com/terms/s/sharperatio.asp``.

        Args:
            riskfree_rate: Return of the zero volatility asset.
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            periods_in_a_year_fixed: Lock periods-in-a-year to simplify tests and
                comparisons.

        Returns:
            Ratio value. Float for OpenTimeSeries, ``Series[float]`` for OpenFrame.
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
        """Sortino ratio or Kappa-3 ratio.

        Sortino: ``(return - riskfree_rate) / downside deviation`` using arithmetic
        mean of returns. Kappa-3 when ``order=3`` penalizes larger downside more
        than Sortino.

        Args:
            riskfree_rate: Return of the zero volatility asset.
            min_accepted_return: Annualized Minimum Accepted Return (MAR).
            order: Order of partial moment (2 for Sortino, 3 for Kappa-3).
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            periods_in_a_year_fixed: Lock periods-in-a-year to simplify tests and
                comparisons.

        Returns:
            Ratio value. Float for OpenTimeSeries, ``Series[float]`` for OpenFrame.
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

        Compares returns above MAR to the total downside risk below MAR.
        Reference: ``https://en.wikipedia.org/wiki/Omega_ratio``.

        Args:
            min_accepted_return: Annualized Minimum Accepted Return (MAR).
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            Omega ratio. Float for OpenTimeSeries, ``Series[float]`` for OpenFrame.
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

        Args:
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            Simple return. Float for OpenTimeSeries, ``Series[float]`` for
            OpenFrame.

        Raises:
            InitialValueZeroError: If initial value is zero.
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

        Args:
            year: Calendar year of the period to calculate.
            month: Calendar month of the period to calculate.

        Returns:
            Simple return for the period. Float for OpenTimeSeries,
            ``Series[float]`` for OpenFrame.
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
        """Downside Value At Risk (VaR).

        Equivalent to ``PERCENTILE.INC(returns, 1-level)`` in Excel. Reference:
        ``https://www.investopedia.com/terms/v/var.asp``.

        Args:
            level: The sought VaR level.
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.
            interpolation: Interpolation used by ``DataFrame.quantile``.

        Returns:
            Downside VaR. Float for OpenTimeSeries, ``Series[float]`` for OpenFrame.
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
        """Most negative percentage change over a rolling window.

        Args:
            observations: Number of observations for the rolling window.
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            Most negative percentage change. Float for OpenTimeSeries,
            ``Series[float]`` for OpenFrame.
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
        """Z-score of the last return.

        Computed as ``(last return - mean return) / std dev of returns``.
        Reference: ``https://www.investopedia.com/terms/z/zscore.asp``.

        Args:
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            Z-score. Float for OpenTimeSeries, ``Series[float]`` for OpenFrame.
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

        Args:
            column: Column position to calculate.
            level: Conditional Value At Risk level.
            observations: Number of observations in the overlapping window.

        Returns:
            DataFrame with rolling annualized downside CVaR.
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

        Args:
            column: Column position to calculate.
            observations: Number of observations in the overlapping window.

        Returns:
            DataFrame with rolling returns.
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
        """Calculate rolling annualized downside Value At Risk (VaR).

        Args:
            column: Column position to calculate.
            level: Value At Risk level.
            observations: Number of observations in the overlapping window.
            interpolation: Interpolation used by ``DataFrame.quantile``.

        Returns:
            DataFrame with rolling annualized downside VaR.
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
        """Calculate rolling annualized volatilities.

        Args:
            column: Column position to calculate.
            observations: Number of observations in the overlapping window.
            periods_in_a_year_fixed: Lock periods-in-a-year to simplify tests and
                comparisons.
            dlta_degr_freedms: Variance bias factor (0 or 1).

        Returns:
            DataFrame with rolling annualized volatilities.
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

    def outliers(
        self: Self,
        threshold: float = 3.0,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> Series | DataFrame:
        """Detect outliers using z-score analysis.

        Identifies data points where the absolute z-score exceeds the threshold.
        For OpenTimeSeries, returns a Series with dates and outlier values. For
        OpenFrame, returns a DataFrame with dates and outlier values for each
        column.

        Args:
            threshold: Z-score threshold; values with ``|z| > threshold`` are
                outliers.
            months_from_last: Number of months offset as positive integer. Overrides
                use of ``from_date`` and ``to_date``.
            from_date: Specific from date.
            to_date: Specific to date.

        Returns:
            For OpenTimeSeries: Series of outliers. For OpenFrame: DataFrame of
            outliers. Empty if none found.
        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )

        # Get the data for the specified date range
        data = self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]

        # Calculate z-scores for each column
        z_scores = (data - data.mean()) / data.std()

        # Find outliers where |z-score| > threshold
        outliers_mask = z_scores.abs() > threshold

        if self.tsdf.shape[1] == 1:
            # OpenTimeSeries case - return Series
            outlier_values = data[outliers_mask].iloc[:, 0].dropna()
            return Series(
                data=outlier_values.values,
                index=outlier_values.index,
                name="Outliers",
                dtype="float64",
            )
        # OpenFrame case - return DataFrame
        outlier_df = data[outliers_mask].dropna(how="all")
        return DataFrame(
            data=outlier_df,
            dtype="float64",
        )
