"""
Defining the OpenTimeSeries class
"""
from __future__ import annotations
from copy import deepcopy
import datetime as dt
from re import compile as re_compile
from typing import cast, Optional, Type, Union, TypeVar
from numpy import (
    array,
    cumprod,
    float64,
    insert,
    log,
    sqrt,
)
from numpy.typing import NDArray
from pandas import (
    DataFrame,
    DatetimeIndex,
    date_range,
    MultiIndex,
    Series,
)
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from stdnum import isin as isincode
from stdnum.exceptions import InvalidChecksum

from openseries.common_model import CommonModel
from openseries.common_tools import (
    do_resample_to_business_period_ends,
    get_calc_range,
    check_if_none,
)
from openseries.datefixer import (
    date_fix,
    align_dataframe_to_local_cdays,
)
from openseries.types import (
    CountriesType,
    CurrencyStringType,
    DatabaseIdStringType,
    DateListType,
    ValueListType,
    LiteralQuantileInterp,
    LiteralBizDayFreq,
    LiteralPandasResampleConvention,
    LiteralPandasReindexMethod,
    LiteralSeriesProps,
    OpenTimeSeriesPropertiesList,
    ValueType,
)
from openseries.risk import (
    cvar_down_calc,
    var_down_calc,
    drawdown_details,
    ewma_calc,
)

TypeOpenTimeSeries = TypeVar("TypeOpenTimeSeries", bound="OpenTimeSeries")


class OpenTimeSeries(BaseModel, CommonModel):
    """Object of the class OpenTimeSeries. Subclass of the Pydantic BaseModel

    Parameters
    ----------
    timeseriesId : DatabaseIdStringType
        Database identifier of the timeseries
    instrumentId: DatabaseIdStringType
        Database identifier of the instrument associated with the timeseries
    name : str
        string identifier of the timeseries and/or instrument
    valuetype : openseries.types.ValueType
        Identifies if the series is a series of values or returns
    dates : DateListType
        Dates of the individual timeseries items
        These dates will not be altered by methods
    values : ValueListType
        The value or return values of the timeseries items
        These values will not be altered by methods
    local_ccy: bool
        Boolean flag indicating if timeseries is in local currency
    tsdf: pandas.DataFrame
        Pandas object holding dates and values that can be altered via methods
    currency : CurrencyStringType
        ISO 4217 currency code of the timeseries
    domestic : CurrencyStringType, default: "SEK"
        ISO 4217 currency code of the user's home currency
    countries: Union[CountryStringType, CountryListType], default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    isin : str, optional
        ISO 6166 identifier code of the associated instrument
    label : str, optional
        Placeholder for a name of the timeseries
    """

    timeseriesId: DatabaseIdStringType
    instrumentId: DatabaseIdStringType
    name: str
    valuetype: ValueType
    dates: DateListType
    values: ValueListType
    local_ccy: bool
    tsdf: DataFrame
    currency: CurrencyStringType
    domestic: CurrencyStringType = "SEK"
    countries: CountriesType = "SE"
    isin: Optional[str] = None
    label: Optional[str] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="always",
        extra="allow",
    )

    # noinspection PyMethodParameters
    @field_validator("isin")
    def check_isincode(  # pylint: disable=no-self-argument
        cls: TypeOpenTimeSeries, isin_code: str
    ) -> str:
        """Pydantic validator to ensure that the ISIN code is valid if provided"""
        if isin_code:
            try:
                isincode.validate(isin_code)
            except InvalidChecksum as exc:
                raise ValueError(
                    "The ISIN code's checksum or check digit is invalid."
                ) from exc
        return isin_code

    @model_validator(mode="after")
    def check_dates_unique(self) -> OpenTimeSeries:
        """Pydantic validator to ensure that the dates are unique"""
        dates_list_length = len(self.dates)
        dates_set_length = len(set(self.dates))
        if dates_list_length != dates_set_length:
            raise ValueError("Dates are not unique")
        return self

    @classmethod
    def setup_class(
        cls: Type[TypeOpenTimeSeries],
        domestic_ccy: CurrencyStringType = "SEK",
        countries: CountriesType = "SE",
    ) -> None:
        """Sets the domestic currency and calendar of the user.

        Parameters
        ----------
        domestic_ccy : str, default: "SEK"
            Currency code according to ISO 4217
        countries: list[str] | str, default: "SE"
            (List of) country code(s) according to ISO 3166-1 alpha-2
        """
        ccy_pattern = re_compile(r"^[A-Z]{3}$")
        ctry_pattern = re_compile(r"^[A-Z]{2}$")
        try:
            ccy_ok = ccy_pattern.match(domestic_ccy)
        except TypeError as exc:
            raise ValueError(
                "domestic currency must be a code according to ISO 4217"
            ) from exc
        if not ccy_ok:
            raise ValueError("domestic currency must be a code according to ISO 4217")
        if isinstance(countries, str):
            if not ctry_pattern.match(countries):
                raise ValueError(
                    "countries must be a country code according to "
                    "ISO 3166-1 alpha-2"
                )
        elif isinstance(countries, list):
            try:
                all_ctries = all(ctry_pattern.match(ctry) for ctry in countries)
            except TypeError as exc:
                raise ValueError(
                    "countries must be a list of country codes "
                    "according to ISO 3166-1 alpha-2"
                ) from exc
            if not all_ctries:
                raise ValueError(
                    "countries must be a list of country codes "
                    "according to ISO 3166-1 alpha-2"
                )
        else:
            raise ValueError(
                "countries must be a (list of) country code(s) "
                "according to ISO 3166-1 alpha-2"
            )

        cls.domestic = domestic_ccy
        cls.countries = countries

    @classmethod
    def from_arrays(
        cls: Type[TypeOpenTimeSeries],
        name: str,
        dates: DateListType,
        values: ValueListType,
        valuetype: ValueType = ValueType.PRICE,
        timeseries_id: DatabaseIdStringType = "",
        instrument_id: DatabaseIdStringType = "",
        isin: Optional[str] = None,
        baseccy: CurrencyStringType = "SEK",
        local_ccy: bool = True,
    ) -> TypeOpenTimeSeries:
        """Creates a timeseries from a Pandas DataFrame or Series

        Parameters
        ----------
        name: str
            string identifier of the timeseries and/or instrument
        dates: DateListType
            List of date strings as ISO 8601 YYYY-MM-DD
        values: ValueListType
            Array of float values
        valuetype : ValueType, default: ValueType.PRICE
            Identifies if the series is a series of values or returns
        timeseries_id : DatabaseIdStringType, optional
            Database identifier of the timeseries
        instrument_id: DatabaseIdStringType, optional
            Database identifier of the instrument associated with the timeseries
        isin : str, optional
            ISO 6166 identifier code of the associated instrument
        baseccy : CurrencyStringType, default: "SEK"
            ISO 4217 currency code of the timeseries
        local_ccy: bool, default: True
            Boolean flag indicating if timeseries is in local currency

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        return cls(
            name=name,
            label=name,
            dates=dates,
            values=values,
            valuetype=valuetype,
            timeseriesId=timeseries_id,
            instrumentId=instrument_id,
            isin=isin,
            currency=baseccy,
            local_ccy=local_ccy,
            tsdf=DataFrame(
                data=values,
                index=[deyt.date() for deyt in DatetimeIndex(dates)],
                columns=[[name], [valuetype]],
                dtype="float64",
            ),
        )

    @classmethod
    def from_df(
        cls: Type[TypeOpenTimeSeries],
        dframe: Union[DataFrame, Series],
        column_nmbr: int = 0,
        valuetype: ValueType = ValueType.PRICE,
        baseccy: CurrencyStringType = "SEK",
        local_ccy: bool = True,
    ) -> TypeOpenTimeSeries:
        """Creates a timeseries from a Pandas DataFrame or Series

        Parameters
        ----------
        dframe: Union[DataFrame, Series]
            Pandas DataFrame or Series
        column_nmbr : int, default: 0
            Using iloc[:, column_nmbr] to pick column
        valuetype : ValueType, default: ValueType.PRICE
            Identifies if the series is a series of values or returns
        baseccy : CurrencyStringType, default: "SEK"
            ISO 4217 currency code of the timeseries
        local_ccy: bool, default: True
            Boolean flag indicating if timeseries is in local currency

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        if isinstance(dframe, Series):
            if isinstance(dframe.name, tuple):
                label, _ = dframe.name
            else:
                label = dframe.name
            values = dframe.values.tolist()
        else:
            values = dframe.iloc[:, column_nmbr].tolist()
            if isinstance(dframe.columns, MultiIndex):
                if check_if_none(
                    dframe.columns.get_level_values(0).values[column_nmbr]
                ):
                    print(
                        "checked item",
                        dframe.columns.get_level_values(0).values[column_nmbr],
                    )
                    label = "Series"
                    print(f"label missing. Adding '{label}' as label")
                else:
                    label = dframe.columns.get_level_values(0).values[column_nmbr]
                if check_if_none(
                    dframe.columns.get_level_values(1).values[column_nmbr]
                ):
                    valuetype = ValueType.PRICE
                    print(
                        f"valuetype missing. Adding '{valuetype.value}' as valuetype"
                    )
                else:
                    valuetype = dframe.columns.get_level_values(1).values[column_nmbr]
            else:
                label = dframe.columns.values[column_nmbr]
        dates = [date_fix(d).strftime("%Y-%m-%d") for d in dframe.index]

        return cls(
            timeseriesId="",
            instrumentId="",
            currency=baseccy,
            dates=dates,
            name=label,
            label=label,
            valuetype=valuetype,
            values=values,
            local_ccy=local_ccy,
            tsdf=DataFrame(
                data=values,
                index=[deyt.date() for deyt in DatetimeIndex(dates)],
                columns=[[label], [valuetype]],
                dtype="float64",
            ),
        )

    @classmethod
    def from_fixed_rate(
        cls: Type[TypeOpenTimeSeries],
        rate: float,
        d_range: Optional[DatetimeIndex] = None,
        days: Optional[int] = None,
        end_dt: Optional[dt.date] = None,
        label: str = "Series",
        valuetype: ValueType = ValueType.PRICE,
        baseccy: CurrencyStringType = "SEK",
        local_ccy: bool = True,
    ) -> TypeOpenTimeSeries:
        """Creates a timeseries from values accruing with a given fixed rate return

        Providing a date_range of type Pandas DatetimeIndex takes priority over
        providing a combination of days and an end date.

        Parameters
        ----------
        rate: float
            The accrual rate
        d_range: DatetimeIndex, optional
            A given range of dates
        days: int, optional
            Number of days to generate when date_range not provided. Must be combined
            with end_dt
        end_dt: datetime.date, optional
            End date of date range to generate when date_range not provided. Must be
            combined with days
        label : str
            Placeholder for a name of the timeseries
        valuetype : ValueType, default: ValueType.PRICE
            Identifies if the series is a series of values or returns
        baseccy : CurrencyStringType, default: "SEK"
            The currency of the timeseries
        local_ccy: bool, default: True
            Boolean flag indicating if timeseries is in local currency

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        if not isinstance(d_range, DatetimeIndex) and all([days, end_dt]):
            d_range = DatetimeIndex(
                [d.date() for d in date_range(periods=days, end=end_dt, freq="D")]
            )
        elif not isinstance(d_range, DatetimeIndex) and not all([days, end_dt]):
            raise ValueError(
                "If d_range is not provided both days and end_dt must be."
            )

        deltas = array(
            [
                i.days
                for i in cast(DatetimeIndex, d_range)[1:]
                - cast(DatetimeIndex, d_range)[:-1]
            ]
        )
        arr = list(cumprod(insert(1 + deltas * rate / 365, 0, 1.0)))
        d_range = [d.strftime("%Y-%m-%d") for d in cast(DatetimeIndex, d_range)]

        return cls(
            timeseriesId="",
            instrumentId="",
            currency=baseccy,
            dates=d_range,
            name=label,
            label=label,
            valuetype=valuetype,
            values=arr,
            local_ccy=local_ccy,
            tsdf=DataFrame(
                data=arr,
                index=[d.date() for d in DatetimeIndex(d_range)],
                columns=[[label], [valuetype]],
                dtype="float64",
            ),
        )

    def from_deepcopy(self: TypeOpenTimeSeries) -> TypeOpenTimeSeries:
        """Creates a copy of an OpenTimeSeries object

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        return deepcopy(self)

    def pandas_df(self: TypeOpenTimeSeries) -> TypeOpenTimeSeries:
        """Sets the .tsdf parameter as a Pandas DataFrame from the .dates and
        .values lists

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        dframe = DataFrame(
            data=self.values,
            index=self.dates,
            columns=[[self.label], [self.valuetype]],
            dtype="float64",
        )
        dframe.index = [d.date() for d in DatetimeIndex(dframe.index)]

        dframe.sort_index(inplace=True)
        self.tsdf = dframe

        return self

    def calc_range(
        self: TypeOpenTimeSeries,
        months_offset: Optional[int] = None,
        from_dt: Optional[dt.date] = None,
        to_dt: Optional[dt.date] = None,
    ) -> tuple[dt.date, dt.date]:
        """Creates user defined date range

        Parameters
        ----------
        months_offset: int, optional
            Number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_dt: datetime.date, optional
            Specific from date
        to_dt: datetime.date, optional
            Specific from date

        Returns
        -------
        (datetime.date, datetime.date)
            Start and end date of the chosen date range
        """
        return get_calc_range(
            data=self.tsdf, months_offset=months_offset, from_dt=from_dt, to_dt=to_dt
        )

    def align_index_to_local_cdays(self: TypeOpenTimeSeries) -> TypeOpenTimeSeries:
        """Changes the index of the associated Pandas DataFrame .tsdf to align with
        local calendar business days

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        self.tsdf = align_dataframe_to_local_cdays(
            data=self.tsdf, countries=self.countries
        )
        return self

    def all_properties(
        self: TypeOpenTimeSeries, properties: Optional[list[LiteralSeriesProps]] = None
    ) -> DataFrame:
        """Calculates the chosen timeseries properties

        Parameters
        ----------
        properties: list[LiteralSeriesProps], optional
            The properties to calculate. Defaults to calculating all available.

        Returns
        -------
        pandas.DataFrame
            Properties of the OpenTimeSeries
        """

        if not properties:
            properties = cast(
                list[LiteralSeriesProps], OpenTimeSeriesPropertiesList.allowed_strings
            )

        props = OpenTimeSeriesPropertiesList(*properties)
        pdf = DataFrame.from_dict({x: getattr(self, x) for x in props}, orient="index")
        pdf.columns = self.tsdf.columns
        return pdf

    @property
    def geo_ret(self: TypeOpenTimeSeries) -> float:
        """https://www.investopedia.com/terms/c/cagr.asp

        Returns
        -------
        float
            Compounded Annual Growth Rate (CAGR)
        """
        return self.geo_ret_func()

    @property
    def arithmetic_ret(self: TypeOpenTimeSeries) -> float:
        """https://www.investopedia.com/terms/a/arithmeticmean.asp

        Returns
        -------
        float
            Annualized arithmetic mean of returns
        """

        return self.arithmetic_ret_func()

    @property
    def value_ret(self: TypeOpenTimeSeries) -> float:
        """
        Returns
        -------
        float
            Simple return
        """
        return self.value_ret_func()

    @property
    def vol(self: TypeOpenTimeSeries) -> float:
        """Based on Pandas .std() which is the equivalent of stdev.s([...])
        in MS Excel \n
        https://www.investopedia.com/terms/v/volatility.asp

        Returns
        -------
        float
            Annualized volatility
        """

        return self.vol_func()

    @property
    def downside_deviation(self: TypeOpenTimeSeries) -> float:
        """The standard deviation of returns that are below a Minimum Accepted
        Return of zero.
        It is used to calculate the Sortino Ratio \n
        https://www.investopedia.com/terms/d/downside-deviation.asp

        Returns
        -------
        float
            Downside deviation
        """
        min_accepted_return: float = 0.0
        return self.downside_deviation_func(min_accepted_return=min_accepted_return)

    @property
    def ret_vol_ratio(self: TypeOpenTimeSeries) -> float:
        """
        Returns
        -------
        float
            Ratio of the annualized arithmetic mean of returns and annualized
            volatility.
        """
        riskfree_rate: float = 0.0
        return self.ret_vol_ratio_func(riskfree_rate=riskfree_rate)

    @property
    def sortino_ratio(self: TypeOpenTimeSeries) -> float:
        """https://www.investopedia.com/terms/s/sortinoratio.asp

        Returns
        -------
        float
        Pandas.Series
            Sortino ratio calculated as the annualized arithmetic mean of returns
            / downside deviation. The ratio implies that the riskfree asset has zero
            volatility, and a minimum acceptable return of zero.
        """
        riskfree_rate: float = 0.0
        return self.sortino_ratio_func(riskfree_rate=riskfree_rate)

    @property
    def z_score(self: TypeOpenTimeSeries) -> float:
        """https://www.investopedia.com/terms/z/zscore.asp

        Returns
        -------
        float
            Z-score as (last return - mean return) / standard deviation of returns.
        """
        return self.z_score_func()

    @property
    def max_drawdown(self: TypeOpenTimeSeries) -> float:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        float
            Maximum drawdown without any limit on date range
        """
        return self.max_drawdown_func()

    @property
    def max_drawdown_date(self: TypeOpenTimeSeries) -> dt.date:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        datetime.date
            Date when the maximum drawdown occurred
        """

        mdddf = self.tsdf.copy()
        mdddf.index = DatetimeIndex(mdddf.index)
        mdd_date = (mdddf / mdddf.expanding(min_periods=1).max()).idxmin().values[0]
        return dt.datetime.strptime(str(mdd_date)[:10], "%Y-%m-%d").date()

    @property
    def worst(self: TypeOpenTimeSeries) -> float:
        """
        Returns
        -------
        float
            Most negative percentage change
        """
        observations: int = 1
        return self.worst_func(observations=observations)

    @property
    def worst_month(self: TypeOpenTimeSeries) -> float:
        """
        Returns
        -------
        float
            Most negative month
        """

        resdf = self.tsdf.copy()
        resdf.index = DatetimeIndex(resdf.index)
        return float((resdf.resample("BM").last().pct_change().min()).iloc[0])

    @property
    def positive_share(self: TypeOpenTimeSeries) -> float:
        """
        Returns
        -------
        float
            The share of percentage changes that are greater than zero
        """
        return self.positive_share_func()

    @property
    def skew(self: TypeOpenTimeSeries) -> float:
        """https://www.investopedia.com/terms/s/skewness.asp

        Returns
        -------
        float
            Skew of the return distribution
        """
        return self.skew_func()

    @property
    def kurtosis(self: TypeOpenTimeSeries) -> float:
        """https://www.investopedia.com/terms/k/kurtosis.asp

        Returns
        -------
        float
            Kurtosis of the return distribution
        """
        return self.kurtosis_func()

    @property
    def cvar_down(self: TypeOpenTimeSeries) -> float:
        """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

        Returns
        -------
        float
            Downside Conditional 95% Value At Risk "CVaR"
        """
        level: float = 0.95
        return self.cvar_down_func(level=level)

    @property
    def var_down(self: TypeOpenTimeSeries) -> float:
        """Downside 95% Value At Risk, "VaR". The equivalent of
        percentile.inc([...], 1-level) over returns in MS Excel \n
        https://www.investopedia.com/terms/v/var.asp

        Returns
        -------
        float
            Downside 95% Value At Risk
        """
        level: float = 0.95
        interpolation: LiteralQuantileInterp = "lower"
        return self.var_down_func(level=level, interpolation=interpolation)

    @property
    def vol_from_var(self: TypeOpenTimeSeries) -> float:
        """
        Returns
        -------
        float
            Implied annualized volatility from the Downside 95% VaR using the
            assumption that returns are normally distributed.
        """
        level: float = 0.95
        interpolation: LiteralQuantileInterp = "lower"
        return self.vol_from_var_func(level=level, interpolation=interpolation)

    def value_to_ret(self: TypeOpenTimeSeries) -> TypeOpenTimeSeries:
        """
        Returns
        -------
        OpenTimeSeries
            The returns of the values in the series
        """

        self.tsdf = self.tsdf.pct_change()
        self.tsdf.iloc[0] = 0
        self.valuetype = ValueType.RTRN
        self.tsdf.columns = [[self.label], [self.valuetype]]
        return self

    def value_to_diff(
        self: TypeOpenTimeSeries, periods: int = 1
    ) -> TypeOpenTimeSeries:
        """Converts a valueseries to a series of its period differences

        Parameters
        ----------
        periods: int, default: 1
            The number of periods between observations over which difference
            is calculated

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.tsdf = self.tsdf.diff(periods=periods)
        self.tsdf.iloc[0] = 0
        self.valuetype = ValueType.RTRN
        self.tsdf.columns = [[self.label], [self.valuetype]]
        return self

    def to_cumret(self: TypeOpenTimeSeries) -> TypeOpenTimeSeries:
        """Converts a returnseries into a cumulative valueseries

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        if not any(
            x == ValueType.RTRN
            for x in cast(MultiIndex, self.tsdf.columns).get_level_values(1).values
        ):
            self.value_to_ret()

        self.tsdf = self.tsdf.add(1.0)
        self.tsdf = self.tsdf.cumprod(axis=0) / self.tsdf.iloc[0]
        self.valuetype = ValueType.PRICE
        self.tsdf.columns = [[self.label], [self.valuetype]]

        return self

    def from_1d_rate_to_cumret(
        self: TypeOpenTimeSeries, days_in_year: int = 365, divider: float = 1.0
    ) -> TypeOpenTimeSeries:
        """Converts a series of 1-day rates into a cumulative valueseries

        Parameters
        ----------
        days_in_year: int, default 365
            Calendar days per year used as divisor
        divider: float, default 100.0
            Convenience divider for when the 1-day rate is not scaled correctly

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        arr = array(self.values) / divider

        deltas = array([i.days for i in self.tsdf.index[1:] - self.tsdf.index[:-1]])
        arr = cumprod(insert(1.0 + deltas * arr[:-1] / days_in_year, 0, 1.0))

        self.dates = [d.strftime("%Y-%m-%d") for d in self.tsdf.index]
        self.values = list(arr)
        self.valuetype = ValueType.PRICE
        self.pandas_df()

        return self

    def resample(
        self: TypeOpenTimeSeries, freq: Union[LiteralBizDayFreq, str] = "BM"
    ) -> TypeOpenTimeSeries:
        """Resamples the timeseries frequency

        Parameters
        ----------
        freq: Union[LiteralBizDayFreq, str], default "BM"
            The date offset string that sets the resampled frequency
            Examples are "7D", "B", "M", "BM", "Q", "BQ", "A", "BA"

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.tsdf.index = DatetimeIndex(self.tsdf.index)
        self.tsdf = self.tsdf.resample(freq).last()
        self.tsdf.index = [d.date() for d in DatetimeIndex(self.tsdf.index)]
        return self

    def resample_to_business_period_ends(
        self: TypeOpenTimeSeries,
        freq: LiteralBizDayFreq = "BM",
        convention: LiteralPandasResampleConvention = "end",
        method: LiteralPandasReindexMethod = "nearest",
    ) -> TypeOpenTimeSeries:
        """Resamples timeseries frequency to the business calendar
        month end dates of each period while leaving any stubs
        in place

        Parameters
        ----------
        freq: LiteralBizDayFreq, default BM
            The date offset string that sets the resampled frequency
        convention: LiteralPandasResampleConvention, default; end
            Controls whether to use the start or end of `rule`.
        method: LiteralPandasReindexMethod, default: nearest
            Controls the method used to align values across columns

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        head = self.tsdf.iloc[0].copy()
        tail = self.tsdf.iloc[-1].copy()
        dates = do_resample_to_business_period_ends(
            data=self.tsdf,
            head=head,
            tail=tail,
            freq=freq,
            countries=self.countries,
            convention=convention,
        )
        self.tsdf = self.tsdf.reindex([deyt.date() for deyt in dates], method=method)
        return self

    def drawdown_details(self: TypeOpenTimeSeries) -> DataFrame:
        """
        Returns
        -------
        Pandas.DataFrame
            Calculates 'Max Drawdown', 'Start of drawdown', 'Date of bottom',
            'Days from start to bottom', & 'Average fall per day'
        """

        dddf = self.tsdf.copy()
        dddf.index = DatetimeIndex(dddf.index)
        return drawdown_details(dddf).to_frame()

    def ewma_vol_func(
        self: TypeOpenTimeSeries,
        lmbda: float = 0.94,
        day_chunk: int = 11,
        dlta_degr_freedms: int = 0,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> Series:
        """Exponentially Weighted Moving Average Model for Volatility.
        https://www.investopedia.com/articles/07/ewma.asp

        Parameters
        ----------
        lmbda: float, default: 0.94
            Scaling factor to determine weighting.
        day_chunk: int, default: 0
            Sampling the data which is assumed to be daily.
        dlta_degr_freedms: int, default: 0
            Variance bias factor taking the value 0 or 1.
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        Pandas.DataFrame
            Series EWMA volatility
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            how_many = self.tsdf.loc[
                cast(int, earlier) : cast(int, later), self.tsdf.columns.values[0]
            ].count()
            fraction = (later - earlier).days / 365.25
            time_factor = how_many / fraction

        data = self.tsdf.loc[cast(int, earlier) : cast(int, later)].copy()

        data[self.label, "Returns"] = (
            data.loc[:, self.tsdf.columns.values[0]].apply(log).diff()
        )

        rawdata = [
            data.loc[:, (self.label, "Returns")]
            .iloc[1:day_chunk]
            .std(ddof=dlta_degr_freedms)
            * sqrt(time_factor)
        ]

        for item in data.loc[:, (self.label, "Returns")].iloc[1:]:
            previous = rawdata[-1]
            rawdata.append(
                ewma_calc(
                    reeturn=item,
                    prev_ewma=previous,
                    time_factor=time_factor,
                    lmbda=lmbda,
                )
            )

        data.loc[:, (self.label, ValueType.EWMA)] = rawdata

        return data.loc[:, (self.label, ValueType.EWMA)]

    def rolling_vol(
        self: TypeOpenTimeSeries,
        observations: int = 21,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> DataFrame:
        """
        Parameters
        ----------
        observations: int, default: 21
            Number of observations in the overlapping window.
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        Pandas.DataFrame
            Rolling annualised volatilities
        """

        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = self.periods_in_a_year
        dframe = self.tsdf.pct_change().copy()
        voldf = dframe.rolling(observations, min_periods=observations).std() * sqrt(
            time_factor
        )
        voldf.dropna(inplace=True)
        voldf.columns = [[self.label], ["Rolling volatility"]]

        return voldf

    def rolling_return(self: TypeOpenTimeSeries, observations: int = 21) -> DataFrame:
        """
        Parameters
        ----------
        observations: int, default: 21
            Number of observations in the overlapping window.

        Returns
        -------
        Pandas.DataFrame
            Rolling returns
        """

        retdf = (
            self.tsdf.pct_change()
            .rolling(observations, min_periods=observations)
            .sum()
        )
        retdf.columns = [[self.label], ["Rolling returns"]]

        return retdf.dropna()

    def rolling_cvar_down(
        self: TypeOpenTimeSeries, level: float = 0.95, observations: int = 252
    ) -> DataFrame:
        """
        Parameters
        ----------
        level: float, default: 0.95
            The sought Conditional Value At Risk level
        observations: int, default: 252
            Number of observations in the overlapping window.

        Returns
        -------
        Pandas.DataFrame
            Rolling annualized downside CVaR
        """

        cvardf = self.tsdf.rolling(observations, min_periods=observations).apply(
            lambda x: cvar_down_calc(x, level=level)
        )
        cvardf = cvardf.dropna()
        cvardf.columns = [[self.label], ["Rolling CVaR"]]

        return cvardf

    def rolling_var_down(
        self: TypeOpenTimeSeries,
        level: float = 0.95,
        observations: int = 252,
        interpolation: LiteralQuantileInterp = "lower",
    ) -> DataFrame:
        """
        Parameters
        ----------
        level: float, default: 0.95
            The sought Value At Risk level
        observations: int, default: 252
            Number of observations in the overlapping window.
        interpolation: LiteralQuantileInterp, default: "lower"
            Type of interpolation in Pandas.DataFrame.quantile() function.

        Returns
        -------
        Pandas.DataFrame
           Rolling annualized downside Value At Risk "VaR"
        """

        vardf = self.tsdf.rolling(observations, min_periods=observations).apply(
            lambda x: var_down_calc(x, level=level, interpolation=interpolation)
        )
        vardf = vardf.dropna()
        vardf.columns = [[self.label], ["Rolling VaR"]]

        return vardf

    def running_adjustment(
        self: TypeOpenTimeSeries, adjustment: float, days_in_year: int = 365
    ) -> TypeOpenTimeSeries:
        """Adds (+) or subtracts (-) a fee from the timeseries return

        Parameters
        ----------
        adjustment: float
            Fee to add or subtract
        days_in_year: int, default: 365
            The calculation divisor and
            assumed number of days in a calendar year

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        values: list[float]
        if any(
            x == ValueType.RTRN
            for x in cast(MultiIndex, self.tsdf.columns).get_level_values(1).values
        ):
            ra_df = self.tsdf.copy()
            values = [1.0]
            returns_input = True
        else:
            values = [self.tsdf.iloc[0, 0]]
            ra_df = self.tsdf.pct_change().copy()
            returns_input = False
        ra_df.dropna(inplace=True)

        prev = self.first_idx
        idx: dt.date
        dates: list[dt.date] = [prev]

        for idx, row in ra_df.iterrows():
            dates.append(idx)
            values.append(
                values[-1]
                * (1 + row.iloc[0] + adjustment * (idx - prev).days / days_in_year)
            )
            prev = idx
        self.tsdf = DataFrame(data=values, index=dates)
        self.valuetype = ValueType.PRICE
        self.tsdf.columns = [[self.label], [self.valuetype]]
        self.tsdf.index = [d.date() for d in DatetimeIndex(self.tsdf.index)]
        if returns_input:
            self.value_to_ret()
        return self

    def set_new_label(
        self: TypeOpenTimeSeries,
        lvl_zero: Optional[str] = None,
        lvl_one: Optional[ValueType] = None,
        delete_lvl_one: bool = False,
    ) -> TypeOpenTimeSeries:
        """Sets the column labels of the .tsdf Pandas Dataframe associated
        with the timeseries

        Parameters
        ----------
        lvl_zero: str, optional
            New level zero label
        lvl_one: str, optional
            New level one label
        delete_lvl_one: bool, default: False
            If True the level one label is deleted

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        if lvl_zero is None and lvl_one is None:
            self.tsdf.columns = MultiIndex.from_arrays(
                [[self.label], [self.valuetype]]
            )
        elif lvl_zero is not None and lvl_one is None:
            self.tsdf.columns = MultiIndex.from_arrays([[lvl_zero], [self.valuetype]])
            self.label = lvl_zero
        elif lvl_zero is None and lvl_one is not None:
            self.tsdf.columns = MultiIndex.from_arrays([[self.label], [lvl_one]])
            self.valuetype = lvl_one
        else:
            self.tsdf.columns = MultiIndex.from_arrays([[lvl_zero], [lvl_one]])
            self.label, self.valuetype = lvl_zero, cast(ValueType, lvl_one)
        if delete_lvl_one:
            self.tsdf.columns = self.tsdf.columns.droplevel(level=1)
        return self


def timeseries_chain(
    front: TypeOpenTimeSeries,
    back: TypeOpenTimeSeries,
    old_fee: float = 0.0,
) -> Union[TypeOpenTimeSeries, OpenTimeSeries]:
    """Chain two timeseries together

    Parameters
    ----------
    front: TypeOpenTimeSeries
        Earlier series to chain with
    back: TypeOpenTimeSeries
        Later series to chain with
    old_fee: bool, default: False
        Fee to apply to earlier series

    Returns
    -------
    TypeOpenTimeSeries
        An OpenTimeSeries object or a subclass thereof
    """
    old = front.from_deepcopy()
    old.running_adjustment(old_fee)
    olddf = old.tsdf.copy()
    new = back.from_deepcopy()
    idx = 0
    first = new.tsdf.index[idx]

    assert (
        old.last_idx >= first
    ), "Timeseries dates must overlap to allow them to be chained."

    while first not in olddf.index:
        idx += 1
        first = new.tsdf.index[idx]
        if first > olddf.index[-1]:
            raise ValueError("Failed to find a matching date between series")

    dates: list[str] = [x.strftime("%Y-%m-%d") for x in olddf.index if x < first]
    values = array([x[0] for x in old.tsdf.values][: len(dates)])
    values = cast(
        NDArray[float64],
        list(values * new.tsdf.iloc[:, 0].loc[first] / olddf.iloc[:, 0].loc[first]),
    )

    dates.extend([x.strftime("%Y-%m-%d") for x in new.tsdf.index])
    values += [x[0] for x in new.tsdf.values]

    if back.__class__.__subclasscheck__(  # pylint: disable=unnecessary-dunder-call
        OpenTimeSeries
    ):
        return OpenTimeSeries(
            timeseriesId=new.timeseriesId,
            instrumentId=new.instrumentId,
            currency=new.currency,
            dates=dates,
            name=new.name,
            label=new.name,
            valuetype=new.valuetype,
            values=list(values),
            local_ccy=new.local_ccy,
            tsdf=DataFrame(
                data=values,
                index=[d.date() for d in DatetimeIndex(dates)],
                columns=[[new.label], [new.valuetype]],
                dtype="float64",
            ),
        )
    return back.__class__(
        timeseriesId=new.timeseriesId,
        instrumentId=new.instrumentId,
        currency=new.currency,
        dates=dates,
        name=new.name,
        label=new.name,
        valuetype=new.valuetype,
        values=list(values),
        local_ccy=new.local_ccy,
        tsdf=DataFrame(
            data=values,
            index=[d.date() for d in DatetimeIndex(dates)],
            columns=[[new.label], [new.valuetype]],
            dtype="float64",
        ),
    )
