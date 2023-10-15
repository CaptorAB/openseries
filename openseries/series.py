"""Defining the OpenTimeSeries class."""
# mypy: disable-error-code="unused-ignore"
from __future__ import annotations

import datetime as dt
from copy import deepcopy
from logging import warning
from typing import Any, Optional, TypeVar, Union, cast

from numpy import (
    append,
    array,
    cumprod,
    insert,
    isnan,
    log,
    sqrt,
)
from pandas import (
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Series,
    date_range,
)
from pydantic import model_validator

from openseries.common_model import CommonModel
from openseries.datefixer import (
    align_dataframe_to_local_cdays,
    date_fix,
    do_resample_to_business_period_ends,
    get_calc_range,
)
from openseries.risk import (
    drawdown_details,
    ewma_calc,
)
from openseries.types import (
    Countries,
    CountriesType,
    Currency,
    CurrencyStringType,
    DatabaseIdStringType,
    DateListType,
    LiteralBizDayFreq,
    LiteralPandasReindexMethod,
    LiteralPandasResampleConvention,
    LiteralSeriesProps,
    OpenTimeSeriesPropertiesList,
    ValueListType,
    ValueType,
)

TypeOpenTimeSeries = TypeVar("TypeOpenTimeSeries", bound="OpenTimeSeries")


class OpenTimeSeries(CommonModel):  # type: ignore[misc]

    """
    OpenTimeSeries objects are at the core of the openseries package.

    The intended use is to allow analyses of financial timeseries.
    It is only intended for daily or less frequent data samples.

    Parameters
    ----------
    timeseries_id : DatabaseIdStringType
        Database identifier of the timeseries
    instrument_id: DatabaseIdStringType
        Database identifier of the instrument associated with the timeseries
    name : str
        string identifier of the timeseries and/or instrument
    valuetype : ValueType
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
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    isin : str, optional
        ISO 6166 identifier code of the associated instrument
    label : str, optional
        Placeholder for a name of the timeseries
    """

    timeseries_id: DatabaseIdStringType
    instrument_id: DatabaseIdStringType
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

    @model_validator(mode="after")  # type: ignore[misc]
    def dates_and_values_validate(self: OpenTimeSeries) -> OpenTimeSeries:
        """Pydantic validator to ensure dates and values are validated."""
        values_list_length = len(self.values)
        dates_list_length = len(self.dates)
        dates_set_length = len(set(self.dates))
        if dates_list_length != dates_set_length:
            msg = "Dates are not unique"
            raise ValueError(msg)
        if values_list_length < 1:
            msg = "There must be at least 1 value"
            raise ValueError(msg)
        if (
            (dates_list_length != values_list_length)
            or (len(self.tsdf.index) != self.tsdf.shape[0])
            or (self.tsdf.shape[1] != 1)
        ):
            msg = "Number of dates and values passed do not match"
            raise ValueError(msg)
        return self

    @classmethod
    def setup_class(
        cls: type[OpenTimeSeries],
        domestic_ccy: CurrencyStringType = "SEK",
        countries: CountriesType = "SE",
    ) -> None:
        """
        Set the domestic currency and calendar of the user.

        Parameters
        ----------
        domestic_ccy : CurrencyStringType, default: "SEK"
            Currency code according to ISO 4217
        countries: CountriesType, default: "SE"
            (List of) country code(s) according to ISO 3166-1 alpha-2
        """
        _ = Currency(ccy=domestic_ccy)
        _ = Countries(countryinput=countries)

        cls.domestic = domestic_ccy
        cls.countries = countries

    @classmethod
    def from_arrays(
        cls: type[OpenTimeSeries],
        name: str,
        dates: DateListType,
        values: ValueListType,
        valuetype: ValueType = ValueType.PRICE,
        timeseries_id: DatabaseIdStringType = "",
        instrument_id: DatabaseIdStringType = "",
        isin: Optional[str] = None,
        baseccy: CurrencyStringType = "SEK",
        local_ccy: bool = True,  # noqa: FBT001, FBT002
    ) -> OpenTimeSeries:
        """
        Create series from a Pandas DataFrame or Series.

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
            timeseries_id=timeseries_id,
            instrument_id=instrument_id,
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
        cls: type[OpenTimeSeries],
        dframe: Union[DataFrame, Series[float]],
        column_nmbr: int = 0,
        valuetype: ValueType = ValueType.PRICE,
        baseccy: CurrencyStringType = "SEK",
        local_ccy: bool = True,  # noqa: FBT001, FBT002
    ) -> OpenTimeSeries:
        """
        Create series from a Pandas DataFrame or Series.

        Parameters
        ----------
        dframe: Union[DataFrame, Series[type[float]]]
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
            values = dframe.to_numpy().tolist()
        else:
            values = cast(DataFrame, dframe).iloc[:, column_nmbr].tolist()
            if isinstance(dframe.columns, MultiIndex):
                if check_if_none(
                    dframe.columns.get_level_values(0).to_numpy()[column_nmbr],
                ):
                    label = "Series"
                    msg = f"Label missing. Adding: {label}"
                    warning(msg=msg)
                else:
                    label = dframe.columns.get_level_values(0).to_numpy()[column_nmbr]
                if check_if_none(
                    dframe.columns.get_level_values(1).to_numpy()[column_nmbr],
                ):
                    valuetype = ValueType.PRICE
                    msg = f"valuetype missing. Adding: {valuetype.value}"
                    warning(msg=msg)
                else:
                    valuetype = dframe.columns.get_level_values(1).to_numpy()[
                        column_nmbr
                    ]
            else:
                label = cast(MultiIndex, dframe.columns).to_numpy()[column_nmbr]
        dates = [date_fix(d).strftime("%Y-%m-%d") for d in dframe.index]

        return cls(
            timeseries_id="",
            instrument_id="",
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
        cls: type[OpenTimeSeries],
        rate: float,
        d_range: Optional[DatetimeIndex] = None,
        days: Optional[int] = None,
        end_dt: Optional[dt.date] = None,
        label: str = "Series",
        valuetype: ValueType = ValueType.PRICE,
        baseccy: CurrencyStringType = "SEK",
        local_ccy: bool = True,  # noqa: FBT001, FBT002
    ) -> OpenTimeSeries:
        """
        Create series from values accruing with a given fixed rate return.

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
                [d.date() for d in date_range(periods=days, end=end_dt, freq="D")],
            )
        elif not isinstance(d_range, DatetimeIndex) and not all([days, end_dt]):
            msg = "If d_range is not provided both days and end_dt must be."
            raise ValueError(
                msg,
            )

        deltas = array(
            [
                i.days
                for i in cast(DatetimeIndex, d_range)[1:]
                - cast(DatetimeIndex, d_range)[:-1]
            ],
        )
        arr = list(cumprod(insert(1 + deltas * rate / 365, 0, 1.0)))
        dates = [d.strftime("%Y-%m-%d") for d in cast(DatetimeIndex, d_range)]

        return cls(
            timeseries_id="",
            instrument_id="",
            currency=baseccy,
            dates=dates,
            name=label,
            label=label,
            valuetype=valuetype,
            values=arr,
            local_ccy=local_ccy,
            tsdf=DataFrame(
                data=arr,
                index=[d.date() for d in DatetimeIndex(dates)],
                columns=[[label], [valuetype]],
                dtype="float64",
            ),
        )

    def from_deepcopy(self: OpenTimeSeries) -> OpenTimeSeries:
        """
        Create copy of OpenTimeSeries object.

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        return deepcopy(self)

    def pandas_df(self: OpenTimeSeries) -> OpenTimeSeries:
        """
        Populate .tsdf Pandas DataFrame from the .dates and .values lists.

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        dframe = DataFrame(
            data=self.values,
            index=[d.date() for d in DatetimeIndex(self.dates)],
            columns=[[self.label], [self.valuetype]],
            dtype="float64",
        )
        self.tsdf = dframe

        return self

    def calc_range(
        self: OpenTimeSeries,
        months_offset: Optional[int] = None,
        from_dt: Optional[dt.date] = None,
        to_dt: Optional[dt.date] = None,
    ) -> tuple[dt.date, dt.date]:
        """
        Create user defined date range.

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
            data=self.tsdf,
            months_offset=months_offset,
            from_dt=from_dt,
            to_dt=to_dt,
        )

    def align_index_to_local_cdays(self: OpenTimeSeries) -> OpenTimeSeries:
        """
        Align the index .tsdf with local calendar business days.

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        self.tsdf = align_dataframe_to_local_cdays(
            data=self.tsdf,
            countries=self.countries,
        )
        return self

    def all_properties(
        self: OpenTimeSeries,
        properties: Optional[list[LiteralSeriesProps]] = None,
    ) -> DataFrame:
        """
        Calculate chosen properties.

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
                list[LiteralSeriesProps],
                OpenTimeSeriesPropertiesList.allowed_strings,
            )

        props = OpenTimeSeriesPropertiesList(*properties)
        pdf = DataFrame.from_dict({x: getattr(self, x) for x in props}, orient="index")
        pdf.columns = self.tsdf.columns
        return pdf

    @property
    def worst_month(self: OpenTimeSeries) -> float:
        """
        Most negative month.

        Returns
        -------
        float
            Most negative month
        """
        resdf = self.tsdf.copy()
        resdf.index = DatetimeIndex(resdf.index)
        return float((resdf.resample("BM").last().ffill().pct_change().min()).iloc[0])

    def value_to_ret(self: OpenTimeSeries) -> OpenTimeSeries:
        """
        Convert series of values into series of returns.

        Returns
        -------
        OpenTimeSeries
            The returns of the values in the series
        """
        self.tsdf = self.tsdf.ffill().pct_change()
        self.tsdf.iloc[0] = 0
        self.valuetype = ValueType.RTRN
        self.tsdf.columns = [  # type: ignore[assignment]
            [self.label],
            [self.valuetype],
        ]
        return self

    def value_to_diff(
        self: OpenTimeSeries,
        periods: int = 1,
    ) -> OpenTimeSeries:
        """
        Convert series of values to series of their period differences.

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
        self.tsdf.columns = [  # type: ignore[assignment]
            [self.label],
            [self.valuetype],
        ]
        return self

    def to_cumret(self: OpenTimeSeries) -> OpenTimeSeries:
        """
        Convert series of returns into cumulative series of values.

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        if not any(
            x == ValueType.RTRN
            for x in cast(MultiIndex, self.tsdf.columns).get_level_values(1).to_numpy()
        ):
            self.value_to_ret()

        self.tsdf = self.tsdf.add(1.0)
        self.tsdf = self.tsdf.cumprod(axis=0) / self.tsdf.iloc[0]
        self.valuetype = ValueType.PRICE
        self.tsdf.columns = [  # type: ignore[assignment]
            [self.label],
            [self.valuetype],
        ]
        return self

    def from_1d_rate_to_cumret(
        self: OpenTimeSeries,
        days_in_year: int = 365,
        divider: float = 1.0,
    ) -> OpenTimeSeries:
        """
        Convert series of 1-day rates into series of cumulative values.

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
        self.tsdf = DataFrame(
            data=self.values,
            index=[d.date() for d in DatetimeIndex(self.dates)],
            columns=[[self.label], [self.valuetype]],
            dtype="float64",
        )

        return self

    def resample(
        self: OpenTimeSeries,
        freq: Union[LiteralBizDayFreq, str] = "BM",
    ) -> OpenTimeSeries:
        """
        Resamples the timeseries frequency.

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
        self.tsdf.index = [  # type: ignore[assignment]
            d.date() for d in DatetimeIndex(self.tsdf.index)
        ]
        return self

    def resample_to_business_period_ends(
        self: OpenTimeSeries,
        freq: LiteralBizDayFreq = "BM",
        convention: LiteralPandasResampleConvention = "end",
        method: LiteralPandasReindexMethod = "nearest",
    ) -> OpenTimeSeries:
        """
        Resamples timeseries frequency to the business calendar month end dates.

        Stubs left in place. Stubs will be aligned to the shortest stub.

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

    def drawdown_details(self: OpenTimeSeries) -> DataFrame:
        """
        Details of the maximum drawdown.

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
        self: OpenTimeSeries,
        lmbda: float = 0.94,
        day_chunk: int = 11,
        dlta_degr_freedms: int = 0,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> DataFrame:
        """
        Exponentially Weighted Moving Average Model for Volatility.

        https://www.investopedia.com/articles/07/ewma.asp.

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
                cast(int, earlier) : cast(int, later),
                self.tsdf.columns.to_numpy()[0],
            ].count()
            fraction = (later - earlier).days / 365.25
            time_factor = how_many / fraction

        data = self.tsdf.loc[cast(int, earlier) : cast(int, later)].copy()

        data[self.label, "Returns"] = (
            data.loc[:, self.tsdf.columns.to_numpy()[0]].apply(log).diff()
        )

        rawdata = [
            data.loc[:, (self.label, "Returns")]  # type: ignore[index]
            .iloc[1:day_chunk]
            .std(ddof=dlta_degr_freedms)
            * sqrt(time_factor),
        ]

        for item in data.loc[:, (self.label, "Returns")].iloc[  # type: ignore[index]
            1:
        ]:
            previous = rawdata[-1]
            rawdata.append(
                ewma_calc(
                    reeturn=cast(float, item),
                    prev_ewma=previous,
                    time_factor=time_factor,
                    lmbda=lmbda,
                ),
            )

        data.loc[:, (self.label, ValueType.EWMA)] = rawdata  # type: ignore[index]

        return data.loc[:, (self.label, ValueType.EWMA)]  # type: ignore[index]

    def running_adjustment(
        self: OpenTimeSeries,
        adjustment: float,
        days_in_year: int = 365,
    ) -> OpenTimeSeries:
        """
        Add or subtract a fee from the timeseries return.

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
            for x in cast(MultiIndex, self.tsdf.columns).get_level_values(1).to_numpy()
        ):
            ra_df = self.tsdf.copy()
            values = [1.0]
            returns_input = True
        else:
            values = [cast(float, self.tsdf.iloc[0, 0])]
            ra_df = self.tsdf.ffill().pct_change()
            returns_input = False
        ra_df = ra_df.dropna()

        prev = self.first_idx
        idx: dt.date
        dates: list[dt.date] = [prev]

        for idx, row in ra_df.iterrows():  # type: ignore[assignment]
            dates.append(idx)
            values.append(
                values[-1]
                * (1 + row.iloc[0] + adjustment * (idx - prev).days / days_in_year),
            )
            prev = idx
        self.tsdf = DataFrame(data=values, index=dates)
        self.valuetype = ValueType.PRICE
        self.tsdf.columns = [  # type: ignore[assignment]
            [self.label],
            [self.valuetype],
        ]
        self.tsdf.index = [  # type: ignore[assignment]
            d.date() for d in DatetimeIndex(self.tsdf.index)
        ]
        if returns_input:
            self.value_to_ret()
        return self

    def set_new_label(
        self: OpenTimeSeries,
        lvl_zero: Optional[str] = None,
        lvl_one: Optional[ValueType] = None,
        delete_lvl_one: bool = False,  # noqa: FBT001, FBT002
    ) -> OpenTimeSeries:
        """
        Set the column labels of the .tsdf Pandas Dataframe.

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
                [[self.label], [self.valuetype]],
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
    """
    Chain two timeseries together.

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
    Union[TypeOpenTimeSeries, OpenTimeSeries]
        An OpenTimeSeries object or a subclass thereof
    """
    old = front.from_deepcopy()
    old.running_adjustment(old_fee)
    new = back.from_deepcopy()
    idx = 0
    first = new.tsdf.index[idx]

    if old.last_idx < first:
        msg = "Timeseries dates must overlap to allow them to be chained."
        raise ValueError(msg)

    while first not in old.tsdf.index:
        idx += 1
        first = new.tsdf.index[idx]
        if first > old.tsdf.index[-1]:
            msg = "Failed to find a matching date between series"
            raise ValueError(msg)

    dates: list[str] = [x.strftime("%Y-%m-%d") for x in old.tsdf.index if x < first]

    values = old.tsdf.iloc[: len(dates), 0]
    values = values.mul(
        new.tsdf.iloc[:, 0].loc[first] / old.tsdf.iloc[:, 0].loc[first],
    )
    values = append(values, new.tsdf.iloc[:, 0])  # type: ignore[assignment]

    dates.extend([x.strftime("%Y-%m-%d") for x in new.tsdf.index])

    if back.__class__.__subclasscheck__(
        OpenTimeSeries,
    ):
        return OpenTimeSeries(
            timeseries_id=new.timeseries_id,
            instrument_id=new.instrument_id,
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
        timeseries_id=new.timeseries_id,
        instrument_id=new.instrument_id,
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


def check_if_none(item: Any) -> bool:  # noqa: ANN401
    """
    Check if a variable is None or equivalent.

    Parameters
    ----------
    item : Any
        variable to be checked

    Returns
    -------
    bool
        Answer to whether the variable is None or equivalent
    """
    try:
        return cast(bool, isnan(item))
    except TypeError:
        if item is None:
            return True
        return len(str(item)) == 0
