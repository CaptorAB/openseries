"""The OpenTimeSeries class."""

from __future__ import annotations

import datetime as dt
from copy import deepcopy
from logging import getLogger
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import Timestamp

from numpy import (
    append,
    array,
    cumprod,
    diff,
    float64,
    insert,
    isnan,
    log,
    sqrt,
    square,
)
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    date_range,
)
from pydantic import field_validator, model_validator

from ._common_model import _calculate_time_factor, _CommonModel
from .datefixer import _do_resample_to_business_period_ends, date_fix
from .owntypes import (
    Countries,
    CountriesType,
    Currency,
    CurrencyStringType,
    DateAlignmentError,
    DateListType,
    DaysInYearType,
    IncorrectArgumentComboError,
    LiteralBizDayFreq,
    LiteralPandasReindexMethod,
    LiteralSeriesProps,
    MarketsNotStringNorListStrError,
    OpenTimeSeriesPropertiesList,
    ResampleDataLossError,
    ValueListType,
    ValueType,
)

logger = getLogger(__name__)

__all__ = ["OpenTimeSeries", "timeseries_chain"]

TypeOpenTimeSeries = TypeVar("TypeOpenTimeSeries", bound="OpenTimeSeries")


class OpenTimeSeries(_CommonModel[float]):
    """OpenTimeSeries objects are at the core of the openseries package.

    The intended use is to allow analyses of financial timeseries.
    It is only intended for daily or less frequent data samples.

    Args:
        timeseries_id: Database identifier of the timeseries.
        instrument_id: Database identifier of the instrument associated with
            the timeseries.
        name: String identifier of the timeseries and/or instrument.
        valuetype: Identifies if the series is a series of values or returns.
        dates: Dates of the individual timeseries items.
            These dates will not be altered by methods.
        values: The value or return values of the timeseries items.
            These values will not be altered by methods.
        local_ccy: Boolean flag indicating if timeseries is in local currency.
        tsdf: Pandas object holding dates and values that can be altered via
            methods.
        currency: ISO 4217 currency code of the timeseries.
        domestic: ISO 4217 currency code of the user's home currency.
            Defaults to "SEK".
        countries: (List of) country code(s) according to ISO 3166-1 alpha-2.
            Defaults to "SE".
        markets: (List of) markets code(s) supported by exchange_calendars.
            Optional.
        isin: ISO 6166 identifier code of the associated instrument. Optional.
        label: Placeholder for a name of the timeseries. Optional.
    """

    timeseries_id: str
    instrument_id: str
    name: str
    valuetype: ValueType
    dates: DateListType
    values: ValueListType
    local_ccy: bool
    tsdf: DataFrame
    currency: CurrencyStringType
    domestic: CurrencyStringType = "SEK"
    countries: CountriesType = "SE"
    markets: list[str] | str | None = None  # type: ignore[assignment]
    isin: str | None = None
    label: str | None = None

    @field_validator("domestic", mode="before")
    @classmethod
    def _validate_domestic(cls, value: CurrencyStringType) -> CurrencyStringType:
        """Pydantic validator to ensure domestic field is validated."""
        Currency(ccy=value)
        return value

    @field_validator("countries", mode="before")
    @classmethod
    def _validate_countries(cls, value: CountriesType) -> CountriesType:
        """Pydantic validator to ensure countries field is validated."""
        Countries(countryinput=value)
        return value

    @field_validator("markets", mode="before")
    @classmethod
    def _validate_markets(
        cls,
        value: list[str] | str | None,
    ) -> list[str] | str | None:
        """Pydantic validator to ensure markets field is validated.

        Raises:
            MarketsNotStringNorListStrError: If ``markets`` is neither a string
                nor a non-empty list of strings.
        """
        msg = (
            "'markets' must be a string or list of strings, "
            f"got {type(value).__name__!r}"
        )
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, list):
            if all(isinstance(item, str) for item in value) and len(value) != 0:
                return value
            item_msg = "All items in 'markets' must be strings."
            raise MarketsNotStringNorListStrError(item_msg)
        raise MarketsNotStringNorListStrError(msg)

    @model_validator(mode="after")
    def _dates_and_values_validate(self: Self) -> Self:
        """Pydantic validator to ensure dates and values are validated.

        Raises:
            ValueError: If dates are not unique or if numbers of dates and values
                do not match the shape of ``tsdf``.
        """
        values_list_length = len(self.values)
        dates_list_length = len(self.dates)
        dates_set_length = len(set(self.dates))
        if dates_list_length != dates_set_length:
            msg = "Dates are not unique"
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
    def from_arrays(
        cls,
        name: str,
        dates: DateListType,
        values: ValueListType,
        valuetype: ValueType = ValueType.PRICE,
        timeseries_id: str = "",
        instrument_id: str = "",
        isin: str | None = None,
        baseccy: CurrencyStringType = "SEK",
        *,
        local_ccy: bool = True,
    ) -> Self:
        """Create series from a list of dates and a list of values.

        Args:
            name: String identifier of the timeseries and/or instrument.
            dates: List of date strings as ISO 8601 YYYY-MM-DD.
            values: Array of float values.
            valuetype: Identifies if the series is a series of values or returns.
                Defaults to ValueType.PRICE.
            timeseries_id: Database identifier of the timeseries. Optional.
            instrument_id: Database identifier of the instrument associated
                with the timeseries. Optional.
            isin: ISO 6166 identifier code of the associated instrument. Optional.
            baseccy: ISO 4217 currency code of the timeseries. Defaults to "SEK".
            local_ccy: Boolean flag indicating if timeseries is in local currency.
                Defaults to True.

        Returns:
            An OpenTimeSeries object.
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
        cls,
        dframe: Series[float] | DataFrame,
        column_nmbr: int = 0,
        valuetype: ValueType = ValueType.PRICE,
        baseccy: CurrencyStringType = "SEK",
        *,
        local_ccy: bool = True,
    ) -> Self:
        """Create series from a Pandas DataFrame or Series.

        Args:
            dframe: Pandas DataFrame or Series.
            column_nmbr: Using iloc[:, column_nmbr] to pick column. Defaults to 0.
            valuetype: Identifies if the series is a series of values or returns.
                Defaults to ValueType.PRICE.
            baseccy: ISO 4217 currency code of the timeseries. Defaults to "SEK".
            local_ccy: Boolean flag indicating if timeseries is in local currency.
                Defaults to True.

        Returns:
            An OpenTimeSeries object.

        Raises:
            TypeError: If ``dframe`` is not a ``pandas.Series`` or a
                ``pandas.DataFrame``.
        """
        msg = "Argument dframe must be pandas Series or DataFrame."
        values: list[float]
        if isinstance(dframe, Series):
            if isinstance(dframe.name, tuple):
                label, _ = dframe.name
            else:
                label = dframe.name
            values = dframe.to_numpy().tolist()
        elif isinstance(dframe, DataFrame):
            values = dframe.iloc[:, column_nmbr].to_list()
            if isinstance(dframe.columns, MultiIndex):
                if _check_if_none(
                    dframe.columns.get_level_values(0).to_numpy()[column_nmbr],
                ):
                    label = "Series"
                    msg = f"Label missing. Adding: {label}"
                    logger.warning(msg)
                else:
                    label = dframe.columns.get_level_values(0).to_numpy()[column_nmbr]
                if _check_if_none(
                    dframe.columns.get_level_values(1).to_numpy()[column_nmbr],
                ):
                    valuetype = ValueType.PRICE
                    msg = f"valuetype missing. Adding: {valuetype.value}"
                    logger.warning(msg)
                else:
                    valuetype = dframe.columns.get_level_values(1).to_numpy()[
                        column_nmbr
                    ]
            else:
                label = dframe.columns.to_numpy()[column_nmbr]
        else:
            raise TypeError(msg)

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
        cls,
        rate: float,
        d_range: DatetimeIndex | None = None,
        days: int | None = None,
        end_dt: dt.date | None = None,
        label: str = "Series",
        valuetype: ValueType = ValueType.PRICE,
        baseccy: CurrencyStringType = "SEK",
        *,
        local_ccy: bool = True,
    ) -> Self:
        """Create series from values accruing with a given fixed rate return.

        Providing a date_range of type Pandas DatetimeIndex takes priority over
        providing a combination of days and an end date.

        Args:
            rate: The accrual rate.
            d_range: A given range of dates. Optional.
            days: Number of days to generate when date_range not provided. Must be
                combined with end_dt. Optional.
            end_dt: End date of date range to generate when date_range not provided.
                Must be combined with days. Optional.
            label: Placeholder for a name of the timeseries.
            valuetype: Identifies if the series is a series of values or returns.
                Defaults to ValueType.PRICE.
            baseccy: The currency of the timeseries. Defaults to "SEK".
            local_ccy: Boolean flag indicating if timeseries is in local currency.
                Defaults to True.

        Returns:
            An OpenTimeSeries object.

        Raises:
            IncorrectArgumentComboError: If ``d_range`` is not provided and the
                combination of ``days`` and ``end_dt`` is incomplete.
        """
        if d_range is None:
            if days is not None and end_dt is not None:
                d_range = DatetimeIndex(
                    [d.date() for d in date_range(periods=days, end=end_dt, freq="D")],
                )
            else:
                msg = "If d_range is not provided both days and end_dt must be."
                raise IncorrectArgumentComboError(msg)
        deltas = array([i.days for i in d_range[1:] - d_range[:-1]])
        arr: list[float] = list(cumprod(insert(1 + deltas * rate / 365, 0, 1.0)))
        dates = [d.strftime("%Y-%m-%d") for d in d_range]

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

    def from_deepcopy(self: Self) -> Self:
        """Create copy of OpenTimeSeries object.

        Returns:
            An OpenTimeSeries object.
        """
        return deepcopy(self)

    def pandas_df(self: Self) -> Self:
        """Populate .tsdf Pandas DataFrame from the .dates and .values lists.

        Returns:
            An OpenTimeSeries object.
        """
        dframe = DataFrame(
            data=self.values,
            index=[d.date() for d in DatetimeIndex(self.dates)],
            columns=[[self.label], [self.valuetype]],
            dtype="float64",
        )
        self.tsdf = dframe

        return self

    def all_properties(
        self: Self,
        properties: list[LiteralSeriesProps] | None = None,
    ) -> DataFrame:
        """Calculate chosen properties.

        Args:
            properties: The properties to calculate. Defaults to calculating all
                available. Optional.

        Returns:
            Properties of the OpenTimeSeries.
        """
        if not properties:
            properties = cast(
                "list[LiteralSeriesProps]",
                OpenTimeSeriesPropertiesList.allowed_strings,
            )

        props = OpenTimeSeriesPropertiesList(*properties)
        pdf = DataFrame.from_dict({x: getattr(self, x) for x in props}, orient="index")
        pdf.columns = self.tsdf.columns
        return pdf

    def value_to_ret(self: Self) -> Self:
        """Convert series of values into series of returns.

        Returns:
            The returns of the values in the series.
        """
        returns = self.tsdf.ffill().pct_change()
        returns.iloc[0] = 0
        self.valuetype = ValueType.RTRN
        arrays = [[self.label], [self.valuetype]]
        returns.columns = MultiIndex.from_arrays(
            arrays=arrays,  # type: ignore[arg-type]
        )
        self.tsdf = returns.copy()
        return self

    def value_to_diff(self: Self, periods: int = 1) -> Self:
        """Convert series of values to series of their period differences.

        Args:
            periods: The number of periods between observations over which difference
                is calculated. Defaults to 1.

        Returns:
            An OpenTimeSeries object.
        """
        self.tsdf = self.tsdf.diff(periods=periods)
        self.tsdf.iloc[0] = 0
        self.valuetype = ValueType.RTRN
        self.tsdf.columns = MultiIndex.from_arrays(
            [
                [self.label],
                [self.valuetype],
            ],
        )
        return self

    def to_cumret(self: Self) -> Self:
        """Convert series of returns into cumulative series of values.

        Returns:
            An OpenTimeSeries object.
        """
        if self.valuetype == ValueType.PRICE:
            self.value_to_ret()

        self.tsdf = self.tsdf.add(1.0)
        self.tsdf = self.tsdf.cumprod(axis=0) / self.tsdf.iloc[0]

        self.valuetype = ValueType.PRICE
        self.tsdf.columns = MultiIndex.from_arrays(
            [
                [self.label],
                [self.valuetype],
            ],
        )
        return self

    def from_1d_rate_to_cumret(
        self: Self,
        days_in_year: int = 365,
        divider: float = 1.0,
    ) -> Self:
        """Convert series of 1-day rates into series of cumulative values.

        Args:
            days_in_year: Calendar days per year used as divisor. Defaults to 365.
            divider: Convenience divider for when the 1-day rate is not scaled
                correctly. Defaults to 1.0.

        Returns:
            An OpenTimeSeries object.
        """
        arr: NDArray[float64] = array(self.values) / divider

        deltas = array([i.days for i in self.tsdf.index[1:] - self.tsdf.index[:-1]])
        arr = cast(
            "NDArray[float64]",
            cumprod(
                a=insert(
                    arr=1.0 + deltas * arr[:-1] / days_in_year, obj=0, values=1.0
                ),
            ),
        )

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
        self: Self,
        freq: LiteralBizDayFreq | str = "BME",
    ) -> Self:
        """Resamples the timeseries frequency.

        Args:
            freq: The date offset string that sets the resampled frequency.
                Defaults to "BME".

        Returns:
            An OpenTimeSeries object.
        """
        self.tsdf.index = DatetimeIndex(self.tsdf.index)
        if self.valuetype == ValueType.RTRN:
            self.tsdf = self.tsdf.resample(freq).sum()
        else:
            self.tsdf = self.tsdf.resample(freq).last()
        self.tsdf.index = Index(DatetimeIndex(self.tsdf.index).date)
        return self

    def resample_to_business_period_ends(
        self: Self,
        freq: LiteralBizDayFreq = "BME",
        method: LiteralPandasReindexMethod = "nearest",
    ) -> Self:
        """Resamples timeseries frequency to the business calendar month end dates.

        Stubs left in place. Stubs will be aligned to the shortest stub.

        Args:
            freq: The date offset string that sets the resampled frequency.
                Defaults to BME.
            method: Controls the method used to align values across columns.
                Defaults to nearest.

        Returns:
            An OpenTimeSeries object.

        Raises:
            ResampleDataLossError: If called on a return series (``valuetype`` is
                ``ValueType.RTRN``), since summation across sparser frequency would
                be required to avoid data loss.
        """
        if self.valuetype == ValueType.RTRN:
            msg = (
                "Do not run resample_to_business_period_ends on return series. "
                "The operation will pick the last data point in the sparser series. "
                "It will not sum returns and therefore data will be lost."
            )
            raise ResampleDataLossError(msg)

        dates = _do_resample_to_business_period_ends(
            data=self.tsdf,
            freq=freq,
            countries=self.countries,
            markets=self.markets,
        )
        self.tsdf = self.tsdf.reindex([deyt.date() for deyt in dates], method=method)
        return self

    def ewma_vol_func(
        self: Self,
        lmbda: float = 0.94,
        day_chunk: int = 11,
        dlta_degr_freedms: int = 0,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> Series[float]:
        """Exponentially Weighted Moving Average Model for Volatility.

        Reference: https://www.investopedia.com/articles/07/ewma.asp.

        Args:
            lmbda: Scaling factor to determine weighting. Defaults to 0.94.
            day_chunk: Sampling the data which is assumed to be daily.
                Defaults to 11.
            dlta_degr_freedms: Variance bias factor taking the value 0 or 1.
                Defaults to 0.
            months_from_last: Number of months offset as positive integer.
                Overrides use of from_date and to_date. Optional.
            from_date: Specific from date. Optional.
            to_date: Specific to date. Optional.
            periods_in_a_year_fixed: Allows locking the periods-in-a-year to simplify
                test cases and comparisons. Optional.

        Returns:
            Series EWMA volatility.
        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        time_factor = _calculate_time_factor(
            data=self.tsdf.loc[
                cast("Timestamp", earlier) : cast("Timestamp", later)
            ].iloc[:, 0],
            earlier=earlier,
            later=later,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

        data = self.tsdf.loc[
            cast("Timestamp", earlier) : cast("Timestamp", later)
        ].copy()

        data.loc[:, (self.label, ValueType.RTRN)] = log(  # type: ignore[index]
            data.loc[:, self.tsdf.columns.to_numpy()[0]],
        ).diff()

        rawdata = [
            data[(self.label, ValueType.RTRN)]
            .iloc[1:day_chunk]
            .std(ddof=dlta_degr_freedms)
            * sqrt(time_factor),
        ]

        for item in data[(self.label, ValueType.RTRN)].iloc[1:]:
            prev = rawdata[-1]
            rawdata.append(
                sqrt(
                    square(item) * time_factor * (1 - lmbda) + square(prev) * lmbda,
                ),
            )

        return Series(
            data=rawdata,
            index=data.index,
            name=(self.label, ValueType.EWMA),
            dtype="float64",
        )

    def running_adjustment(
        self: Self,
        adjustment: float,
        days_in_year: int = 365,
    ) -> Self:
        """Add or subtract a fee from the timeseries return.

        Args:
            adjustment: Fee to add or subtract.
            days_in_year: The calculation divisor and assumed number of days in a
                calendar year. Defaults to 365.

        Returns:
            An OpenTimeSeries object.
        """
        if self.valuetype == ValueType.RTRN:
            ra_df = self.tsdf.copy()
            initial_value = 1.0
            returns_input = True
        else:
            initial_value = cast("float", self.tsdf.iloc[0, 0])
            ra_df = self.tsdf.ffill().pct_change()
            returns_input = False
        ra_df = ra_df.dropna()

        dates_index = DatetimeIndex(ra_df.index)
        dates_list = [self.first_idx] + [d.date() for d in dates_index]

        dates_np = array(
            [dt.datetime.combine(d, dt.time()) for d in dates_list],
            dtype="datetime64[D]",
        )
        date_diffs = cast(
            "NDArray[float64]",
            diff(dates_np).astype("timedelta64[D]").astype(float64),
        )

        returns_array = cast(
            "NDArray[float64]",
            ra_df.iloc[:, 0].to_numpy(),
        )

        adjustment_factors = (
            1.0 + returns_array + adjustment * date_diffs / days_in_year
        )

        values_array = cumprod(insert(adjustment_factors, 0, initial_value))
        values = list(values_array)

        self.tsdf = DataFrame(data=values, index=dates_list)
        self.valuetype = ValueType.PRICE
        self.tsdf.columns = MultiIndex.from_arrays(
            [
                [self.label],
                [self.valuetype],
            ],
        )
        self.tsdf.index = Index(DatetimeIndex(self.tsdf.index).date)
        if returns_input:
            self.value_to_ret()
        return self

    def set_new_label(
        self: Self,
        lvl_zero: str | None = None,
        lvl_one: ValueType | None = None,
        *,
        delete_lvl_one: bool = False,
    ) -> Self:
        """Set the column labels of the .tsdf Pandas Dataframe.

        Args:
            lvl_zero: New level zero label. Optional.
            lvl_one: New level one label. Optional.
            delete_lvl_one: If True the level one label is deleted. Defaults to False.

        Returns:
            An OpenTimeSeries object.
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
            self.label, self.valuetype = lvl_zero, cast("ValueType", lvl_one)
        if delete_lvl_one:
            self.tsdf.columns = self.tsdf.columns.droplevel(level=1)
        return self


def timeseries_chain(
    front: TypeOpenTimeSeries,
    back: TypeOpenTimeSeries,
    old_fee: float = 0.0,
) -> TypeOpenTimeSeries:
    """Chain two timeseries together.

    The function assumes that the two series have at least one date in common.

    Args:
        front: Earlier series to chain with.
        back: Later series to chain with.
        old_fee: Fee to apply to earlier series. Defaults to 0.0.

    Returns:
        An OpenTimeSeries object or a subclass thereof.
    """
    old = front.from_deepcopy()
    old.running_adjustment(old_fee)
    new = back.from_deepcopy()
    idx = 0
    first = new.tsdf.index[idx]

    if old.last_idx < first:
        msg = "Timeseries dates must overlap to allow them to be chained."
        raise DateAlignmentError(msg)

    while first not in old.tsdf.index:
        idx += 1
        first = new.tsdf.index[idx]
        if first > old.tsdf.index[-1]:
            msg = "Failed to find a matching date between series"
            raise DateAlignmentError(msg)

    dates: list[str] = [x.strftime("%Y-%m-%d") for x in old.tsdf.index if x < first]

    old_values = Series(old.tsdf.iloc[: len(dates), 0])
    old_values = old_values.mul(
        Series(new.tsdf.iloc[:, 0]).loc[first]
        / Series(old.tsdf.iloc[:, 0]).loc[first],
    )
    values = append(old_values, new.tsdf.iloc[:, 0])

    dates.extend([x.strftime("%Y-%m-%d") for x in new.tsdf.index])

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


def _check_if_none(item: Any) -> bool:  # noqa: ANN401
    """Check if a variable is None or equivalent.

    Args:
        item: Variable to be checked.

    Returns:
        Answer to whether the variable is None or equivalent.
    """
    try:
        return cast("bool", isnan(item))
    except TypeError:
        if item is None:
            return True
        return len(str(item)) == 0
