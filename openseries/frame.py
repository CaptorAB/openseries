"""Defining the OpenFrame class."""

# mypy: disable-error-code="index,assignment,arg-type"
from __future__ import annotations

from copy import deepcopy
from functools import reduce
from logging import warning
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import datetime as dt  # pragma: no cover

import statsmodels.api as sm  # type: ignore[import-untyped,unused-ignore]
from numpy import (
    cov,
    cumprod,
    divide,
    isinf,
    log,
    nan,
    sqrt,
    square,
    std,
)
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Int64Dtype,
    MultiIndex,
    Series,
    concat,
    merge,
)
from pydantic import field_validator

# noinspection PyProtectedMember
from statsmodels.regression.linear_model import (  # type: ignore[import-untyped,unused-ignore]
    OLSResults,
)
from typing_extensions import Self

from ._common_model import _CommonModel
from .datefixer import _do_resample_to_business_period_ends
from .series import OpenTimeSeries
from .types import (
    CountriesType,
    DaysInYearType,
    LiteralBizDayFreq,
    LiteralCaptureRatio,
    LiteralFrameProps,
    LiteralHowMerge,
    LiteralOlsFitCovType,
    LiteralOlsFitMethod,
    LiteralPandasReindexMethod,
    LiteralPortfolioWeightings,
    LiteralTrunc,
    OpenFramePropertiesList,
    ValueType,
)

__all__ = ["OpenFrame"]


# noinspection PyUnresolvedReferences,PyTypeChecker
class OpenFrame(_CommonModel):
    """OpenFrame objects hold OpenTimeSeries in the list constituents.

    The intended use is to allow comparisons across these timeseries.

    Parameters
    ----------
    constituents: list[OpenTimeSeries]
        List of objects of Class OpenTimeSeries
    weights: list[float], optional
        List of weights in float format.

    Returns
    -------
    OpenFrame
        Object of the class OpenFrame

    """

    constituents: list[OpenTimeSeries]
    tsdf: DataFrame = DataFrame(dtype="float64")
    weights: list[float] | None = None

    # noinspection PyMethodParameters
    @field_validator("constituents")  # type: ignore[misc]
    def _check_labels_unique(
        cls: OpenFrame,  # noqa: N805
        tseries: list[OpenTimeSeries],
    ) -> list[OpenTimeSeries]:
        """Pydantic validator ensuring that OpenFrame labels are unique."""
        labls = [x.label for x in tseries]
        if len(set(labls)) != len(labls):
            msg = "TimeSeries names/labels must be unique"
            raise ValueError(msg)
        return tseries

    def __init__(
        self: Self,
        constituents: list[OpenTimeSeries],
        weights: list[float] | None = None,
    ) -> None:
        """OpenFrame objects hold OpenTimeSeries in the list constituents.

        The intended use is to allow comparisons across these timeseries.

        Parameters
        ----------
        constituents: list[OpenTimeSeries]
            List of objects of Class OpenTimeSeries
        weights: list[float], optional
            List of weights in float format.

        Returns
        -------
        OpenFrame
            Object of the class OpenFrame

        """
        super().__init__(  # type: ignore[call-arg]
            constituents=constituents,
            weights=weights,
        )

        self.constituents = constituents
        self.weights = weights
        self._set_tsdf()

    def _set_tsdf(self: Self) -> None:
        """Set the tsdf DataFrame."""
        if self.constituents is not None and len(self.constituents) != 0:
            self.tsdf = reduce(
                lambda left, right: concat([left, right], axis="columns", sort=True),
                [x.tsdf for x in self.constituents],
            )
        else:
            warning("OpenFrame() was passed an empty list.")

    def from_deepcopy(self: Self) -> Self:
        """Create copy of the OpenFrame object.

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        return deepcopy(self)

    def merge_series(
        self: Self,
        how: LiteralHowMerge = "outer",
    ) -> Self:
        """Merge index of Pandas Dataframes of the constituent OpenTimeSeries.

        Parameters
        ----------
        how: LiteralHowMerge, default: "outer"
            The Pandas merge method.

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        lvl_zero = list(self.columns_lvl_zero)
        self.tsdf = reduce(
            lambda left, right: merge(
                left=left,
                right=right,
                how=how,
                left_index=True,
                right_index=True,
            ),
            [x.tsdf for x in self.constituents],
        )

        mapper = dict(zip(self.columns_lvl_zero, lvl_zero))
        self.tsdf = self.tsdf.rename(columns=mapper, level=0)

        if self.tsdf.empty:
            msg = (
                "Merging OpenTimeSeries DataFrames with "
                f"argument how={how} produced an empty DataFrame."
            )
            raise ValueError(msg)

        if how == "inner":
            for xerie in self.constituents:
                xerie.tsdf = xerie.tsdf.loc[self.tsdf.index]
        return self

    def all_properties(
        self: Self,
        properties: list[LiteralFrameProps] | None = None,
    ) -> DataFrame:
        """Calculate chosen timeseries properties.

        Parameters
        ----------
        properties: list[LiteralFrameProps], optional
            The properties to calculate. Defaults to calculating all available.

        Returns
        -------
        pandas.DataFrame
            Properties of the contituent OpenTimeSeries

        """
        if properties:
            props = OpenFramePropertiesList(*properties)
            prop_list = [getattr(self, x) for x in props]
        else:
            prop_list = [
                getattr(self, x) for x in OpenFramePropertiesList.allowed_strings
            ]
        return cast(DataFrame, concat(prop_list, axis="columns").T)

    @property
    def lengths_of_items(self: Self) -> Series[int]:
        """Number of observations of all constituents.

        Returns
        -------
        Pandas.Series[int]
            Number of observations of all constituents

        """
        return Series(
            data=[int(self.tsdf.loc[:, d].count()) for d in self.tsdf],
            index=self.tsdf.columns,
            name="observations",
            dtype=Int64Dtype(),
        )

    @property
    def item_count(self: Self) -> int:
        """Number of constituents.

        Returns
        -------
        int
            Number of constituents

        """
        return len(self.constituents)

    @property
    def columns_lvl_zero(self: Self) -> list[str]:
        """Level 0 values of the MultiIndex columns in the .tsdf DataFrame.

        Returns
        -------
        list[str]
            Level 0 values of the MultiIndex columns in the .tsdf DataFrame

        """
        return list(self.tsdf.columns.get_level_values(0))

    @property
    def columns_lvl_one(self: Self) -> list[ValueType]:
        """Level 1 values of the MultiIndex columns in the .tsdf DataFrame.

        Returns
        -------
        list[ValueType]
            Level 1 values of the MultiIndex columns in the .tsdf DataFrame

        """
        return list(self.tsdf.columns.get_level_values(1))

    @property
    def first_indices(self: Self) -> Series[dt.date]:
        """The first dates in the timeseries of all constituents.

        Returns
        -------
        Pandas.Series[dt.date]
            The first dates in the timeseries of all constituents

        """
        return Series(
            data=[i.first_idx for i in self.constituents],
            index=self.tsdf.columns,
            name="first indices",
            dtype="datetime64[ns]",
        ).dt.date

    @property
    def last_indices(self: Self) -> Series[dt.date]:
        """The last dates in the timeseries of all constituents.

        Returns
        -------
        Pandas.Series[dt.date]
            The last dates in the timeseries of all constituents

        """
        return Series(
            data=[i.last_idx for i in self.constituents],
            index=self.tsdf.columns,
            name="last indices",
            dtype="datetime64[ns]",
        ).dt.date

    @property
    def span_of_days_all(self: Self) -> Series[int]:
        """Number of days from the first date to the last for all items in the frame.

        Returns
        -------
        Pandas.Series[int]
            Number of days from the first date to the last for all
            items in the frame.

        """
        return Series(
            data=[c.span_of_days for c in self.constituents],
            index=self.tsdf.columns,
            name="span of days",
            dtype=Int64Dtype(),
        )

    def value_to_ret(self: Self) -> Self:
        """Convert series of values into series of returns.

        Returns
        -------
        OpenFrame
            The returns of the values in the series

        """
        self.tsdf = self.tsdf.pct_change()
        self.tsdf.iloc[0] = 0
        new_labels = [ValueType.RTRN] * self.item_count
        arrays = [self.tsdf.columns.get_level_values(0), new_labels]
        self.tsdf.columns = MultiIndex.from_arrays(arrays)
        return self

    def value_to_diff(self: Self, periods: int = 1) -> Self:
        """Convert series of values to series of their period differences.

        Parameters
        ----------
        periods: int, default: 1
            The number of periods between observations over which difference
            is calculated

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        self.tsdf = self.tsdf.diff(periods=periods)
        self.tsdf.iloc[0] = 0
        new_labels = [ValueType.RTRN] * self.item_count
        arrays = [self.tsdf.columns.get_level_values(0), new_labels]
        self.tsdf.columns = MultiIndex.from_arrays(arrays)
        return self

    def to_cumret(self: Self) -> Self:
        """Convert series of returns into cumulative series of values.

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        if any(
            x == ValueType.PRICE
            for x in self.tsdf.columns.get_level_values(1).to_numpy()
        ):
            self.value_to_ret()

        self.tsdf = self.tsdf.add(1.0)
        self.tsdf = self.tsdf.apply(cumprod, axis="index") / self.tsdf.iloc[0]
        new_labels = [ValueType.PRICE] * self.item_count
        arrays = [self.tsdf.columns.get_level_values(0), new_labels]
        self.tsdf.columns = MultiIndex.from_arrays(arrays)
        return self

    def resample(
        self: Self,
        freq: LiteralBizDayFreq | str = "BME",
    ) -> Self:
        """Resample the timeseries frequency.

        Parameters
        ----------
        freq: LiteralBizDayFreq | str, default "BME"
            The date offset string that sets the resampled frequency

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        self.tsdf.index = DatetimeIndex(self.tsdf.index)
        self.tsdf = self.tsdf.resample(freq).last()
        self.tsdf.index = Index(d.date() for d in DatetimeIndex(self.tsdf.index))
        for xerie in self.constituents:
            xerie.tsdf.index = DatetimeIndex(xerie.tsdf.index)
            xerie.tsdf = xerie.tsdf.resample(freq).last()
            xerie.tsdf.index = Index(
                dejt.date() for dejt in DatetimeIndex(xerie.tsdf.index)
            )

        return self

    def resample_to_business_period_ends(
        self: Self,
        freq: LiteralBizDayFreq = "BME",
        countries: CountriesType = "SE",
        method: LiteralPandasReindexMethod = "nearest",
    ) -> Self:
        """Resamples timeseries frequency to the business calendar month end dates.

        Stubs left in place. Stubs will be aligned to the shortest stub.

        Parameters
        ----------
        freq: LiteralBizDayFreq, default "BME"
            The date offset string that sets the resampled frequency
        countries: CountriesType, default: "SE"
            (List of) country code(s) according to ISO 3166-1 alpha-2
            to create a business day calendar used for date adjustments
        method: LiteralPandasReindexMethod, default: nearest
            Controls the method used to align values across columns

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        for xerie in self.constituents:
            dates = _do_resample_to_business_period_ends(
                data=xerie.tsdf,
                freq=freq,
                countries=countries,
            )
            xerie.tsdf = xerie.tsdf.reindex(
                [deyt.date() for deyt in dates],
                method=method,
            )

        self._set_tsdf()

        return self

    def ewma_risk(
        self: Self,
        lmbda: float = 0.94,
        day_chunk: int = 11,
        dlta_degr_freedms: int = 0,
        first_column: int = 0,
        second_column: int = 1,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> DataFrame:
        """Exponentially Weighted Moving Average Volatilities and Correlation.

        Exponentially Weighted Moving Average (EWMA) for Volatilities and
        Correlation. https://www.investopedia.com/articles/07/ewma.asp.

        Parameters
        ----------
        lmbda: float, default: 0.94
            Scaling factor to determine weighting.
        day_chunk: int, default: 11
            Sampling the data which is assumed to be daily.
        dlta_degr_freedms: int, default: 0
            Variance bias factor taking the value 0 or 1.
        first_column: int, default: 0
            Column of first timeseries.
        second_column: int, default: 1
            Column of second timeseries.
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

        Returns
        -------
        Pandas.DataFrame
            Series volatilities and correlation

        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed is None:
            fraction = (later - earlier).days / 365.25
            how_many = (
                self.tsdf.loc[cast(int, earlier) : cast(int, later)].count().iloc[0]
            )
            time_factor = how_many / fraction
        else:
            time_factor = periods_in_a_year_fixed

        corr_label = (
            cast(tuple[str, str], self.tsdf.iloc[:, first_column].name)[0]
            + "_VS_"
            + cast(tuple[str, str], self.tsdf.iloc[:, second_column].name)[0]
        )
        cols = [
            cast(tuple[str, str], self.tsdf.iloc[:, first_column].name)[0],
            cast(tuple[str, str], self.tsdf.iloc[:, second_column].name)[0],
        ]

        data = self.tsdf.loc[cast(int, earlier) : cast(int, later)].copy()

        for rtn in cols:
            data[rtn, ValueType.RTRN] = (
                data.loc[:, (rtn, ValueType.PRICE)].apply(log).diff()
            )

        raw_one = [
            data.loc[:, (cols[0], ValueType.RTRN)]
            .iloc[1:day_chunk]
            .std(ddof=dlta_degr_freedms)
            * sqrt(time_factor),
        ]
        raw_two = [
            data.loc[:, (cols[1], ValueType.RTRN)]
            .iloc[1:day_chunk]
            .std(ddof=dlta_degr_freedms)
            * sqrt(time_factor),
        ]
        raw_cov = [
            cov(
                m=data.loc[:, (cols[0], ValueType.RTRN)].iloc[1:day_chunk].to_numpy(),
                y=data.loc[:, (cols[1], ValueType.RTRN)].iloc[1:day_chunk].to_numpy(),
                ddof=dlta_degr_freedms,
            )[0][1],
        ]
        raw_corr = [raw_cov[0] / (2 * raw_one[0] * raw_two[0])]

        for _, row in data.iloc[1:].iterrows():
            tmp_raw_one = sqrt(
                square(row.loc[cols[0], ValueType.RTRN]) * time_factor * (1 - lmbda)
                + square(raw_one[-1]) * lmbda,
            )
            tmp_raw_two = sqrt(
                square(row.loc[cols[1], ValueType.RTRN]) * time_factor * (1 - lmbda)
                + square(raw_two[-1]) * lmbda,
            )
            tmp_raw_cov = (
                row.loc[cols[0], ValueType.RTRN]
                * row.loc[cols[1], ValueType.RTRN]
                * time_factor
                * (1 - lmbda)
                + raw_cov[-1] * lmbda
            )
            tmp_raw_corr = tmp_raw_cov / (2 * tmp_raw_one * tmp_raw_two)
            raw_one.append(tmp_raw_one)
            raw_two.append(tmp_raw_two)
            raw_cov.append(tmp_raw_cov)
            raw_corr.append(tmp_raw_corr)

        return DataFrame(
            index=[*cols, corr_label],
            columns=data.index,
            data=[raw_one, raw_two, raw_corr],
        ).T

    @property
    def correl_matrix(self: Self) -> DataFrame:
        """Correlation matrix.

        Returns
        -------
        Pandas.DataFrame
            Correlation matrix

        """
        corr_matrix = self.tsdf.pct_change().corr(
            method="pearson",
            min_periods=1,
        )
        corr_matrix.columns = corr_matrix.columns.droplevel(level=1)
        corr_matrix.index = corr_matrix.index.droplevel(level=1)
        corr_matrix.index.name = "Correlation"
        return corr_matrix

    def add_timeseries(
        self: Self,
        new_series: OpenTimeSeries,
    ) -> Self:
        """To add an OpenTimeSeries object.

        Parameters
        ----------
        new_series: OpenTimeSeries
            The timeseries to add

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        self.constituents += [new_series]
        self.tsdf = concat([self.tsdf, new_series.tsdf], axis="columns", sort=True)
        return self

    def delete_timeseries(self: Self, lvl_zero_item: str) -> Self:
        """To delete an OpenTimeSeries object.

        Parameters
        ----------
        lvl_zero_item: str
            The .tsdf column level 0 value of the timeseries to delete

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        if self.weights:
            new_c, new_w = [], []
            for serie, weight in zip(self.constituents, self.weights):
                if serie.label != lvl_zero_item:
                    new_c.append(serie)
                    new_w.append(weight)
            self.constituents = new_c
            self.weights = new_w
        else:
            self.constituents = [
                item for item in self.constituents if item.label != lvl_zero_item
            ]
        self.tsdf = self.tsdf.drop(lvl_zero_item, axis="columns", level=0)
        return self

    def trunc_frame(
        self: Self,
        start_cut: dt.date | None = None,
        end_cut: dt.date | None = None,
        where: LiteralTrunc = "both",
    ) -> Self:
        """Truncate DataFrame such that all timeseries have the same time span.

        Parameters
        ----------
        start_cut: datetime.date, optional
            New first date
        end_cut: datetime.date, optional
            New last date
        where: LiteralTrunc, default: both
            Determines where dataframe is truncated also when start_cut
            or end_cut is None.

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        if not start_cut and where in ["before", "both"]:
            start_cut = self.first_indices.max()
        if not end_cut and where in ["after", "both"]:
            end_cut = self.last_indices.min()
        self.tsdf = self.tsdf.sort_index()
        self.tsdf = self.tsdf.truncate(before=start_cut, after=end_cut, copy=False)

        for xerie in self.constituents:
            xerie.tsdf = xerie.tsdf.truncate(
                before=start_cut,
                after=end_cut,
                copy=False,
            )
        if len(set(self.first_indices)) != 1:
            msg = (
                f"One or more constituents still "
                f"not truncated to same start dates.\n"
                f"{self.tsdf.head()}"
            )
            warning(msg=msg)
        if len(set(self.last_indices)) != 1:
            msg = (
                f"One or more constituents still "
                f"not truncated to same end dates.\n"
                f"{self.tsdf.tail()}"
            )
            warning(msg=msg)
        return self

    def relative(
        self: Self,
        long_column: int = 0,
        short_column: int = 1,
        *,
        base_zero: bool = True,
    ) -> None:
        """Calculate cumulative relative return between two series.

        Parameters
        ----------
        long_column: int, default: 0
            Column number of timeseries bought
        short_column: int, default: 1
            Column number of timeseries sold
        base_zero: bool, default: True
            If set to False 1.0 is added to allow for a capital base and
            to allow a volatility calculation

        """
        rel_label = (
            cast(tuple[str, str], self.tsdf.iloc[:, long_column].name)[0]
            + "_over_"
            + cast(tuple[str, str], self.tsdf.iloc[:, short_column].name)[0]
        )
        if base_zero:
            self.tsdf[rel_label, ValueType.RELRTRN] = (
                self.tsdf.iloc[:, long_column] - self.tsdf.iloc[:, short_column]
            )
        else:
            self.tsdf[rel_label, ValueType.RELRTRN] = (
                1.0 + self.tsdf.iloc[:, long_column] - self.tsdf.iloc[:, short_column]
            )
        self.constituents += [
            OpenTimeSeries.from_df(self.tsdf.iloc[:, -1]),
        ]

    def tracking_error_func(
        self: Self,
        base_column: tuple[str, ValueType] | int = -1,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> Series[float]:
        """Tracking Error.

        Calculates Tracking Error which is the standard deviation of the
        difference between the fund and its index returns.
        https://www.investopedia.com/terms/t/trackingerror.asp.

        Parameters
        ----------
        base_column: tuple[str, ValueType] | int, default: -1
            Column of timeseries that is the denominator in the ratio.
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

        Returns
        -------
        Pandas.Series[float]
            Tracking Errors

        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        fraction = (later - earlier).days / 365.25

        msg = "base_column should be a tuple[str, ValueType] or an integer."
        if isinstance(base_column, tuple):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                :,
                base_column,
            ]
            short_item = base_column
            short_label = cast(
                tuple[str, ValueType],
                self.tsdf.loc[:, base_column].name,
            )[0]
        elif isinstance(base_column, int):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[
                :,
                base_column,
            ]
            short_item = self.tsdf.iloc[
                :,
                base_column,
            ].name
            short_label = cast(tuple[str, str], self.tsdf.iloc[:, base_column].name)[0]
        else:
            raise TypeError(msg)

        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = float(shortdf.count() / fraction)

        terrors = []
        for item in self.tsdf:
            if item == short_item:
                terrors.append(0.0)
            else:
                longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                    :,
                    item,
                ]
                relative = 1.0 + longdf - shortdf
                vol = float(
                    relative.pct_change().std() * sqrt(time_factor),
                )
                terrors.append(vol)

        return Series(
            data=terrors,
            index=self.tsdf.columns,
            name=f"Tracking Errors vs {short_label}",
            dtype="float64",
        )

    def info_ratio_func(
        self: Self,
        base_column: tuple[str, ValueType] | int = -1,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> Series[float]:
        """Information Ratio.

        The Information Ratio equals ( fund return less index return ) divided
        by the Tracking Error. And the Tracking Error is the standard deviation of
        the difference between the fund and its index returns.
        The ratio is calculated using the annualized arithmetic mean of returns.

        Parameters
        ----------
        base_column: tuple[str, ValueType] | int, default: -1
            Column of timeseries that is the denominator in the ratio.
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

        Returns
        -------
        Pandas.Series[float]
            Information Ratios

        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        fraction = (later - earlier).days / 365.25

        msg = "base_column should be a tuple[str, ValueType] or an integer."
        if isinstance(base_column, tuple):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                :,
                base_column,
            ]
            short_item = base_column
            short_label = cast(
                tuple[str, str],
                self.tsdf.loc[:, base_column].name,
            )[0]
        elif isinstance(base_column, int):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[
                :,
                base_column,
            ]
            short_item = self.tsdf.iloc[
                :,
                base_column,
            ].name
            short_label = cast(tuple[str, str], self.tsdf.iloc[:, base_column].name)[0]
        else:
            raise TypeError(msg)

        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = float(shortdf.count() / fraction)

        ratios = []
        for item in self.tsdf:
            if item == short_item:
                ratios.append(0.0)
            else:
                longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                    :,
                    item,
                ]
                relative = 1.0 + longdf - shortdf
                ret = float(
                    relative.pct_change().mean() * time_factor,
                )
                vol = float(
                    relative.pct_change().std() * sqrt(time_factor),
                )
                ratios.append(ret / vol)

        return Series(
            data=ratios,
            index=self.tsdf.columns,
            name=f"Info Ratios vs {short_label}",
            dtype="float64",
        )

    def capture_ratio_func(  # noqa: C901
        self: Self,
        ratio: LiteralCaptureRatio,
        base_column: tuple[str, ValueType] | int = -1,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> Series[float]:
        """Capture Ratio.

        The Up (Down) Capture Ratio is calculated by dividing the CAGR
        of the asset during periods that the benchmark returns are positive (negative)
        by the CAGR of the benchmark during the same periods.
        CaptureRatio.BOTH is the Up ratio divided by the Down ratio.
        Source: 'Capture Ratios: A Popular Method of Measuring Portfolio Performance
        in Practice', Don R. Cox and Delbert C. Goff, Journal of Economics and
        Finance Education (Vol 2 Winter 2013).
        https://www.economics-finance.org/jefe/volume12-2/11ArticleCox.pdf.

        Parameters
        ----------
        ratio: LiteralCaptureRatio
            The ratio to calculate
        base_column: tuple[str, ValueType] | int, default: -1
            Column of timeseries that is the denominator in the ratio.
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

        Returns
        -------
        Pandas.Series[float]
            Capture Ratios

        """
        loss_limit: float = 0.0
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        fraction = (later - earlier).days / 365.25

        msg = "base_column should be a tuple[str, ValueType] or an integer."
        if isinstance(base_column, tuple):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                :,
                base_column,
            ]
            short_item = base_column
            short_label = cast(
                tuple[str, str],
                self.tsdf.loc[:, base_column].name,
            )[0]
        elif isinstance(base_column, int):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[
                :,
                base_column,
            ]
            short_item = self.tsdf.iloc[
                :,
                base_column,
            ].name
            short_label = cast(tuple[str, str], self.tsdf.iloc[:, base_column].name)[0]
        else:
            raise TypeError(msg)

        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = float(shortdf.count() / fraction)

        ratios = []
        for item in self.tsdf:
            if item == short_item:
                ratios.append(0.0)
            else:
                longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                    :,
                    item,
                ]
                msg = "ratio must be one of 'up', 'down' or 'both'."
                if ratio == "up":
                    uparray = (
                        longdf.pct_change()[
                            shortdf.pct_change().to_numpy() > loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    up_rtrn = uparray.prod() ** (1 / (len(uparray) / time_factor)) - 1
                    upidxarray = (
                        shortdf.pct_change()[
                            shortdf.pct_change().to_numpy() > loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    up_idx_return = (
                        upidxarray.prod() ** (1 / (len(upidxarray) / time_factor)) - 1
                    )
                    ratios.append(up_rtrn / up_idx_return)
                elif ratio == "down":
                    downarray = (
                        longdf.pct_change()[
                            shortdf.pct_change().to_numpy() < loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    down_return = (
                        downarray.prod() ** (1 / (len(downarray) / time_factor)) - 1
                    )
                    downidxarray = (
                        shortdf.pct_change()[
                            shortdf.pct_change().to_numpy() < loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    down_idx_return = (
                        downidxarray.prod() ** (1 / (len(downidxarray) / time_factor))
                        - 1
                    )
                    ratios.append(down_return / down_idx_return)
                elif ratio == "both":
                    uparray = (
                        longdf.pct_change()[
                            shortdf.pct_change().to_numpy() > loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    up_rtrn = uparray.prod() ** (1 / (len(uparray) / time_factor)) - 1
                    upidxarray = (
                        shortdf.pct_change()[
                            shortdf.pct_change().to_numpy() > loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    up_idx_return = (
                        upidxarray.prod() ** (1 / (len(upidxarray) / time_factor)) - 1
                    )
                    downarray = (
                        longdf.pct_change()[
                            shortdf.pct_change().to_numpy() < loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    down_return = (
                        downarray.prod() ** (1 / (len(downarray) / time_factor)) - 1
                    )
                    downidxarray = (
                        shortdf.pct_change()[
                            shortdf.pct_change().to_numpy() < loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    down_idx_return = (
                        downidxarray.prod() ** (1 / (len(downidxarray) / time_factor))
                        - 1
                    )
                    ratios.append(
                        (up_rtrn / up_idx_return) / (down_return / down_idx_return),
                    )
                else:
                    raise ValueError(msg)

        if ratio == "up":
            resultname = f"Up Capture Ratios vs {short_label}"
        elif ratio == "down":
            resultname = f"Down Capture Ratios vs {short_label}"
        else:
            resultname = f"Up-Down Capture Ratios vs {short_label}"

        return Series(
            data=ratios,
            index=self.tsdf.columns,
            name=resultname,
            dtype="float64",
        )

    def beta(
        self: Self,
        asset: tuple[str, ValueType] | int,
        market: tuple[str, ValueType] | int,
        dlta_degr_freedms: int = 1,
    ) -> float:
        """Market Beta.

        Calculates Beta as Co-variance of asset & market divided by Variance
        of the market. https://www.investopedia.com/terms/b/beta.asp.

        Parameters
        ----------
        asset: tuple[str, ValueType] | int
            The column of the asset
        market: tuple[str, ValueType] | int
            The column of the market against which Beta is measured
        dlta_degr_freedms: int, default: 1
            Variance bias factor taking the value 0 or 1.

        Returns
        -------
        float
            Beta as Co-variance of x & y divided by Variance of x

        """
        if all(
            x_value == ValueType.RTRN
            for x_value in self.tsdf.columns.get_level_values(1).to_numpy()
        ):
            msg = "asset should be a tuple[str, ValueType] or an integer."
            if isinstance(asset, tuple):
                y_value = self.tsdf.loc[:, asset]
            elif isinstance(asset, int):
                y_value = self.tsdf.iloc[:, asset]
            else:
                raise TypeError(msg)

            msg = "market should be a tuple[str, ValueType] or an integer."
            if isinstance(market, tuple):
                x_value = self.tsdf.loc[:, market]
            elif isinstance(market, int):
                x_value = self.tsdf.iloc[:, market]
            else:
                raise TypeError(msg)
        else:
            msg = "asset should be a tuple[str, ValueType] or an integer."
            if isinstance(asset, tuple):
                y_value = log(
                    self.tsdf.loc[:, asset] / self.tsdf.loc[:, asset].iloc[0],
                )
            elif isinstance(asset, int):
                y_value = log(
                    self.tsdf.iloc[:, asset] / cast(float, self.tsdf.iloc[0, asset]),
                )
            else:
                raise TypeError(msg)

            msg = "market should be a tuple[str, ValueType] or an integer."
            if isinstance(market, tuple):
                x_value = log(
                    self.tsdf.loc[:, market] / self.tsdf.loc[:, market].iloc[0],
                )
            elif isinstance(market, int):
                x_value = log(
                    self.tsdf.iloc[:, market] / cast(float, self.tsdf.iloc[0, market]),
                )
            else:
                raise TypeError(msg)

        covariance = cov(y_value, x_value, ddof=dlta_degr_freedms)
        beta = covariance[0, 1] / covariance[1, 1]

        return float(beta)

    def ord_least_squares_fit(
        self: Self,
        y_column: tuple[str, ValueType] | int,
        x_column: tuple[str, ValueType] | int,
        method: LiteralOlsFitMethod = "pinv",
        cov_type: LiteralOlsFitCovType = "nonrobust",
        *,
        fitted_series: bool = True,
    ) -> OLSResults:
        """Ordinary Least Squares fit.

        Performs a linear regression and adds a new column with a fitted line
        using Ordinary Least Squares fit
        https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html.

        Parameters
        ----------
        y_column: tuple[str, ValueType] | int
            The column level values of the dependent variable y
        x_column: tuple[str, ValueType] | int
            The column level values of the exogenous variable x
        method: LiteralOlsFitMethod, default: pinv
            Method to solve least squares problem
        cov_type: LiteralOlsFitCovType, default: nonrobust
            Covariance estimator
        fitted_series: bool, default: True
            If True the fit is added as a new column in the .tsdf Pandas.DataFrame

        Returns
        -------
        OLSResults
            The Statsmodels regression output

        """
        msg = "y_column should be a tuple[str, ValueType] or an integer."
        if isinstance(y_column, tuple):
            y_value = self.tsdf.loc[:, y_column]
            y_label = cast(
                tuple[str, str],
                self.tsdf.loc[:, y_column].name,
            )[0]
        elif isinstance(y_column, int):
            y_value = self.tsdf.iloc[:, y_column]
            y_label = cast(tuple[str, str], self.tsdf.iloc[:, y_column].name)[0]
        else:
            raise TypeError(msg)

        msg = "x_column should be a tuple[str, ValueType] or an integer."
        if isinstance(x_column, tuple):
            x_value = self.tsdf.loc[:, x_column]
            x_label = cast(
                tuple[str, str],
                self.tsdf.loc[:, x_column].name,
            )[0]
        elif isinstance(x_column, int):
            x_value = self.tsdf.iloc[:, x_column]
            x_label = cast(tuple[str, str], self.tsdf.iloc[:, x_column].name)[0]
        else:
            raise TypeError(msg)

        results = sm.OLS(y_value, x_value).fit(method=method, cov_type=cov_type)
        if fitted_series:
            self.tsdf[y_label, x_label] = results.predict(x_value)

        return cast(OLSResults, results)

    def jensen_alpha(  # noqa: C901
        self: Self,
        asset: tuple[str, ValueType] | int,
        market: tuple[str, ValueType] | int,
        riskfree_rate: float = 0.0,
        dlta_degr_freedms: int = 1,
    ) -> float:
        """Jensen's alpha.

        The Jensen's measure, or Jensen's alpha, is a risk-adjusted performance
        measure that represents the average return on a portfolio or investment,
        above or below that predicted by the capital asset pricing model (CAPM),
        given the portfolio's or investment's beta and the average market return.
        This metric is also commonly referred to as simply alpha.
        https://www.investopedia.com/terms/j/jensensmeasure.asp.

        Parameters
        ----------
        asset: tuple[str, ValueType] | int
            The column of the asset
        market: tuple[str, ValueType] | int
            The column of the market against which Jensen's alpha is measured
        riskfree_rate : float, default: 0.0
            The return of the zero volatility riskfree asset
        dlta_degr_freedms: int, default: 1
            Variance bias factor taking the value 0 or 1.

        Returns
        -------
        float
            Jensen's alpha

        """
        full_year = 1.0
        if all(
            x == ValueType.RTRN
            for x in self.tsdf.columns.get_level_values(1).to_numpy()
        ):
            msg = "asset should be a tuple[str, ValueType] or an integer."
            if isinstance(asset, tuple):
                asset_log = self.tsdf.loc[:, asset]
                asset_cagr = asset_log.mean()
            elif isinstance(asset, int):
                asset_log = self.tsdf.iloc[:, asset]
                asset_cagr = asset_log.mean()
            else:
                raise TypeError(msg)

            msg = "market should be a tuple[str, ValueType] or an integer."
            if isinstance(market, tuple):
                market_log = self.tsdf.loc[:, market]
                market_cagr = market_log.mean()
            elif isinstance(market, int):
                market_log = self.tsdf.iloc[:, market]
                market_cagr = market_log.mean()
            else:
                raise TypeError(msg)
        else:
            msg = "asset should be a tuple[str, ValueType] or an integer."
            if isinstance(asset, tuple):
                asset_log = log(
                    self.tsdf.loc[:, asset] / self.tsdf.loc[:, asset].iloc[0],
                )
                if self.yearfrac > full_year:
                    asset_cagr = (
                        self.tsdf.loc[:, asset].iloc[-1]
                        / self.tsdf.loc[:, asset].iloc[0]
                    ) ** (1 / self.yearfrac) - 1
                else:
                    asset_cagr = (
                        self.tsdf.loc[:, asset].iloc[-1]
                        / self.tsdf.loc[:, asset].iloc[0]
                        - 1
                    )
            elif isinstance(asset, int):
                asset_log = log(
                    self.tsdf.iloc[:, asset] / cast(float, self.tsdf.iloc[0, asset]),
                )
                if self.yearfrac > full_year:
                    asset_cagr = (
                        cast(float, self.tsdf.iloc[-1, asset])
                        / cast(float, self.tsdf.iloc[0, asset])
                    ) ** (1 / self.yearfrac) - 1
                else:
                    asset_cagr = (
                        cast(float, self.tsdf.iloc[-1, asset])
                        / cast(float, self.tsdf.iloc[0, asset])
                        - 1
                    )
            else:
                raise TypeError(msg)

            msg = "market should be a tuple[str, ValueType] or an integer."
            if isinstance(market, tuple):
                market_log = log(
                    self.tsdf.loc[:, market] / self.tsdf.loc[:, market].iloc[0],
                )
                if self.yearfrac > full_year:
                    market_cagr = (
                        self.tsdf.loc[:, market].iloc[-1]
                        / self.tsdf.loc[:, market].iloc[0]
                    ) ** (1 / self.yearfrac) - 1
                else:
                    market_cagr = (
                        self.tsdf.loc[:, market].iloc[-1]
                        / self.tsdf.loc[:, market].iloc[0]
                        - 1
                    )
            elif isinstance(market, int):
                market_log = log(
                    self.tsdf.iloc[:, market] / cast(float, self.tsdf.iloc[0, market]),
                )
                if self.yearfrac > full_year:
                    market_cagr = (
                        cast(float, self.tsdf.iloc[-1, market])
                        / cast(float, self.tsdf.iloc[0, market])
                    ) ** (1 / self.yearfrac) - 1
                else:
                    market_cagr = (
                        cast(float, self.tsdf.iloc[-1, market])
                        / cast(float, self.tsdf.iloc[0, market])
                        - 1
                    )
            else:
                raise TypeError(msg)

        covariance = cov(asset_log, market_log, ddof=dlta_degr_freedms)
        beta = covariance[0, 1] / covariance[1, 1]

        return float(asset_cagr - riskfree_rate - beta * (market_cagr - riskfree_rate))

    def make_portfolio(
        self: Self,
        name: str,
        weight_strat: LiteralPortfolioWeightings | None = None,
    ) -> DataFrame:
        """Calculate a basket timeseries based on the supplied weights.

        Parameters
        ----------
        name: str
            Name of the basket timeseries
        weight_strat: LiteralPortfolioWeightings, optional
            weight calculation strategies

        Returns
        -------
        Pandas.DataFrame
            A basket timeseries

        """
        if self.weights is None and weight_strat is None:
            msg = (
                "OpenFrame weights property must be provided "
                "to run the make_portfolio method."
            )
            raise ValueError(msg)
        dframe = self.tsdf.copy()
        if not any(
            x == ValueType.RTRN
            for x in self.tsdf.columns.get_level_values(1).to_numpy()
        ):
            dframe = dframe.pct_change()
            dframe.iloc[0] = 0

        msg = "Weight strategy not implemented"
        if weight_strat:
            if weight_strat == "eq_weights":
                self.weights = [1.0 / self.item_count] * self.item_count
            elif weight_strat == "inv_vol":
                vol = divide(1.0, std(dframe, axis=0, ddof=1))
                vol[isinf(vol)] = nan
                self.weights = list(divide(vol, vol.sum()))
            else:
                raise NotImplementedError(msg)

        return DataFrame(
            data=(dframe @ self.weights).add(1.0).cumprod(),
            index=self.tsdf.index,
            columns=[[name], [ValueType.PRICE]],
            dtype="float64",
        )

    def rolling_info_ratio(
        self: Self,
        long_column: int = 0,
        short_column: int = 1,
        observations: int = 21,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> DataFrame:
        """Calculate rolling Information Ratio.

        The Information Ratio equals ( fund return less index return ) divided by
        the Tracking Error. And the Tracking Error is the standard deviation of the
        difference between the fund and its index returns.

        Parameters
        ----------
        long_column: int, default: 0
            Column of timeseries that is the numerator in the ratio.
        short_column: int, default: 1
            Column of timeseries that is the denominator in the ratio.
        observations: int, default: 21
            The length of the rolling window to use is set as number of observations.
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        Pandas.DataFrame
            Rolling Information Ratios

        """
        long_label = cast(
            tuple[str, str],
            self.tsdf.iloc[:, long_column].name,
        )[0]
        short_label = cast(
            tuple[str, str],
            self.tsdf.iloc[:, short_column].name,
        )[0]
        ratio_label = f"{long_label} / {short_label}"
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = self.periods_in_a_year

        relative = (
            1.0 + self.tsdf.iloc[:, long_column] - self.tsdf.iloc[:, short_column]
        )

        retseries = (
            relative.pct_change().rolling(observations, min_periods=observations).sum()
        )
        retdf = retseries.dropna().to_frame()

        voldf = relative.pct_change().rolling(
            observations,
            min_periods=observations,
        ).std() * sqrt(time_factor)
        voldf = voldf.dropna().to_frame()

        ratiodf = (retdf.iloc[:, 0] / voldf.iloc[:, 0]).to_frame()
        ratiodf.columns = [[ratio_label], ["Information Ratio"]]

        return DataFrame(ratiodf)

    def rolling_beta(
        self: Self,
        asset_column: int = 0,
        market_column: int = 1,
        observations: int = 21,
        dlta_degr_freedms: int = 1,
    ) -> DataFrame:
        """Calculate rolling Market Beta.

        Calculates Beta as Co-variance of asset & market divided by Variance
        of the market. https://www.investopedia.com/terms/b/beta.asp.

        Parameters
        ----------
        asset_column: int, default: 0
            Column of timeseries that is the asset.
        market_column: int, default: 1
            Column of timeseries that is the market.
        observations: int, default: 21
            The length of the rolling window to use is set as number of observations.
        dlta_degr_freedms: int, default: 1
            Variance bias factor taking the value 0 or 1.

        Returns
        -------
        Pandas.DataFrame
            Rolling Betas

        """
        market_label = cast(tuple[str, str], self.tsdf.iloc[:, market_column].name)[0]
        asset_label = cast(tuple[str, str], self.tsdf.iloc[:, asset_column].name)[0]
        beta_label = f"{asset_label} / {market_label}"

        rolling = self.tsdf.pct_change().rolling(
            observations,
            min_periods=observations,
        )

        rcov = rolling.cov(ddof=dlta_degr_freedms)
        rcov = rcov.dropna()

        rollbetaseries = rcov.iloc[:, asset_column].xs(
            market_label,
            level=1,
        ) / rcov.iloc[
            :,
            market_column,
        ].xs(
            market_label,
            level=1,
        )
        rollbeta = rollbetaseries.to_frame()
        rollbeta.index = rollbeta.index.droplevel(level=1)
        rollbeta.columns = MultiIndex.from_arrays([[beta_label], ["Beta"]])

        return rollbeta

    def rolling_corr(
        self: Self,
        first_column: int = 0,
        second_column: int = 1,
        observations: int = 21,
    ) -> DataFrame:
        """Calculate rolling Correlation.

        Calculates correlation between two series. The period with
        at least the given number of observations is the first period calculated.

        Parameters
        ----------
        first_column: int, default: 0
            The position as integer of the first timeseries to compare
        second_column: int, default: 1
            The position as integer of the second timeseries to compare
        observations: int, default: 21
            The length of the rolling window to use is set as number of observations

        Returns
        -------
        Pandas.DataFrame
            Rolling Correlations

        """
        corr_label = (
            cast(tuple[str, str], self.tsdf.iloc[:, first_column].name)[0]
            + "_VS_"
            + cast(tuple[str, str], self.tsdf.iloc[:, second_column].name)[0]
        )
        first_series = (
            self.tsdf.iloc[:, first_column]
            .pct_change()[1:]
            .rolling(observations, min_periods=observations)
        )
        second_series = self.tsdf.iloc[:, second_column].pct_change()[1:]
        corrdf = first_series.corr(other=second_series).dropna().to_frame()
        corrdf.columns = MultiIndex.from_arrays(
            [
                [corr_label],
                ["Rolling correlation"],
            ],
        )

        return DataFrame(corrdf)
