"""The OpenFrame class."""

from __future__ import annotations

from copy import deepcopy
from functools import reduce
from logging import getLogger
from typing import TYPE_CHECKING, Any, Self, cast

from numpy import (
    array,
    asarray,
    bool_,
    concatenate,
    corrcoef,
    cov,
    diff,
    divide,
    float64,
    isinf,
    isnan,
    linalg,
    log,
    nan,
    sqrt,
    std,
)
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    concat,
    merge,
)

if TYPE_CHECKING:  # pragma: no cover
    import datetime as dt

    from numpy.typing import NDArray
    from pandas import Series as _Series
    from pandas import Timestamp

    SeriesFloat = _Series[float]
else:
    SeriesFloat = Series

from pydantic import field_validator
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

from ._common_model import _calculate_time_factor, _CommonModel, _get_base_column_data
from .datefixer import _do_resample_to_business_period_ends
from .owntypes import (
    DaysInYearType,
    LabelsNotUniqueError,
    LiteralBizDayFreq,
    LiteralCaptureRatio,
    LiteralFrameProps,
    LiteralHowMerge,
    LiteralPandasReindexMethod,
    LiteralPortfolioWeightings,
    LiteralTrunc,
    MaxDiversificationNaNError,
    MaxDiversificationNegativeWeightsError,
    MergingResultedInEmptyError,
    MixedValuetypesError,
    MultipleCurrenciesError,
    NoWeightsError,
    OpenFramePropertiesList,
    PortfolioItemsNotWithinFrameError,
    RatioInputError,
    ResampleDataLossError,
    ValueType,
    WeightsNotProvidedError,
)
from .series import OpenTimeSeries

logger = getLogger(__name__)

__all__ = ["OpenFrame"]


class OpenFrame(_CommonModel[SeriesFloat]):
    """OpenFrame objects hold OpenTimeSeries in the list constituents.

    The intended use is to allow comparisons across these timeseries.

    Args:
        constituents: List of objects of Class OpenTimeSeries.
        weights: List of weights in float format. Optional.
    """

    constituents: list[OpenTimeSeries]
    tsdf: DataFrame = DataFrame(dtype="float64")
    weights: list[float] | None = None

    @field_validator("constituents")
    def _check_labels_unique(
        cls: type[OpenFrame],  # noqa: N805
        tseries: list[OpenTimeSeries],
    ) -> list[OpenTimeSeries]:
        """Pydantic validator ensuring that OpenFrame labels are unique."""
        labls = [x.label for x in tseries]
        if len(set(labls)) != len(labls):
            msg = "TimeSeries names/labels must be unique"
            raise LabelsNotUniqueError(msg)
        return tseries

    def __init__(
        self: Self,
        constituents: list[OpenTimeSeries],
        weights: list[float] | None = None,
    ) -> None:
        """OpenFrame objects hold OpenTimeSeries in the list constituents.

        The intended use is to allow comparisons across these timeseries.

        Args:
            constituents: List of objects of Class OpenTimeSeries.
            weights: List of weights in float format. Optional.
        """
        copied_constituents = [ts.from_deepcopy() for ts in constituents]

        super().__init__(  # type: ignore[call-arg]
            constituents=copied_constituents,
            weights=weights,
        )
        self._set_tsdf()

    def _set_tsdf(self: Self) -> None:
        """Set the tsdf DataFrame."""
        if self.constituents is not None and len(self.constituents) != 0:
            if len(self.constituents) == 1:
                self.tsdf = self.constituents[0].tsdf.copy()
            else:
                self.tsdf = concat(
                    [x.tsdf for x in self.constituents], axis="columns", sort=True
                )
        else:
            logger.warning("OpenFrame() was passed an empty list.")

    def from_deepcopy(self: Self) -> Self:
        """Create copy of the OpenFrame object.

        Returns:
            An OpenFrame object.
        """
        return deepcopy(self)

    def merge_series(
        self: Self,
        how: LiteralHowMerge = "outer",
    ) -> Self:
        """Merge index of Pandas Dataframes of the constituent OpenTimeSeries.

        Args:
            how: The Pandas merge method. Defaults to "outer".

        Returns:
            An OpenFrame object.
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

        mapper = dict(zip(self.columns_lvl_zero, lvl_zero, strict=True))
        self.tsdf = self.tsdf.rename(columns=mapper, level=0)

        if self.tsdf.empty:
            msg = (
                "Merging OpenTimeSeries DataFrames with "
                f"argument how={how} produced an empty DataFrame."
            )
            raise MergingResultedInEmptyError(msg)

        if how == "inner":
            for xerie in self.constituents:
                xerie.tsdf = xerie.tsdf.loc[self.tsdf.index]
        return self

    def all_properties(
        self: Self,
        properties: list[LiteralFrameProps] | None = None,
    ) -> DataFrame:
        """Calculate chosen timeseries properties.

        Args:
            properties: The properties to calculate. Defaults to calculating all
                available. Optional.

        Returns:
            Properties of the constituent OpenTimeSeries.
        """
        if properties:
            props = OpenFramePropertiesList(*properties)
            prop_list = [getattr(self, x) for x in props]
        else:
            prop_list = [
                getattr(self, x) for x in OpenFramePropertiesList.allowed_strings
            ]
        return cast("DataFrame", concat(prop_list, axis="columns").T)

    @property
    def lengths_of_items(self: Self) -> Series[int]:
        """Number of observations of all constituents.

        Returns:
            Number of observations of all constituents.
        """
        return Series(
            data=[self.tsdf[col].count() for col in self.tsdf.columns],
            index=self.tsdf.columns,
            name="observations",
        ).astype(int)

    @property
    def item_count(self: Self) -> int:
        """Number of constituents.

        Returns:
            Number of constituents.
        """
        return len(self.constituents)

    @property
    def columns_lvl_zero(self: Self) -> list[str]:
        """Level 0 values of the MultiIndex columns in the .tsdf DataFrame.

        Returns:
            Level 0 values of the MultiIndex columns in the .tsdf DataFrame.
        """
        return list(self.tsdf.columns.get_level_values(0))

    @property
    def columns_lvl_one(self: Self) -> list[ValueType]:
        """Level 1 values of the MultiIndex columns in the .tsdf DataFrame.

        Returns:
            Level 1 values of the MultiIndex columns in the .tsdf DataFrame.
        """
        return list(self.tsdf.columns.get_level_values(1))

    @property
    def _value_types(self: Self) -> list[bool]:
        """Cached value type checks for efficiency.

        Returns:
            List of booleans indicating if each column is ValueType.RTRN.
        """
        return [x == ValueType.RTRN for x in self.tsdf.columns.get_level_values(1)]

    @property
    def first_indices(self: Self) -> Series[dt.date]:
        """The first dates in the timeseries of all constituents.

        Returns:
            The first dates in the timeseries of all constituents.
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

        Returns:
            The last dates in the timeseries of all constituents.
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

        Returns:
            Number of days from the first date to the last for all
            items in the frame.
        """
        return Series(
            data=[c.span_of_days for c in self.constituents],
            index=self.tsdf.columns,
            name="span of days",
        ).astype(int)

    def value_to_ret(self: Self) -> Self:
        """Convert series of values into series of returns.

        Returns:
            The returns of the values in the series.
        """
        returns = self.tsdf.ffill().pct_change()
        returns.iloc[0] = 0
        new_labels: list[ValueType] = [ValueType.RTRN] * self.item_count
        arrays: list[Index[Any], list[ValueType]] = [  # type: ignore[type-arg]
            self.tsdf.columns.get_level_values(0),
            new_labels,
        ]
        returns.columns = MultiIndex.from_arrays(arrays=arrays)
        self.tsdf = returns.copy()
        return self

    def value_to_diff(self: Self, periods: int = 1) -> Self:
        """Convert series of values to series of their period differences.

        Args:
            periods: The number of periods between observations over which
                difference is calculated. Defaults to 1.

        Returns:
            An OpenFrame object.
        """
        self.tsdf = self.tsdf.diff(periods=periods)
        self.tsdf.iloc[0] = 0
        new_labels: list[ValueType] = [ValueType.RTRN] * self.item_count
        arrays: list[Index[Any], list[ValueType]] = [  # type: ignore[type-arg]
            self.tsdf.columns.get_level_values(0),
            new_labels,
        ]
        self.tsdf.columns = MultiIndex.from_arrays(arrays)
        return self

    def to_cumret(self: Self) -> Self:
        """Convert series of returns into cumulative series of values.

        Returns:
            An OpenFrame object.
        """
        vtypes = self._value_types
        if not any(vtypes):
            returns = self.tsdf.ffill().pct_change()
            returns.iloc[0] = 0
        elif all(vtypes):
            returns = self.tsdf.copy()
            returns.iloc[0] = 0
        else:
            msg = "Mix of series types will give inconsistent results"
            raise MixedValuetypesError(msg)

        returns = returns.add(1.0)
        self.tsdf = returns.cumprod(axis=0) / returns.iloc[0]

        new_labels: list[ValueType] = [ValueType.PRICE] * self.item_count
        arrays: list[Index[Any], list[ValueType]] = [  # type: ignore[type-arg]
            self.tsdf.columns.get_level_values(0),
            new_labels,
        ]
        self.tsdf.columns = MultiIndex.from_arrays(arrays)
        return self

    def resample(
        self: Self,
        freq: LiteralBizDayFreq | str = "BME",
    ) -> Self:
        """Resample the timeseries frequency.

        Args:
            freq: The date offset string that sets the resampled frequency.
                Defaults to "BME".

        Returns:
            An OpenFrame object.
        """
        vtypes = self._value_types
        if not any(vtypes):
            value_type = ValueType.PRICE
        elif all(vtypes):
            value_type = ValueType.RTRN
        else:
            msg = "Mix of series types will give inconsistent results"
            raise MixedValuetypesError(msg)

        self.tsdf.index = DatetimeIndex(self.tsdf.index)
        if value_type == ValueType.PRICE:
            self.tsdf = self.tsdf.resample(freq).last()
        else:
            self.tsdf = self.tsdf.resample(freq).sum()
        self.tsdf.index = Index(DatetimeIndex(self.tsdf.index).date)
        for xerie in self.constituents:
            xerie.tsdf.index = DatetimeIndex(xerie.tsdf.index)
            if value_type == ValueType.PRICE:
                xerie.tsdf = xerie.tsdf.resample(freq).last()
            else:
                xerie.tsdf = xerie.tsdf.resample(freq).sum()
            xerie.tsdf.index = Index(DatetimeIndex(xerie.tsdf.index).date)

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
                Defaults to "BME".
            method: Controls the method used to align values across columns.
                Defaults to nearest.

        Returns:
            An OpenFrame object.
        """
        vtypes = self._value_types
        if any(vtypes):
            msg = (
                "Do not run resample_to_business_period_ends on return series. "
                "The operation will pick the last data point in the sparser series. "
                "It will not sum returns and therefore data will be lost."
            )
            raise ResampleDataLossError(msg)

        for xerie in self.constituents:
            dates = _do_resample_to_business_period_ends(
                data=xerie.tsdf,
                freq=freq,
                countries=xerie.countries,
                markets=xerie.markets,
            )
            xerie.tsdf = xerie.tsdf.reindex(
                [deyt.date() for deyt in dates],
                method=method,
            )

        arrays = [
            self.tsdf.columns.get_level_values(0),
            self.tsdf.columns.get_level_values(1),
        ]

        self._set_tsdf()

        self.tsdf.columns = MultiIndex.from_arrays(arrays)

        return self

    def ewma_risk(
        self: Self,
        lmbda: float = 0.94,
        day_chunk: int = 11,
        dlta_degr_freedms: int = 0,
        first_column: int = 0,
        second_column: int = 1,
        corr_scale: float = 2.0,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: DaysInYearType | None = None,
    ) -> DataFrame:
        """Exponentially Weighted Moving Average Volatilities and Correlation.

        Exponentially Weighted Moving Average (EWMA) for Volatilities and
        Correlation.

        Reference: https://www.investopedia.com/articles/07/ewma.asp.

        Args:
            lmbda: Scaling factor to determine weighting. Defaults to 0.94.
            day_chunk: Sampling the data which is assumed to be daily. Defaults to 11.
            dlta_degr_freedms: Variance bias factor taking the value 0 or 1.
                Defaults to 0.
            first_column: Column of first timeseries. Defaults to 0.
            second_column: Column of second timeseries. Defaults to 1.
            corr_scale: Correlation scale factor. Defaults to 2.0.
            months_from_last: Number of months offset as positive integer. Overrides
                use of from_date and to_date. Optional.
            from_date: Specific from date. Optional.
            to_date: Specific to date. Optional.
            periods_in_a_year_fixed: Allows locking the periods-in-a-year to simplify
                test cases and comparisons. Optional.

        Returns:
            Series volatilities and correlation.
        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        if periods_in_a_year_fixed is None:
            fraction = (later - earlier).days / 365.25
            how_many = (
                self.tsdf.loc[cast("Timestamp", earlier) : cast("Timestamp", later)]
                .count()
                .iloc[0]
            )
            time_factor = how_many / fraction
        else:
            time_factor = periods_in_a_year_fixed

        corr_label = (
            cast("tuple[str, str]", self.tsdf.iloc[:, first_column].name)[0]
            + "_VS_"
            + cast("tuple[str, str]", self.tsdf.iloc[:, second_column].name)[0]
        )
        cols = [
            cast("tuple[str, str]", self.tsdf.iloc[:, first_column].name)[0],
            cast("tuple[str, str]", self.tsdf.iloc[:, second_column].name)[0],
        ]

        data = self.tsdf.loc[
            cast("Timestamp", earlier) : cast("Timestamp", later)
        ].copy()

        for rtn in cols:
            arr = concatenate([array([nan]), diff(log(data[(rtn, ValueType.PRICE)]))])
            data[rtn, ValueType.RTRN] = arr

        raw_one = [
            data[(cols[0], ValueType.RTRN)]
            .iloc[1:day_chunk]
            .std(ddof=dlta_degr_freedms)
            * sqrt(time_factor),
        ]
        raw_two = [
            data[(cols[1], ValueType.RTRN)]
            .iloc[1:day_chunk]
            .std(ddof=dlta_degr_freedms)
            * sqrt(time_factor),
        ]
        rm = data[(cols[0], ValueType.RTRN)].iloc[1:day_chunk]
        m: NDArray[float64] = asarray(rm, dtype=float64)
        ry = data[(cols[1], ValueType.RTRN)].iloc[1:day_chunk]
        y: NDArray[float64] = asarray(ry, dtype=float64)

        raw_cov = [cov(m=m, y=y, ddof=dlta_degr_freedms)[0][1]]

        r1 = data[(cols[0], ValueType.RTRN)]
        r2 = data[(cols[1], ValueType.RTRN)]

        alpha = 1.0 - lmbda

        s1 = r1.pow(2) * time_factor
        s2 = r2.pow(2) * time_factor
        sc = r1 * r2 * time_factor

        s1.iloc[0] = float(raw_one[0] ** 2)
        s2.iloc[0] = float(raw_two[0] ** 2)
        sc.iloc[0] = float(raw_cov[0])

        m1 = s1.ewm(alpha=alpha, adjust=False).mean()
        m2 = s2.ewm(alpha=alpha, adjust=False).mean()
        mc = sc.ewm(alpha=alpha, adjust=False).mean()

        m1v = m1.to_numpy(copy=False)
        m2v = m2.to_numpy(copy=False)
        mcv = mc.to_numpy(copy=False)

        vol1 = sqrt(m1v)
        vol2 = sqrt(m2v)
        denom = corr_scale * vol1 * vol2

        corr = mcv / denom
        corr[denom == 0.0] = nan

        return DataFrame(
            index=[*cols, corr_label],
            columns=data.index,
            data=[vol1, vol2, corr],
        ).T

    @property
    def correl_matrix(self: Self) -> DataFrame:
        """Correlation matrix.

        This property returns the correlation matrix of the time series
        in the frame.

        Returns:
            Correlation matrix of the time series in the frame.
        """
        corr_matrix = (
            self.tsdf.ffill()
            .pct_change()
            .corr(
                method="pearson",
                min_periods=1,
            )
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

        Args:
            new_series: The timeseries to add.

        Returns:
            An OpenFrame object.
        """
        self.constituents += [new_series]
        self.tsdf = concat([self.tsdf, new_series.tsdf], axis="columns", sort=True)
        return self

    def delete_timeseries(self: Self, lvl_zero_item: str) -> Self:
        """To delete an OpenTimeSeries object.

        Args:
            lvl_zero_item: The .tsdf column level 0 value of the timeseries to delete.

        Returns:
            An OpenFrame object.
        """
        if self.weights:
            new_c, new_w = [], []
            for serie, weight in zip(self.constituents, self.weights, strict=True):
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

        Args:
            start_cut: New first date. Optional.
            end_cut: New last date. Optional.
            where: Determines where dataframe is truncated also when start_cut
                or end_cut is None. Defaults to both.

        Returns:
            An OpenFrame object.
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
            logger.warning(msg)
        if len(set(self.last_indices)) != 1:
            msg = (
                f"One or more constituents still "
                f"not truncated to same end dates.\n"
                f"{self.tsdf.tail()}"
            )
            logger.warning(msg)
        return self

    def relative(
        self: Self,
        long_column: int = 0,
        short_column: int = 1,
        *,
        base_zero: bool = True,
    ) -> None:
        """Calculate cumulative relative return between two series.

        Args:
            long_column: Column number of timeseries bought. Defaults to 0.
            short_column: Column number of timeseries sold. Defaults to 1.
            base_zero: If set to False 1.0 is added to allow for a capital base and
                to allow a volatility calculation. Defaults to True.
        """
        rel_label = (
            cast("tuple[str, str]", self.tsdf.iloc[:, long_column].name)[0]
            + "_over_"
            + cast("tuple[str, str]", self.tsdf.iloc[:, short_column].name)[0]
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

        Reference: https://www.investopedia.com/terms/t/trackingerror.asp.

        Args:
            base_column: Column of timeseries that is the denominator in the ratio.
                Defaults to -1.
            months_from_last: Number of months offset as positive integer. Overrides
                use of from_date and to_date. Optional.
            from_date: Specific from date. Optional.
            to_date: Specific to date. Optional.
            periods_in_a_year_fixed: Allows locking the periods-in-a-year to simplify
                test cases and comparisons. Optional.

        Returns:
            Tracking Errors.
        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )

        shortdf, short_item, short_label = _get_base_column_data(
            self=self,
            base_column=base_column,
            earlier=earlier,
            later=later,
        )

        time_factor = _calculate_time_factor(
            data=shortdf,
            earlier=earlier,
            later=later,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

        shortdf_returns = shortdf.ffill().pct_change()

        terrors = []
        for item in self.tsdf:
            if item == short_item:
                terrors.append(0.0)
            else:
                longdf = self.tsdf.loc[
                    cast("Timestamp", earlier) : cast("Timestamp", later)
                ][item]
                relative = longdf.ffill().pct_change() - shortdf_returns
                vol = float(relative.std() * sqrt(time_factor))
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

        Args:
            base_column: Column of timeseries that is the denominator in the ratio.
                Defaults to -1.
            months_from_last: Number of months offset as positive integer. Overrides
                use of from_date and to_date. Optional.
            from_date: Specific from date. Optional.
            to_date: Specific to date. Optional.
            periods_in_a_year_fixed: Allows locking the periods-in-a-year to simplify
                test cases and comparisons. Optional.

        Returns:
            Information Ratios.
        """
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )

        shortdf, short_item, short_label = _get_base_column_data(
            self=self,
            base_column=base_column,
            earlier=earlier,
            later=later,
        )

        time_factor = _calculate_time_factor(
            data=shortdf,
            earlier=earlier,
            later=later,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

        shortdf_returns = shortdf.ffill().pct_change()

        ratios = []
        for item in self.tsdf:
            if item == short_item:
                ratios.append(0.0)
            else:
                longdf = self.tsdf.loc[
                    cast("Timestamp", earlier) : cast("Timestamp", later)
                ][item]
                relative = longdf.ffill().pct_change() - shortdf_returns
                ret = float(relative.mean() * time_factor)
                vol = float(relative.std() * sqrt(time_factor))
                ratios.append(ret / vol)

        return Series(
            data=ratios,
            index=self.tsdf.columns,
            name=f"Info Ratios vs {short_label}",
            dtype="float64",
        )

    def _calculate_cagr_from_returns(
        self: Self,
        returns_array: NDArray[float64],
        mask: NDArray[bool_],
        time_factor: float,
    ) -> float:
        """Calculate CAGR from returns array with mask.

        Args:
            returns_array: Returns array.
            mask: Boolean mask.
            time_factor: Time factor for annualization.

        Returns:
            CAGR value.
        """
        masked_array = returns_array[mask] + 1.0
        if len(masked_array) == 0:
            return 0.0
        exponent = 1 / (len(masked_array) / time_factor)
        return float(masked_array.prod() ** exponent - 1)

    def _calculate_capture_ratio_for_item(
        self: Self,
        ratio: LiteralCaptureRatio,
        longdf_returns_np: NDArray[float64],
        shortdf_returns_np: NDArray[float64],
        up_mask: NDArray[bool_],
        down_mask: NDArray[bool_],
        time_factor: float,
    ) -> float:
        """Calculate capture ratio for a single item.

        Args:
            ratio: Ratio type to calculate.
            longdf_returns_np: Long returns array.
            shortdf_returns_np: Short returns array.
            up_mask: Up mask.
            down_mask: Down mask.
            time_factor: Time factor.

        Returns:
            Capture ratio value.

        Raises:
            RatioInputError: If ratio is invalid.
        """
        if ratio == "up":
            up_rtrn = self._calculate_cagr_from_returns(
                longdf_returns_np, up_mask, time_factor
            )
            up_idx_return = self._calculate_cagr_from_returns(
                shortdf_returns_np, up_mask, time_factor
            )
            if up_idx_return == 0.0:
                return 0.0
            return up_rtrn / up_idx_return

        if ratio == "down":
            down_return = self._calculate_cagr_from_returns(
                longdf_returns_np, down_mask, time_factor
            )
            down_idx_return = self._calculate_cagr_from_returns(
                shortdf_returns_np, down_mask, time_factor
            )
            if down_idx_return == 0.0:
                return 0.0
            return down_return / down_idx_return

        if ratio == "both":
            up_rtrn = self._calculate_cagr_from_returns(
                longdf_returns_np, up_mask, time_factor
            )
            up_idx_return = self._calculate_cagr_from_returns(
                shortdf_returns_np, up_mask, time_factor
            )
            down_return = self._calculate_cagr_from_returns(
                longdf_returns_np, down_mask, time_factor
            )
            down_idx_return = self._calculate_cagr_from_returns(
                shortdf_returns_np, down_mask, time_factor
            )
            if up_idx_return == 0.0 or down_idx_return == 0.0:
                return 0.0
            return (up_rtrn / up_idx_return) / (down_return / down_idx_return)

        msg = "ratio must be one of 'up', 'down' or 'both'."
        raise RatioInputError(msg)

    def capture_ratio_func(
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

        Reference: https://www.economics-finance.org/jefe/volume12-2/11ArticleCox.pdf.

        Args:
            ratio: The ratio to calculate.
            base_column: Column of timeseries that is the denominator in the ratio.
                Defaults to -1.
            months_from_last: Number of months offset as positive integer. Overrides
                use of from_date and to_date. Optional.
            from_date: Specific from date. Optional.
            to_date: Specific to date. Optional.
            periods_in_a_year_fixed: Allows locking the periods-in-a-year to simplify
                test cases and comparisons. Optional.

        Returns:
            Capture Ratios.
        """
        loss_limit: float = 0.0
        earlier, later = self.calc_range(
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        fraction: float = (later - earlier).days / 365.25

        shortdf, short_item, short_label = _get_base_column_data(
            self=self,
            base_column=base_column,
            earlier=earlier,
            later=later,
        )

        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = shortdf.count() / fraction

        shortdf_returns = shortdf.ffill().pct_change()
        shortdf_returns_np = cast("NDArray[float64]", shortdf_returns.to_numpy())
        up_mask = shortdf_returns_np > loss_limit
        down_mask = shortdf_returns_np < loss_limit

        ratios = []
        for item in self.tsdf:
            if item == short_item:
                ratios.append(0.0)
            else:
                longdf = self.tsdf.loc[
                    cast("Timestamp", earlier) : cast("Timestamp", later)
                ][item]
                longdf_returns = longdf.ffill().pct_change()
                longdf_returns_np = cast("NDArray[float64]", longdf_returns.to_numpy())
                ratio_value = self._calculate_capture_ratio_for_item(
                    ratio=ratio,
                    longdf_returns_np=longdf_returns_np,
                    shortdf_returns_np=shortdf_returns_np,
                    up_mask=up_mask,
                    down_mask=down_mask,
                    time_factor=time_factor,
                )
                ratios.append(ratio_value)

        ratio_names = {
            "up": f"Up Capture Ratios vs {short_label}",
            "down": f"Down Capture Ratios vs {short_label}",
            "both": f"Up-Down Capture Ratios vs {short_label}",
        }
        resultname = ratio_names[ratio]

        return Series(
            data=ratios,
            index=self.tsdf.columns,
            name=resultname,
            dtype="float64",
        )

    def _extract_column_value(
        self: Self,
        column: tuple[str, ValueType] | int,
        vtypes: list[bool],
        param_name: str = "column",
    ) -> Series[float]:
        """Extract column value based on value types.

        Args:
            column: Column reference.
            vtypes: Value types list.
            param_name: Parameter name for error messages.

        Returns:
            Series value.

        Raises:
            TypeError: If column type is invalid.
        """
        msg = f"{param_name} should be a tuple[str, ValueType] or an integer."
        if isinstance(column, tuple):
            if all(vtypes):
                return self.tsdf[column]
            return self.tsdf[column].ffill().pct_change().iloc[1:]
        if isinstance(column, int):
            if all(vtypes):
                return self.tsdf.iloc[:, column]
            return self.tsdf.iloc[:, column].ffill().pct_change().iloc[1:]
        raise TypeError(msg)

    def beta(
        self: Self,
        asset: tuple[str, ValueType] | int,
        market: tuple[str, ValueType] | int,
        dlta_degr_freedms: int = 1,
    ) -> float:
        """Market Beta.

        Calculates Beta as Co-variance of asset & market divided by Variance
        of the market.

        Reference: https://www.investopedia.com/terms/b/beta.asp.

        Args:
            asset: The column of the asset.
            market: The column of the market against which Beta is measured.
            dlta_degr_freedms: Variance bias factor taking the value 0 or 1.
                Defaults to 1.

        Returns:
            Beta as Co-variance of x & y divided by Variance of x.
        """
        vtypes = self._value_types
        if not (all(vtypes) or not any(vtypes)):
            msg = "Mix of series types will give inconsistent results"
            raise MixedValuetypesError(msg)

        y_value = self._extract_column_value(asset, vtypes, param_name="asset")
        x_value = self._extract_column_value(market, vtypes, param_name="market")

        covariance = cov(m=y_value, y=x_value, ddof=dlta_degr_freedms)
        beta = covariance[0, 1] / covariance[1, 1]

        return float(beta)

    def ord_least_squares_fit(
        self: Self,
        y_column: tuple[str, ValueType] | int,
        x_column: tuple[str, ValueType] | int,
        *,
        fitted_series: bool = True,
    ) -> dict[str, float]:
        """Ordinary Least Squares fit.

        Performs a linear regression and adds a new column with a fitted line
        using Ordinary Least Squares fit.

        Args:
            y_column: The column level values of the dependent variable y.
            x_column: The column level values of the exogenous variable x.
            fitted_series: If True the fit is added as a new column in the .tsdf
                Pandas.DataFrame. Defaults to True.

        Returns:
            A dictionary with the coefficient, intercept and rsquared outputs.
        """
        msg = "y_column should be a tuple[str, ValueType] or an integer."
        if isinstance(y_column, tuple):
            y_value = self.tsdf[y_column].to_numpy()
            y_label = cast(
                "tuple[str, str]",
                self.tsdf[y_column].name,
            )[0]
        elif isinstance(y_column, int):
            y_value = self.tsdf.iloc[:, y_column].to_numpy()
            y_label = cast("tuple[str, str]", self.tsdf.iloc[:, y_column].name)[0]
        else:
            raise TypeError(msg)

        msg = "x_column should be a tuple[str, ValueType] or an integer."
        if isinstance(x_column, tuple):
            x_value = self.tsdf[x_column].to_numpy().reshape(-1, 1)
            x_label = cast(
                "tuple[str, str]",
                self.tsdf[x_column].name,
            )[0]
        elif isinstance(x_column, int):
            x_value = self.tsdf.iloc[:, x_column].to_numpy().reshape(-1, 1)
            x_label = cast("tuple[str, str]", self.tsdf.iloc[:, x_column].name)[0]
        else:
            raise TypeError(msg)

        model = LinearRegression(fit_intercept=True)
        model.fit(x_value, y_value)
        if fitted_series:
            self.tsdf[y_label, x_label] = model.predict(x_value)
        return {
            "coefficient": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "rsquared": model.score(x_value, y_value),
        }

    def jensen_alpha(
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

        Reference: https://www.investopedia.com/terms/j/jensensmeasure.asp.

        Args:
            asset: The column of the asset.
            market: The column of the market against which Jensen's alpha is measured.
            riskfree_rate: The return of the zero volatility riskfree asset.
                Defaults to 0.0.
            dlta_degr_freedms: Variance bias factor taking the value 0 or 1.
                Defaults to 1.

        Returns:
            Jensen's alpha.
        """
        vtypes = self._value_types
        if not (all(vtypes) or not any(vtypes)):
            msg = "Mix of series types will give inconsistent results"
            raise MixedValuetypesError(msg)

        asset_rtn = self._extract_column_value(asset, vtypes, param_name="asset")
        market_rtn = self._extract_column_value(market, vtypes, param_name="market")

        asset_rtn_mean = float(asset_rtn.mean() * self.periods_in_a_year)
        market_rtn_mean = float(market_rtn.mean() * self.periods_in_a_year)

        covariance = cov(m=asset_rtn, y=market_rtn, ddof=dlta_degr_freedms)
        beta = covariance[0, 1] / covariance[1, 1]

        return float(
            asset_rtn_mean - riskfree_rate - beta * (market_rtn_mean - riskfree_rate),
        )

    def _prepare_returns_for_portfolio(self: Self) -> DataFrame:
        """Prepare returns DataFrame for portfolio calculation.

        Returns:
            Returns DataFrame.

        Raises:
            MixedValuetypesError: If series types are mixed.
        """
        vtypes = self._value_types
        if not any(vtypes):
            returns = self.tsdf.ffill().pct_change()
            returns.iloc[0] = 0
            return returns
        if all(vtypes):
            return self.tsdf
        msg = "Mix of series types will give inconsistent results"
        raise MixedValuetypesError(msg)

    def _calculate_eq_weights(self: Self) -> list[float]:
        """Calculate equal weights.

        Returns:
            List of equal weights.
        """
        return [1.0 / self.item_count] * self.item_count

    def _calculate_inv_vol_weights(self: Self, returns: DataFrame) -> list[float]:
        """Calculate inverse volatility weights.

        Args:
            returns: Returns DataFrame.

        Returns:
            List of inverse volatility weights.
        """
        vol = divide(1.0, std(returns, axis=0, ddof=1))
        vol[isinf(vol)] = nan
        return list(divide(vol, vol.sum()))

    def _calculate_max_div_weights(self: Self, returns: DataFrame) -> list[float]:
        """Calculate maximum diversification weights.

        Args:
            returns: Returns DataFrame.

        Returns:
            List of maximum diversification weights.

        Raises:
            MaxDiversificationNaNError: If correlation matrix has NaN values.
            MaxDiversificationNegativeWeightsError: If weights are negative.
        """
        corr_matrix = corrcoef(returns.T)
        corr_matrix[isinf(corr_matrix)] = nan
        corr_matrix[isnan(corr_matrix)] = nan

        msga = "max_div weight strategy failed: correlation matrix contains NaN values"
        if isnan(corr_matrix).any():
            raise MaxDiversificationNaNError(msga)

        try:
            inv_corr_sum = linalg.inv(corr_matrix).sum(axis=1)

            msgb = (
                "max_div weight strategy failed: "
                "inverse correlation matrix sum contains NaN values"
            )
            if isnan(inv_corr_sum).any():
                raise MaxDiversificationNaNError(msgb)

            weights = list(divide(inv_corr_sum, inv_corr_sum.sum()))

            msgc = "max_div weight strategy failed: final weights contain NaN values"
            if any(isnan(weight) for weight in weights):  # pragma: no cover
                raise MaxDiversificationNaNError(msgc)

            msgd = (
                "max_div weight strategy failed: negative weights detected"
                f" - weights: {[round(w, 6) for w in weights]}"
            )
            if any(weight < 0 for weight in weights):
                raise MaxDiversificationNegativeWeightsError(msgd)

        except linalg.LinAlgError as e:
            msge = (
                "max_div weight strategy failed: "
                f"correlation matrix is singular - {e!s}"
            )
            raise MaxDiversificationNaNError(msge) from e
        else:
            return weights

    def _calculate_min_vol_overweight_weights(
        self: Self,
        returns: DataFrame,
    ) -> list[float]:
        """Calculate minimum volatility overweight weights.

        Args:
            returns: Returns DataFrame.

        Returns:
            List of minimum volatility overweight weights.
        """
        vols = std(returns, axis=0, ddof=1)
        min_vol_idx = vols.argmin()
        min_vol_weight = 0.6
        remaining_weight = 0.4
        weights = [remaining_weight / (self.item_count - 1)] * self.item_count
        weights[min_vol_idx] = min_vol_weight
        return weights

    def _calculate_weights_from_strategy(
        self: Self,
        weight_strat: LiteralPortfolioWeightings,
        returns: DataFrame,
    ) -> list[float]:
        """Calculate weights based on strategy.

        Args:
            weight_strat: Weight calculation strategy.
            returns: Returns DataFrame.

        Returns:
            List of weights.

        Raises:
            NotImplementedError: If strategy is not implemented.
        """
        if weight_strat == "eq_weights":
            return self._calculate_eq_weights()
        if weight_strat == "inv_vol":
            return self._calculate_inv_vol_weights(returns)
        if weight_strat == "max_div":
            return self._calculate_max_div_weights(returns)
        if weight_strat == "min_vol_overweight":
            return self._calculate_min_vol_overweight_weights(returns)

        msg = "Weight strategy not implemented"
        raise NotImplementedError(msg)

    def make_portfolio(
        self: Self,
        name: str,
        weight_strat: LiteralPortfolioWeightings | None = None,
    ) -> DataFrame:
        """Calculate a basket timeseries based on the supplied weights.

        Args:
            name: Name of the basket timeseries.
            weight_strat: Weight calculation strategies. Optional.

        Returns:
            A basket timeseries.
        """
        if self.weights is None and weight_strat is None:
            msg = (
                "OpenFrame weights property must be provided "
                "to run the make_portfolio method."
            )
            raise NoWeightsError(msg)

        returns = self._prepare_returns_for_portfolio()

        if weight_strat:
            self.weights = self._calculate_weights_from_strategy(
                weight_strat=weight_strat,
                returns=returns,
            )

        return DataFrame(
            data=(returns @ array(self.weights)).add(1.0).cumprod(),
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

        Args:
            long_column: Column of timeseries that is the numerator in the ratio.
                Defaults to 0.
            short_column: Column of timeseries that is the denominator in the ratio.
                Defaults to 1.
            observations: The length of the rolling window to use is set as number of
                observations. Defaults to 21.
            periods_in_a_year_fixed: Allows locking the periods-in-a-year to simplify
                test cases and comparisons. Optional.

        Returns:
            Rolling Information Ratios.
        """
        long_label = cast(
            "tuple[str, str]",
            self.tsdf.iloc[:, long_column].name,
        )[0]
        short_label = cast(
            "tuple[str, str]",
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
            relative.ffill()
            .pct_change()
            .rolling(observations, min_periods=observations)
            .sum()
        )
        retdf = retseries.dropna().to_frame()

        voldf = relative.ffill().pct_change().rolling(
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
        of the market.

        Reference: https://www.investopedia.com/terms/b/beta.asp.

        Args:
            asset_column: Column of timeseries that is the asset. Defaults to 0.
            market_column: Column of timeseries that is the market. Defaults to 1.
            observations: The length of the rolling window to use is set as number of
                observations. Defaults to 21.
            dlta_degr_freedms: Variance bias factor taking the value 0 or 1.
                Defaults to 1.

        Returns:
            Rolling Betas.
        """
        market_label = cast("tuple[str, str]", self.tsdf.iloc[:, market_column].name)[
            0
        ]
        asset_label = cast("tuple[str, str]", self.tsdf.iloc[:, asset_column].name)[0]
        beta_label = f"{asset_label} / {market_label}"

        rolling = (
            self.tsdf.ffill()
            .pct_change()
            .rolling(
                observations,
                min_periods=observations,
            )
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

        Args:
            first_column: The position as integer of the first timeseries to compare.
                Defaults to 0.
            second_column: The position as integer of the second timeseries to compare.
                Defaults to 1.
            observations: The length of the rolling window to use is set as number of
                observations. Defaults to 21.

        Returns:
            Rolling Correlations.
        """
        corr_label = (
            cast("tuple[str, str]", self.tsdf.iloc[:, first_column].name)[0]
            + "_VS_"
            + cast("tuple[str, str]", self.tsdf.iloc[:, second_column].name)[0]
        )
        first_series = (
            self.tsdf.iloc[:, first_column]
            .ffill()
            .pct_change()[1:]
            .rolling(observations, min_periods=observations)
        )
        second_series = self.tsdf.iloc[:, second_column].ffill().pct_change()[1:]
        corrdf = first_series.corr(other=second_series).dropna().to_frame()
        corrdf.columns = MultiIndex.from_arrays(
            [
                [corr_label],
                ["Rolling correlation"],
            ],
        )

        return DataFrame(corrdf)

    def multi_factor_linear_regression(
        self: Self,
        dependent_column: tuple[str, ValueType],
    ) -> tuple[DataFrame, OpenTimeSeries]:
        """Perform a multi-factor linear regression.

        This function treats one specified column in the DataFrame as the dependent
        variable (y) and uses all remaining columns as independent variables (X).
        It utilizes a scikit-learn LinearRegression model and returns a DataFrame
        with summary output and an OpenTimeSeries of predicted values.

        Args:
            dependent_column: A tuple key to select the column in the
                OpenFrame.tsdf.columns to use as the dependent variable.

        Returns:
            A tuple containing:
            - A DataFrame with the R-squared, the intercept and the regression
              coefficients
            - An OpenTimeSeries of predicted values

        Raises:
            KeyError: If the column tuple is not found in the OpenFrame.tsdf.columns.
            ValueError: If not all series are returnseries (ValueType.RTRN).
        """
        key_msg = (
            f"Tuple ({dependent_column[0]}, "
            f"{dependent_column[1].value}) not found in data."
        )
        if dependent_column not in self.tsdf.columns:
            raise KeyError(key_msg)

        vtype_msg = "All series should be of ValueType.RTRN."
        if not all(x == ValueType.RTRN for x in self.tsdf.columns.get_level_values(1)):
            raise MixedValuetypesError(vtype_msg)

        dependent = self.tsdf[dependent_column]
        factors = self.tsdf.drop(columns=[dependent_column])
        indx = ["R-square", "Intercept", *factors.columns.droplevel(level=1)]

        model = LinearRegression()
        model.fit(factors, dependent)

        predictions = OpenTimeSeries.from_arrays(
            name=f"Predicted {dependent_column[0]}",
            dates=[date.strftime("%Y-%m-%d") for date in self.tsdf.index],
            values=list(model.predict(factors)),
            valuetype=ValueType.RTRN,
        )

        output = [model.score(factors, dependent), model.intercept_, *model.coef_]

        result = DataFrame(data=output, index=indx, columns=[dependent_column[0]])

        return result, predictions.to_cumret()

    def _validate_and_prepare_rebalance_inputs(
        self: Self,
        items: list[str] | None,
        bal_weights: list[float] | None,
        *,
        equal_weights: bool,
    ) -> tuple[list[str], list[float]]:
        """Validate and prepare inputs for rebalanced portfolio.

        Args:
            items: List of items to include. If None, uses all items.
            bal_weights: List of weights. If None, uses frame weights.
            equal_weights: If True, use equal weights.

        Returns:
            Tuple of (validated items, validated weights).

        Raises:
            WeightsNotProvidedError: If weights are required but not provided.
            TypeError: If items is not a list.
            PortfolioItemsNotWithinFrameError: If items are invalid.
        """
        if bal_weights is None and not equal_weights:
            if self.weights is None:
                msg = "Weights must be provided."
                raise WeightsNotProvidedError(msg)
            bal_weights = list(self.weights)

        if items is None:
            items = list(self.columns_lvl_zero)
        else:
            msg = "Items must be passed as list."
            if not isinstance(items, list):
                raise TypeError(msg)
            if not items:
                msg = "Items for portfolio must be within SeriesFrame items."
                raise PortfolioItemsNotWithinFrameError(msg)
            if not set(items) <= set(self.columns_lvl_zero):
                msg = "Items for portfolio must be within SeriesFrame items."
                raise PortfolioItemsNotWithinFrameError(msg)

        if equal_weights:
            bal_weights = [1 / len(items)] * len(items)

        return items, cast("list[float]", bal_weights)

    def _initialize_rebalance_output(
        self: Self,
        items: list[str],
        name: str,
        cash_values: list[float],
    ) -> dict[str, dict[str, list[float]]]:
        """Initialize output structure for rebalanced portfolio.

        Args:
            items: List of items in portfolio.
            name: Name of the portfolio.
            cash_values: Cash index values.

        Returns:
            Initialized output dictionary.
        """
        output = {
            item: {
                ValueType.PRICE: [],
                "buysell_qty": [0.0] * self.length,
                "position": [0.0] * self.length,
                "value": [0.0] * self.length,
                "twr": [0.0] * self.length,
                "settle": [0.0] * self.length,
            }
            for item in items
        }
        output.update(
            {
                "cash": {
                    ValueType.PRICE: cash_values,
                    "buysell_qty": [0.0] * self.length,
                    "position": [0.0] * self.length,
                    "value": [0.0] * self.length,
                    "twr": [0.0] * self.length,
                    "settle": [0.0] * self.length,
                },
                name: {
                    ValueType.PRICE: [1.0] + [0.0] * (self.length - 1),
                    "buysell_qty": [-1.0] + [0.0] * (self.length - 1),
                    "position": [-1.0] + [0.0] * (self.length - 1),
                    "value": [-1.0] + [0.0] * (self.length - 1),
                    "twr": [1.0] + [0.0] * (self.length - 1),
                    "settle": [1.0] + [0.0] * (self.length - 1),
                },
            },
        )
        return output

    def _initialize_first_day_positions(
        self: Self,
        items: list[str],
        bal_weights: list[float],
        output: dict[str, dict[str, list[float]]],
        name: str,
    ) -> None:
        """Initialize positions for the first day.

        Args:
            items: List of items in portfolio.
            bal_weights: Weights for each item.
            output: Output dictionary to update.
            name: Name of the portfolio.
        """
        for item, weight in zip(items, bal_weights, strict=False):
            output[item][ValueType.PRICE] = cast(
                "list[float]",
                self.tsdf[(item, ValueType.PRICE)].to_numpy().tolist(),
            )
            output[item]["buysell_qty"][0] = (
                weight / self.tsdf[(item, ValueType.PRICE)].iloc[0]
            )
            output[item]["position"][0] = output[item]["buysell_qty"][0]
            output[item]["value"][0] = (
                output[item]["position"][0] * output[item][ValueType.PRICE][0]
            )
            output[item]["settle"][0] = (
                -output[item]["buysell_qty"][0] * output[item][ValueType.PRICE][0]
            )
            output["cash"]["buysell_qty"][0] += output[item]["settle"][0]
            output[item]["twr"][0] = (
                output[item]["value"][0] / -output[item]["settle"][0]
            )

        output["cash"]["position"][0] = (
            output["cash"]["buysell_qty"][0] + output[name]["settle"][0]
        )
        output["cash"]["settle"][0] = -output["cash"]["position"][0]

    def _process_rebalancing_day(
        self: Self,
        day: int,
        items: list[str],
        bal_weights: list[float],
        output: dict[str, dict[str, list[float]]],
        name: str,
    ) -> tuple[float, float]:
        """Process a rebalancing day.

        Args:
            day: Current day index.
            items: List of items in portfolio.
            bal_weights: Target weights for rebalancing.
            output: Output dictionary to update.
            name: Name of the portfolio.

        Returns:
            Tuple of (portfolio_value, settle_value).
        """
        portfolio_value = 0.0
        settle_value = 0.0

        for item, weight in zip(items, bal_weights, strict=False):
            output[item]["buysell_qty"][day] = (
                weight
                - output[item]["value"][day - 1] / -output[name]["value"][day - 1]
            ) / output[item][ValueType.PRICE][day]
            output[item]["position"][day] = (
                output[item]["position"][day - 1] + output[item]["buysell_qty"][day]
            )
            output[item]["value"][day] = (
                output[item]["position"][day] * output[item][ValueType.PRICE][day]
            )
            portfolio_value += output[item]["value"][day]
            output[item]["twr"][day] = (
                output[item]["value"][day]
                / (output[item]["value"][day - 1] - output[item]["settle"][day])
                * output[item]["twr"][day - 1]
            )
            output[item]["settle"][day] = (
                -output[item]["buysell_qty"][day] * output[item][ValueType.PRICE][day]
            )
            settle_value += output[item]["settle"][day]

        return portfolio_value, settle_value

    def _process_non_rebalancing_day(
        self: Self,
        day: int,
        items: list[str],
        output: dict[str, dict[str, list[float]]],
    ) -> float:
        """Process a non-rebalancing day.

        Args:
            day: Current day index.
            items: List of items in portfolio.
            output: Output dictionary to update.

        Returns:
            Portfolio value.
        """
        portfolio_value = 0.0

        for item in items:
            output[item]["position"][day] = output[item]["position"][day - 1]
            output[item]["value"][day] = (
                output[item]["position"][day] * output[item][ValueType.PRICE][day]
            )
            portfolio_value += output[item]["value"][day]
            output[item]["twr"][day] = (
                output[item]["value"][day]
                / (output[item]["value"][day - 1] - output[item]["settle"][day])
                * output[item]["twr"][day - 1]
            )

        return portfolio_value

    def _update_cash_and_portfolio(
        self: Self,
        day: int,
        portfolio_value: float,
        settle_value: float,
        output: dict[str, dict[str, list[float]]],
        name: str,
    ) -> None:
        """Update cash and portfolio values for a day.

        Args:
            day: Current day index.
            portfolio_value: Total portfolio value (before cash).
            settle_value: Total settle value.
            output: Output dictionary to update.
            name: Name of the portfolio.
        """
        output["cash"]["buysell_qty"][day] = settle_value
        output["cash"]["position"][day] = (
            output["cash"]["position"][day - 1]
            * output["cash"][ValueType.PRICE][day]
            / output["cash"][ValueType.PRICE][day - 1]
            + output["cash"]["buysell_qty"][day]
        )
        output["cash"]["value"][day] = output["cash"]["position"][day]
        total_portfolio_value = portfolio_value + output["cash"]["value"][day]
        output[name]["position"][day] = output[name]["position"][day - 1]
        output[name]["value"][day] = -total_portfolio_value
        output[name]["twr"][day] = (
            output[name]["value"][day] / output[name]["position"][day]
        )
        output[name][ValueType.PRICE][day] = output[name]["twr"][day]

    def _build_rebalance_result(
        self: Self,
        output: dict[str, dict[str, list[float]]],
        instruments: list[str],
        subheaders: list[str | ValueType],
    ) -> DataFrame:
        """Build result DataFrame from output dictionary.

        Args:
            output: Output dictionary with all calculated values.
            instruments: List of instrument names.
            subheaders: List of subheader names.

        Returns:
            DataFrame with MultiIndex columns.
        """
        result = DataFrame()
        for outvalue in output.values():
            result = concat(
                [
                    result,
                    DataFrame(data=outvalue, index=self.tsdf.index),
                ],
                axis="columns",
            )
        lvlone, lvltwo = [], []
        for instr in instruments:
            lvlone.extend([instr] * 6)
            lvltwo.extend(subheaders)
        result.columns = MultiIndex.from_arrays([lvlone, lvltwo])
        return result

    def rebalanced_portfolio(
        self: Self,
        name: str,
        items: list[str] | None = None,
        bal_weights: list[float] | None = None,
        frequency: int = 1,
        cash_index: OpenTimeSeries | None = None,
        *,
        equal_weights: bool = False,
        drop_extras: bool = True,
    ) -> OpenFrame:
        """Create a rebalanced portfolio from the OpenFrame constituents.

        Args:
            name: Name of the portfolio.
            items: List of items to include in the portfolio. If None, uses all items.
                Optional.
            bal_weights: List of weights for rebalancing. If None, uses frame weights.
                Optional.
            frequency: Rebalancing frequency. Defaults to 1.
            cash_index: Cash index series for cash component. Optional.
            equal_weights: If True, use equal weights for all items. Defaults to False.
            drop_extras: If True, only return TWR series; if False, return all details.
                Defaults to True.

        Returns:
            OpenFrame containing the rebalanced portfolio.
        """
        items, bal_weights = self._validate_and_prepare_rebalance_inputs(
            items,
            bal_weights,
            equal_weights=equal_weights,
        )

        if cash_index:
            cash_index.tsdf = cash_index.tsdf.reindex(self.tsdf.index)
            cash_values: list[float] = cast(
                "list[float]", cash_index.tsdf.iloc[:, 0].to_numpy().tolist()
            )
        else:
            cash_values = [1.0] * self.length

        if self.tsdf.isna().to_numpy().any():
            self.value_nan_handle()

        ccies = list({serie.currency for serie in self.constituents})
        if len(ccies) != 1:
            msg = "Items for portfolio must be denominated in same currency."
            raise MultipleCurrenciesError(msg)
        currency = ccies[0]

        instruments = [*items, "cash", name]
        subheaders = [
            ValueType.PRICE,
            "buysell_qty",
            "position",
            "value",
            "twr",
            "settle",
        ]

        output = self._initialize_rebalance_output(
            items=items,
            name=name,
            cash_values=cash_values,
        )

        self._initialize_first_day_positions(
            items=items,
            bal_weights=bal_weights,
            output=output,
            name=name,
        )

        counter = 1
        for day in range(1, self.length):
            if day == frequency * counter:
                portfolio_value, settle_value = self._process_rebalancing_day(
                    day=day,
                    items=items,
                    bal_weights=bal_weights,
                    output=output,
                    name=name,
                )
                counter += 1
            else:
                portfolio_value = self._process_non_rebalancing_day(
                    day=day,
                    items=items,
                    output=output,
                )
                settle_value = 0.0

            self._update_cash_and_portfolio(
                day=day,
                portfolio_value=portfolio_value,
                settle_value=settle_value,
                output=output,
                name=name,
            )

        result = self._build_rebalance_result(
            output=output,
            instruments=instruments,
            subheaders=subheaders,
        )

        series = []
        if drop_extras:
            used_constituents = [
                item for item in self.constituents if item.label in items
            ]
            series.extend(
                [
                    OpenTimeSeries.from_df(
                        dframe=result[(item.label, "twr")],
                        valuetype=ValueType.PRICE,
                        baseccy=item.currency,
                        local_ccy=item.local_ccy,
                    )
                    for item in used_constituents
                ]
            )
            series.append(
                OpenTimeSeries.from_df(
                    dframe=result[(name, "twr")],
                    valuetype=ValueType.PRICE,
                    baseccy=currency,
                    local_ccy=True,
                ),
            )
        else:
            series.extend(
                [
                    OpenTimeSeries.from_df(
                        dframe=result.loc[:, col],
                        valuetype=ValueType.PRICE,
                        baseccy=currency,
                        local_ccy=True,
                    ).set_new_label(f"{col[0]}, {col[1]!s}")
                    for col in result.columns
                ]
            )

        return OpenFrame(series)
