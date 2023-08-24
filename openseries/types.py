"""Declaring types used throughout the project."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import StringConstraints, confloat, conint, conlist, constr

CountryStringType = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Z]{2}$",
        to_upper=True,
        min_length=2,
        max_length=2,
        strict=True,
    ),
]
CountryListType = conlist(
    constr(
        pattern=r"^[A-Z]{2}$",
        to_upper=True,
        min_length=2,
        max_length=2,
        strict=True,
    ),
    min_length=1,
)
CountriesType = Union[CountryListType, CountryStringType]  # type: ignore[valid-type]

CurrencyStringType = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Z]{3}$",
        to_upper=True,
        min_length=3,
        max_length=3,
        strict=True,
    ),
]

DateListType = Annotated[
    list[str],
    conlist(
        constr(pattern=r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"),
        min_length=2,
    ),
]

ValueListType = Annotated[list[float], conlist(float, min_length=2)]

DatabaseIdStringType = Annotated[str, StringConstraints(pattern=r"^([0-9a-f]{24})?$")]

DaysInYearType = Annotated[int, conint(strict=True, ge=1, le=366)]

TradingDaysType = Annotated[int, conint(strict=True, gt=1)]

SimCountType = Annotated[int, conint(strict=True, ge=1)]

VolatilityType = Annotated[float, confloat(strict=True, gt=0.0)]

LiteralLinePlotMode = Literal[
    "lines",
    "markers",
    "lines+markers",
    "lines+text",
    "markers+text",
    "lines+markers+text",
]
LiteralHowMerge = Literal["outer", "inner"]
LiteralQuantileInterp = Literal["linear", "lower", "higher", "midpoint", "nearest"]
LiteralBizDayFreq = Literal["BM", "BQ", "BA"]
LiteralPandasResampleConvention = Literal["start", "s", "end", "e"]
LiteralPandasReindexMethod = Literal[
    None,
    "pad",
    "ffill",
    "backfill",
    "bfill",
    "nearest",
]
LiteralNanMethod = Literal["fill", "drop"]
LiteralCaptureRatio = Literal["up", "down", "both"]
LiteralBarPlotMode = Literal["stack", "group", "overlay", "relative"]
LiteralPlotlyOutput = Literal["file", "div"]
LiteralOlsFitMethod = Literal["pinv", "qr"]
LiteralPortfolioWeightings = Literal["eq_weights", "eq_risk", "inv_vol", "mean_var"]
LiteralCovMethod = Literal["ledoit-wolf", "standard"]
LiteralRiskParityMethod = Literal["ccd", "slsqp"]
LiteralOlsFitCovType = Literal[
    "nonrobust",
    "fixed scale",
    "HC0",
    "HC1",
    "HC2",
    "HC3",
    "HAC",
    "hac-panel",
    "hac-groupsum",
    "cluster",
]

LiteralSeriesProps = Literal[
    "value_ret",
    "geo_ret",
    "arithmetic_ret",
    "vol",
    "downside_deviation",
    "ret_vol_ratio",
    "sortino_ratio",
    "z_score",
    "skew",
    "kurtosis",
    "positive_share",
    "var_down",
    "cvar_down",
    "vol_from_var",
    "worst",
    "worst_month",
    "max_drawdown_cal_year",
    "max_drawdown",
    "max_drawdown_date",
    "first_idx",
    "last_idx",
    "length",
    "span_of_days",
    "yearfrac",
    "periods_in_a_year",
]
LiteralFrameProps = Literal[
    "value_ret",
    "geo_ret",
    "arithmetic_ret",
    "vol",
    "downside_deviation",
    "ret_vol_ratio",
    "sortino_ratio",
    "z_score",
    "skew",
    "kurtosis",
    "positive_share",
    "var_down",
    "cvar_down",
    "vol_from_var",
    "worst",
    "worst_month",
    "max_drawdown",
    "max_drawdown_date",
    "max_drawdown_cal_year",
    "first_indices",
    "last_indices",
    "lengths_of_items",
    "span_of_days_all",
]


class OpenTimeSeriesPropertiesList(list[str]):
    """Allowed property arguments for the OpenTimeSeries class."""

    allowed_strings = {
        "value_ret",
        "geo_ret",
        "arithmetic_ret",
        "vol",
        "downside_deviation",
        "ret_vol_ratio",
        "sortino_ratio",
        "z_score",
        "skew",
        "kurtosis",
        "positive_share",
        "var_down",
        "cvar_down",
        "vol_from_var",
        "worst",
        "worst_month",
        "max_drawdown_cal_year",
        "max_drawdown",
        "max_drawdown_date",
        "first_idx",
        "last_idx",
        "length",
        "span_of_days",
        "yearfrac",
        "periods_in_a_year",
    }

    def __init__(self, *args: LiteralSeriesProps) -> None:
        super().__init__(args)
        self._validate()

    def _validate(self) -> None:
        seen = set()
        for item in self:
            if item not in self.allowed_strings:
                raise ValueError(
                    f"Invalid string: {item}. Allowed strings: {self.allowed_strings}",
                )
            if item in seen:
                raise ValueError(f"Duplicate string: {item}")
            seen.add(item)


class OpenFramePropertiesList(list[str]):
    """Allowed property arguments for the OpenFrame class."""

    allowed_strings = {
        "value_ret",
        "geo_ret",
        "arithmetic_ret",
        "vol",
        "downside_deviation",
        "ret_vol_ratio",
        "sortino_ratio",
        "z_score",
        "skew",
        "kurtosis",
        "positive_share",
        "var_down",
        "cvar_down",
        "vol_from_var",
        "worst",
        "worst_month",
        "max_drawdown",
        "max_drawdown_date",
        "max_drawdown_cal_year",
        "first_indices",
        "last_indices",
        "lengths_of_items",
        "span_of_days_all",
    }

    def __init__(self, *args: LiteralFrameProps) -> None:
        super().__init__(args)
        self._validate()

    def _validate(self) -> None:
        seen = set()
        for item in self:
            if item not in self.allowed_strings:
                raise ValueError(
                    f"Invalid string: {item}. Allowed strings: {self.allowed_strings}",
                )
            if item in seen:
                raise ValueError(f"Duplicate string: {item}")
            seen.add(item)


class ValueType(str, Enum):
    """Class defining the different timeseries types within the project."""

    EWMA = "EWMA"
    PRICE = "Price(Close)"
    RTRN = "Return(Total)"
    RELRTRN = "Relative return"
    ROLLBETA = "Beta"
    ROLLCORR = "Rolling correlation"
    ROLLCVAR = "Rolling CVaR"
    ROLLINFORATIO = "Information Ratio"
    ROLLRTRN = "Rolling returns"
    ROLLVAR = "Rolling VaR"
    ROLLVOL = "Rolling volatility"
