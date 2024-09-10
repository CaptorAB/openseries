"""Declaring types used throughout the project."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from pprint import pformat
from typing import Annotated, ClassVar, Literal, Union

from numpy import datetime64
from pandas import Timestamp
from pydantic import BaseModel, Field, StringConstraints, conlist, conset
from typing_extensions import Self

__all__ = ["ValueType"]


CountryStringType = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[A-Z]{2}$",
        to_upper=True,
        min_length=2,
        max_length=2,
        strict=True,
    ),
]
CountryListType = conset(
    item_type=CountryStringType,
    min_length=1,
)
CountriesType = Union[CountryListType, CountryStringType]  # type: ignore[valid-type]


class Countries(BaseModel):
    """Declare Countries."""

    countryinput: CountriesType


CurrencyStringType = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Z]{3}$",
        to_upper=True,
        min_length=3,
        max_length=3,
        strict=True,
        strip_whitespace=True,
    ),
]


class Currency(BaseModel):
    """Declare Currency."""

    ccy: CurrencyStringType


DateListType = Annotated[
    list[str],
    conset(
        Annotated[
            str,
            StringConstraints(
                pattern=r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$",
                strip_whitespace=True,
                strict=True,
                min_length=10,
                max_length=10,
            ),
        ],
        min_length=1,
    ),
]

ValueListType = Annotated[list[float], conlist(float, min_length=1)]

DatabaseIdStringType = Annotated[
    str,
    StringConstraints(
        pattern=r"^([0-9a-f]{24})?$",
        strict=True,
        strip_whitespace=True,
        max_length=24,
    ),
]

DaysInYearType = Annotated[int, Field(strict=True, ge=1, le=366)]

DateType = str | dt.date | dt.datetime | datetime64 | Timestamp

HolidayType = (
    dict[dt.date | dt.datetime | str | float | int, str]
    | list[dt.date | dt.datetime | str | float | int]
    | dt.date
    | dt.datetime
    | str
    | float
    | int
)

PlotlyLayoutType = dict[
    str,
    str
    | int
    | float
    | bool
    | list[str]
    | dict[str, str | int | float | bool | list[str]],
]

CaptorLogoType = dict[str, str | float]

LiteralJsonOutput = Literal["values", "tsdf"]
LiteralTrunc = Literal["before", "after", "both"]
LiteralLinePlotMode = Literal[
    "lines",
    "markers",
    "lines+markers",
    "lines+text",
    "markers+text",
    "lines+markers+text",
    None,
]
LiteralHowMerge = Literal["outer", "inner"]
LiteralQuantileInterp = Literal["linear", "lower", "higher", "midpoint", "nearest"]
LiteralBizDayFreq = Literal["B", "BME", "BQE", "BYE"]
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
LiteralPlotlyJSlib = Literal[True, False, "cdn"]
LiteralOlsFitMethod = Literal["pinv", "qr"]
LiteralPortfolioWeightings = Literal["eq_weights", "inv_vol"]
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

LiteralMinimizeMethods = Literal[
    "SLSQP",
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
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


class PropertiesList(list[str]):
    """Base class for allowed property arguments definition."""

    allowed_strings: ClassVar[set[str]] = {
        "value_ret",
        "geo_ret",
        "arithmetic_ret",
        "vol",
        "downside_deviation",
        "ret_vol_ratio",
        "sortino_ratio",
        "omega_ratio",
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
    }

    def _validate(self: Self) -> None:
        """Validate the string input of the all_properties method."""
        seen = set()
        invalids = set()
        duplicates = set()
        msg = ""
        for item in self:
            if item not in self.allowed_strings:
                invalids.add(item)
            if item in seen:
                duplicates.add(item)
            seen.add(item)
        if len(invalids) != 0:
            msg += (
                f"Invalid string(s): {list(invalids)}.\nAllowed strings are:"
                f"\n{pformat(self.allowed_strings)}\n"
            )
        if len(duplicates) != 0:
            msg += f"Duplicate string(s): {list(duplicates)}."
        if len(msg) != 0:
            raise ValueError(msg)


class OpenTimeSeriesPropertiesList(PropertiesList):
    """Allowed property arguments for the OpenTimeSeries class."""

    allowed_strings: ClassVar[set[str]] = PropertiesList.allowed_strings | {
        "first_idx",
        "last_idx",
        "length",
        "span_of_days",
        "yearfrac",
        "periods_in_a_year",
    }

    def __init__(
        self: Self,
        *args: LiteralSeriesProps,
    ) -> None:
        """Property arguments for the OpenTimeSeries class."""
        super().__init__(args)
        self._validate()


class OpenFramePropertiesList(PropertiesList):
    """Allowed property arguments for the OpenFrame class."""

    allowed_strings: ClassVar[set[str]] = PropertiesList.allowed_strings | {
        "first_indices",
        "last_indices",
        "lengths_of_items",
        "span_of_days_all",
    }

    def __init__(self: Self, *args: LiteralFrameProps) -> None:
        """Property arguments for the OpenFrame class."""
        super().__init__(args)
        self._validate()


class ValueType(str, Enum):
    """Enum types of OpenTimeSeries to identify the output."""

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
