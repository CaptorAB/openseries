"""
Defining the OpenTimeSeries class
"""
from copy import deepcopy
import datetime as dt
from enum import Enum
from json import dump
from math import ceil
from os import path
from pathlib import Path
from re import compile as re_compile
from typing import Any, cast, Dict, List, Optional, Tuple, TypeVar, Union
from numpy import (
    array,
    cumprod,
    dtype,
    insert,
    isnan,
    log,
    ndarray,
    sqrt,
    square,
)
from dateutil.relativedelta import relativedelta
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from pandas import (
    concat,
    DataFrame,
    DatetimeIndex,
    date_range,
    MultiIndex,
    Series,
)
from pandas.tseries.offsets import CustomBusinessDay
from plotly.graph_objs import Figure
from plotly.offline import plot
from pydantic import BaseModel, conlist, field_validator, model_validator
from scipy.stats import kurtosis, norm, skew
from stdnum import isin as isincode
from stdnum.exceptions import InvalidChecksum

from openseries.datefixer import date_offset_foll, date_fix, holiday_calendar
from openseries.load_plotly import load_plotly_dict
from openseries.types import (
    CountryStringType,
    CurrencyStringType,
    DatabaseIdStringType,
    DateListType,
    LiteralQuantileInterp,
    LiteralBizDayFreq,
    LiteralPandasResampleConvention,
    LiteralPandasReindexMethod,
    LiteralNanMethod,
    LiteralLinePlotMode,
    LiteralBarPlotMode,
    LiteralPlotlyOutput,
    LiteralSeriesProps,
    OpenTimeSeriesPropertiesList,
)
from openseries.risk import (
    cvar_down,
    var_down,
    drawdown_series,
    drawdown_details,
)


def check_if_none(item: Any) -> bool:
    """Function to check if a variable is None or equivalent

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


def ewma_calc(
    reeturn: float, prev_ewma: float, time_factor: float, lmbda: float = 0.94
) -> float:
    """Helper function for EWMA calculation

    Parameters
    ----------
    reeturn : float
        Return value
    prev_ewma : float
        Previous EWMA volatility value
    time_factor : float
        Scaling factor to annualize
    lmbda: float, default: 0.94
        Scaling factor to determine weighting.

    Returns
    -------
    float
        EWMA volatility value
    """
    return cast(
        float,
        sqrt(square(reeturn) * time_factor * (1 - lmbda) + square(prev_ewma) * lmbda),
    )


class ValueType(str, Enum):
    """Class defining the different timeseries types within the project"""

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


class OpenTimeSeries(
    BaseModel,
    arbitrary_types_allowed=True,
    validate_assignment=True,
    revalidate_instances="always",
    extra="allow",
):
    """Object of the class OpenTimeSeries. Subclass of the Pydantic BaseModel

    Parameters
    ----------
    timeseriesId : str
        Database identifier of the timeseries
    instrumentId: str
        Database identifier of the instrument associated with the timeseries
    name : str
        string identifier of the timeseries and/or instrument
    valuetype : ValueType
        Identifies if the series is a series of values or returns
    dates : List[str]
        Dates of the individual timeseries items
        These dates will not be altered by methods
    values : List[float]
        The value or return values of the timeseries items
        These values will not be altered by methods
    local_ccy: bool
        Boolean flag indicating if timeseries is in local currency
    tsdf: pandas.DataFrame
        Pandas object holding dates and values that can be altered via methods
    currency : str
        ISO 4217 currency code of the timeseries
    domestic : str, default: "SEK"
        ISO 4217 currency code of the user's home currency
    countries: str, default: "SE"
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
    values: conlist(float, min_length=2)
    local_ccy: bool
    tsdf: DataFrame
    currency: CurrencyStringType
    domestic: CurrencyStringType = "SEK"
    countries: CountryStringType = "SE"
    isin: Optional[str] = None
    label: Optional[str] = None

    @field_validator("isin")
    @classmethod
    def check_isincode(cls, isin_code: str) -> str:
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
    def check_dates_unique(self) -> "OpenTimeSeries":
        """Pydantic validator to ensure that the dates are unique"""
        dates_list_length = len(self.dates)
        dates_set_length = len(set(self.dates))
        if dates_list_length != dates_set_length:
            raise ValueError("Dates are not unique")
        return self

    @classmethod
    def setup_class(
        cls, domestic_ccy: str = "SEK", countries: List[str] | str = "SE"
    ) -> None:
        """Sets the domestic currency and calendar of the user.

        Parameters
        ----------
        domestic_ccy : str, default: "SEK"
            Currency code according to ISO 4217
        countries: List[str] | str, default: "SE"
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
        cls,
        name: str,
        dates: List[str],
        values: List[float],
        valuetype: ValueType = ValueType.PRICE,
        timeseries_id: str = "",
        instrument_id: str = "",
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ) -> "OpenTimeSeries":
        """Creates a timeseries from a Pandas DataFrame or Series

        Parameters
        ----------
        name: str
            string identifier of the timeseries and/or instrument
        dates: List[str]
            Array of date strings as ISO 8601
        values: List[float]
            Array of values
        valuetype : ValueType, default: ValueType.PRICE
            Identifies if the series is a series of values or returns
        timeseries_id : str
            Database identifier of the timeseries
        instrument_id: str
            Database identifier of the instrument associated with the timeseries
        baseccy : str, default: "SEK"
            The currency of the timeseries
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
            currency=baseccy,
            local_ccy=local_ccy,
            tsdf=DataFrame(
                data=values,
                index=[dejt.date() for dejt in DatetimeIndex(dates)],
                columns=[[name], [valuetype]],
                dtype="float64",
            ),
        )

    @classmethod
    def from_df(
        cls,
        dframe: DataFrame | Series,
        column_nmbr: int = 0,
        valuetype: ValueType = ValueType.PRICE,
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ) -> "OpenTimeSeries":
        """Creates a timeseries from a Pandas DataFrame or Series

        Parameters
        ----------
        dframe: DataFrame | Series
            Pandas DataFrame or Series
        column_nmbr : int, default: 0
            Using iloc[:, column_nmbr] to pick column
        valuetype : ValueType, default: ValueType.PRICE
            Identifies if the series is a series of values or returns
        baseccy : str, default: "SEK"
            The currency of the timeseries
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
                index=[dejt.date() for dejt in DatetimeIndex(dates)],
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
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ) -> "OpenTimeSeries":
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
        baseccy : str, default: "SEK"
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

    def from_deepcopy(self: "OpenTimeSeries") -> "OpenTimeSeries":
        """Creates a copy of an OpenTimeSeries object

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        return deepcopy(self)

    def to_xlsx(
        self: "OpenTimeSeries",
        filename: str,
        sheet_title: str | None = None,
        directory: str | None = None,
    ) -> str:
        """Saves the data in the .tsdf DataFrame to an Excel spreadsheet file

        Parameters
        ----------
        filename: str
            Filename that should include .xlsx
        sheet_title: str, optional
            Name of the sheet in the Excel file
        directory: str, optional
            The file directory where the Excel file is saved.
        Returns
        -------
        str
            The Excel file path
        """

        if filename[-5:].lower() != ".xlsx":
            raise NameError("Filename must end with .xlsx")
        if directory:
            sheetfile = path.join(directory, filename)
        else:
            script_path = path.abspath(__file__)
            sheetfile = path.join(path.dirname(script_path), filename)

        wrkbook = Workbook()
        wrksheet = wrkbook.active

        if sheet_title:
            wrksheet.title = sheet_title

        for row in dataframe_to_rows(df=self.tsdf, index=True, header=True):
            wrksheet.append(row)

        wrkbook.save(sheetfile)

        return sheetfile

    def to_json(
        self: "OpenTimeSeries", filename: str, directory: str | None = None
    ) -> Dict[str, str | bool | ValueType | List[str] | List[float]]:
        """Dumps timeseries data into a json file

        The label and tsdf parameters are deleted before the json file is saved

        Parameters
        ----------
        filename: str
            Filename including filetype
        directory: str, optional
            File folder location
        Returns
        -------
        dict
            A dictionary
        """
        if not directory:
            directory = path.dirname(path.abspath(__file__))

        data = self.__dict__

        cleaner_list = ["label", "tsdf"]
        for item in cleaner_list:
            data.pop(item)

        with open(path.join(directory, filename), "w", encoding="utf-8") as jsonfile:
            dump(data, jsonfile, indent=2, sort_keys=False)

        return data

    def pandas_df(self: "OpenTimeSeries") -> "OpenTimeSeries":
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
        self: "OpenTimeSeries",
        months_offset: int | None = None,
        from_dt: dt.date | None = None,
        to_dt: dt.date | None = None,
    ) -> Tuple[dt.date, dt.date]:
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
        earlier, later = self.first_idx, self.last_idx
        if months_offset is not None or from_dt is not None or to_dt is not None:
            if months_offset is not None:
                earlier = date_offset_foll(
                    raw_date=self.last_idx,
                    months_offset=-months_offset,
                    adjust=False,
                    following=True,
                )
                assert (
                    earlier >= self.first_idx
                ), "Function calc_range returned earlier date < series start"
                later = self.last_idx
            else:
                if from_dt is not None and to_dt is None:
                    assert (
                        from_dt >= self.first_idx
                    ), "Function calc_range returned earlier date < series start"
                    earlier, later = from_dt, self.last_idx
                elif from_dt is None and to_dt is not None:
                    assert (
                        to_dt <= self.last_idx
                    ), "Function calc_range returned later date > series end"
                    earlier, later = self.first_idx, to_dt
                elif from_dt is not None or to_dt is not None:
                    assert (
                        cast(dt.date, to_dt) <= self.last_idx
                        and cast(dt.date, from_dt) >= self.first_idx
                    ), "Function calc_range returned dates outside series range"
                    earlier, later = cast(dt.date, from_dt), cast(dt.date, to_dt)
            if earlier is not None:
                while earlier not in self.tsdf.index.tolist():
                    earlier -= dt.timedelta(days=1)
            if later is not None:
                while later not in self.tsdf.index.tolist():
                    later += dt.timedelta(days=1)

        return earlier, later

    def align_index_to_local_cdays(self: "OpenTimeSeries") -> "OpenTimeSeries":
        """Changes the index of the associated Pandas DataFrame .tsdf to align with
        local calendar business days

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        startyear = self.first_idx.year
        endyear = self.last_idx.year
        calendar = holiday_calendar(
            startyear=startyear, endyear=endyear, countries=self.countries
        )
        d_range = [
            d.date()
            for d in date_range(
                start=self.tsdf.first_valid_index(),
                end=self.tsdf.last_valid_index(),
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]

        self.tsdf = self.tsdf.reindex(d_range, method=None, copy=False)

        return self

    def all_properties(
        self: "OpenTimeSeries", properties: List[LiteralSeriesProps] | None = None
    ) -> DataFrame:
        """Calculates the chosen timeseries properties

        Parameters
        ----------
        properties: List[LiteralSeriesProps], optional
            The properties to calculate. Defaults to calculating all available.

        Returns
        -------
        pandas.DataFrame
            Properties of the OpenTimeSeries
        """

        if not properties:
            properties = cast(
                List[LiteralSeriesProps], OpenTimeSeriesPropertiesList.allowed_strings
            )

        props = OpenTimeSeriesPropertiesList(*properties)
        pdf = DataFrame.from_dict({x: getattr(self, x) for x in props}, orient="index")
        pdf.columns = self.tsdf.columns
        return pdf

    @property
    def length(self: "OpenTimeSeries") -> int:
        """
        Returns
        -------
        int
            Number of observations
        """

        return len(self.tsdf.index)

    @property
    def first_idx(self: "OpenTimeSeries") -> dt.date:
        """
        Returns
        -------
        datetime.date
            The first date in the timeseries
        """

        return cast(dt.date, self.tsdf.index[0])

    @property
    def last_idx(self: "OpenTimeSeries") -> dt.date:
        """
        Returns
        -------
        datetime.date
            The last date in the timeseries
        """

        return cast(dt.date, self.tsdf.index[-1])

    @property
    def span_of_days(self: "OpenTimeSeries") -> int:
        """
        Returns
        -------
        int
            Number of days from the first date to the last
        """

        return (self.last_idx - self.first_idx).days

    @property
    def yearfrac(self: "OpenTimeSeries") -> float:
        """
        Returns
        -------
        float
            Length of the timeseries expressed in years assuming all years
            have 365.25 days
        """

        return self.span_of_days / 365.25

    @property
    def periods_in_a_year(self: "OpenTimeSeries") -> float:
        """
        Returns
        -------
        float
            The average number of observations per year
        """

        return self.length / self.yearfrac

    @property
    def geo_ret(self: "OpenTimeSeries") -> float:
        """https://www.investopedia.com/terms/c/cagr.asp

        Returns
        -------
        float
            Compounded Annual Growth Rate (CAGR)
        """

        if (
            self.tsdf.loc[self.first_idx, self.tsdf.columns.values[0]] == 0.0
            or self.tsdf.lt(0.0).values.any()
        ):
            raise ValueError(
                "Geometric return cannot be calculated due to an initial "
                "value being zero or a negative value."
            )
        return cast(
            float,
            (
                self.tsdf.loc[self.last_idx, self.tsdf.columns.values[0]]
                / self.tsdf.loc[self.first_idx, self.tsdf.columns.values[0]]
            )
            ** (1 / self.yearfrac)
            - 1,
        )

    def geo_ret_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> float:
        """https://www.investopedia.com/terms/c/cagr.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Compounded Annual Growth Rate (CAGR)
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        fraction = (later - earlier).days / 365.25

        if (
            self.tsdf.loc[earlier, self.tsdf.columns.values[0]] == 0.0
            or self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .lt(0.0)
            .values.any()
        ):
            raise ValueError(
                "Geometric return cannot be calculated due to an initial "
                "value being zero or a negative value."
            )

        return cast(
            float,
            (
                self.tsdf.loc[later, self.tsdf.columns.values[0]]
                / self.tsdf.loc[earlier, self.tsdf.columns.values[0]]
            )
            ** (1 / fraction)
            - 1,
        )

    @property
    def arithmetic_ret(self: "OpenTimeSeries") -> float:
        """https://www.investopedia.com/terms/a/arithmeticmean.asp

        Returns
        -------
        float
            Annualized arithmetic mean of returns
        """

        return float((self.tsdf.pct_change().mean() * self.periods_in_a_year).iloc[0])

    def arithmetic_ret_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: int | None = None,
    ) -> float:
        """https://www.investopedia.com/terms/a/arithmeticmean.asp

        Parameters
        ----------
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
        float
            Annualized arithmetic mean of returns
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[
                cast(int, earlier) : cast(int, later), self.tsdf.columns.values[0]
            ].count()
            time_factor = how_many / fraction
        return cast(
            float,
            (
                self.tsdf.loc[cast(int, earlier) : cast(int, later)]
                .pct_change()
                .mean()
                * time_factor
            ).iloc[0],
        )

    @property
    def value_ret(self: "OpenTimeSeries") -> float:
        """
        Returns
        -------
        float
            Simple return
        """

        if self.tsdf.iloc[0, 0] == 0.0:
            raise ValueError(
                "Simple Return cannot be calculated due to an initial value being "
                "zero."
            )
        return float((self.tsdf.iloc[-1] / self.tsdf.iloc[0] - 1).iloc[0])

    def value_ret_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> float:
        """
        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Simple return
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if self.tsdf.loc[earlier, self.tsdf.columns.values[0]] == 0.0:
            raise ValueError(
                "Simple Return cannot be calculated due to an initial value being "
                "zero."
            )
        return float((self.tsdf.loc[later] / self.tsdf.loc[earlier] - 1).iloc[0])

    def value_ret_calendar_period(
        self: "OpenTimeSeries", year: int, month: int | None = None
    ) -> float:
        """
        Parameters
        ----------
        year : int
            Calendar year of the period to calculate.
        month : int, optional
            Calendar month of the period to calculate.

        Returns
        -------
        float
            Simple return for a specific calendar period
        """

        caldf = self.tsdf.copy()
        caldf.index = DatetimeIndex(caldf.index)
        if month is None:
            period = str(year)
        else:
            period = "-".join([str(year), str(month).zfill(2)])
        rtn = caldf.copy().pct_change()
        rtn = rtn.loc[period] + 1
        return float((rtn.apply(cumprod, axis="index").iloc[-1] - 1).iloc[0])

    @property
    def vol(self: "OpenTimeSeries") -> float:
        """Based on Pandas .std() which is the equivalent of stdev.s([...])
        in MS Excel \n
        https://www.investopedia.com/terms/v/volatility.asp

        Returns
        -------
        float
            Annualized volatility
        """

        return cast(
            float,
            (self.tsdf.pct_change().std() * sqrt(self.periods_in_a_year)).iloc[0],
        )

    def vol_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: int | None = None,
    ) -> float:
        """Based on Pandas .std() which is the equivalent of stdev.s([...])
        in MS Excel \n
        https://www.investopedia.com/terms/v/volatility.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases
            and comparisons

        Returns
        -------
        float
            Annualized volatility
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[
                cast(int, earlier) : cast(int, later), self.tsdf.columns.values[0]
            ].count()
            time_factor = how_many / fraction
        return cast(
            float,
            (
                self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change().std()
                * sqrt(time_factor)
            ).iloc[0],
        )

    @property
    def downside_deviation(self: "OpenTimeSeries") -> float:
        """The standard deviation of returns that are below a Minimum Accepted
        Return of zero.
        It is used to calculate the Sortino Ratio \n
        https://www.investopedia.com/terms/d/downside-deviation.asp

        Returns
        -------
        float
            Downside deviation
        """

        dddf = self.tsdf.pct_change()

        return cast(
            float,
            sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / self.length)
            * sqrt(self.periods_in_a_year),
        )

    def downside_deviation_func(
        self: "OpenTimeSeries",
        min_accepted_return: float = 0.0,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: int | None = None,
    ) -> float:
        """The standard deviation of returns that are below a Minimum Accepted
        Return of zero.
        It is used to calculate the Sortino Ratio \n
        https://www.investopedia.com/terms/d/downside-deviation.asp

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
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        float
            Downside deviation
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        how_many = (
            self.tsdf.loc[
                cast(int, earlier) : cast(int, later), self.tsdf.columns.values[0]
            ]
            .pct_change()
            .count()
        )
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            fraction = (later - earlier).days / 365.25
            time_factor = how_many / fraction

        dddf = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()
            .sub(min_accepted_return / time_factor)
        )

        return cast(
            float,
            sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / how_many)
            * sqrt(time_factor),
        )

    @property
    def ret_vol_ratio(self: "OpenTimeSeries") -> float:
        """
        Returns
        -------
        float
            Ratio of the annualized arithmetic mean of returns and annualized
            volatility.
        """
        return self.arithmetic_ret / self.vol

    def ret_vol_ratio_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        riskfree_rate: float = 0.0,
    ) -> float:
        """The ratio of annualized arithmetic mean of returns and annualized
        volatility or, if risk-free return provided, Sharpe ratio calculated
        as ( geometric return - risk-free return ) / volatility. The latter ratio
        implies that the riskfree asset has zero volatility. \n
        https://www.investopedia.com/terms/s/sharperatio.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        riskfree_rate : float, optional
            The return of the zero volatility asset used to calculate Sharpe ratio

        Returns
        -------
        float
            Ratio of the annualized arithmetic mean of returns and annualized
            volatility or,
            if risk-free return provided, Sharpe ratio
        """

        return (
            self.arithmetic_ret_func(months_from_last, from_date, to_date)
            - riskfree_rate
        ) / self.vol_func(months_from_last, from_date, to_date)

    @property
    def sortino_ratio(self: "OpenTimeSeries") -> float:
        """https://www.investopedia.com/terms/s/sortinoratio.asp

        Returns
        -------
        float
        Pandas.Series
            Sortino ratio calculated as the annualized arithmetic mean of returns
            / downside deviation. The ratio implies that the riskfree asset has zero
            volatility, and a minimum acceptable return of zero.
        """

        return self.arithmetic_ret / self.downside_deviation

    def sortino_ratio_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        riskfree_rate: float = 0.0,
    ) -> float:
        """The Sortino ratio calculated as ( asset return - risk free return )
        / downside deviation. The ratio implies that the riskfree asset has
        zero volatility. \n
        https://www.investopedia.com/terms/s/sortinoratio.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        riskfree_rate : float, optional
            The return of the zero volatility asset

        Returns
        -------
        float
        Pandas.Series
            Sortino ratio calculated as the annualized arithmetic mean of returns
            / downside deviation.
        """

        return (
            self.arithmetic_ret_func(months_from_last, from_date, to_date)
            - riskfree_rate
        ) / self.downside_deviation_func(
            min_accepted_return=0.0,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
        )

    @property
    def z_score(self: "OpenTimeSeries") -> float:
        """https://www.investopedia.com/terms/z/zscore.asp

        Returns
        -------
        float
            Z-score as (last return - mean return) / standard deviation of returns.
        """

        return cast(
            float,
            (
                (self.tsdf.pct_change().iloc[-1] - self.tsdf.pct_change().mean())
                / self.tsdf.pct_change().std()
            ).iloc[0],
        )

    def z_score_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> float:
        """https://www.investopedia.com/terms/z/zscore.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Z-score as (last return - mean return) / standard deviation of returns
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        part = self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change().copy()
        return float(((part.iloc[-1] - part.mean()) / part.std()).iloc[0])

    @property
    def max_drawdown(self: "OpenTimeSeries") -> float:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        float
            Maximum drawdown without any limit on date range
        """

        return cast(
            float,
            ((self.tsdf / self.tsdf.expanding(min_periods=1).max()).min() - 1).iloc[0],
        )

    @property
    def max_drawdown_date(self: "OpenTimeSeries") -> dt.date:
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

    def max_drawdown_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> float:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Maximum drawdown without any limit on date range
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return cast(
            float,
            (
                (
                    self.tsdf.loc[cast(int, earlier) : cast(int, later)]
                    / self.tsdf.loc[cast(int, earlier) : cast(int, later)]
                    .expanding(min_periods=1)
                    .max()
                ).min()
                - 1
            ).iloc[0],
        )

    @property
    def max_drawdown_cal_year(self: "OpenTimeSeries") -> float:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        float
            Maximum drawdown in a single calendar year.
        """
        years = [d.year for d in self.tsdf.index]
        return cast(
            float,
            (
                self.tsdf.groupby(years)
                .apply(lambda x: (x / x.expanding(min_periods=1).max()).min() - 1)
                .min()
            ).iloc[0],
        )

    @property
    def worst(self: "OpenTimeSeries") -> float:
        """
        Returns
        -------
        float
            Most negative percentage change
        """

        return float((self.tsdf.pct_change().min()).iloc[0])

    @property
    def worst_month(self: "OpenTimeSeries") -> float:
        """
        Returns
        -------
        float
            Most negative month
        """

        resdf = self.tsdf.copy()
        resdf.index = DatetimeIndex(resdf.index)
        return float((resdf.resample("BM").last().pct_change().min()).iloc[0])

    def worst_func(
        self: "OpenTimeSeries",
        observations: int = 1,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> float:
        """
        Parameters
        ----------
        observations: int, default: 1
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Most negative percentage change over a rolling number of observations
            within
            a chosen date range
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return cast(
            float,
            (
                self.tsdf.loc[cast(int, earlier) : cast(int, later)]
                .pct_change()
                .rolling(observations, min_periods=observations)
                .sum()
                .min()
            ).iloc[0],
        )

    @property
    def positive_share(self: "OpenTimeSeries") -> float:
        """
        Returns
        -------
        float
            The share of percentage changes that are greater than zero
        """
        pos = self.tsdf.pct_change()[1:][
            self.tsdf.pct_change()[1:].values >= 0.0
        ].count()
        tot = self.tsdf.pct_change()[1:].count()
        return float((pos / tot).iloc[0])

    def positive_share_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> float:
        """
        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            The share of percentage changes that are greater than zero
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        period = self.tsdf.loc[cast(int, earlier) : cast(int, later)].copy()
        return cast(
            float,
            (
                period[period.pct_change().ge(0.0)].count()
                / period.pct_change().count()
            ).iloc[0],
        )

    @property
    def skew(self: "OpenTimeSeries") -> float:
        """https://www.investopedia.com/terms/s/skewness.asp

        Returns
        -------
        float
            Skew of the return distribution
        """

        return cast(
            float,
            skew(a=self.tsdf.pct_change().values, bias=True, nan_policy="omit")[0],
        )

    def skew_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> float:
        """https://www.investopedia.com/terms/s/skewness.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Skew of the return distribution
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return cast(
            float,
            skew(
                self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change(),
                bias=True,
                nan_policy="omit",
            )[0],
        )

    @property
    def kurtosis(self: "OpenTimeSeries") -> float:
        """https://www.investopedia.com/terms/k/kurtosis.asp

        Returns
        -------
        float
            Kurtosis of the return distribution
        """
        return cast(
            float,
            kurtosis(
                self.tsdf.pct_change(),
                fisher=True,
                bias=True,
                nan_policy="omit",
            )[0],
        )

    def kurtosis_func(
        self: "OpenTimeSeries",
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> float:
        """https://www.investopedia.com/terms/k/kurtosis.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Kurtosis of the return distribution
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)

        return cast(
            float,
            kurtosis(
                self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change(),
                fisher=True,
                bias=True,
                nan_policy="omit",
            )[0],
        )

    @property
    def cvar_down(self: "OpenTimeSeries") -> float:
        """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

        Returns
        -------
        float
            Downside Conditional 95% Value At Risk "CVaR"
        """
        level: float = 0.95
        items = self.tsdf.iloc[:, 0].pct_change().count()
        return cast(
            float,
            (
                self.tsdf.iloc[:, 0]
                .pct_change()
                .sort_values()
                .iloc[: int(ceil((1 - level) * items))]
                .mean()
            ),
        )

    def cvar_down_func(
        self: "OpenTimeSeries",
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
    ) -> float:
        """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

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

        Returns
        -------
        float
            Downside Conditional Value At Risk "CVaR"
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        how_many = (
            self.tsdf.loc[
                cast(int, earlier) : cast(int, later), self.tsdf.columns.values[0]
            ]
            .pct_change()
            .count()
        )
        return cast(
            float,
            (
                self.tsdf.loc[
                    cast(int, earlier) : cast(int, later), self.tsdf.columns.values[0]
                ]
                .pct_change()
                .sort_values()
                .iloc[: int(ceil((1 - level) * how_many))]
                .mean()
            ),
        )

    @property
    def var_down(self: "OpenTimeSeries") -> float:
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
        return cast(
            float,
            (
                self.tsdf.pct_change()
                .quantile(1 - level, interpolation=interpolation)
                .iloc[0]
            ),
        )

    def var_down_func(
        self: "OpenTimeSeries",
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        interpolation: LiteralQuantileInterp = "lower",
    ) -> float:
        """https://www.investopedia.com/terms/v/var.asp
        Downside Value At Risk, "VaR". The equivalent of
        percentile.inc([...], 1-level) over returns in MS Excel.

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

        Returns
        -------
        float
            Downside Value At Risk
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return cast(
            float,
            (
                self.tsdf.loc[cast(int, earlier) : cast(int, later)]
                .pct_change()
                .quantile(q=1 - level, interpolation=interpolation)
            ).iloc[0],
        )

    @property
    def vol_from_var(self: "OpenTimeSeries") -> float:
        """
        Returns
        -------
        float
            Implied annualized volatility from the Downside 95% VaR using the
            assumption that returns are normally distributed.
        """
        level: float = 0.95
        interpolation: LiteralQuantileInterp = "lower"
        return cast(
            float,
            (
                -sqrt(self.periods_in_a_year)
                * self.var_down_func(level, interpolation=interpolation)
                / norm.ppf(level)
            ),
        )

    def vol_from_var_func(
        self: "OpenTimeSeries",
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        interpolation: LiteralQuantileInterp = "lower",
        drift_adjust: bool = False,
        periods_in_a_year_fixed: int | None = None,
    ) -> float:
        """
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
        drift_adjust: bool, default: False
            An adjustment to remove the bias implied by the average return
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        float
            Implied annualized volatility from the Downside VaR using the
            assumption that returns are normally distributed.
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[
                cast(int, earlier) : cast(int, later), self.tsdf.columns.values[0]
            ].count()
            time_factor = how_many / fraction
        if drift_adjust:
            return cast(
                float,
                (
                    (-sqrt(time_factor) / norm.ppf(level))
                    * (
                        self.var_down_func(
                            level,
                            months_from_last,
                            from_date,
                            to_date,
                            interpolation,
                        )
                        - self.tsdf.loc[cast(int, earlier) : cast(int, later)]
                        .pct_change()
                        .sum()
                        / len(
                            self.tsdf.loc[
                                cast(int, earlier) : cast(int, later)
                            ].pct_change()
                        )
                    )
                ).iloc[0],
            )

        return cast(
            float,
            (
                -sqrt(time_factor)
                * self.var_down_func(
                    level, months_from_last, from_date, to_date, interpolation
                )
                / norm.ppf(level)
            ),
        )

    def target_weight_from_var(
        self: "OpenTimeSeries",
        target_vol: float = 0.175,
        min_leverage_local: float = 0.0,
        max_leverage_local: float = 99999.0,
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        interpolation: LiteralQuantileInterp = "lower",
        drift_adjust: bool = False,
        periods_in_a_year_fixed: int | None = None,
    ) -> float:
        """A position weight multiplier from the ratio between a VaR implied
        volatility and a given target volatility. Multiplier = 1.0 -> target met

        Parameters
        ----------
        target_vol: float, default: 0.175
            Target Volatility
        min_leverage_local: float, default: 0.0
            A minimum adjustment factor
        max_leverage_local: float, default: 99999.0
            A maximum adjustment factor
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
        drift_adjust: bool, default: False
            An adjustment to remove the bias implied by the average return
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        float
            A position weight multiplier from the ratio between a VaR implied
            volatility and a given target volatility. Multiplier = 1.0 -> target met
        """

        return max(
            min_leverage_local,
            min(
                target_vol
                / self.vol_from_var_func(
                    level=level,
                    months_from_last=months_from_last,
                    from_date=from_date,
                    to_date=to_date,
                    interpolation=interpolation,
                    drift_adjust=drift_adjust,
                    periods_in_a_year_fixed=periods_in_a_year_fixed,
                ),
                max_leverage_local,
            ),
        )

    def value_to_ret(self: "OpenTimeSeries") -> "OpenTimeSeries":
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

    def value_to_diff(self: "OpenTimeSeries", periods: int = 1) -> "OpenTimeSeries":
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

    def value_to_log(self: "OpenTimeSeries") -> "OpenTimeSeries":
        """Converts a valueseries into logarithmic return series \n
        Equivalent to LN(value[t] / value[t=0]) in MS Excel

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.tsdf = log(self.tsdf / self.tsdf.iloc[0])
        return self

    def to_cumret(self: "OpenTimeSeries") -> "OpenTimeSeries":
        """Converts a returnseries into a cumulative valueseries

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        if not any(
            x == ValueType.RTRN for x in self.tsdf.columns.get_level_values(1).values
        ):
            self.value_to_ret()

        self.tsdf = self.tsdf.add(1.0)
        self.tsdf = self.tsdf.cumprod(axis=0) / self.tsdf.iloc[0]
        self.valuetype = ValueType.PRICE
        self.tsdf.columns = [[self.label], [self.valuetype]]

        return self

    def from_1d_rate_to_cumret(
        self: "OpenTimeSeries", days_in_year: int = 365, divider: float = 1.0
    ) -> "OpenTimeSeries":
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
        self: "OpenTimeSeries", freq: Union[LiteralBizDayFreq, str] = "BM"
    ) -> "OpenTimeSeries":
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
        self: "OpenTimeSeries",
        freq: LiteralBizDayFreq = "BM",
        convention: LiteralPandasResampleConvention = "end",
        method: LiteralPandasReindexMethod = "nearest",
    ) -> "OpenTimeSeries":
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
        head = head.to_frame().T
        tail = self.tsdf.iloc[-1].copy()
        tail = tail.to_frame().T
        self.tsdf.index = DatetimeIndex(self.tsdf.index)
        self.tsdf = self.tsdf.resample(rule=freq, convention=convention).last()
        self.tsdf.drop(index=self.tsdf.index[-1], inplace=True)
        self.tsdf.index = [d.date() for d in DatetimeIndex(self.tsdf.index)]

        if head.index[0] not in self.tsdf.index:
            self.tsdf = concat([self.tsdf, head])

        if tail.index[0] not in self.tsdf.index:
            self.tsdf = concat([self.tsdf, tail])

        self.tsdf.sort_index(inplace=True)

        dates = DatetimeIndex(
            [self.tsdf.index[0]]
            + [
                date_offset_foll(
                    dt.date(d.year, d.month, 1)
                    + relativedelta(months=1)
                    - dt.timedelta(days=1),
                    countries=self.countries,
                    months_offset=0,
                    adjust=True,
                    following=False,
                )
                for d in self.tsdf.index[1:-1]
            ]
            + [self.tsdf.index[-1]]
        )
        dates = dates.drop_duplicates()
        self.tsdf = self.tsdf.reindex([d.date() for d in dates], method=method)
        return self

    def to_drawdown_series(self: "OpenTimeSeries") -> "OpenTimeSeries":
        """Converts the timeseries into a drawdown series

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.tsdf = drawdown_series(self.tsdf)
        self.tsdf.columns = [[self.label], ["Drawdowns"]]

        return self

    def drawdown_details(self: "OpenTimeSeries") -> DataFrame:
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
        self: "OpenTimeSeries",
        lmbda: float = 0.94,
        day_chunk: int = 11,
        dlta_degr_freedms: int = 0,
        months_from_last: int | None = None,
        from_date: dt.date | None = None,
        to_date: dt.date | None = None,
        periods_in_a_year_fixed: int | None = None,
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
        self: "OpenTimeSeries",
        observations: int = 21,
        periods_in_a_year_fixed: int | None = None,
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

    def rolling_return(self: "OpenTimeSeries", observations: int = 21) -> DataFrame:
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
        self: "OpenTimeSeries", level: float = 0.95, observations: int = 252
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
            lambda x: cvar_down(x, level=level)
        )
        cvardf = cvardf.dropna()
        cvardf.columns = [[self.label], ["Rolling CVaR"]]

        return cvardf

    def rolling_var_down(
        self: "OpenTimeSeries",
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
            lambda x: var_down(x, level=level, interpolation=interpolation)
        )
        vardf = vardf.dropna()
        vardf.columns = [[self.label], ["Rolling VaR"]]

        return vardf

    def value_nan_handle(
        self: "OpenTimeSeries", method: LiteralNanMethod = "fill"
    ) -> "OpenTimeSeries":
        """Handling of missing values in a valueseries

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with last known or drop

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        assert method in [
            "fill",
            "drop",
        ], "Method must be either fill or drop passed as string."
        if method == "fill":
            self.tsdf.fillna(method="ffill", inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def return_nan_handle(
        self: "OpenTimeSeries", method: LiteralNanMethod = "fill"
    ) -> "OpenTimeSeries":
        """Handling of missing values in a returnseries

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with zero or drop

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        assert method in [
            "fill",
            "drop",
        ], "Method must be either fill or drop passed as string."
        if method == "fill":
            self.tsdf.fillna(value=0.0, inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def running_adjustment(
        self: "OpenTimeSeries", adjustment: float, days_in_year: int = 365
    ) -> "OpenTimeSeries":
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
        values: List[float]
        if any(
            x == ValueType.RTRN for x in self.tsdf.columns.get_level_values(1).values
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
        dates: List[dt.date] = [prev]

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
        self: "OpenTimeSeries",
        lvl_zero: str | None = None,
        lvl_one: ValueType | None = None,
        delete_lvl_one: bool = False,
    ) -> "OpenTimeSeries":
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

    def plot_series(
        self: "OpenTimeSeries",
        mode: LiteralLinePlotMode = "lines",
        tick_fmt: str | None = None,
        directory: str | None = None,
        auto_open: bool = True,
        add_logo: bool = True,
        show_last: bool = False,
        output_type: LiteralPlotlyOutput = "file",
    ) -> Tuple[Figure, str]:
        """Creates a Plotly Figure

        Parameters
        ----------
        mode: LiteralLinePlotMode, default: "lines"
            The type of scatter to use
        tick_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show
        directory: str, optional
            Directory where Plotly html file is saved
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot
        show_last: bool, default: False
            If True the last data point is highlighted as red dot with a label
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type

        Returns
        -------
        (plotly.go.Figure, str)
            Plotly Figure and html filename with location
        """

        if not directory:
            directory = path.join(str(Path.home()), "Documents")
        filename = (
            cast(str, self.label)
            .replace("/", "")
            .replace("#", "")
            .replace(" ", "")
            .upper()
        )
        plotfile = path.join(path.abspath(directory), f"{filename}.html")

        fig, logo = load_plotly_dict()
        figure = Figure(fig)
        figure.add_scatter(
            x=self.tsdf.index,
            y=self.tsdf.iloc[:, 0],
            mode=mode,
            line={"width": 2.5, "dash": "solid"},
            hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
            showlegend=True,
            name=self.label,
        )
        figure.update_layout(yaxis={"tickformat": tick_fmt})

        if add_logo:
            figure.add_layout_image(logo)

        if show_last is True:
            if tick_fmt:
                txt = f"Last {{:{tick_fmt}}}"
            else:
                txt = "Last {}"

            figure.add_scatter(
                x=[self.last_idx],
                y=[self.tsdf.iloc[-1, 0]],
                mode="markers + text",
                marker={"color": "red", "size": 12},
                hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                showlegend=False,
                name=self.label,
                text=[txt.format(self.tsdf.iloc[-1, 0])],
                textposition="top center",
            )

        plot(
            figure,
            filename=plotfile,
            auto_open=auto_open,
            link_text="",
            include_plotlyjs="cdn",
            config=fig["config"],
            output_type=output_type,
        )

        return figure, plotfile

    def plot_bars(
        self: "OpenTimeSeries",
        mode: LiteralBarPlotMode = "group",
        tick_fmt: str | None = None,
        directory: str | None = None,
        auto_open: bool = True,
        add_logo: bool = True,
        output_type: LiteralPlotlyOutput = "file",
    ) -> Tuple[Figure, str]:
        """Creates a Plotly Bar Figure

        Parameters
        ----------
        mode: LiteralBarPlotMode, default: "group"
            The type of bar to use
        tick_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show
        directory: str, optional
            Directory where Plotly html file is saved
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type

        Returns
        -------
        (plotly.go.Figure, str)
            Plotly Figure and html filename with location
        """
        if not directory:
            directory = path.join(str(Path.home()), "Documents")
        filename = (
            cast(str, self.label)
            .replace("/", "")
            .replace("#", "")
            .replace(" ", "")
            .upper()
        )
        plotfile = path.join(path.abspath(directory), f"{filename}.html")

        fig, logo = load_plotly_dict()
        figure = Figure(fig)
        figure.add_bar(
            x=self.tsdf.index,
            y=self.tsdf.iloc[:, 0],
            hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
            name=self.label,
        )
        figure.update_layout(barmode=mode, yaxis={"tickformat": tick_fmt})

        if add_logo:
            figure.add_layout_image(logo)

        plot(
            figure,
            filename=plotfile,
            auto_open=auto_open,
            link_text="",
            include_plotlyjs="cdn",
            config=fig["config"],
            output_type=output_type,
        )

        return figure, plotfile


TypeOpenTimeSeries = TypeVar("TypeOpenTimeSeries", bound=OpenTimeSeries)


def timeseries_chain(
    front: Union[TypeOpenTimeSeries, type(OpenTimeSeries)],
    back: Union[TypeOpenTimeSeries, type(OpenTimeSeries)],
    old_fee: float = 0.0,
) -> Union[TypeOpenTimeSeries, OpenTimeSeries]:
    """Chain two timeseries together

    Parameters
    ----------
    front: Union[TypeOpenTimeSeries, type(OpenTimeSeries)]
        Earlier series to chain with
    back: Union[TypeOpenTimeSeries, type(OpenTimeSeries)]
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

    dates: List[str] = [x.strftime("%Y-%m-%d") for x in olddf.index if x < first]
    values = array([x[0] for x in old.tsdf.values][: len(dates)])
    values = cast(
        ndarray[Any, dtype[Any]],
        list(values * new.tsdf.iloc[:, 0].loc[first] / olddf.iloc[:, 0].loc[first]),
    )

    dates.extend([x.strftime("%Y-%m-%d") for x in new.tsdf.index])
    values += [x[0] for x in new.tsdf.values]

    if back.__class__.__subclasscheck__(OpenTimeSeries):
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
