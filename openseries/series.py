from copy import deepcopy
from datetime import date, datetime, timedelta
from json import dump, load
from jsonschema import Draft7Validator
from math import ceil
from numpy import array, busdaycalendar, cumprod, insert, log, sqrt, square, zeros
from os import path
from pandas import DataFrame, DatetimeIndex, date_range, MultiIndex, Series
from pandas.tseries.offsets import CustomBusinessDay
from pathlib import Path
from plotly.graph_objs import Figure, Scatter
from plotly.offline import plot
from stdnum import isin as isincode
from scipy.stats import kurtosis, norm, skew
from typing import List, Literal, TypedDict

from openseries.datefixer import date_offset_foll, date_fix, holiday_calendar
from openseries.load_plotly import load_plotly_dict
from openseries.risk import (
    cvar_down,
    var_down,
    drawdown_series,
    drawdown_details,
)


class TimeSerie(TypedDict, total=False):
    """Class to hold the type of input data for the OpenTimeSeries class.

    Parameters
    ----------
    _id : str
        Database identifier of the timeseries
    instrumentId: str
        Database identifier of the instrument associated with the timeseries
    currency : str
        ISO 4217 currency code of the timeseries
    dates : List[str]
        Dates of the individual timeseries items
        These dates will not be altered by methods
    domestic : str
        ISO 4217 currency code of the user's home currency
    name : str
        string identifier of the timeseries and/or instrument
    isin : str
        ISO 6166 identifier code of the associated instrument
    label : str
        Placeholder for a name of the timeseries
    calendar : numpy.busdaycalendar
        Placeholder for a business calendar
    valuetype : str
        Identifies if the series is a series of values or returns
    values : List[float]
        The value or return values of the timeseries items
        These values will not be altered by methods
    local_ccy: bool
        Boolean flag indicating if timeseries is in local currency
    tsdf: pandas.DataFrame
        Pandas object holding dates and values that can be altered via methods
    """

    _id: str
    instrumentId: str
    currency: str
    dates: List[str]
    domestic: str
    name: str
    isin: str
    label: str
    calendar: busdaycalendar
    valuetype: str
    values: List[float]
    local_ccy: bool
    tsdf: DataFrame


class OpenTimeSeries(object):
    _id: str
    instrumentId: str
    currency: str
    dates: List[str]
    domestic: str
    name: str
    isin: str
    label: str
    calendar: busdaycalendar
    valuetype: str
    values: List[float]
    local_ccy: bool
    tsdf: DataFrame

    @classmethod
    def setup_class(cls, domestic_ccy: str = "SEK", country: str = "SE"):
        """Sets the domestic currency and calendar of the user.

        Parameters
        ----------
        domestic_ccy : str, default: "SEK"
            Currency code according to ISO 4217
        country: str, default: "SE"
            Country code according to ISO 3166-1 alpha-2
        """

        cls.domestic = domestic_ccy
        cls.calendar = holiday_calendar(country=country)

    def __init__(self, d: TimeSerie):
        """Instantiates an object of the class OpenTimeSeries
         The data can have daily frequency, but not more frequent

        Parameters
        ----------
        d: TimeSerie
            A subclass of TypedDict with the required and optional parameters

        Returns
        -------
        OpenTimeSeries
            Object of the class OpenTimeSeries
        """

        schema_file = path.join(path.dirname(path.abspath(__file__)), "openseries.json")
        with open(file=schema_file, mode="r", encoding="utf-8") as f:
            series_schema = load(f)

        Draft7Validator.check_schema(schema=series_schema)
        validator = Draft7Validator(series_schema)
        validator.validate(d)

        if d.get("isin", None):
            isincode.validate(d["isin"])

        self.__dict__ = d

        if self.name != "":
            self.label = self.name

        self.pandas_df()

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            A representation of an OpenTimeSeries object
        """
        return (
            "{}(name={}, _id={}, instrumentId={}, valuetype={}, "
            "currency={}, start={}, end={}, local_ccy={})"
        ).format(
            self.__class__.__name__,
            self.name,
            self._id,
            self.instrumentId,
            self.valuetype,
            self.currency,
            self.first_idx.strftime("%Y-%m-%d"),
            self.last_idx.strftime("%Y-%m-%d"),
            self.local_ccy,
        )

    @classmethod
    def from_df(
        cls,
        df: DataFrame | Series,
        column_nmbr: int = 0,
        valuetype: str = "Price(Close)",
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ):
        """Creates a timeseries from a Pandas DataFrame or Series

        Parameters
        ----------
        df: DataFrame | Series
            Pandas DataFrame or Series
        column_nmbr : int, default: 0
            Using iloc[:, column_nmbr] to pick column
        valuetype : str, default: "Price(Close)"
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

        if isinstance(df, Series):
            if isinstance(df.name, tuple):
                label, _ = df.name
            else:
                label = df.name
            values = df.values.tolist()
        else:
            values = df.iloc[:, column_nmbr].tolist()
            if isinstance(df.columns, MultiIndex):
                label = df.columns.get_level_values(0).values[column_nmbr]
                valuetype = df.columns.get_level_values(1).values[column_nmbr]
            else:
                label = df.columns.values[column_nmbr]
        dates = [date_fix(d).strftime("%Y-%m-%d") for d in df.index]
        output = TimeSerie(
            _id="",
            currency=baseccy,
            instrumentId="",
            isin="",
            local_ccy=local_ccy,
            name=label,
            valuetype=valuetype,
            dates=dates,
            values=values,
        )

        return cls(d=output)

    @classmethod
    def from_frame(
        cls,
        frame,
        label: str,
        valuetype: str = "Price(Close)",
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ):
        """Creates a timeseries from an openseries.frame.OpenFrame

        Parameters
        ----------
        frame: OpenFrame
            openseries.frame.OpenFrame
        label : str
            Placeholder for a name of the timeseries
        valuetype : str, default: "Price(Close)"
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

        df = frame.tsdf.loc[:, (label, valuetype)]
        dates = [d.strftime("%Y-%m-%d") for d in df.index]

        output = TimeSerie(
            _id="",
            currency=baseccy,
            instrumentId="",
            isin="",
            local_ccy=local_ccy,
            name=df.name[0],
            valuetype=df.name[1],
            dates=dates,
            values=df.values.tolist(),
        )

        return cls(d=output)

    def from_deepcopy(self):
        """Creates a copy of an OpenTimeSeries object

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        return deepcopy(self)

    @classmethod
    def from_fixed_rate(
        cls,
        rate: float,
        d_range: DatetimeIndex | None = None,
        days: int | None = None,
        end_dt: date | None = None,
        label: str = "Series",
        valuetype: str = "Price(Close)",
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ):
        """Creates a timeseries from a series of values accruing with a given fixed rate

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
        valuetype : str, default: "Price(Close)"
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
        if d_range is None:
            d_range = DatetimeIndex(
                [d.date() for d in date_range(periods=days, end=end_dt, freq="D")]
            )
        deltas = array([i.days for i in d_range[1:] - d_range[:-1]])
        arr = list(cumprod(insert(1 + deltas * rate / 365, 0, 1.0)))
        d_range = [d.strftime("%Y-%m-%d") for d in d_range]

        output = TimeSerie(
            _id="",
            name=label,
            currency=baseccy,
            instrumentId="",
            isin="",
            local_ccy=local_ccy,
            valuetype=valuetype,
            dates=d_range,
            values=arr,
        )

        return cls(d=output)

    def to_json(self, filename: str, directory: str | None = None) -> dict:
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

        with open(path.join(directory, filename), "w") as ff:
            dump(data, ff, indent=2, sort_keys=False)

        return data

    def pandas_df(self):
        """Sets the .tsdf parameter as a Pandas DataFrame from the .dates and
        .values lists

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        df = DataFrame(
            data=self.values,
            index=self.dates,
            columns=[[self.label], [self.valuetype]],
            dtype="float64",
        )
        df.index = [d.date() for d in DatetimeIndex(df.index)]

        df.sort_index(inplace=True)
        self.tsdf = df

        return self

    def calc_range(
        self,
        months_offset: int | None = None,
        from_dt: date | None = None,
        to_dt: date | None = None,
    ) -> (date, date):
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
        earlier, later = None, None
        if months_offset is not None or from_dt is not None or to_dt is not None:
            if months_offset is not None:
                self.setup_class()
                earlier = date_offset_foll(
                    self.last_idx,
                    months_offset=-months_offset,
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
                        to_dt <= self.last_idx and from_dt >= self.first_idx
                    ), "Function calc_range returned dates outside series range"
                    earlier, later = from_dt, to_dt
            if earlier is not None:
                while not self.tsdf.index.isin([earlier]).any():
                    earlier -= timedelta(days=1)
            if later is not None:
                while not self.tsdf.index.isin([later]).any():
                    later += timedelta(days=1)
        else:
            earlier, later = self.first_idx, self.last_idx

        return earlier, later

    def align_index_to_local_cdays(self):
        """Changes the index of the associated Pandas DataFrame .tsdf to align with
        local calendar business days

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.setup_class()
        d_range = [
            d.date()
            for d in date_range(
                start=self.tsdf.first_valid_index(),
                end=self.tsdf.last_valid_index(),
                freq=CustomBusinessDay(calendar=self.calendar),
            )
        ]

        self.tsdf = self.tsdf.reindex(d_range, method=None, copy=False)

        return self

    def all_properties(self, properties: list | None = None) -> DataFrame:
        """Calculates the chosen timeseries properties

        Parameters
        ----------
        properties: list, optional
            The properties to calculate. Defaults to calculating all available.

        Returns
        -------
        pandas.DataFrame
            Properties of the OpenTimeSeries
        """

        if not properties:
            properties = [
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

        pdf = DataFrame.from_dict(
            {x: getattr(self, x) for x in properties}, orient="index"
        )

        pdf.columns = self.tsdf.columns

        return pdf

    @property
    def length(self) -> int:
        """
        Returns
        -------
        int
            Number of observations
        """

        return len(self.tsdf.index)

    @property
    def first_idx(self) -> date:
        """
        Returns
        -------
        datetime.date
            The first date in the timeseries
        """

        return self.tsdf.index[0]

    @property
    def last_idx(self) -> date:
        """
        Returns
        -------
        datetime.date
            The last date in the timeseries
        """

        return self.tsdf.index[-1]

    @property
    def span_of_days(self) -> int:
        """
        Returns
        -------
        int
            Number of days from the first date to the last
        """

        return (self.last_idx - self.first_idx).days

    @property
    def yearfrac(self) -> float:
        """
        Returns
        -------
        float
            Length of the timeseries expressed in years assuming all years
            have 365.25 days
        """

        return self.span_of_days / 365.25

    @property
    def periods_in_a_year(self) -> float:
        """
        Returns
        -------
        float
            The average number of observations per year
        """

        return self.length / self.yearfrac

    @property
    def geo_ret(self) -> float:
        """https://www.investopedia.com/terms/c/cagr.asp

        Returns
        -------
        float
            Compounded Annual Growth Rate (CAGR)
        """

        if (
            float(self.tsdf.loc[self.first_idx]) == 0.0
            or self.tsdf.lt(0.0).values.any()
        ):
            raise Exception(
                "Geometric return cannot be calculated due to an initial "
                "value being zero or a negative value."
            )
        return float(
            (self.tsdf.loc[self.last_idx] / self.tsdf.loc[self.first_idx])
            ** (1 / self.yearfrac)
            - 1
        )

    def geo_ret_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
            float(self.tsdf.loc[earlier]) == 0.0
            or self.tsdf.loc[earlier:later].lt(0.0).values.any()
        ):
            raise Exception(
                "Geometric return cannot be calculated due to an initial "
                "value being zero or a negative value."
            )

        return float(
            (self.tsdf.loc[later] / self.tsdf.loc[earlier]) ** (1 / fraction) - 1
        )

    @property
    def arithmetic_ret(self) -> float:
        """https://www.investopedia.com/terms/a/arithmeticmean.asp

        Returns
        -------
        float
            Annualized arithmetic mean of returns
        """

        return float(self.tsdf.pct_change().mean() * self.periods_in_a_year)

    def arithmetic_ret_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            how_many = int(self.tsdf.loc[earlier:later].count(numeric_only=True))
            time_factor = how_many / fraction
        return float(self.tsdf.loc[earlier:later].pct_change().mean() * time_factor)

    @property
    def value_ret(self) -> float:
        """
        Returns
        -------
        float
            Simple return
        """

        if float(self.tsdf.iloc[0]) == 0.0:
            raise Exception(
                "Simple Return cannot be calculated due to an initial value being "
                "zero."
            )
        return float(self.tsdf.iloc[-1] / self.tsdf.iloc[0] - 1)

    def value_ret_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
        if float(self.tsdf.loc[earlier]) == 0.0:
            raise Exception(
                "Simple Return cannot be calculated due to an initial value being "
                "zero."
            )
        return float(self.tsdf.loc[later] / self.tsdf.loc[earlier] - 1)

    def value_ret_calendar_period(self, year: int, month: int | None = None) -> float:
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
        return float(rtn.apply(cumprod, axis="index").iloc[-1] - 1)

    @property
    def vol(self) -> float:
        """Based on Pandas .std() which is the equivalent of stdev.s([...])
        in MS Excel \n
        https://www.investopedia.com/terms/v/volatility.asp

        Returns
        -------
        float
            Annualized volatility
        """

        return float(self.tsdf.pct_change().std() * sqrt(self.periods_in_a_year))

    def vol_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            how_many = int(self.tsdf.loc[earlier:later].count(numeric_only=True))
            time_factor = how_many / fraction

        return float(
            self.tsdf.loc[earlier:later].pct_change().std() * sqrt(time_factor)
        )

    @property
    def downside_deviation(self) -> float:
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

        return float(
            sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / self.length)
            * sqrt(self.periods_in_a_year)
        )

    def downside_deviation_func(
        self,
        min_accepted_return: float = 0.0,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
        how_many = float(
            self.tsdf.loc[earlier:later].pct_change().count(numeric_only=True)
        )
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            time_factor = how_many / fraction

        dddf = (
            self.tsdf.loc[earlier:later]
            .pct_change()
            .sub(min_accepted_return / time_factor)
        )

        return float(
            sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / how_many)
            * sqrt(time_factor)
        )

    @property
    def ret_vol_ratio(self) -> float:
        """
        Returns
        -------
        float
            Ratio of the annualized arithmetic mean of returns and annualized
            volatility.
        """
        return self.arithmetic_ret / self.vol

    def ret_vol_ratio_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
    def sortino_ratio(self) -> float:
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
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
    def z_score(self) -> float:
        """https://www.investopedia.com/terms/z/zscore.asp

        Returns
        -------
        float
            Z-score as (last return - mean return) / standard deviation of returns.
        """

        return float(
            (self.tsdf.pct_change().iloc[-1] - self.tsdf.pct_change().mean())
            / self.tsdf.pct_change().std()
        )

    def z_score_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
        part = self.tsdf.loc[earlier:later].pct_change().copy()
        return float((part.iloc[-1] - part.mean()) / part.std())

    @property
    def max_drawdown(self) -> float:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        float
            Maximum drawdown without any limit on date range
        """

        return float((self.tsdf / self.tsdf.expanding(min_periods=1).max()).min() - 1)

    @property
    def max_drawdown_date(self) -> date:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        datetime.date
            Date when the maximum drawdown occurred
        """

        mdddf = self.tsdf.copy()
        mdddf.index = DatetimeIndex(mdddf.index)
        mdd_date = (
            (mdddf / mdddf.expanding(min_periods=1).max())
            .idxmin()
            .values[0]
            .astype(datetime)
        )
        return datetime.fromtimestamp(mdd_date / 1e9).date()

    def max_drawdown_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
        return float(
            (
                self.tsdf.loc[earlier:later]
                / self.tsdf.loc[earlier:later].expanding(min_periods=1).max()
            ).min()
            - 1
        )

    @property
    def max_drawdown_cal_year(self) -> float:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        float
            Maximum drawdown in a single calendar year.
        """

        return float(
            self.tsdf.groupby([DatetimeIndex(self.tsdf.index).year])
            .apply(lambda x: (x / x.expanding(min_periods=1).max()).min() - 1)
            .min()
        )

    @property
    def worst(self) -> float:
        """
        Returns
        -------
        float
            Most negative percentage change
        """

        return float(self.tsdf.pct_change().min())

    @property
    def worst_month(self) -> float:
        """
        Returns
        -------
        float
            Most negative month
        """

        resdf = self.tsdf.copy()
        resdf.index = DatetimeIndex(resdf.index)
        return float(resdf.resample("BM").last().pct_change().min())

    def worst_func(
        self,
        observations: int = 1,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
        return float(
            self.tsdf.loc[earlier:later]
            .pct_change()
            .rolling(observations, min_periods=observations)
            .sum()
            .min()
        )

    @property
    def positive_share(self) -> float:
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
        return float(pos / tot)

    def positive_share_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
        period = self.tsdf.loc[earlier:later].copy()
        return float(
            period[period.pct_change().ge(0.0)].count(numeric_only=True)
            / period.pct_change().count(numeric_only=True)
        )

    @property
    def skew(self) -> float:
        """https://www.investopedia.com/terms/s/skewness.asp

        Returns
        -------
        float
            Skew of the return distribution
        """

        return float(
            skew(a=self.tsdf.pct_change().values, bias=True, nan_policy="omit")
        )

    def skew_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
        return float(
            skew(
                self.tsdf.loc[earlier:later].pct_change(),
                bias=True,
                nan_policy="omit",
            )
        )

    @property
    def kurtosis(self) -> float:
        """https://www.investopedia.com/terms/k/kurtosis.asp

        Returns
        -------
        float
            Kurtosis of the return distribution
        """
        return float(
            kurtosis(
                self.tsdf.pct_change(),
                fisher=True,
                bias=True,
                nan_policy="omit",
            )
        )

    def kurtosis_func(
        self,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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

        return float(
            kurtosis(
                self.tsdf.loc[earlier:later].pct_change(),
                fisher=True,
                bias=True,
                nan_policy="omit",
            )
        )

    @property
    def cvar_down(self, level: float = 0.95) -> float:
        """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

        Parameters
        ----------
        level: float, default: 0.95
            The sought CVaR level

        Returns
        -------
        float
            Downside Conditional Value At Risk "CVaR"
        """

        items = self.tsdf.iloc[:, 0].pct_change().count()
        return (
            self.tsdf.iloc[:, 0]
            .pct_change()
            .sort_values()
            .iloc[: int(ceil((1 - level) * items))]
            .mean()
        )

    def cvar_down_func(
        self,
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
            self.tsdf.loc[earlier:later, self.tsdf.columns.values[0]]
            .pct_change()
            .count()
        )
        return (
            self.tsdf.loc[earlier:later, self.tsdf.columns.values[0]]
            .pct_change()
            .sort_values()
            .iloc[: int(ceil((1 - level) * how_many))]
            .mean()
        )

    @property
    def var_down(
        self,
        level: float = 0.95,
        interpolation: Literal[
            "linear", "lower", "higher", "midpoint", "nearest"
        ] = "lower",
    ) -> float:
        """Downside Value At Risk, "VaR". The equivalent of
        percentile.inc([...], 1-level) over returns in MS Excel \n
        https://www.investopedia.com/terms/v/var.asp

        Parameters
        ----------

        level: float, default: 0.95
            The sought VaR level
        interpolation: Literal["linear", "lower", "higher", "midpoint",
        "nearest"], default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.

        Returns
        -------
        float
            Downside Value At Risk
        """

        return float(
            self.tsdf.pct_change().quantile(1 - level, interpolation=interpolation)
        )

    def var_down_func(
        self,
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        interpolation: Literal[
            "linear", "lower", "higher", "midpoint", "nearest"
        ] = "lower",
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
        interpolation: Literal["linear", "lower", "higher", "midpoint",
        "nearest"], default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.

        Returns
        -------
        float
            Downside Value At Risk
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float(
            self.tsdf.loc[earlier:later]
            .pct_change()
            .quantile(q=1 - level, interpolation=interpolation)
        )

    @property
    def vol_from_var(
        self,
        level: float = 0.95,
        interpolation: Literal[
            "linear", "lower", "higher", "midpoint", "nearest"
        ] = "lower",
    ) -> float:
        """
        Parameters
        ----------

        level: float, default: 0.95
            The sought VaR level
        interpolation: Literal["linear", "lower", "higher", "midpoint",
        "nearest"], default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.

        Returns
        -------
        float
            Implied annualized volatility from the Downside VaR using the
            assumption that returns are normally distributed.
        """

        return float(
            -sqrt(self.periods_in_a_year)
            * self.var_down_func(level, interpolation=interpolation)
            / norm.ppf(level)
        )

    def vol_from_var_func(
        self,
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        interpolation: Literal[
            "linear", "lower", "higher", "midpoint", "nearest"
        ] = "lower",
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
        interpolation: Literal["linear", "lower", "higher", "midpoint",
        "nearest"], default: "lower"
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
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            how_many = int(self.tsdf.loc[earlier:later].count(numeric_only=True))
            time_factor = how_many / fraction
        if drift_adjust:
            return float(
                (-sqrt(time_factor) / norm.ppf(level))
                * (
                    self.var_down_func(
                        level,
                        months_from_last,
                        from_date,
                        to_date,
                        interpolation,
                    )
                    - self.tsdf.loc[earlier:later].pct_change().sum()
                    / len(self.tsdf.loc[earlier:later].pct_change())
                )
            )
        else:
            return float(
                -sqrt(time_factor)
                * self.var_down_func(
                    level, months_from_last, from_date, to_date, interpolation
                )
                / norm.ppf(level)
            )

    def target_weight_from_var(
        self,
        target_vol: float = 0.175,
        min_leverage_local: float = 0.0,
        max_leverage_local: float = 99999.0,
        level: float = 0.95,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        interpolation: Literal[
            "linear", "lower", "higher", "midpoint", "nearest"
        ] = "lower",
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
        interpolation: Literal["linear", "lower", "higher", "midpoint",
        "nearest"], default: "lower"
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

    def value_to_ret(self):
        """
        Returns
        -------
        OpenTimeSeries
            The returns of the values in the series
        """

        self.tsdf = self.tsdf.pct_change()
        self.tsdf.iloc[0] = 0
        self.valuetype = "Return(Total)"
        self.tsdf.columns = [[self.label], [self.valuetype]]
        return self

    def value_to_diff(self, periods: int = 1):
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
        self.valuetype = "Return(Total)"
        self.tsdf.columns = [[self.label], [self.valuetype]]
        return self

    def value_to_log(self):
        """Converts a valueseries into logarithmic return series \n
        Equivalent to LN(value[t] / value[t=0]) in MS Excel

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.tsdf = log(self.tsdf / self.tsdf.iloc[0])
        return self

    def to_cumret(self):
        """Converts a returnseries into a cumulative valueseries

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        if not any(
            [
                True if x == "Return(Total)" else False
                for x in self.tsdf.columns.get_level_values(1).values
            ]
        ):
            self.value_to_ret()

        self.tsdf = self.tsdf.add(1.0)
        self.tsdf = self.tsdf.cumprod(axis=0) / self.tsdf.iloc[0]
        self.valuetype = "Price(Close)"
        self.tsdf.columns = [[self.label], [self.valuetype]]

        return self

    def from_1d_rate_to_cumret(self, days_in_year: int = 365, divider: float = 1.0):
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
        self.valuetype = "Price(Close)"
        self.pandas_df()

        return self

    def resample(self, freq: str = "BM"):
        """Resamples the timeseries frequency

        Parameters
        ----------
        freq: str, default "BM"
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

    def to_drawdown_series(self):
        """Converts the timeseries into a drawdown series

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.tsdf = drawdown_series(self.tsdf)
        self.tsdf.columns = [[self.label], ["Drawdowns"]]

        return self

    def drawdown_details(self) -> DataFrame:
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
        self,
        lmbda: float = 0.94,
        day_chunk: int = 11,
        dlta_degr_freedms: int = 0,
        months_from_last: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
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
            time_factor = periods_in_a_year_fixed
            how_many = int(self.length)
        else:
            fraction = (later - earlier).days / 365.25
            how_many = int(self.tsdf.loc[earlier:later].count(numeric_only=True))
            time_factor = how_many / fraction

        data = self.tsdf.loc[earlier:later].copy()

        data[self.label, "Returns"] = log(
            data.loc[:, (self.label, "Price(Close)")]
        ).diff()
        data[self.label, "EWMA"] = zeros(how_many)
        data.loc[:, (self.label, "EWMA")].iloc[0] = data.loc[
            :, (self.label, "Returns")
        ].iloc[1:day_chunk].std(ddof=dlta_degr_freedms) * sqrt(time_factor)

        prev = data.loc[self.first_idx]
        for _, row in data.iloc[1:].iterrows():
            row.loc[self.label, "EWMA"] = sqrt(
                square(row.loc[self.label, "Returns"]) * time_factor * (1 - lmbda)
                + square(prev.loc[self.label, "EWMA"]) * lmbda
            )
            prev = row.copy()

        return data.loc[:, (self.label, "EWMA")]

    def rolling_vol(
        self, observations: int = 21, periods_in_a_year_fixed: int | None = None
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
            time_factor = periods_in_a_year_fixed
        else:
            time_factor = self.periods_in_a_year
        df = self.tsdf.pct_change().copy()
        voldf = df.rolling(observations, min_periods=observations).std() * sqrt(
            time_factor
        )
        voldf.dropna(inplace=True)
        voldf.columns = [[self.label], ["Rolling volatility"]]

        return voldf

    def rolling_return(self, observations: int = 21) -> DataFrame:
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
            self.tsdf.pct_change().rolling(observations, min_periods=observations).sum()
        )
        retdf.columns = [[self.label], ["Rolling returns"]]

        return retdf.dropna()

    def rolling_cvar_down(
        self, level: float = 0.95, observations: int = 252
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
        self,
        level: float = 0.95,
        observations: int = 252,
        interpolation: Literal[
            "linear", "lower", "higher", "midpoint", "nearest"
        ] = "lower",
    ) -> DataFrame:
        """
        Parameters
        ----------
        level: float, default: 0.95
            The sought Value At Risk level
        observations: int, default: 252
            Number of observations in the overlapping window.
        interpolation: Literal["linear", "lower", "higher", "midpoint",
        "nearest"], default: "lower"
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

    def value_nan_handle(self, method: Literal["fill", "drop"] = "fill"):
        """Handling of missing values in a valueseries

        Parameters
        ----------
        method: Literal["fill", "drop"], default: "fill"
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

    def return_nan_handle(self, method: Literal["fill", "drop"] = "fill"):
        """Handling of missing values in a returnseries

        Parameters
        ----------
        method: Literal["fill", "drop"], default: "fill"
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

    def running_adjustment(self, adjustment: float, days_in_year: int = 365):
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

        if any(
            [
                True if x == "Return(Total)" else False
                for x in self.tsdf.columns.get_level_values(1).values
            ]
        ):
            ra_df = self.tsdf.copy()
            values: list = [1.0]
            returns_input = True
        else:
            values: list = [float(self.tsdf.iloc[0])]
            ra_df = self.tsdf.pct_change().copy()
            returns_input = False
        ra_df.dropna(inplace=True)

        prev = self.first_idx
        idx: date
        dates: list = [prev]

        for idx, row in ra_df.iterrows():
            dates.append(idx)
            values.append(
                values[-1]
                * (1 + float(row) + adjustment * (idx - prev).days / days_in_year)
            )
            prev = idx
        self.tsdf = DataFrame(data=values, index=dates)
        self.valuetype = "Price(Close)"
        self.tsdf.columns = [[self.label], [self.valuetype]]
        self.tsdf.index = [d.date() for d in DatetimeIndex(self.tsdf.index)]
        if returns_input:
            self.value_to_ret()
        return self

    def set_new_label(
        self,
        lvl_zero: str | None = None,
        lvl_one: str | None = None,
        delete_lvl_one: bool = False,
    ):
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
            self.tsdf.columns = [[self.label], [self.valuetype]]
        elif lvl_zero is not None and lvl_one is None:
            self.tsdf.columns = [[lvl_zero], [self.valuetype]]
            self.label = lvl_zero
        elif lvl_zero is None and lvl_one is not None:
            self.tsdf.columns = [[self.label], [lvl_one]]
            self.valuetype = lvl_one
        else:
            self.tsdf.columns = [[lvl_zero], [lvl_one]]
            self.label, self.valuetype = lvl_zero, lvl_one
        if delete_lvl_one:
            # noinspection PyUnresolvedReferences
            self.tsdf.columns = self.tsdf.columns.droplevel(level=1)
        return self

    def plot_series(
        self,
        mode: Literal["lines", "markers", "lines+markers"] = "lines",
        tick_fmt: str | None = None,
        directory: str | None = None,
        auto_open: bool = True,
        add_logo: bool = True,
        show_last: bool = False,
        output_type: Literal["file", "div"] = "file",
    ) -> (Figure, str):
        """Creates a Plotly Figure

        Parameters
        ----------
        mode: Literal["lines", "markers", "lines+markers"], default: "lines"
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
        output_type: str, default: "file"
            file or div

        Returns
        -------
        (plotly.go.Figure, str)
            Plotly Figure and html filename with location
        """

        if not directory:
            directory = path.join(str(Path.home()), "Documents")
        filename = self.label.replace("/", "").replace("#", "").replace(" ", "").upper()
        plotfile = path.join(path.abspath(directory), "{}.html".format(filename))

        values = [float(x) for x in self.tsdf.iloc[:, 0].tolist()]

        data = [
            Scatter(
                x=self.tsdf.index,
                y=values,
                hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                line=dict(width=2.5, dash="solid"),
                mode=mode,
                name=self.label,
            )
        ]

        fig, logo = load_plotly_dict()
        fig["data"] = data
        figure = Figure(fig)
        figure.update_layout(yaxis=dict(tickformat=tick_fmt))

        if add_logo:
            figure.add_layout_image(logo)

        if show_last is True:
            if tick_fmt:
                txt = "Last " + "{:" + "{}".format(tick_fmt) + "}"
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


def timeseries_chain(front, back, old_fee: float = 0.0) -> OpenTimeSeries:
    """Chain two timeseries together

    Parameters
    ----------
    front: OpenTimeSeries
        Earlier series to chain with
    back: OpenTimeSeries
        Later series to chain with
    old_fee: bool, default: False
        Fee to apply to earlier series

    Returns
    -------
    OpenTimeSeries
        An OpenTimeSeries object
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
            raise Exception("Failed to find a matching date between series")

    dates = [x.strftime("%Y-%m-%d") for x in olddf.index if x < first]
    values = array([float(x) for x in old.tsdf.values][: len(dates)])
    values = list(values * float(new.tsdf.loc[first]) / float(olddf.loc[first]))

    dates.extend([x.strftime("%Y-%m-%d") for x in new.tsdf.index])
    values.extend([float(x) for x in new.tsdf.values])

    new_dict = dict(new.__dict__)
    cleaner_list = ["label", "tsdf"]
    for item in cleaner_list:
        new_dict.pop(item)
    new_dict.update(dates=dates, values=values)
    return type(back)(new_dict)
