"""Date related utilities."""
from __future__ import annotations

import datetime as dt
from typing import Optional, Union, cast

from dateutil.relativedelta import relativedelta
from holidays import (  # type: ignore[import-untyped,unused-ignore]
    country_holidays,
    list_supported_countries,
)
from numpy import array, busdaycalendar, datetime64, is_busday, where
from pandas import DataFrame, DatetimeIndex, Series, Timestamp, concat, date_range
from pandas.tseries.offsets import CustomBusinessDay
from pydantic import PositiveInt

from openseries.types import (
    CountriesType,
    DateType,
    HolidayType,
    LiteralBizDayFreq,
    LiteralPandasResampleConvention,
)


def holiday_calendar(
    startyear: int,
    endyear: int,
    countries: CountriesType = "SE",
    custom_holidays: Optional[HolidayType] = None,
) -> busdaycalendar:
    """
    Generate a business calendar.

    Parameters
    ----------
    startyear: int
        First year in date range generated
    endyear: int
        Last year in date range generated
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: HolidayType, optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    numpy.busdaycalendar
        Generate a business calendar
    """
    startyear -= 1
    endyear += 1
    if startyear == endyear:
        endyear += 1
    years = list(range(startyear, endyear))

    if isinstance(countries, str) and countries in list_supported_countries():
        staging = country_holidays(country=countries, years=years)
        if custom_holidays is not None:
            staging.update(custom_holidays)
        hols = array(sorted(staging.keys()), dtype="datetime64[D]")
    elif isinstance(countries, list) and all(
        country in list_supported_countries() for country in countries
    ):
        country: str
        countryholidays: list[Union[dt.date, str]] = []
        for i, country in enumerate(countries):
            staging = country_holidays(country=country, years=years)
            if i == 0 and custom_holidays is not None:
                staging.update(custom_holidays)
            countryholidays += list(staging)
        hols = array(sorted(set(countryholidays)), dtype="datetime64[D]")
    else:
        msg = (
            "Argument countries must be a string country code or "
            "a list of string country codes according to ISO 3166-1 alpha-2."
        )
        raise ValueError(
            msg,
        )

    return busdaycalendar(holidays=hols)


def date_fix(
    fixerdate: DateType,
) -> dt.date:
    """
    Parse different date formats into datetime.date.

    Parameters
    ----------
    fixerdate: DateType
        The data item to parse

    Returns
    -------
    datetime.date
        Parsed date
    """
    if isinstance(fixerdate, (Timestamp, dt.datetime)):
        return fixerdate.date()
    if isinstance(fixerdate, dt.date):
        return fixerdate
    if isinstance(fixerdate, datetime64):
        return (
            dt.datetime.strptime(str(fixerdate)[:10], "%Y-%m-%d")
            .replace(tzinfo=dt.timezone.utc)
            .date()
        )
    if isinstance(fixerdate, str):
        return (
            dt.datetime.strptime(fixerdate, "%Y-%m-%d")
            .replace(tzinfo=dt.timezone.utc)
            .date()
        )
    msg = f"Unknown date format {fixerdate!s} of type {type(fixerdate)!s} encountered"
    raise TypeError(
        msg,
    )


def date_offset_foll(
    raw_date: DateType,
    months_offset: int = 12,
    adjust: bool = False,  # noqa: FBT001, FBT002
    following: bool = True,  # noqa: FBT001, FBT002
    countries: CountriesType = "SE",
    custom_holidays: Optional[HolidayType] = None,
) -> dt.date:
    """
    Offset dates according to a given calendar.

    Parameters
    ----------
    raw_date: DateType
        The date to offset from
    months_offset: int, default: 12
        Number of months as integer
    adjust: bool, default: False
        Determines if offset should adjust for business days
    following: bool, default: True
        Determines if days should be offset forward (following) or backward
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: HolidayType, optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    datetime.date
        Off-set date
    """
    raw_date = date_fix(raw_date)
    month_delta = relativedelta(months=months_offset)

    day_delta = relativedelta(days=1) if following else relativedelta(days=-1)

    new_date = raw_date + month_delta

    if adjust:
        startyear = min([raw_date.year, new_date.year])
        endyear = max([raw_date.year, new_date.year])
        calendar = holiday_calendar(
            startyear=startyear,
            endyear=endyear,
            countries=countries,
            custom_holidays=custom_holidays,
        )
        while not is_busday(dates=new_date, busdaycal=calendar):
            new_date += day_delta

    return new_date


def get_previous_business_day_before_today(
    today: Optional[dt.date] = None,
    countries: CountriesType = "SE",
    custom_holidays: Optional[HolidayType] = None,
) -> dt.date:
    """
    Bump date backwards to find the previous business day.

    Parameters
    ----------
    today: datetime.date, optional
        Manual input of the day from where the previous business day is found
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: HolidayType, optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    datetime.date
        The previous business day
    """
    if today is None:
        today = dt.datetime.now(tz=dt.timezone.utc).date()

    return date_offset_foll(
        today - dt.timedelta(days=1),
        countries=countries,
        custom_holidays=custom_holidays,
        months_offset=0,
        adjust=True,
        following=False,
    )


def offset_business_days(
    ddate: dt.date,
    days: int,
    countries: CountriesType = "SE",
    custom_holidays: Optional[HolidayType] = None,
) -> dt.date:
    """
    Bump date by business days.

    It first adjusts to a valid business day and then bumps with given
    number of business days from there.

    Parameters
    ----------
    ddate: datetime.date
        A starting date that does not have to be a business day
    days: int
        The number of business days to offset from the business day that is
        the closest preceding the day given
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: HolidayType, optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    datetime.date
        The new offset business day
    """
    if days <= 0:
        scaledtoyeardays = int((days * 372 / 250) // 1) - 365
        ndate = ddate + dt.timedelta(days=scaledtoyeardays)
        calendar = holiday_calendar(
            startyear=ndate.year,
            endyear=ddate.year,
            countries=countries,
            custom_holidays=custom_holidays,
        )
        local_bdays: list[dt.date] = [
            bday.date()
            for bday in date_range(
                periods=abs(scaledtoyeardays),
                end=ddate,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]
    else:
        scaledtoyeardays = int((days * 372 / 250) // 1) + 365
        ndate = ddate + dt.timedelta(days=scaledtoyeardays)
        calendar = holiday_calendar(
            startyear=ddate.year,
            endyear=ndate.year,
            countries=countries,
            custom_holidays=custom_holidays,
        )
        local_bdays = [
            bday.date()
            for bday in date_range(
                start=ddate,
                periods=scaledtoyeardays,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]

    while ddate not in local_bdays:
        if days <= 0:
            ddate -= dt.timedelta(days=1)
        else:
            ddate += dt.timedelta(days=1)

    idx = where(array(local_bdays) == ddate)[0]

    return cast(dt.date, local_bdays[idx[0] + days])


def generate_calender_date_range(
    trading_days: PositiveInt,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    countries: CountriesType = "SE",
) -> list[dt.date]:
    """
    Generate a list of business day calender dates.

    Parameters
    ----------
    trading_days: PositiveInt
        Number of days to generate
    start: datetime.date, optional
        Date when the range starts
    end: datetime.date, optional
        Date when the range ends
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2

    Returns
    -------
    list[dt.date]
        List of business day calender dates
    """
    if start and not end:
        tmp_range = date_range(
            start=start,
            periods=trading_days * 365 // 252,
            freq="D",
        )
        calendar = holiday_calendar(
            startyear=start.year,
            endyear=tmp_range[-1].year,
            countries=countries,
        )
        return [
            d.date()
            for d in date_range(
                start=start,
                periods=trading_days,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]

    if end and not start:
        tmp_range = date_range(end=end, periods=trading_days * 365 // 252, freq="D")
        calendar = holiday_calendar(
            startyear=tmp_range[0].year,
            endyear=end.year,
            countries=countries,
        )
        return [
            d.date()
            for d in date_range(
                end=end,
                periods=trading_days,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]

    msg = (
        "Provide one of start or end date, but not both. "
        "Date range is inferred from number of trading days."
    )
    raise ValueError(
        msg,
    )


def align_dataframe_to_local_cdays(
    data: DataFrame,
    countries: CountriesType = "SE",
) -> DataFrame:
    """
    Align the index .tsdf with local calendar business days.

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2

    Returns
    -------
    pandas.DataFrame
        An OpenFrame object
    """
    startyear = data.index[0].year
    endyear = data.index[-1].year
    calendar = holiday_calendar(
        startyear=startyear,
        endyear=endyear,
        countries=countries,
    )

    d_range = [
        d.date()
        for d in date_range(
            start=cast(dt.date, data.first_valid_index()),
            end=cast(dt.date, data.last_valid_index()),
            freq=CustomBusinessDay(calendar=calendar),
        )
    ]
    return data.reindex(d_range, method=None, copy=False)


def do_resample_to_business_period_ends(
    data: DataFrame,
    head: Series[type[float]],
    tail: Series[type[float]],
    freq: LiteralBizDayFreq,
    countries: CountriesType,
    convention: LiteralPandasResampleConvention,
) -> DatetimeIndex:
    """
    Resample timeseries frequency to business calendar month end dates.

    Stubs left in place. Stubs will be aligned to the shortest stub.

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    head: pandas:Series
        Data point at maximum first date of all series
    tail: pandas:Series
        Data point at minimum last date of all series
    freq: LiteralBizDayFreq
        The date offset string that sets the resampled frequency
    countries: CountriesType
        (List of) country code(s) according to ISO 3166-1 alpha-2
        to create a business day calendar used for date adjustments
    convention: LiteralPandasResampleConvention
        Controls whether to use the start or end of `rule`.

    Returns
    -------
    Pandas.DatetimeIndex
        A date range aligned to business period ends
    """
    newhead = head.to_frame().T
    newtail = tail.to_frame().T
    data.index = DatetimeIndex(data.index)
    data = data.resample(rule=freq, convention=convention).last()
    data = data.drop(index=data.index[-1])
    data.index = [  # type: ignore[assignment,unused-ignore]
        d.date() for d in DatetimeIndex(data.index)
    ]

    if newhead.index[0] not in data.index:
        data = concat([data, newhead])

    if newtail.index[0] not in data.index:
        data = concat([data, newtail])

    data = data.sort_index()

    dates = DatetimeIndex(
        [data.index[0]]
        + [
            date_offset_foll(
                dt.date(d.year, d.month, 1)
                + relativedelta(months=1)
                - dt.timedelta(days=1),
                countries=countries,
                months_offset=0,
                adjust=True,
                following=False,
            )
            for d in data.index[1:-1]
        ]
        + [data.index[-1]],
    )
    return dates.drop_duplicates()


def get_calc_range(  # noqa: C901
    data: DataFrame,
    months_offset: Optional[int] = None,
    from_dt: Optional[dt.date] = None,
    to_dt: Optional[dt.date] = None,
) -> tuple[dt.date, dt.date]:
    """
    Create user defined date range.

    Parameters
    ----------
    data: pandas.DataFrame
        The data with the date range
    months_offset: int, optional
        Number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_dt: datetime.date, optional
        Specific from date
    to_dt: datetime.date, optional
        Specific from date

    Returns
    -------
    tuple[datetime.date, datetime.date]
        Start and end date of the chosen date range
    """
    earlier, later = data.index[0], data.index[-1]
    if any([months_offset, from_dt, to_dt]):
        if months_offset is not None:
            earlier = date_offset_foll(
                raw_date=data.index[-1],
                months_offset=-months_offset,
                adjust=False,
                following=True,
            )
            if earlier < data.index[0]:
                msg = "Function calc_range returned earlier date < series start"
                raise ValueError(
                    msg,
                )
            later = data.index[-1]
        elif from_dt is not None and to_dt is None:
            if from_dt < data.index[0]:
                msg = "Given from_dt date < series start"
                raise ValueError(msg)
            earlier, later = from_dt, data.index[-1]
        elif from_dt is None and to_dt is not None:
            if to_dt > data.index[-1]:
                msg = "Given to_dt date > series end"
                raise ValueError(msg)
            earlier, later = data.index[0], to_dt
        elif from_dt is not None and to_dt is not None:
            if to_dt > data.index[-1] or from_dt < data.index[0]:
                msg = "Given from_dt or to_dt dates outside series range"
                raise ValueError(
                    msg,
                )
            earlier, later = from_dt, to_dt
        while earlier not in data.index.tolist():
            earlier -= dt.timedelta(days=1)
        while later not in data.index.tolist():
            later += dt.timedelta(days=1)

    return earlier, later
