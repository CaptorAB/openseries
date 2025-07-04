"""Date related utilities.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, cast

import exchange_calendars as exchcal
from dateutil.relativedelta import relativedelta
from holidays import (
    country_holidays,
    list_supported_countries,
)
from numpy import array, busdaycalendar, datetime64, is_busday, where
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Timestamp,
    concat,
    date_range,
)
from pandas.tseries.offsets import CustomBusinessDay

from .owntypes import (
    BothStartAndEndError,
    CountriesNotStringNorListStrError,
    MarketsNotStringNorListStrError,
    TradingDaysNotAboveZeroError,
)

if TYPE_CHECKING:
    from .owntypes import (  # pragma: no cover
        CountriesType,
        DateType,
        LiteralBizDayFreq,
    )

__all__ = [
    "date_fix",
    "date_offset_foll",
    "generate_calendar_date_range",
    "get_previous_business_day_before_today",
    "holiday_calendar",
    "offset_business_days",
]


def market_holidays(
    startyear: int,
    endyear: int,
    markets: str | list[str],
) -> list[str]:
    """Return a dict of holiday dates mapping to list of markets closed.

    Parameters
    ----------
    startyear: int
        First year (inclusive) to consider.
    endyear: int
        Last year (inclusive) to consider.
    markets: str | list[str]
        String or list of market codes supported by pandas_market_calendars.

    Returns:
    -------
    list[str]
        list of holiday dates.
    """
    market_list = [markets] if isinstance(markets, str) else list(markets)

    supported = exchcal.get_calendar_names()

    if not all(m in supported for m in market_list):
        msg = (
            "Argument markets must be a string market code or a list of market "
            "codes supported by pandas_market_calendars."
        )
        raise MarketsNotStringNorListStrError(msg)

    holidays: list[str] = []
    for m in market_list:
        cal = exchcal.get_calendar(m)
        cal_hols = cal.regular_holidays.holidays()
        my_hols: list[str] = [
            date.date().strftime("%Y-%m-%d")
            for date in cal_hols
            if (startyear <= date.date().year <= endyear)
        ]
        holidays.extend(my_hols)

    return list(set(holidays))


def holiday_calendar(
    startyear: int,
    endyear: int,
    countries: CountriesType = "SE",
    markets: list[str] | str | None = None,
    custom_holidays: list[str] | str | None = None,
) -> busdaycalendar:
    """Generate a business calendar.

    Parameters
    ----------
    startyear: int
        First year in date range generated
    endyear: int
        Last year in date range generated
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    markets: list[str] | str, optional
        (List of) markets code(s) according to pandas-market-calendars
    custom_holidays: list[str] | str, optional
        Argument where missing holidays can be added

    Returns:
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
        hols = list(staging.keys())
    elif isinstance(countries, (list, set)) and all(
        country in list_supported_countries() for country in countries
    ):
        country: str
        countryholidays: list[dt.date | str] = []
        for country in countries:
            staging = country_holidays(country=country, years=years)
            countryholidays += list(staging)
        hols = list(countryholidays)
    else:
        msg = (
            "Argument countries must be a string country code or "
            "a list of string country codes according to ISO 3166-1 alpha-2."
        )
        raise CountriesNotStringNorListStrError(msg)

    if markets:
        market_hols = market_holidays(
            startyear=startyear, endyear=endyear, markets=markets
        )
        dt_mkt_hols = [date_fix(fixerdate=ddate) for ddate in market_hols]
        hols.extend(dt_mkt_hols)

    if custom_holidays:
        custom_list = (
            [custom_holidays]
            if isinstance(custom_holidays, str)
            else list(custom_holidays)  # type: ignore[arg-type]
        )
        hols.extend([date_fix(fixerdate=ddate) for ddate in custom_list])

    return busdaycalendar(holidays=array(sorted(set(hols)), dtype="datetime64[D]"))


def date_fix(
    fixerdate: DateType,
) -> dt.date:
    """Parse different date formats into datetime.date.

    Parameters
    ----------
    fixerdate: DateType
        The data item to parse

    Returns:
    -------
    datetime.date
        Parsed date

    """
    msg = f"Unknown date format {fixerdate!s} of type {type(fixerdate)!s} encountered"
    if isinstance(fixerdate, Timestamp | dt.datetime):
        return fixerdate.date()
    if isinstance(fixerdate, dt.date):
        return fixerdate
    if isinstance(fixerdate, datetime64):
        return (
            dt.datetime.strptime(str(fixerdate)[:10], "%Y-%m-%d").astimezone().date()
        )
    if isinstance(fixerdate, str):
        return dt.datetime.strptime(fixerdate, "%Y-%m-%d").astimezone().date()
    raise TypeError(msg)


def date_offset_foll(
    raw_date: DateType,
    months_offset: int = 12,
    countries: CountriesType = "SE",
    markets: list[str] | str | None = None,
    custom_holidays: list[str] | str | None = None,
    *,
    adjust: bool = False,
    following: bool = True,
) -> dt.date:
    """Offset dates according to a given calendar.

    Parameters
    ----------
    raw_date: DateType
        The date to offset from
    months_offset: int, default: 12
        Number of months as integer
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    markets: list[str] | str, optional
        (List of) markets code(s) according to pandas-market-calendars
    custom_holidays: list[str] | str, optional
        Argument where missing holidays can be added
    adjust: bool, default: False
        Determines if offset should adjust for business days
    following: bool, default: True
        Determines if days should be offset forward (following) or backward

    Returns:
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
            markets=markets,
            custom_holidays=custom_holidays,
        )
        while not is_busday(dates=new_date, busdaycal=calendar):
            new_date += day_delta

    return new_date  # type: ignore[no-any-return]


def get_previous_business_day_before_today(
    today: dt.date | None = None,
    countries: CountriesType = "SE",
    markets: list[str] | str | None = None,
    custom_holidays: list[str] | str | None = None,
) -> dt.date:
    """Bump date backwards to find the previous business day.

    Parameters
    ----------
    today: datetime.date, optional
        Manual input of the day from where the previous business day is found
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    markets: list[str] | str, optional
        (List of) markets code(s) according to pandas-market-calendars
    custom_holidays: list[str] | str, optional
        Argument where missing holidays can be added

    Returns:
    -------
    datetime.date
        The previous business day

    """
    if today is None:
        today = dt.datetime.now().astimezone().date()

    return date_offset_foll(
        raw_date=today - dt.timedelta(days=1),
        months_offset=0,
        countries=countries,
        markets=markets,
        custom_holidays=custom_holidays,
        adjust=True,
        following=False,
    )


def offset_business_days(
    ddate: dt.date,
    days: int,
    countries: CountriesType = "SE",
    markets: list[str] | str | None = None,
    custom_holidays: list[str] | str | None = None,
) -> dt.date:
    """Bump date by business days.

    It first adjusts to a valid business day and then bumps with given
    number of business days from there.

    Parameters
    ----------
    ddate: datetime.date
        A starting date that does not have to be a business day
    days: int
        The number of business days to offset from the business day that is given
        If days is set as anything other than an integer its value is set to zero
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    markets: list[str] | str, optional
        (List of) markets code(s) according to pandas-market-calendars
    custom_holidays: list[str] | str, optional
        Argument where missing holidays can be added

    Returns:
    -------
    datetime.date
        The new offset business day

    """
    try:
        days = int(days)
    except TypeError:
        days = 0

    if days <= 0:
        scaledtoyeardays = int((days * 372 / 250) // 1) - 365
        ndate = ddate + dt.timedelta(days=scaledtoyeardays)
        calendar = holiday_calendar(
            startyear=ndate.year,
            endyear=ddate.year,
            countries=countries,
            markets=markets,
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
            markets=markets,
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

    return cast("dt.date", local_bdays[idx[0] + days])


def generate_calendar_date_range(
    trading_days: int,
    start: dt.date | None = None,
    end: dt.date | None = None,
    countries: CountriesType = "SE",
    markets: list[str] | str | None = None,
    custom_holidays: list[str] | str | None = None,
) -> list[dt.date]:
    """Generate a list of business day calendar dates.

    Parameters
    ----------
    trading_days: int
        Number of days to generate. Must be greater than zero
    start: datetime.date, optional
        Date when the range starts
    end: datetime.date, optional
        Date when the range ends
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    markets: list[str] | str, optional
        (List of) markets code(s) according to pandas-market-calendars
    custom_holidays: list[str] | str, optional
        Argument where missing holidays can be added

    Returns:
    -------
    list[dt.date]
        List of business day calendar dates

    """
    if trading_days < 1:
        msg = "Argument trading_days must be greater than zero."
        raise TradingDaysNotAboveZeroError(msg)

    if start and not end:
        tmp_range = date_range(
            start=start,
            periods=trading_days * 365 // 252,
            freq="D",
        )
        calendar = holiday_calendar(
            startyear=start.year,
            endyear=date_fix(tmp_range.tolist()[-1]).year,
            countries=countries,
            markets=markets,
            custom_holidays=custom_holidays,
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
            startyear=date_fix(tmp_range.tolist()[0]).year,
            endyear=end.year,
            countries=countries,
            markets=markets,
            custom_holidays=custom_holidays,
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
    raise BothStartAndEndError(msg)


def _do_resample_to_business_period_ends(
    data: DataFrame,
    freq: LiteralBizDayFreq,
    countries: CountriesType,
    markets: list[str] | str | None = None,
    custom_holidays: list[str] | str | None = None,
) -> DatetimeIndex:
    """Resample timeseries frequency to business calendar month end dates.

    Stubs left in place. Stubs will be aligned to the shortest stub.

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    freq: LiteralBizDayFreq
        The date offset string that sets the resampled frequency
    countries: CountriesType
        (List of) country code(s) according to ISO 3166-1 alpha-2
    markets: list[str] | str, optional
        (List of) markets code(s) according to pandas-market-calendars
    custom_holidays: list[str] | str, optional
        Argument where missing holidays can be added

    Returns:
    -------
    Pandas.DatetimeIndex
        A date range aligned to business period ends

    """
    copydata = data.copy()
    copydata.index = DatetimeIndex(copydata.index)
    copydata = copydata.resample(rule=freq).last()
    copydata = copydata.drop(index=copydata.index[-1])
    copydata.index = Index(d.date() for d in DatetimeIndex(copydata.index))

    copydata = concat([data.head(n=1), copydata, data.tail(n=1)]).sort_index()

    dates = DatetimeIndex(
        [copydata.index[0]]
        + [
            date_offset_foll(  # type: ignore[misc]
                raw_date=dt.date(d.year, d.month, 1)
                + relativedelta(months=1)
                - dt.timedelta(days=1),
                months_offset=0,
                countries=countries,
                markets=markets,
                custom_holidays=custom_holidays,
                adjust=True,
                following=False,
            )
            for d in copydata.index[1:-1]
        ]
        + [copydata.index[-1]],
    )
    return DatetimeIndex(dates.drop_duplicates())
