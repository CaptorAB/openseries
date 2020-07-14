# -*- coding: utf-8 -*-
import datetime as dt
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import holidays
import logging
import numpy as np
import pandas as pd
from pandas.tseries.offsets import CDay
from typing import Union

from OpenSeries.sweden_holidays import CaptorHolidayCalendar, holidays_sw


def np_biz_calendar(country: str = 'SE') -> np.busdaycalendar:

    all_dates = np.arange('1970-12-30', '2070-12-30', dtype='datetime64[D]')

    years = [y for y in range(1970, 2071)]
    country_holidays = holidays.CountryHoliday(country=country, years=years)
    hols = []
    for date in sorted(country_holidays.keys()):
        hols.append(np.datetime64(date))

    hols = np.array(hols, dtype='datetime64[D]')

    while hols[0] < all_dates[0]:
        hols = hols[1:]
    while hols[-1] > all_dates[-1]:
        hols = hols[:-1]

    return np.busdaycalendar(holidays=hols)


def np_biz_date_range(start_dt: Union[np.datetime64, dt.date, str], end_dt: Union[np.datetime64, dt.date, str],
                      bdays: np.busdaycalendar = None, country: str = 'SE') -> np.ndarray:

    if isinstance(start_dt, np.datetime64):
        start_dt = str(np.datetime_as_string(start_dt))
    elif isinstance(start_dt, dt.date):
        start_dt = start_dt.strftime('%Y-%m-%d')

    if isinstance(end_dt, np.datetime64):
        end_dt = str(np.datetime_as_string(end_dt))
    elif isinstance(end_dt, dt.date):
        end_dt = end_dt.strftime('%Y-%m-%d')

    all_dates = np.arange(start_dt, end_dt, dtype='datetime64[D]')

    if bdays is None:
        first_year = int(str(all_dates[0]).split('-')[0])
        last_year = int(str(all_dates[-1]).split('-')[0]) + 1
        years = [y for y in range(first_year, last_year)]

        country_holidays = holidays.CountryHoliday(country=country, years=years)
        hols = []
        for date in sorted(country_holidays.keys()):
            hols.append(np.datetime64(date))

        hols = np.array(hols, dtype='datetime64[D]')
        while hols[0] < all_dates[0]:
            hols = hols[1:]
        while hols[-1] > all_dates[-1]:
            hols = hols[:-1]

        bdays = np.busdaycalendar(holidays=hols)

    return all_dates[np.is_busday(all_dates, busdaycal=bdays)]


def date_fix(d: Union[str, dt.date, dt.datetime]) -> dt.date:
    """
    Function to parse date or timestamp string into datetime.date
    :param d: A string containing a date/time stamp.
    :returns : datetime.date
    """
    try:
        if isinstance(d, str):
            temp_dt = parse(d)
        elif isinstance(d, dt.datetime) or isinstance(d, dt.date):
            temp_dt = parse(d.strftime('%Y-%m-%d'))
        else:
            raise ValueError('Argument passed to date_fix cannot be parsed.')
        return dt.date(temp_dt.year, temp_dt.month, temp_dt.day)
    except ValueError as e:
        logging.exception(e)


def date_offset_foll(raw_date: Union[str, dt.date, dt.datetime], calendar: CDay, months_offset: int = 12,
                     adjust: bool = False, following: bool = True) -> dt.date:
    """
    Function to offset dates according to a given calendar.
    :param raw_date: The date to offset from.
    :param calendar:
    :param months_offset: Number of months as integer
    :param adjust: Boolean condition controlling if offset should adjust for business days.
    :param following: Boolean condition controlling days should be offset forward (following=True) or backward.
    :returns : datetime.date
    """
    start_dt = dt.date(1970, 12, 30)
    end_dt = dt.date(start_dt.year + 90, 12, 30)
    local_bdays = pd.date_range(start=start_dt, end=end_dt, freq=calendar)
    raw_date = date_fix(raw_date)
    assert isinstance(raw_date, dt.date), 'Error when parsing raw_date.'
    month_delta = relativedelta(months=months_offset)
    if following:
        day_delta = relativedelta(days=1)
    else:
        day_delta = relativedelta(days=-1)
    new_date = raw_date + month_delta
    if adjust:
        while new_date not in local_bdays:
            new_date += day_delta
    return new_date


def get_previous_sweden_business_day_before_today():
    sweden = CaptorHolidayCalendar(rules=holidays_sw)
    return date_offset_foll(dt.date.today() - dt.timedelta(days=1), calendar=CDay(calendar=sweden),
                            months_offset=0, adjust=True, following=False)
