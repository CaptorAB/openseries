# -*- coding: utf-8 -*-
import datetime as dt
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import logging
import pandas as pd
from pandas.tseries.offsets import CDay
from typing import Union

from OpenSeries.sweden_holidays import CaptorHolidayCalendar, holidays_sw


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
