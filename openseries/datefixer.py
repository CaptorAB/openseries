import datetime as dt
from dateutil.relativedelta import relativedelta
import holidays
import numpy as np
import pandas as pd


def holiday_calendar(country: str = "SE") -> np.busdaycalendar:
    """Function to generate a business calendar

    Parameters
    ----------
    country: str, default: "SE"
        Numpy busdaycalendar country code

    Returns
    -------
    numpy.busdaycalendar
        Numpy busdaycalendar object
    """

    all_dates = np.arange("1970-12-30", "2070-12-30", dtype="datetime64[D]")

    years = [y for y in range(1970, 2071)]
    country_holidays = holidays.country_holidays(country=country, years=years)
    hols = []
    for date in sorted(country_holidays.keys()):
        hols.append(np.datetime64(date))

    hols = np.array(hols, dtype="datetime64[D]")

    while hols[0] < all_dates[0]:
        hols = hols[1:]
    while hols[-1] > all_dates[-1]:
        hols = hols[:-1]

    return np.busdaycalendar(holidays=hols)


def date_fix(d: str | dt.date | dt.datetime | np.datetime64 | pd.Timestamp) -> dt.date:
    """Function to parse from different date formats into datetime.date

    Parameters
    ----------
    d: str | dt.date | dt.datetime | np.datetime64 | pd.Timestamp
        The data item to parse

    Returns
    -------
    datetime.date
        Parsed date
    """

    if isinstance(d, dt.datetime) or isinstance(d, pd.Timestamp):
        return d.date()
    elif isinstance(d, dt.date):
        return d
    elif isinstance(d, np.datetime64):
        return pd.to_datetime(str(d)).date()
    elif isinstance(d, str):
        return dt.datetime.strptime(d, "%Y-%m-%d").date()
    else:
        raise Exception(
            f"Unknown date format {str(d)} of type {str(type(d))} encountered"
        )


def date_offset_foll(
    raw_date: str | dt.date | dt.datetime | np.datetime64 | pd.Timestamp,
    country: str = "SE",
    months_offset: int = 12,
    adjust: bool = False,
    following: bool = True,
) -> dt.date:
    """Function to offset dates according to a given calendar

    Parameters
    ----------
    raw_date: str | datetime.date | datetime.datetime | numpy.datetime64 | pandas.Timestamp
        The date to offset from
    country: str, default: "SE"
        Numpy busdaycalendar country code
    months_offset: int, default: 12
        Number of months as integer
    adjust: bool, default: False
        Determines if offset should adjust for business days
    following: bool, default: True
        Determines if days should be offset forward (following) or backward

    Returns
    -------
    datetime.date
        Off-set date
    """

    calendar = holiday_calendar(country=country)
    raw_date = date_fix(raw_date)
    month_delta = relativedelta(months=months_offset)

    if following:
        day_delta = relativedelta(days=1)
    else:
        day_delta = relativedelta(days=-1)

    new_date = raw_date + month_delta

    if adjust:
        while not np.is_busday(dates=new_date, busdaycal=calendar):
            new_date += day_delta

    return new_date


def get_previous_sweden_business_day_before_today(today: dt.date | None = None):
    """Function to bump backwards to find the previous Swedish business day before today

    Parameters
    ----------
    today: datetime.date, optional
        Manual input of the day from where the previous business day is found

    Returns
    -------
    datetime.date
        The previous Swedish business day
    """

    if today is None:
        today = dt.date.today()

    return date_offset_foll(
        today - dt.timedelta(days=1),
        country="SE",
        months_offset=0,
        adjust=True,
        following=False,
    )
