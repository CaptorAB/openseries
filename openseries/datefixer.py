from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from holidays import country_holidays
from numpy import array, arange, busdaycalendar, datetime64, is_busday
from pandas import Timestamp, to_datetime


def holiday_calendar(country: str = "SE") -> busdaycalendar:
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

    all_dates = arange("1970-12-30", "2070-12-30", dtype="datetime64[D]")

    years = [y for y in range(1970, 2071)]
    countryholidays = country_holidays(country=country, years=years)
    hols = []
    for dte in sorted(countryholidays.keys()):
        hols.append(datetime64(dte))

    hols = array(hols, dtype="datetime64[D]")

    while hols[0] < all_dates[0]:
        hols = hols[1:]
    while hols[-1] > all_dates[-1]:
        hols = hols[:-1]

    return busdaycalendar(holidays=hols)


def date_fix(d: str | date | datetime | datetime64 | Timestamp) -> date:
    """Function to parse from different date formats into datetime.date

    Parameters
    ----------
    d: str | date | datetime | datetime64 | Timestamp
        The data item to parse

    Returns
    -------
    datetime.date
        Parsed date
    """

    if isinstance(d, datetime) or isinstance(d, Timestamp):
        return d.date()
    elif isinstance(d, date):
        return d
    elif isinstance(d, datetime64):
        return to_datetime(str(d)).date()
    elif isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    else:
        raise Exception(
            f"Unknown date format {str(d)} of type {str(type(d))} encountered"
        )


def date_offset_foll(
    raw_date: str | date | datetime | datetime64 | Timestamp,
    country: str = "SE",
    months_offset: int = 12,
    adjust: bool = False,
    following: bool = True,
) -> date:
    """Function to offset dates according to a given calendar

    Parameters
    ----------
    raw_date: str | date | datetime | datetime64 | Timestamp
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
        while not is_busday(dates=new_date, busdaycal=calendar):
            new_date += day_delta

    return new_date


def get_previous_sweden_business_day_before_today(today: date | None = None):
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
        today = date.today()

    return date_offset_foll(
        today - timedelta(days=1),
        country="SE",
        months_offset=0,
        adjust=True,
        following=False,
    )
