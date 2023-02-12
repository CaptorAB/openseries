import datetime as dt
from dateutil.relativedelta import relativedelta
from holidays import country_holidays, list_supported_countries
from numpy import array, busdaycalendar, datetime64, is_busday, where
from pandas import date_range, Timestamp, to_datetime
from pandas.tseries.offsets import CustomBusinessDay


def holiday_calendar(
    startyear: int,
    endyear: int,
    countries: list | str = "SE",
    custom_holidays: list | dict | None = None,
) -> busdaycalendar:
    """Function to generate a business calendar

    Parameters
    ----------
    startyear: int
        First year in date range generated
    endyear: int
        Last year in date range generated
    countries: list | str, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: list | dict | None, default: None
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    numpy.busdaycalendar
        Numpy busdaycalendar object
    """
    startyear -= 1
    endyear += 1
    if startyear == endyear:
        endyear += 1
    years = [y for y in range(startyear, endyear)]

    if isinstance(countries, str) and countries in list_supported_countries():
        staging = country_holidays(country=countries, years=years)
        if custom_holidays is not None:
            staging.update(custom_holidays)
        hols = array(sorted(staging.keys()), dtype="datetime64[D]")
    elif isinstance(countries, list) and all(
        country in list_supported_countries() for country in countries
    ):
        countryholidays = []
        for i, country in enumerate(countries):
            staging = country_holidays(country=country, years=years)
            if i == 0 and custom_holidays is not None:
                staging.update(custom_holidays)
            countryholidays.extend(staging.keys())
        hols = array(sorted(list(set(countryholidays))), dtype="datetime64[D]")
    else:
        raise Exception(
            "Argument countries must be a string country code or a list "
            "of string country codes according to ISO 3166-1 alpha-2."
        )

    return busdaycalendar(holidays=hols)


def date_fix(d: str | dt.date | dt.datetime | datetime64 | Timestamp) -> dt.date:
    """Function to parse from different date formats into datetime.date

    Parameters
    ----------
    d: str | datetime.date | datetime.datetime | numpy.datetime64 | pandas.Timestamp
        The data item to parse

    Returns
    -------
    datetime.date
        Parsed date
    """

    if isinstance(d, dt.datetime) or isinstance(d, Timestamp):
        return d.date()
    elif isinstance(d, dt.date):
        return d
    elif isinstance(d, datetime64):
        return to_datetime(str(d)).date()
    elif isinstance(d, str):
        return dt.datetime.strptime(d, "%Y-%m-%d").date()
    else:
        raise Exception(
            f"Unknown date format {str(d)} of type {str(type(d))} encountered"
        )


def date_offset_foll(
    raw_date: str | dt.date | dt.datetime | datetime64 | Timestamp,
    months_offset: int = 12,
    adjust: bool = False,
    following: bool = True,
    countries: str | list = "SE",
    custom_holidays: list | dict | None = None,
) -> dt.date:
    """Function to offset dates according to a given calendar

    Parameters
    ----------
    raw_date: str | datetime.date | datetime.datetime | numpy.datetime64 |
    pandas.Timestamp
        The date to offset from
    months_offset: int, default: 12
        Number of months as integer
    adjust: bool, default: False
        Determines if offset should adjust for business days
    following: bool, default: True
        Determines if days should be offset forward (following) or backward
    countries: list | str, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: list | dict | None, default: None
        Argument where missing holidays can be added as
        {"2023-02-12": "Jack's birthday"} or ["2023-02-12"]

    Returns
    -------
    datetime.date
        Off-set date
    """

    raw_date = date_fix(raw_date)
    month_delta = relativedelta(months=months_offset)

    if following:
        day_delta = relativedelta(days=1)
    else:
        day_delta = relativedelta(days=-1)

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
    today: dt.date | None = None,
    countries: str | list = "SE",
    custom_holidays: list | dict | None = None,
):
    """Function to bump backwards to find the previous business day before today

    Parameters
    ----------
    today: datetime.date, optional
        Manual input of the day from where the previous business day is found
    countries: list | str, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: list | dict | None, default: None
        Argument where missing holidays can be added as
        {"2023-02-12": "Jack's birthday"} or ["2023-02-12"]

    Returns
    -------
    datetime.date
        The previous business day
    """

    if today is None:
        today = dt.date.today()

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
    countries: list | str = "SE",
    custom_holidays: list | dict | None = None,
) -> dt.date:
    """Function to bump a date by business days instead of calendar days

    Parameters
    ----------
    ddate: datetime.date
        Manual input of the day from where the previous business day is found
    days: int
        The number of business days to offset the original date
    countries: list | str, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: list | dict | None, default: None
        Argument where missing holidays can be added as
        {"2023-02-12": "Jack's birthday"} or ["2023-02-12"]

    Returns
    -------
    datetime.date
        The new offset business day
    """
    ndate = ddate + dt.timedelta(days=days)
    if days < 0:
        calendar = holiday_calendar(
            startyear=ndate.year,
            endyear=ddate.year,
            countries=countries,
            custom_holidays=custom_holidays,
        )
        local_bdays = date_range(
            periods=abs(days + 10), end=ddate, freq=CustomBusinessDay(calendar=calendar)
        ).date
    else:
        calendar = holiday_calendar(
            startyear=ddate.year,
            endyear=ndate.year,
            countries=countries,
            custom_holidays=custom_holidays,
        )
        local_bdays = date_range(
            start=ddate, periods=days + 10, freq=CustomBusinessDay(calendar=calendar)
        ).date

    while ddate not in local_bdays:
        if days < 0:
            ddate -= dt.timedelta(days=1)
        else:
            ddate += dt.timedelta(days=1)

    idx = where(local_bdays == ddate)[0]

    return local_bdays[idx + days][0]
