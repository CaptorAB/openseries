"""Test suite for the openseries/datefixer.py module.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, cast

import pytest
from numpy import datetime64
from pandas import Timestamp

from openseries.datefixer import (
    date_fix,
    date_offset_foll,
    generate_calendar_date_range,
    get_previous_business_day_before_today,
    holiday_calendar,
    market_holidays,
    offset_business_days,
)
from openseries.owntypes import (
    BothStartAndEndError,
    CountriesNotStringNorListStrError,
    MarketsNotStringNorListStrError,
    TradingDaysNotAboveZeroError,
)
from openseries.series import OpenTimeSeries

if TYPE_CHECKING:
    from openseries.owntypes import CountriesType, DateType


class DatefixerTestError(Exception):
    """Custom exception used for signaling test failures."""


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    ("date", "offset", "country"),
    [
        (dt.date(2022, 6, 2), -1, "SE"),
        (dt.date(2022, 6, 3), 0, "SE"),
        (dt.date(2022, 6, 8), 1, "SE"),
        (dt.date(2022, 6, 8), "1", "SE"),
        (dt.date(2022, 6, 3), None, "SE"),
        (dt.date(2022, 6, 3), -1, "US"),
        (dt.date(2022, 6, 6), 0, "US"),
        (dt.date(2022, 6, 7), 1, "US"),
        (dt.date(2022, 6, 7), "1", "US"),
        (dt.date(2022, 6, 6), None, "US"),
    ],
)
def test_offset_business_days(
    date: dt.date,
    offset: int,
    country: CountriesType,
) -> None:
    """Test offset_business_days function."""
    se_nationalday = dt.date(2022, 6, 6)
    offsetdate = offset_business_days(
        ddate=se_nationalday,
        days=offset,
        countries=country,
    )
    if offsetdate != date:
        msg = f"Unintended result from offset_business_days {offsetdate}"
        raise DatefixerTestError(msg)


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "fixerdate",
    [
        "2022-07-15",
        dt.date(year=2022, month=7, day=15),
        dt.datetime(year=2022, month=7, day=15, tzinfo=dt.timezone.utc),
        Timestamp(year=2022, month=7, day=15),
        datetime64("2022-07-15"),
    ],
)
def test_date_fix(fixerdate: DateType) -> None:
    """Test date_fix argument types."""
    output = dt.date(2022, 7, 15)
    if output != date_fix(fixerdate=fixerdate):
        msg = (
            f"Unknown date format {fixerdate!s} of"
            f" type {type(fixerdate)!s} encountered"
        )
        raise TypeError(msg)

    int_arg: int = 3
    with pytest.raises(
        expected_exception=TypeError,
        match="Unknown date format 3 of type <class 'int'> encountered",
    ):
        _ = date_fix(fixerdate=cast("str", int_arg))

    str_arg: str = "abcdef"
    with pytest.raises(
        expected_exception=ValueError,
        match=f"time data '{str_arg!s}' does not match format '%Y-%m-%d'",
    ):
        _ = date_fix(fixerdate="abcdef")


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    ("today", "countries", "intention"),
    [
        (dt.date(2022, 6, 7), "SE", dt.date(2022, 6, 3)),
        (dt.date(2022, 6, 7), "US", dt.date(2022, 6, 6)),
        (dt.date(2022, 6, 7), ["SE", "US"], dt.date(2022, 6, 3)),
        (dt.date(2022, 7, 5), "SE", dt.date(2022, 7, 4)),
        (dt.date(2022, 7, 5), "US", dt.date(2022, 7, 1)),
        (dt.date(2022, 7, 5), ["SE", "US"], dt.date(2022, 7, 1)),
    ],
)
def test_get_previous_business_day_before_today(
    today: dt.date,
    countries: CountriesType,
    intention: dt.date,
) -> None:
    """Test get_previous_business_day_before_today function."""
    result = get_previous_business_day_before_today(
        today=today,
        countries=countries,
    )
    if result != intention:
        msg = f"{result} does not equal {intention}"
        raise DatefixerTestError(msg)

    before_today_one = get_previous_business_day_before_today(
        today=None,
        countries=countries,
    )
    before_today_two = date_offset_foll(
        raw_date=dt.datetime.now().astimezone().date() - dt.timedelta(days=1),
        months_offset=0,
        countries=countries,
        adjust=True,
        following=False,
    )
    if before_today_one != before_today_two:
        msg = (
            "Inconsistent results from get_previous_business_day_before_today "
            "and date_offset_foll methods."
        )
        raise DatefixerTestError(msg)


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    ("ddate", "countries", "intention"),
    [
        (dt.date(2022, 6, 7), "SE", dt.date(2022, 6, 2)),
        (dt.date(2022, 6, 7), "US", dt.date(2022, 6, 3)),
        (dt.date(2022, 6, 7), ["SE", "US"], dt.date(2022, 6, 2)),
        (dt.date(2022, 7, 5), "SE", dt.date(2022, 7, 1)),
        (dt.date(2022, 7, 5), "US", dt.date(2022, 6, 30)),
        (dt.date(2022, 7, 5), ["SE", "US"], dt.date(2022, 6, 30)),
    ],
)
def test_offset_business_days_calendar_options(
    ddate: dt.date,
    countries: CountriesType,
    intention: dt.date,
) -> None:
    """Test offset_business_days function with different calendar combinations."""
    result = offset_business_days(
        ddate=ddate,
        days=-2,
        countries=countries,
    )
    if result != intention:
        msg = "Unintended result from offset_business_days"
        raise DatefixerTestError(msg)


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    ("raw_date", "countries", "following", "intention"),
    [
        (dt.date(2022, 6, 5), "SE", False, dt.date(2022, 6, 3)),
        (dt.date(2022, 7, 3), "SE", False, dt.date(2022, 7, 1)),
        (dt.date(2022, 6, 5), "US", False, dt.date(2022, 6, 3)),
        (dt.date(2022, 7, 3), "US", False, dt.date(2022, 7, 1)),
        (dt.date(2022, 6, 5), ["SE", "US"], False, dt.date(2022, 6, 3)),
        (dt.date(2022, 7, 3), ["SE", "US"], False, dt.date(2022, 7, 1)),
        (dt.date(2022, 6, 5), "SE", True, dt.date(2022, 6, 7)),
        (dt.date(2022, 7, 3), "SE", True, dt.date(2022, 7, 4)),
        (dt.date(2022, 6, 5), "US", True, dt.date(2022, 6, 6)),
        (dt.date(2022, 7, 3), "US", True, dt.date(2022, 7, 5)),
        (dt.date(2022, 6, 5), ["SE", "US"], True, dt.date(2022, 6, 7)),
        (dt.date(2022, 7, 3), ["SE", "US"], True, dt.date(2022, 7, 5)),
    ],
)
def test_date_offset_foll(
    raw_date: dt.date,
    countries: CountriesType,
    intention: dt.date,
    *,
    following: bool,
) -> None:
    """Test date_offset_foll function."""
    result = date_offset_foll(
        raw_date=raw_date,
        months_offset=0,
        countries=countries,
        adjust=True,
        following=following,
    )
    if result != intention:
        msg = f"Unintended result from date_offset_foll: {result}"
        raise DatefixerTestError(msg)

    static = date_offset_foll(
        raw_date=raw_date,
        months_offset=0,
        adjust=False,
    )
    if raw_date != static:
        msg = "Unintended result from date_offset_foll"
        raise DatefixerTestError(msg)

    offset = date_offset_foll(
        raw_date=raw_date,
        months_offset=1,
        adjust=False,
    )
    if offset != dt.date(
        raw_date.year,
        raw_date.month + 1,
        raw_date.day,
    ):
        msg = "Unintended result from date_offset_foll"
        raise DatefixerTestError(msg)

    nonsense: str = "abcdef"
    with pytest.raises(
        expected_exception=ValueError,
        match=f"time data '{nonsense!s}' does not match format '%Y-%m-%d'",
    ):
        _ = date_offset_foll(
            raw_date=nonsense,
        )

    with pytest.raises(
        expected_exception=CountriesNotStringNorListStrError,
        match="Argument countries must be a string country code",
    ):
        _ = date_offset_foll(
            raw_date=dt.datetime.now().astimezone().date(),
            adjust=True,
            countries="ZZ",
        )


class TestDateFixer:
    """class to run tests on the module datefixer.py."""

    def test_holiday_calendar(self: TestDateFixer) -> None:
        """Test holiday_calendar function."""
        date = dt.date(2025, 5, 18)
        series = OpenTimeSeries.from_fixed_rate(
            rate=0.02, days=1000, end_dt=date, baseccy="USD"
        )

        msg = "holiday_calendar not working as intended"

        caldays_limit = 360
        if not (series.periods_in_a_year > caldays_limit and series.countries == "SE"):
            raise DatefixerTestError(msg)

        countries = ["ID", "US"]
        series.countries = countries  # type: ignore[assignment]
        series.align_index_to_local_cdays()

        trddays_limit = 260
        if not (
            series.periods_in_a_year < trddays_limit
            and series.countries == set(countries)  # type: ignore[comparison-overlap]
        ):
            raise DatefixerTestError(msg)

        twentytwentythreeholidays = [
            datetime64("2023-01-06"),
            datetime64("2023-04-07"),
            datetime64("2023-04-10"),
            datetime64("2023-05-01"),
            datetime64("2023-05-18"),
            datetime64("2023-06-06"),
            datetime64("2023-06-23"),
            datetime64("2023-12-25"),
            datetime64("2023-12-26"),
        ]

        msg_inv = "holiday_calendar input invalid"
        for start, end in zip([2023, 2024], [2023, 2022], strict=True):
            cdr = holiday_calendar(startyear=start, endyear=end, countries="SE")
            if not all(
                date_str in list(cdr.holidays)
                for date_str in twentytwentythreeholidays
            ):
                raise DatefixerTestError(msg_inv)

    def test_holiday_calendar_with_custom_days(self: TestDateFixer) -> None:
        """Test holiday_calendar with custom input."""
        year = 2021
        twentytwentyoneholidays = [
            dt.date(year, 1, 1),
            dt.date(year, 1, 6),
            dt.date(year, 4, 2),
            dt.date(year, 4, 5),
            dt.date(year, 5, 13),
            dt.date(year, 6, 25),
            dt.date(year, 12, 24),
            dt.date(year, 12, 31),
        ]
        cdr_without = holiday_calendar(startyear=year, endyear=year, countries="SE")
        hols_without = [
            date_fix(d) for d in list(cdr_without.holidays) if date_fix(d).year == year
        ]
        if twentytwentyoneholidays != hols_without:
            msg = "Holidays not matching as intended"
            raise DatefixerTestError(msg)

        jacks_birthday = f"{year}-02-12"
        cdr_with = holiday_calendar(
            startyear=year,
            endyear=year,
            countries="SE",
            custom_holidays=jacks_birthday,
        )
        hols_with = [
            date_fix(d) for d in list(cdr_with.holidays) if date_fix(d).year == year
        ]
        compared = set(twentytwentyoneholidays).symmetric_difference(set(hols_with))
        if len(compared) == 0:
            msg = f"Holidays not the same are: {compared}"
            raise DatefixerTestError(msg)

        twentytwentyoneholidays.append(date_fix(fixerdate=jacks_birthday))
        twentytwentyoneholidays.sort()

        if twentytwentyoneholidays != hols_with:
            msg = "Holidays not matching as intended"
            raise DatefixerTestError(msg)

    def test_market_holidays_as_custom_days(self: TestDateFixer) -> None:
        """Test holiday_calendar with custom input."""
        start = 2025
        end = 2025
        markets = ["NYSE", "LSE"]
        countries = ["US", "GB"]
        exchange_holidays = market_holidays(
            startyear=start, endyear=end, markets=markets
        )

        calendar = holiday_calendar(
            startyear=start,
            endyear=end,
            countries=countries,
            custom_holidays=exchange_holidays,  # type: ignore[arg-type]
        )
        country_holidays = [
            date_fix(fixerdate=date).strftime("%Y-%m-%d") for date in calendar.holidays
        ]

        expected_diff = 16

        msg = "market_holidays() not working as intended"
        if len(country_holidays) - len(exchange_holidays) != expected_diff:
            raise DatefixerTestError(msg)

        exchange_holidays.extend(country_holidays)
        holidays_combined = list(set(exchange_holidays))

        new_expected_diff = 0

        if len(country_holidays) - len(holidays_combined) != new_expected_diff:
            raise DatefixerTestError(msg)

        xmarkets = ["STO", "XSTO"]
        with pytest.raises(
            expected_exception=MarketsNotStringNorListStrError,
            match=(
                "Argument markets must be a string market code or a list "
                "of market codes supported by pandas_market_calendars."
            ),
        ):
            _ = market_holidays(startyear=start, endyear=end, markets=xmarkets)

    def test_offset_business_days_with_custom_days(self: TestDateFixer) -> None:
        """Test offset_business_days function with custom input."""
        day_after_jacks_birthday = dt.date(2021, 2, 15)

        offsetdate_without = offset_business_days(
            ddate=day_after_jacks_birthday,
            days=-2,
            countries=["SE", "US"],
        )
        if offsetdate_without != dt.date(2021, 2, 10):
            msg = "Unintended result from offset_business_days"
            raise DatefixerTestError(msg)

        offsetdate_with = offset_business_days(
            ddate=day_after_jacks_birthday,
            days=-2,
            countries=["SE", "US"],
            custom_holidays="2021-02-12",  # Jack's birthday
        )
        if offsetdate_with != dt.date(2021, 2, 9):
            msg = "Unintended result from offset_business_days"
            raise DatefixerTestError(msg)

    def test_offset_business_days_many_days(self: TestDateFixer) -> None:
        """Test offset_business_days function with many days."""
        startdate = dt.date(2023, 4, 13)
        forward = 2421
        forwarddate = dt.date(2033, 4, 13)
        backward = -forward
        backwarddate = dt.date(2013, 4, 23)

        offsetdate_forward = offset_business_days(
            ddate=startdate,
            days=forward,
            countries=["SE", "US"],
        )
        if offsetdate_forward != forwarddate:
            msg = (
                "Unintended result from offset_business_days: "
                f"{offsetdate_forward.strftime('%Y-%m-%d')}"
            )
            raise DatefixerTestError(msg)

        offsetdate_backward = offset_business_days(
            ddate=startdate,
            days=backward,
            countries=["SE", "US"],
        )
        if offsetdate_backward != backwarddate:
            msg = "Unintended result from offset_business_days"
            raise DatefixerTestError(msg)

    def test_generate_calendar_date_range(self: TestDateFixer) -> None:
        """Test generate_calendar_date_range function with wrong date input."""
        start = dt.date(2009, 6, 30)
        trd_days: int = 506
        end = dt.date(2011, 6, 30)

        d_range = generate_calendar_date_range(
            trading_days=trd_days, start=start, markets=["XSTO"]
        )

        if len(d_range) != trd_days:
            msg = "Unintended result from generate_calendar_date_range"
            raise DatefixerTestError(msg)

        if d_range[-1] != end:
            msg = "Unintended result from generate_calendar_date_range"
            raise DatefixerTestError(msg)

        neg_days = -5
        with pytest.raises(
            expected_exception=TradingDaysNotAboveZeroError,
            match="Argument trading_days must be greater than zero.",
        ):
            _ = generate_calendar_date_range(
                trading_days=neg_days,
                start=start,
            )

        with pytest.raises(
            expected_exception=BothStartAndEndError,
            match=(
                "Provide one of start or end date, but not both. "
                "Date range is inferred from number of trading days."
            ),
        ):
            _ = generate_calendar_date_range(
                trading_days=trd_days,
                start=start,
                end=end,
            )

        with pytest.raises(
            expected_exception=BothStartAndEndError,
            match=(
                "Provide one of start or end date, but not both. "
                "Date range is inferred from number of trading days."
            ),
        ):
            _ = generate_calendar_date_range(trading_days=trd_days)
