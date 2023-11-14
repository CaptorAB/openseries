"""Test suite for the openseries/datefixer.py module."""
from __future__ import annotations

import datetime as dt
from typing import Union, cast
from unittest import TestCase

import pytest
from numpy import datetime64
from pandas import Timestamp

from openseries.datefixer import (
    date_fix,
    date_offset_foll,
    generate_calendar_date_range,
    get_previous_business_day_before_today,
    holiday_calendar,
    offset_business_days,
)
from openseries.types import DateType, HolidayType


class TestDateFixer(TestCase):

    """class to run unittests on the module datefixer.py."""

    def test_arg_types(self: TestDateFixer) -> None:
        """Test date_fix argument types."""
        formats = [
            "2022-07-15",
            dt.date(year=2022, month=7, day=15),
            dt.datetime(year=2022, month=7, day=15, tzinfo=dt.timezone.utc),
            Timestamp(year=2022, month=7, day=15),
            datetime64("2022-07-15"),
        ]

        output = dt.date(2022, 7, 15)

        for fmt in formats:
            if output != date_fix(cast(DateType, fmt)):
                msg = f"Unknown date format {fmt!s} of type {type(fmt)!s} encountered"
                raise TypeError(
                    msg,
                )

    def test_arg_type_error(self: TestDateFixer) -> None:
        """Test date_fix to raise TypeError when appropriate."""
        with pytest.raises(TypeError):
            _ = date_fix(cast(str, 3))

        str_arg: str = "abcdef"
        with pytest.raises(
            ValueError,
            match=f"time data '{str_arg!s}' does not match format '%Y-%m-%d'",
        ):
            _ = date_fix("abcdef")

    def test_get_previous_business_day_before_today(self: TestDateFixer) -> None:
        """Test get_previous_business_day_before_today function."""
        day_after_se_nationalday = dt.date(2022, 6, 7)
        se_dte_swehol = get_previous_business_day_before_today(
            today=day_after_se_nationalday,
            countries="SE",
        )
        us_dte_swehol = get_previous_business_day_before_today(
            today=day_after_se_nationalday,
            countries="US",
        )
        se_us_dte_swehol = get_previous_business_day_before_today(
            today=day_after_se_nationalday,
            countries=["SE", "US"],
        )
        for result, intention in zip(
            [se_dte_swehol, us_dte_swehol, se_us_dte_swehol],
            [dt.date(2022, 6, 3), dt.date(2022, 6, 6), dt.date(2022, 6, 3)],
        ):
            if result != intention:
                msg = f"{result} does not equal {intention}"
                raise ValueError(msg)

        day_after_us_independenceday = dt.date(2022, 7, 5)
        se_dte_usahol = get_previous_business_day_before_today(
            today=day_after_us_independenceday,
            countries="SE",
        )
        us_dte_usahol = get_previous_business_day_before_today(
            today=day_after_us_independenceday,
            countries="US",
        )
        se_us_dte_usahol = get_previous_business_day_before_today(
            today=day_after_us_independenceday,
            countries=["SE", "US"],
        )
        for result, intention in zip(
            [se_dte_usahol, us_dte_usahol, se_us_dte_usahol],
            [dt.date(2022, 7, 4), dt.date(2022, 7, 1), dt.date(2022, 7, 1)],
        ):
            if result != intention:
                msg = f"{result} does not equal {intention}"
                raise ValueError(msg)

        before_today_one = get_previous_business_day_before_today(
            countries=["SE", "US"],
        )
        before_today_two = date_offset_foll(
            raw_date=dt.datetime.now(tz=dt.timezone.utc).date() - dt.timedelta(days=1),
            months_offset=0,
            countries=["SE", "US"],
            adjust=True,
            following=False,
        )
        if before_today_one != before_today_two:
            msg = (
                "Inconsistent results from get_previous_business_day_before_today "
                "and date_offset_foll methods."
            )
            raise ValueError(msg)

    def test_date_offset_foll(self: TestDateFixer) -> None:
        """Test date_offset_foll function."""
        originals = [dt.date(2022, 6, 5), dt.date(2022, 7, 3)]
        country_sets: list[Union[str, list[str]]] = ["SE", "US", ["SE", "US"]]
        earliers = [
            dt.date(2022, 6, 3),
            dt.date(2022, 7, 1),
            dt.date(2022, 6, 3),
            dt.date(2022, 7, 1),
            dt.date(2022, 6, 3),
            dt.date(2022, 7, 1),
        ]
        laters = [
            dt.date(2022, 6, 7),
            dt.date(2022, 7, 4),
            dt.date(2022, 6, 6),
            dt.date(2022, 7, 5),
            dt.date(2022, 6, 7),
            dt.date(2022, 7, 5),
        ]
        counter = 0

        for countries in country_sets:
            for original in originals:
                earlier = date_offset_foll(
                    raw_date=original,
                    months_offset=0,
                    countries=countries,
                    adjust=True,
                    following=False,
                )
                later = date_offset_foll(
                    raw_date=original,
                    months_offset=0,
                    countries=countries,
                    adjust=True,
                    following=True,
                )
                if earliers[counter] != earlier:
                    msg = "Unintended result from date_offset_foll preceeding"
                    raise ValueError(
                        msg,
                    )
                if laters[counter] != later:
                    msg = "Unintended result from date_offset_foll following"
                    raise ValueError(
                        msg,
                    )
                counter += 1

        static = date_offset_foll(
            raw_date=originals[0],
            months_offset=0,
            adjust=False,
        )
        if originals[0] != static:
            msg = "Unintended result from date_offset_foll"
            raise ValueError(msg)

        offset = date_offset_foll(
            raw_date=originals[0],
            months_offset=1,
            adjust=False,
        )
        if offset != dt.date(
            originals[0].year,
            originals[0].month + 1,
            originals[0].day,
        ):
            msg = "Unintended result from date_offset_foll"
            raise ValueError(msg)

        nonsense: str = "abcdef"
        with pytest.raises(
            ValueError,
            match=f"time data '{nonsense!s}' does not match format '%Y-%m-%d'",
        ):
            _ = date_offset_foll(
                raw_date=nonsense,
            )

        with pytest.raises(
            ValueError,
            match="Argument countries must be a string country code",
        ):
            _ = date_offset_foll(
                raw_date=dt.datetime.now(tz=dt.timezone.utc).date(),
                adjust=True,
                countries="ZZ",
            )

    def test_holiday_calendar(self: TestDateFixer) -> None:
        """Test holiday_calendar function."""
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
        for start, end in zip([2023, 2024], [2023, 2022]):
            cdr = holiday_calendar(startyear=start, endyear=end, countries="SE")
            if not all(
                date_str in list(cdr.holidays)
                for date_str in twentytwentythreeholidays
            ):
                msg = "holiday_calendar input invalid"
                raise ValueError(msg)

    def test_holiday_calendar_with_custom_days(self: TestDateFixer) -> None:
        """Test holiday_calendar with custom input."""
        year: int = 2021
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
            raise ValueError(msg)

        jacks_birthday: HolidayType = {
            f"{year}-02-12": "Jack's birthday",
        }
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
            raise ValueError(msg)

        jbirth = cast(dict[str, str], jacks_birthday)
        twentytwentyoneholidays.append(date_fix(next(iter(jbirth.keys()))))
        twentytwentyoneholidays.sort()

        if twentytwentyoneholidays != hols_with:
            msg = "Holidays not matching as intended"
            raise ValueError(msg)

    def test_offset_business_days(self: TestDateFixer) -> None:
        """Test offset_business_days function."""
        se_nationalday = dt.date(2022, 6, 6)
        dates = [
            (dt.date(2022, 6, 2), dt.date(2022, 6, 3)),
            (dt.date(2022, 6, 3), dt.date(2022, 6, 6)),
            (dt.date(2022, 6, 8), dt.date(2022, 6, 7)),
        ]
        offsets = [-1, 0, 1]
        for date, offset in zip(dates, offsets):
            se_offsetdate = offset_business_days(
                ddate=se_nationalday,
                days=offset,
                countries="SE",
            )
            if se_offsetdate != date[0]:
                msg = "Unintended result from offset_business_days"
                raise ValueError(msg)
            us_offsetdate = offset_business_days(
                ddate=se_nationalday,
                days=offset,
                countries="US",
            )
            if us_offsetdate != date[1]:
                msg = "Unintended result from offset_business_days"
                raise ValueError(msg)

    def test_offset_business_days_calendar_options(self: TestDateFixer) -> None:
        """Test offset_business_days function with different calendar combinations."""
        day_after_se_nationalday = dt.date(2022, 6, 7)
        se_enddate = offset_business_days(
            ddate=day_after_se_nationalday,
            days=-2,
            countries="SE",
        )
        if se_enddate != dt.date(2022, 6, 2):
            msg = "Unintended result from offset_business_days"
            raise ValueError(msg)
        us_enddate = offset_business_days(
            ddate=day_after_se_nationalday,
            days=-2,
            countries="US",
        )
        if us_enddate != dt.date(2022, 6, 3):
            msg = "Unintended result from offset_business_days"
            raise ValueError(msg)
        se_us_enddate = offset_business_days(
            ddate=day_after_se_nationalday,
            days=-2,
            countries=["SE", "US"],
        )
        if se_us_enddate != dt.date(2022, 6, 2):
            msg = "Unintended result from offset_business_days"
            raise ValueError(msg)

        day_after_us_independenceday = dt.date(2022, 7, 5)
        se_nddate = offset_business_days(
            ddate=day_after_us_independenceday,
            days=-2,
            countries="SE",
        )
        if se_nddate != dt.date(2022, 7, 1):
            msg = "Unintended result from offset_business_days"
            raise ValueError(msg)
        us_nddate = offset_business_days(
            ddate=day_after_us_independenceday,
            days=-2,
            countries="US",
        )
        if us_nddate != dt.date(2022, 6, 30):
            msg = "Unintended result from offset_business_days"
            raise ValueError(msg)
        se_us_nddate = offset_business_days(
            ddate=day_after_us_independenceday,
            days=-2,
            countries=["SE", "US"],
        )
        if se_us_nddate != dt.date(2022, 6, 30):
            msg = "Unintended result from offset_business_days"
            raise ValueError(msg)

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
            raise ValueError(msg)

        offsetdate_with = offset_business_days(
            ddate=day_after_jacks_birthday,
            days=-2,
            countries=["SE", "US"],
            custom_holidays={"2021-02-12": "Jack's birthday"},
        )
        if offsetdate_with != dt.date(2021, 2, 9):
            msg = "Unintended result from offset_business_days"
            raise ValueError(msg)

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
            msg = "Unintended result from offset_business_days"
            raise ValueError(msg)

        offsetdate_backward = offset_business_days(
            ddate=startdate,
            days=backward,
            countries=["SE", "US"],
        )
        if offsetdate_backward != backwarddate:
            msg = "Unintended result from offset_business_days"
            raise ValueError(msg)

    def test_generate_calendar_date_range(self: TestDateFixer) -> None:
        """Test generate_calendar_date_range function with wrong date input."""
        start = dt.date(2009, 6, 30)
        trd_days: int = 506
        end = dt.date(2011, 6, 30)

        d_range = generate_calendar_date_range(trading_days=trd_days, start=start)

        if len(d_range) != trd_days:
            msg = "Unintended result from generate_calendar_date_range"
            raise ValueError(msg)

        if d_range[-1] != end:
            msg = "Unintended result from generate_calendar_date_range"
            raise ValueError(msg)

        with pytest.raises(
            ValueError,
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
            ValueError,
            match=(
                "Provide one of start or end date, but not both. "
                "Date range is inferred from number of trading days."
            ),
        ):
            _ = generate_calendar_date_range(trading_days=trd_days)
