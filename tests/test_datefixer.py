"""Test suite for the openseries/datefixer.py module."""
from __future__ import annotations

import datetime as dt
from typing import Union, cast
from unittest import TestCase

from numpy import datetime64
from pandas import Timestamp

from openseries.datefixer import (
    date_fix,
    date_offset_foll,
    generate_calender_date_range,
    get_previous_business_day_before_today,
    holiday_calendar,
    offset_business_days,
)
from openseries.types import HolidayType


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
            self.assertEqual(output, date_fix(fmt))

    def test_arg_type_error(self: TestDateFixer) -> None:
        """Test date_fix to raise TypeError when appropriate."""
        with self.assertRaises(TypeError) as e_type:
            digit = cast(str, 3)
            _ = date_fix(digit)

        self.assertIsInstance(e_type.exception, TypeError)

        with self.assertRaises(ValueError) as e_nonsense:
            nonsense = "abcdef"
            _ = date_fix(nonsense)

        self.assertIsInstance(e_nonsense.exception, ValueError)

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
        self.assertEqual(dt.date(2022, 6, 3), se_dte_swehol)
        self.assertEqual(dt.date(2022, 6, 6), us_dte_swehol)
        self.assertEqual(dt.date(2022, 6, 3), se_us_dte_swehol)

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
        self.assertEqual(dt.date(2022, 7, 4), se_dte_usahol)
        self.assertEqual(dt.date(2022, 7, 1), us_dte_usahol)
        self.assertEqual(dt.date(2022, 7, 1), se_us_dte_usahol)

        self.assertEqual(
            get_previous_business_day_before_today(countries=["SE", "US"]),
            date_offset_foll(
                raw_date=dt.datetime.now(tz=dt.timezone.utc).date()
                - dt.timedelta(days=1),
                months_offset=0,
                countries=["SE", "US"],
                adjust=True,
                following=False,
            ),
        )

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
                self.assertEqual(earliers[counter], earlier)
                self.assertEqual(laters[counter], later)
                counter += 1

        static = date_offset_foll(
            raw_date=originals[0],
            months_offset=0,
            adjust=False,
        )
        self.assertEqual(originals[0], static)

        offset = date_offset_foll(
            raw_date=originals[0],
            months_offset=1,
            adjust=False,
        )
        self.assertEqual(
            dt.date(originals[0].year, originals[0].month + 1, originals[0].day),
            offset,
        )

        nonsense = "abcdef"

        with self.assertRaises(ValueError) as e_date:
            _ = date_offset_foll(
                raw_date=nonsense,
            )

        self.assertIsInstance(e_date.exception, ValueError)

        with self.assertRaises(Exception) as e_country:
            _ = date_offset_foll(
                raw_date=dt.datetime.now(tz=dt.timezone.utc).date(),
                adjust=True,
                countries="ZZ",
            )

        self.assertIn(
            member="Argument countries must be a string country code",
            container=e_country.exception.args[0],
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
            check = all(
                date_str in list(cdr.holidays)
                for date_str in twentytwentythreeholidays
            )
            self.assertTrue(check)

    def test_holiday_calendar_with_custom_days(self: TestDateFixer) -> None:
        """Test holiday_calendar with custom input."""
        twentytwentyoneholidays = [
            dt.date(2021, 1, 1),
            dt.date(2021, 1, 6),
            dt.date(2021, 4, 2),
            dt.date(2021, 4, 5),
            dt.date(2021, 5, 13),
            dt.date(2021, 6, 25),
            dt.date(2021, 12, 24),
            dt.date(2021, 12, 31),
        ]
        cdr_without = holiday_calendar(startyear=2021, endyear=2021, countries="SE")
        hols_without = [
            date_fix(d) for d in list(cdr_without.holidays) if date_fix(d).year == 2021
        ]
        self.assertListEqual(list1=twentytwentyoneholidays, list2=hols_without)

        jacks_birthday: HolidayType = {
            "2021-02-12": "Jack's birthday",
        }
        cdr_with = holiday_calendar(
            startyear=2021,
            endyear=2021,
            countries="SE",
            custom_holidays=jacks_birthday,
        )
        hols_with = [
            date_fix(d) for d in list(cdr_with.holidays) if date_fix(d).year == 2021
        ]

        with self.assertRaises(AssertionError) as e_jack:
            self.assertListEqual(list1=twentytwentyoneholidays, list2=hols_with)
        self.assertIsInstance(e_jack.exception, AssertionError)

        jbirth = cast(dict[str, str], jacks_birthday)
        twentytwentyoneholidays.append(date_fix(next(iter(jbirth.keys()))))
        twentytwentyoneholidays.sort()

        self.assertListEqual(list1=twentytwentyoneholidays, list2=hols_with)

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
            self.assertEqual(se_offsetdate, date[0])
            us_offsetdate = offset_business_days(
                ddate=se_nationalday,
                days=offset,
                countries="US",
            )
            self.assertEqual(us_offsetdate, date[1])

    def test_offset_business_days_calender_options(self: TestDateFixer) -> None:
        """Test offset_business_days function with different calendar combinations."""
        day_after_se_nationalday = dt.date(2022, 6, 7)
        se_enddate = offset_business_days(
            ddate=day_after_se_nationalday,
            days=-2,
            countries="SE",
        )
        self.assertEqual(dt.date(2022, 6, 2), se_enddate)
        us_enddate = offset_business_days(
            ddate=day_after_se_nationalday,
            days=-2,
            countries="US",
        )
        self.assertEqual(dt.date(2022, 6, 3), us_enddate)
        se_us_enddate = offset_business_days(
            ddate=day_after_se_nationalday,
            days=-2,
            countries=["SE", "US"],
        )
        self.assertEqual(dt.date(2022, 6, 2), se_us_enddate)

        day_after_us_independenceday = dt.date(2022, 7, 5)
        se_nddate = offset_business_days(
            ddate=day_after_us_independenceday,
            days=-2,
            countries="SE",
        )
        self.assertEqual(dt.date(2022, 7, 1), se_nddate)
        us_nddate = offset_business_days(
            ddate=day_after_us_independenceday,
            days=-2,
            countries="US",
        )
        self.assertEqual(dt.date(2022, 6, 30), us_nddate)
        se_us_nddate = offset_business_days(
            ddate=day_after_us_independenceday,
            days=-2,
            countries=["SE", "US"],
        )
        self.assertEqual(dt.date(2022, 6, 30), se_us_nddate)

    def test_offset_business_days_with_custom_days(self: TestDateFixer) -> None:
        """Test offset_business_days function with custom input."""
        day_after_jacks_birthday = dt.date(2021, 2, 15)

        offsetdate_without = offset_business_days(
            ddate=day_after_jacks_birthday,
            days=-2,
            countries=["SE", "US"],
        )
        self.assertEqual(dt.date(2021, 2, 10), offsetdate_without)

        offsetdate_with = offset_business_days(
            ddate=day_after_jacks_birthday,
            days=-2,
            countries=["SE", "US"],
            custom_holidays={"2021-02-12": "Jack's birthday"},
        )
        self.assertEqual(dt.date(2021, 2, 9), offsetdate_with)

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
        self.assertEqual(offsetdate_forward, forwarddate)

        offsetdate_backward = offset_business_days(
            ddate=startdate,
            days=backward,
            countries=["SE", "US"],
        )
        self.assertEqual(offsetdate_backward, backwarddate)

    def test_generate_calender_date_range(self: TestDateFixer) -> None:
        """Test generate_calender_date_range function with wrong date input."""
        start = dt.date(2009, 6, 30)
        trd_days = 506
        end = dt.date(2011, 6, 30)

        d_range = generate_calender_date_range(trading_days=trd_days, start=start)

        self.assertEqual(len(d_range), 506)
        self.assertEqual(d_range[-1], end)

        with self.assertRaises(ValueError) as e_both:
            _ = generate_calender_date_range(
                trading_days=trd_days,
                start=start,
                end=end,
            )
        self.assertIn(
            member=(
                "Provide one of start or end date, but not both. "
                "Date range is inferred from number of trading days."
            ),
            container=str(e_both.exception),
        )

        with self.assertRaises(ValueError) as e_none:
            _ = generate_calender_date_range(trading_days=trd_days)
        self.assertIn(
            member=(
                "Provide one of start or end date, but not both. "
                "Date range is inferred from number of trading days."
            ),
            container=str(e_none.exception),
        )
