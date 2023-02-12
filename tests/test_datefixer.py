import datetime as dt
from numpy import datetime64
from pandas import Timestamp
from typing import TypeVar
from unittest import TestCase

from openseries.datefixer import (
    date_fix,
    date_offset_foll,
    get_previous_business_day_before_today,
    holiday_calendar,
    offset_business_days,
)

TTestDateFixer = TypeVar("TTestDateFixer", bound="TestDateFixer")


class TestDateFixer(TestCase):
    def test_date_fix_arg_types(self: TTestDateFixer):
        formats = [
            "2022-07-15",
            dt.date(year=2022, month=7, day=15),
            dt.datetime(year=2022, month=7, day=15),
            Timestamp(year=2022, month=7, day=15),
            datetime64("2022-07-15"),
        ]

        output = dt.date(2022, 7, 15)

        for fmt in formats:
            self.assertEqual(output, date_fix(fmt))

    def test_date_fix_arg_type_error(self: TTestDateFixer):
        digit: int = 3

        with self.assertRaises(Exception) as e_type:
            # noinspection PyTypeChecker
            _ = date_fix(digit)

        self.assertIsInstance(e_type.exception, Exception)

        nonsense = "abcdef"

        with self.assertRaises(Exception) as e_nonsense:
            _ = date_fix(nonsense)

        self.assertIsInstance(e_nonsense.exception, Exception)

    def test_get_previous_business_day_before_today(self: TTestDateFixer):
        day_after_se_nationalday = dt.date(2022, 6, 7)
        se_dte_swehol = get_previous_business_day_before_today(
            today=day_after_se_nationalday, countries="SE"
        )
        us_dte_swehol = get_previous_business_day_before_today(
            today=day_after_se_nationalday, countries="US"
        )
        se_us_dte_swehol = get_previous_business_day_before_today(
            today=day_after_se_nationalday, countries=["SE", "US"]
        )
        self.assertEqual(dt.date(2022, 6, 3), se_dte_swehol)
        self.assertEqual(dt.date(2022, 6, 6), us_dte_swehol)
        self.assertEqual(dt.date(2022, 6, 3), se_us_dte_swehol)

        day_after_us_independenceday = dt.date(2022, 7, 5)
        se_dte_usahol = get_previous_business_day_before_today(
            today=day_after_us_independenceday, countries="SE"
        )
        us_dte_usahol = get_previous_business_day_before_today(
            today=day_after_us_independenceday, countries="US"
        )
        se_us_dte_usahol = get_previous_business_day_before_today(
            today=day_after_us_independenceday, countries=["SE", "US"]
        )
        self.assertEqual(dt.date(2022, 7, 4), se_dte_usahol)
        self.assertEqual(dt.date(2022, 7, 1), us_dte_usahol)
        self.assertEqual(dt.date(2022, 7, 1), se_us_dte_usahol)

        self.assertEqual(
            get_previous_business_day_before_today(countries=["SE", "US"]),
            date_offset_foll(
                raw_date=dt.date.today() - dt.timedelta(days=1),
                months_offset=0,
                countries=["SE", "US"],
                adjust=True,
                following=False,
            ),
        )

    def test_date_offset_foll(self: TTestDateFixer):
        originals = [dt.date(2022, 6, 5), dt.date(2022, 7, 3)]
        country_sets = ["SE", "US", ["SE", "US"]]
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
            dt.date(originals[0].year, originals[0].month + 1, originals[0].day), offset
        )

        nonsense = "abcdef"

        with self.assertRaises(ValueError) as e_date:
            _ = date_offset_foll(
                raw_date=nonsense,
            )

        self.assertIsInstance(e_date.exception, ValueError)

        with self.assertRaises(Exception) as e_country:
            _ = date_offset_foll(
                raw_date=dt.date.today(),
                adjust=True,
                countries="ZZ",
            )

        self.assertIn(
            member="Argument countries must be a string country code",
            container=e_country.exception.args[0],
        )

    def test_holiday_calendar(self: TTestDateFixer):
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
        for st, en in zip([2023, 2024], [2023, 2022]):
            cdr = holiday_calendar(startyear=st, endyear=en, countries="SE")
            check = all(
                date_str in cdr.holidays for date_str in twentytwentythreeholidays
            )
            self.assertTrue(check)

    def test_holiday_calendar_with_custom_days(self: TTestDateFixer):
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
            date_fix(d) for d in cdr_without.holidays if date_fix(d).year == 2021
        ]
        self.assertListEqual(list1=twentytwentyoneholidays, list2=hols_without)

        jacks_birthday = {"2021-02-12": "Jack's birthday"}
        cdr_with = holiday_calendar(
            startyear=2021, endyear=2021, countries="SE", custom_holidays=jacks_birthday
        )
        hols_with = [date_fix(d) for d in cdr_with.holidays if date_fix(d).year == 2021]

        with self.assertRaises(AssertionError) as e_jack:
            self.assertListEqual(list1=twentytwentyoneholidays, list2=hols_with)
        self.assertIsInstance(e_jack.exception, AssertionError)

        twentytwentyoneholidays.append(date_fix(list(jacks_birthday.keys())[0]))
        twentytwentyoneholidays.sort()

        self.assertListEqual(list1=twentytwentyoneholidays, list2=hols_with)

    def test_offset_business_days(self: TTestDateFixer):
        se_nationalday = dt.date(2022, 6, 6)
        dates = [
            (dt.date(2022, 6, 2), dt.date(2022, 6, 3)),
            (dt.date(2022, 6, 8), dt.date(2022, 6, 7)),
        ]
        offsets = [-1, 1]
        for date, offset in zip(dates, offsets):
            se_offsetdate = offset_business_days(
                ddate=se_nationalday, days=offset, countries="SE"
            )
            self.assertEqual(se_offsetdate, date[0])
            us_offsetdate = offset_business_days(
                ddate=se_nationalday, days=offset, countries="US"
            )
            self.assertEqual(us_offsetdate, date[1])

    def test_offset_business_days_calender_options(self: TTestDateFixer):
        day_after_se_nationalday = dt.date(2022, 6, 7)
        se_enddate = offset_business_days(
            ddate=day_after_se_nationalday, days=-2, countries="SE"
        )
        self.assertEqual(dt.date(2022, 6, 2), se_enddate)
        us_enddate = offset_business_days(
            ddate=day_after_se_nationalday, days=-2, countries="US"
        )
        self.assertEqual(dt.date(2022, 6, 3), us_enddate)
        se_us_enddate = offset_business_days(
            ddate=day_after_se_nationalday, days=-2, countries=["SE", "US"]
        )
        self.assertEqual(dt.date(2022, 6, 2), se_us_enddate)

        day_after_us_independenceday = dt.date(2022, 7, 5)
        se_nddate = offset_business_days(
            ddate=day_after_us_independenceday, days=-2, countries="SE"
        )
        self.assertEqual(dt.date(2022, 7, 1), se_nddate)
        us_nddate = offset_business_days(
            ddate=day_after_us_independenceday, days=-2, countries="US"
        )
        self.assertEqual(dt.date(2022, 6, 30), us_nddate)
        se_us_nddate = offset_business_days(
            ddate=day_after_us_independenceday, days=-2, countries=["SE", "US"]
        )
        self.assertEqual(dt.date(2022, 6, 30), se_us_nddate)

    def test_offset_business_days_with_custom_days(self: TTestDateFixer):
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
