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
            "2023-01-06",
            "2023-04-07",
            "2023-04-10",
            "2023-05-01",
            "2023-05-18",
            "2023-06-06",
            "2023-06-23",
            "2023-12-25",
            "2023-12-26",
        ]
        for st, en in zip([2023, 2024], [2023, 2022]):
            cdr = holiday_calendar(startyear=st, endyear=en, countries="SE")
            hols = [str(hol) for hol in cdr.holidays]
            check = all(date_str in hols for date_str in twentytwentythreeholidays)
            self.assertTrue(check)
