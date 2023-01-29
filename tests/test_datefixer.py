from datetime import date, datetime, timedelta
from numpy import datetime64
from pandas import Timestamp
from typing import TypeVar
from unittest import TestCase

from openseries.datefixer import (
    date_fix,
    get_previous_business_day_before_today,
    date_offset_foll,
)

TTestDateFixer = TypeVar("TTestDateFixer", bound="TestDateFixer")


class TestDateFixer(TestCase):
    def test_date_fix_arg_types(self: TTestDateFixer):
        formats = [
            "2022-07-15",
            date(year=2022, month=7, day=15),
            datetime(year=2022, month=7, day=15),
            Timestamp(year=2022, month=7, day=15),
            datetime64("2022-07-15"),
        ]

        output = date(2022, 7, 15)

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
        today = date(2022, 6, 7)

        dte = get_previous_business_day_before_today(today=today)

        self.assertEqual(date(2022, 6, 3), dte)

        self.assertEqual(
            get_previous_business_day_before_today(),
            date_offset_foll(
                raw_date=date.today() - timedelta(days=1),
                months_offset=0,
                adjust=True,
                following=False,
            ),
        )

    def test_date_offset_foll(self: TTestDateFixer):
        original = date(2022, 6, 5)

        earlier = date_offset_foll(
            raw_date=original,
            months_offset=0,
            adjust=True,
            following=False,
        )
        later = date_offset_foll(
            raw_date=original,
            months_offset=0,
            adjust=True,
            following=True,
        )

        self.assertEqual(date(2022, 6, 3), earlier)
        self.assertEqual(date(2022, 6, 7), later)

        static = date_offset_foll(
            raw_date=original,
            months_offset=0,
            adjust=False,
        )
        self.assertEqual(original, static)

        offset = date_offset_foll(
            raw_date=original,
            months_offset=1,
            adjust=False,
        )
        self.assertEqual(date(original.year, original.month + 1, original.day), offset)

        nonsense = "abcdef"

        with self.assertRaises(ValueError) as e_date:
            _ = date_offset_foll(
                raw_date=nonsense,
            )

        self.assertIsInstance(e_date.exception, ValueError)
