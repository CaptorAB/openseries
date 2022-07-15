# -*- coding: utf-8 -*-
import datetime as dt
from dateutil.parser import ParserError
from openseries.datefixer import (
    date_fix,
    get_previous_sweden_business_day_before_today,
    date_offset_foll,
)
from openseries.sweden_holidays import SwedenHolidayCalendar, holidays_sw
from pandas.tseries.offsets import CDay
import unittest


class TestDateFixer(unittest.TestCase):
    def test_date_fix_arg_type_error(self):

        digit: int = 3

        with self.assertRaises(ValueError) as e_type:
            # noinspection PyTypeChecker
            _ = date_fix(digit)

        self.assertIsInstance(e_type.exception, ValueError)

    def test_date_fix_parser_error(self):

        nonsense = "abcdef"

        with self.assertRaises(ParserError) as e_nonsense:
            _ = date_fix(nonsense)

        self.assertIsInstance(e_nonsense.exception, ParserError)

    def test_get_previous_sweden_business_day_before_today(self):

        today = dt.date(2022, 6, 7)

        date = get_previous_sweden_business_day_before_today(today=today)

        self.assertEqual(dt.date(2022, 6, 3), date)

        self.assertEqual(
            get_previous_sweden_business_day_before_today(),
            date_offset_foll(
                raw_date=dt.date.today() - dt.timedelta(days=1),
                calendar=CDay(calendar=SwedenHolidayCalendar(rules=holidays_sw)),
                months_offset=0,
                adjust=True,
                following=False,
            ),
        )

    def test_date_offset_foll(self):

        original = dt.date(2022, 6, 5)

        earlier = date_offset_foll(
            raw_date=original,
            calendar=CDay(calendar=SwedenHolidayCalendar(rules=holidays_sw)),
            months_offset=0,
            adjust=True,
            following=False,
        )
        later = date_offset_foll(
            raw_date=original,
            calendar=CDay(calendar=SwedenHolidayCalendar(rules=holidays_sw)),
            months_offset=0,
            adjust=True,
            following=True,
        )

        self.assertEqual(dt.date(2022, 6, 3), earlier)
        self.assertEqual(dt.date(2022, 6, 7), later)

        static = date_offset_foll(
            raw_date=original,
            calendar=CDay(calendar=SwedenHolidayCalendar(rules=holidays_sw)),
            months_offset=0,
            adjust=False,
        )
        self.assertEqual(original, static)

        offset = date_offset_foll(
            raw_date=original,
            calendar=CDay(calendar=SwedenHolidayCalendar(rules=holidays_sw)),
            months_offset=1,
            adjust=False,
        )
        self.assertEqual(
            dt.date(original.year, original.month + 1, original.day), offset
        )

        nonsense = "abcdef"

        with self.assertRaises(ParserError) as e_date:
            _ = date_offset_foll(
                raw_date=nonsense,
                calendar=CDay(calendar=SwedenHolidayCalendar(rules=holidays_sw)),
            )

        self.assertIsInstance(e_date.exception, ParserError)
