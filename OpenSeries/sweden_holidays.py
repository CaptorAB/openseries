# -*- coding: utf-8 -*-
from pandas.tseries.holiday import AbstractHolidayCalendar, DateOffset, Holiday, Day, FR, GoodFriday, \
    EasterMonday, Easter

holidays_sw = [
        Holiday('Nyårsdagen', month=1, day=1),
        Holiday('Trettondag jul', month=1, day=6),
        GoodFriday,
        EasterMonday,
        Holiday('Första maj', month=5, day=1),
        Holiday('Kristi himmelfärd', month=1, day=1, offset=[Easter(), Day(39)]),
        Holiday('Nationaldagen', month=6, day=6),
        Holiday('Midsommarafton', month=6, day=19, offset=DateOffset(weekday=FR(1))),
        Holiday('Julafton', month=12, day=24),
        Holiday('Juldagen', month=12, day=25),
        Holiday('Annandag jul', month=12, day=26),
        Holiday('Nyårsafton', month=12, day=31)
    ]


class SwedenHolidayCalendar(AbstractHolidayCalendar):
    rules = holidays_sw


class CaptorHolidayCalendar(AbstractHolidayCalendar):

    def __init__(self, rules: list):
        super().__init__()
        self.rules = rules
