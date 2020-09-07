# -*- coding: utf-8 -*-
import copy
import datetime as dt
import json
import jsonschema
from jsonschema.exceptions import ValidationError
import logging
import math
import numpy as np
import os
import pandas as pd
from pandas.core.series import Series
from pandas.tseries.offsets import CDay
from pathlib import Path
from plotly.offline import plot
import plotly.graph_objs as go
import requests
import scipy.stats as ss
from typing import Union, Tuple, List

from OpenSeries.captor_open_api_sdk import CaptorOpenApiService
from OpenSeries.datefixer import date_offset_foll, date_fix
from OpenSeries.load_plotly import load_plotly_dict
from OpenSeries.risk import cvar_down, var_down, drawdown_series
from OpenSeries.sweden_holidays import CaptorHolidayCalendar, holidays_sw


class OpenTimeSeries(object):

    _id: str  # Captor database identifier for the timeseries
    instrumentId: str  # Captor database identifier for the instrument associated with the timeseries
    currency: str  # Currency of the timeseries. Only used if conversion/hedging methods are added.
    dates: List[str]  # Dates of the timeseries. Not edited by any method to allow reversion to original.
    domestic: str  # Domestic currency of the user / investor. Only used if conversion/hedging methods are added.
    name: str  # An identifier field.
    isin: str  # ISIN code of the associated instrument. If any.
    label: str  # Field used in outputs.
    schema: dict  # Jsonschema to validate against in the __init__ method.
    sweden: CaptorHolidayCalendar  # A calendar object used to generate business days.
    valuetype: str  # "Price(Close)" if a series of values and "Return(Total)" if a series of returns.
    values: List[float]  # Values of the timeseries. Not edited by any method to allow reversion to original.
    local_ccy: bool  # Indicates if series should be in its local currency or the domestic currency of the user.
    tsdf: pd.DataFrame  # The Pandas DataFrame which gets edited by the class methods.

    @classmethod
    def setup_class(cls):

        cls.domestic = 'SEK'
        cls.sweden = CaptorHolidayCalendar(holidays_sw)

    def __init__(self, d):

        schema_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openseries.json')
        with open(file=schema_file, mode='r', encoding='utf-8') as f:
            series_schema = json.load(f)

        try:
            jsonschema.validate(instance=d, schema=series_schema)
        except ValidationError as e:
            raise Exception(d.get('_id', None), d.get('name', None), e)

        self.__dict__ = d

        if self.name != '':
            self.label = self.name

        self.pandas_df()

    def __repr__(self):

        return '{}(label={}, _id={}, valuetype={}, currency= {}, start={}, end={})' \
            .format(self.__class__.__name__,
                    self.label,
                    self._id,
                    self.valuetype,
                    self.currency,
                    self.first_idx.strftime('%Y-%m-%d'),
                    self.last_idx.strftime('%Y-%m-%d'))

    def __str__(self):

        return '{}(label={}, _id={}, valuetype={}, currency= {}, start={}, end={})' \
            .format(self.__class__.__name__,
                    self.label,
                    self._id,
                    self.valuetype,
                    self.currency,
                    self.first_idx.strftime('%Y-%m-%d'),
                    self.last_idx.strftime('%Y-%m-%d'))

    @classmethod
    def from_open_api(cls, timeseries_id: str, label: str = 'series', baseccy: str = 'SEK', local_ccy: bool = True):

        captor = CaptorOpenApiService()
        data = captor.get_timeseries(timeseries_id)

        output = {'_id': data['id'],
                  'name': label,
                  'currency': baseccy,
                  'instrumentId': '',
                  'isin': '',
                  'local_ccy': local_ccy,
                  'valuetype': data['type'],
                  'dates': data['dates'],
                  'values': [float(val) for val in data['values']]}

        return cls(d=output)

    @classmethod
    def from_open_nav(cls, isin: str, valuetype: str = 'Price(Close)', local_ccy: bool = True):

        captor = CaptorOpenApiService()
        data = captor.get_nav(isin=isin)

        output = {'_id': data['_id'],
                  'name': data['longName'],
                  'currency': data['currency'],
                  'instrumentId': '',
                  'isin': isin,
                  'local_ccy': local_ccy,
                  'valuetype': valuetype,
                  'dates': data['dates'],
                  'values': [float(val) for val in data['navPerUnit']]}

        return cls(d=output)

    @classmethod
    def from_open_fundinfo(cls, isin: str, report_date: dt.date = None, valuetype: str = 'Price(Close)',
                           local_ccy: bool = True):

        captor = CaptorOpenApiService()
        data = captor.get_fundinfo(isins=[isin], report_date=report_date)

        fundinfo = data[0]['classes'][0]

        if isin != fundinfo['isin']:
            raise Exception('Method OpenTimeSeries.from_open_fundinfo() returned the wrong isin.')

        output = {'_id': '',
                  'name': fundinfo['name'],
                  'currency': fundinfo['navCurrency'],
                  'instrumentId': '',
                  'isin': fundinfo['isin'],
                  'local_ccy': local_ccy,
                  'valuetype': valuetype,
                  'dates': fundinfo['returnTimeSeries']['dates'],
                  'values': [float(val) for val in fundinfo['returnTimeSeries']['values']]}

        return cls(d=output)

    @classmethod
    def from_df(cls, df: Union[pd.DataFrame, pd.Series], column_nmbr: int = 0, valuetype: str = 'Price(Close)',
                baseccy: str = 'SEK', local_ccy: bool = True):

        if isinstance(df, Series):
            if isinstance(df.name, tuple):
                label, _ = df.name
            else:
                label = df.name
            values = df.values.tolist()
        else:
            values = df.iloc[:, column_nmbr].tolist()
            if isinstance(df.columns, pd.MultiIndex):
                label = df.columns.get_level_values(0).values[column_nmbr]
                valuetype = df.columns.get_level_values(1).values[column_nmbr]
            else:
                label = df.columns.values[column_nmbr]
        dates = [date_fix(d).strftime('%Y-%m-%d') for d in df.index]
        output = {
            '_id': '',
            'currency': baseccy,
            'instrumentId': '',
            'isin': '',
            'local_ccy': local_ccy,
            'name': label,
            'valuetype': valuetype,
            'dates': dates,
            'values': values}

        return cls(d=output)

    @classmethod
    def from_frame(cls, frame, label: str, valuetype: str = 'Price(Close)',
                   baseccy: str = 'SEK', local_ccy: bool = True):

        df = frame.tsdf.loc[:, (label, valuetype)]
        dates = [date_fix(d).strftime('%Y-%m-%d') for d in df.index]

        output = {
            '_id': '',
            'currency': baseccy,
            'instrumentId': '',
            'isin': '',
            'local_ccy': local_ccy,
            'name': df.name[0],
            'valuetype': df.name[1],
            'dates': dates,
            'values': df.values.tolist()}

        return cls(d=output)

    @classmethod
    def from_quandl(cls, database_code: str, dataset_code: str, field: Union[int, str] = 1, baseccy: str = 'SEK',
                    local_ccy: bool = True):

        home_dir = str(Path.home())
        apikey_file = os.path.join(home_dir, '.quandl_apikey')
        with open(apikey_file, 'r', encoding='utf-8') as ff:
            api_key = ff.read().rstrip()

        url = f'https://www.quandl.com/api/v3/datasets/{database_code}/{dataset_code}/data.json?api_key={api_key}'

        response = requests.get(url=url, headers={'accept': 'application/json'})

        if response.status_code // 100 != 2:
            raise Exception(f'{response.status_code}, {response.text}')

        data = response.json()['dataset_data']

        if isinstance(field, str):
            idx = data['column_names'].index(field)
        elif isinstance(field, int):
            idx = field
            field = data['column_names'][idx]
        else:
            raise Exception('field must be string or integer')

        dates, values = [], []
        for item in data['data']:
            if item[idx]:
                dates.append(item[0])
                values.append(item[idx])

        output = {
            '_id': '',
            'name': dataset_code,
            'currency': baseccy,
            'local_ccy': local_ccy,
            'instrumentId': '',
            'isin': '',
            'valuetype': field,
            'dates': dates,
            'values': values}

        return cls(d=output)

    def from_deepcopy(self):
        return copy.deepcopy(self)

    @classmethod
    def from_fixed_rate(cls, rate: float, days: int, end_dt: dt.date, label: str = 'Series',
                        valuetype: str = 'Price(Close)', baseccy: str = 'SEK', local_ccy: bool = True):
        """

        :param rate:
        :param days:
        :param end_dt:
        :param label:
        :param valuetype:
        :param baseccy:
        :param local_ccy:
        """
        date_range = pd.date_range(periods=days, end=end_dt, freq='D')
        deltas = np.array([i.days for i in date_range[1:] - date_range[:-1]])
        array = np.cumprod(np.insert(1 + deltas * rate / 365, 0, 1.0)).tolist()
        date_range = [d.strftime('%Y-%m-%d') for d in date_range]

        output = {'_id': '',
                  'name': label,
                  'currency': baseccy,
                  'instrumentId': '',
                  'isin': '',
                  'local_ccy': local_ccy,
                  'valuetype': valuetype,
                  'dates': date_range,
                  'values': array}

        return cls(d=output)

    def to_json(self, filename: str, directory: str = None) -> dict:

        if not directory:
            directory = os.path.dirname(os.path.abspath(__file__))

        data = self.__dict__

        cleaner_list = ['label', 'tsdf']
        for item in cleaner_list:
            data.pop(item)

        with open(os.path.join(directory, filename), 'w') as ff:
            json.dump(data, ff, indent=2, sort_keys=False)

        return data

    def pandas_df(self):

        df = pd.DataFrame(data=self.values, index=self.dates, dtype='float64')
        df.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        df.index = pd.DatetimeIndex(df.index)
        df.sort_index(inplace=True)

        if any(df.index.duplicated()):
            duplicates = df.loc[df.loc[df.index.duplicated()].index]
            logging.warning(f'\nData used to create {type(self).__name__} contains duplicate(s).\n {duplicates}'
                            f'\nKeeping the last data point of each duplicate.')
            df = df[~df.index.duplicated(keep='last')]

        self.tsdf = df

        return self

    def validate_vs_schema(self):

        cleaned_dict = self.__dict__

        extra_keys = ['api', 'tsdf', 'local_ccy']

        for kay in extra_keys:
            try:
                del cleaned_dict[kay]
            except KeyError:
                raise Exception(f'Key {kay} not present in self.__dict__')

        jsonschema.validate(cleaned_dict, self.schema)

    def calc_range(self, months_offset: int = None, from_dt: Union[dt.date, None] = None,
                   to_dt: Union[dt.date, None] = None) -> Tuple[dt.date, dt.date]:
        """
        Function to create user defined time frame.

        :param months_offset: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_dt: Specific from date
        :param to_dt: Specific to date
        """
        self.setup_class()
        if months_offset is not None or from_dt is not None or to_dt is not None:
            if months_offset is not None:
                earlier = date_offset_foll(self.last_idx, calendar=CDay(calendar=self.sweden),
                                           months_offset=-months_offset)
                assert earlier >= self.first_idx, 'Function calc_range returned earlier date < series start'
                later = self.last_idx
            else:
                if from_dt is not None and to_dt is None:
                    assert from_dt >= self.first_idx, 'Function calc_range returned earlier date < series start'
                    earlier, later = from_dt, self.last_idx
                elif from_dt is None and to_dt is not None:
                    assert to_dt <= self.last_idx, 'Function calc_range returned later date > series end'
                    earlier, later = self.first_idx, to_dt
                elif from_dt is not None and to_dt is not None:
                    assert to_dt <= self.last_idx and \
                           from_dt >= self.first_idx, 'Function calc_range returned dates outside series range'
                    earlier, later = from_dt, to_dt
                else:
                    earlier, later = from_dt, to_dt

            earlier = date_fix(earlier)
            later = date_fix(later)

            while not self.tsdf.index.isin([earlier]).any():
                earlier -= dt.timedelta(days=1)

            while not self.tsdf.index.isin([later]).any():
                later += dt.timedelta(days=1)

        else:
            earlier, later = self.first_idx, self.last_idx

        return earlier, later

    def align_index_to_local_cdays(self):
        """
        Changes the index of the associated pd.DataFrame tsdf to align with local calendar business days.
        """
        self.setup_class()
        date_range = pd.date_range(start=self.tsdf.first_valid_index(),
                                   end=self.tsdf.last_valid_index(), freq=CDay(calendar=self.sweden))

        self.tsdf = self.tsdf.reindex(date_range, method='pad', copy=False)

        return self

    def all_properties(self, properties: list = None) -> pd.DataFrame:

        if not properties:
            properties = ['value_ret', 'geo_ret', 'arithmetic_ret', 'twr_ret', 'vol', 'ret_vol_ratio', 'z_score',
                          'skew', 'kurtosis', 'positive_share', 'var_down', 'cvar_down', 'vol_from_var', 'worst',
                          'worst_month', 'max_drawdown_cal_year', 'max_drawdown', 'max_drawdown_date', 'first_idx',
                          'last_idx', 'length', 'yearfrac', 'periods_in_a_year']

        pdf = pd.DataFrame.from_dict({x: getattr(self, x) for x in properties}, orient='index')

        pdf.columns = self.tsdf.columns

        return pdf

    @property
    def length(self) -> int:

        return len(self.tsdf.index)

    @property
    def first_idx(self) -> dt.date:

        return self.tsdf.first_valid_index().date()

    @property
    def last_idx(self) -> dt.date:

        return self.tsdf.last_valid_index().date()

    @property
    def yearfrac(self) -> float:
        """
        Length of timeseries expressed as np.float64 fraction of a year with 365.25 days.
        """
        return (self.last_idx - self.first_idx).days / 365.25

    @property
    def periods_in_a_year(self) -> float:
        """
        The number of observations in an average year for all days in the data.
        """
        return self.length / self.yearfrac

    @property
    def geo_ret(self) -> float:
        """
        Geometric annualized return.
        """
        if float(self.tsdf.loc[self.first_idx]) == 0.0:
            raise Exception('First data point == 0.0')
        return float((self.tsdf.loc[self.last_idx] / self.tsdf.loc[self.first_idx]) ** (1 / self.yearfrac) - 1)

    def geo_ret_func(self, months_from_last: int = None, from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        Geometric annualized return.

        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        fraction = (later - earlier).days / 365.25
        if float(self.tsdf.loc[earlier]) == 0.0:
            raise Exception('First data point == 0.0')
        return float((self.tsdf.loc[later] / self.tsdf.loc[earlier]) ** (1 / fraction) - 1)

    @property
    def arithmetic_ret(self) -> float:
        """
        Arithmetic annualized log return.
        """
        return float(np.log(self.tsdf).diff().mean() * self.periods_in_a_year)

    def arithmetic_ret_func(self, months_from_last: int = None, from_date: dt.date = None, to_date: dt.date = None,
                            periods_in_a_year_fixed: int = None) -> float:
        """
        Arithmetic annualized log return.

        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        :param periods_in_a_year_fixed:
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[earlier:later].count(numeric_only=True)
            time_factor = how_many / fraction
        return float(np.log(self.tsdf.loc[earlier:later]).diff().mean() * time_factor)

    @property
    def twr_ret(self) -> float:
        """
        Annualized time weighted return.
        """
        if float(self.tsdf.iloc[0]) == 0.0:
            raise Exception('First data point == 0.0')
        return float(((self.tsdf.iloc[-1] / self.tsdf.iloc[0]) ** (1 / self.length) - 1) * self.periods_in_a_year)

    def twr_ret_func(self, months_from_last: int = None, from_date: dt.date = None, to_date: dt.date = None,
                     periods_in_a_year_fixed: int = None) -> float:
        """
        Annualized time weighted return.

        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        :param periods_in_a_year_fixed:
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        how_many = self.tsdf.loc[earlier:later].count(numeric_only=True)
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            time_factor = how_many / fraction
        if float(self.tsdf.loc[earlier]) == 0.0:
            raise Exception('First data point == 0.0')
        return float(((self.tsdf.loc[later] / self.tsdf.loc[earlier]) ** (1 / how_many) - 1) * time_factor)

    @property
    def value_ret(self) -> float:
        """
        Simple return from first to last observation.
        """
        if float(self.tsdf.iloc[0]) == 0.0:
            raise Exception('First data point == 0.0')
        return float(self.tsdf.iloc[-1] / self.tsdf.iloc[0] - 1)

    def value_ret_func(self, logret: bool = False, months_from_last: int = None,
                       from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        Simple return

        :param logret: Boolean set to True for log return and False for simple return.
        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if float(self.tsdf.loc[earlier]) == 0.0:
            raise Exception('First data point == 0.0')
        if logret:
            ret = np.log(self.tsdf.loc[later] / self.tsdf.loc[earlier])
        else:
            ret = self.tsdf.loc[later] / self.tsdf.loc[earlier] - 1
        return float(ret)

    def value_ret_calendar_period(self, year: int, month: int = None) -> float:
        """
        Function to calculate simple return for a specific calendar period.

        :param year: Year of the period to calculate.
        :param month: Optional month of the period to calculate.
        """
        if month is None:
            period = str(year)
        else:
            period = '-'.join([str(year), str(month).zfill(2)])
        rtn = self.tsdf.copy().pct_change()
        rtn = rtn.loc[period] + 1
        return float(rtn.apply(np.cumprod, axis='index').iloc[-1] - 1)

    @property
    def vol(self) -> float:
        """
        Annualized volatility. Pandas .std() is the equivalent of stdev.s([...]) in MS excel.
        """
        return float(self.tsdf.pct_change().std() * np.sqrt(self.periods_in_a_year))

    def vol_func(self, months_from_last: int = None, from_date: dt.date = None, to_date: dt.date = None,
                 periods_in_a_year_fixed: int = None) -> float:
        """
        Annualized volatility.

        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        :param periods_in_a_year_fixed:
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[earlier:later].count(numeric_only=True)
            time_factor = how_many / fraction
        return float(self.tsdf.loc[earlier:later].pct_change().std() * np.sqrt(time_factor))

    @property
    def ret_vol_ratio(self) -> float:
        """
        Ratio of geometric return and annualized volatility.
        """
        return self.geo_ret / self.vol

    def ret_vol_ratio_func(self, months_from_last: int = None,
                           from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        Ratio of geometric return and annualized volatility.
        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        return self.geo_ret_func(months_from_last, from_date, to_date) / self.vol_func(months_from_last, from_date,
                                                                                       to_date)

    @property
    def z_score(self) -> float:
        """
        Z-score as (last return - mean return) / standard deviation of returns.
        """
        return float((self.tsdf.pct_change().iloc[-1] - self.tsdf.pct_change().mean()) / self.tsdf.pct_change().std())

    def z_score_func(self, months_from_last: int = None, from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        Z-score as (last return - mean return) / standard deviation of returns.
        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        part = self.tsdf.loc[earlier:later].pct_change().copy()
        return float((part.iloc[-1] - part.mean()) / part.std())

    @property
    def max_drawdown(self) -> float:
        """
        Max drawdown.
        """
        return float((self.tsdf / self.tsdf.expanding(min_periods=1).max()).min() - 1)

    @property
    def max_drawdown_date(self) -> dt.date:
        """
        Date when the maximum drawdown occurred.
        """
        mdd_date = (self.tsdf / self.tsdf.expanding(min_periods=1).max()).idxmin().values[0].astype(dt.datetime)
        return dt.datetime.fromtimestamp(mdd_date / 1e9).date()

    def max_drawdown_func(self, months_from_last: int = None,
                          from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        Maximum drawdown.

        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float((self.tsdf.loc[earlier:later] /
                      self.tsdf.loc[earlier:later].expanding(min_periods=1).max()).min() - 1)

    @property
    def max_drawdown_cal_year(self) -> float:
        """
        Maximum drawdown in a single calendar year.
        """
        return float(self.tsdf.groupby([self.tsdf.index.year]).apply(
            lambda x: (x / x.expanding(min_periods=1).max()).min() - 1).min())

    @property
    def worst(self) -> float:
        """
        Most negative percentage change.
        """
        return float(self.tsdf.pct_change().min())

    @property
    def worst_month(self) -> float:
        """
        Most negative month.
        """
        return float(self.tsdf.resample('BM').last().pct_change().min())

    def worst_func(self, months_from_last: int = None,
                   from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        Most negative percentage change.

        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float(self.tsdf.loc[earlier:later].pct_change().min())

    @property
    def positive_share(self) -> float:
        """
        The share of percentage changes that are positive.
        """
        pos = self.tsdf.pct_change()[1:][self.tsdf.pct_change()[1:].values > 0.0].count()
        tot = self.tsdf.pct_change()[1:].count()
        return float(pos / tot)

    def positive_share_func(self, months_from_last: int = None,
                            from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        The share of percentage changes that are positive.

        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        period = self.tsdf.loc[earlier:later].copy()
        return float(period[period.pct_change().ge(0.0)].count(numeric_only=True) /
                     period.pct_change().count(numeric_only=True))

    @property
    def skew(self) -> float:
        """
        Skew of the return distribution.
        """
        return float(ss.skew(self.tsdf.pct_change().values, bias=True, nan_policy='omit'))

    def skew_func(self, months_from_last: int = None, from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        Skew of the return distribution.

        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float(ss.skew(self.tsdf.loc[earlier:later].pct_change(), bias=True, nan_policy='omit'))

    @property
    def kurtosis(self) -> float:
        """
        Kurtosis of the return distribution.
        """
        return float(ss.kurtosis(self.tsdf.pct_change(), fisher=True, bias=True, nan_policy='omit'))

    def kurtosis_func(self, months_from_last: int = None, from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        Kurtosis of the return distribution.

        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float(ss.kurtosis(self.tsdf.loc[earlier:later].pct_change(), fisher=True, bias=True, nan_policy='omit'))

    @property
    def cvar_down(self, level: float = 0.95) -> float:
        """
        Downside Conditional Value At Risk, "CVaR".

        :param level: The sought CVaR level as a float
        """
        items = self.tsdf.iloc[:, 0].pct_change().count()
        return self.tsdf.iloc[:, 0].pct_change().sort_values().iloc[:int(math.ceil((1 - level) * items))].mean()

    def cvar_down_func(self, level: float = 0.95, months_from_last: int = None,
                       from_date: dt.date = None, to_date: dt.date = None) -> float:
        """
        Downside Conditional Value At Risk, "CVaR".

        :param level: The sought CVaR level as a float
        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        how_many = self.tsdf.loc[earlier:later, self.tsdf.columns.values[0]].pct_change().count()
        return self.tsdf.loc[earlier:later, self.tsdf.columns.values[0]].pct_change() \
                   .sort_values().iloc[:int(math.ceil((1 - level) * how_many))].mean()

    @property
    def var_down(self, level: float = 0.95, interpolation: str = 'lower') -> float:
        """
        Downside Value At Risk, "VaR". The equivalent of percentile.inc([...], 1-level) over returns in MS Excel.

        :param level: The sought VaR level as a float
        :param interpolation: type of interpolation in quantile function (default value in quantile is linear)
        """
        return float(self.tsdf.pct_change().quantile(1 - level, interpolation=interpolation))

    def var_down_func(self, level: float = 0.95, months_from_last: int = None,
                      from_date: dt.date = None, to_date: dt.date = None, interpolation: str = 'lower') -> float:
        """
        Downside Value At Risk, "VaR". The equivalent of percentile.inc([...], 1-level) over returns in MS Excel.

        :param level: The sought VaR level as a float
        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        :param interpolation: type of interpolation in quantile function (default value in quantile is linear)
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float(self.tsdf.loc[earlier:later].pct_change().quantile(q=1 - level, interpolation=interpolation))

    @property
    def vol_from_var(self, level: float = 0.95, interpolation: str = 'lower') -> float:
        """
        Implied annualized volatility from the Downside VaR using the assumption that returns
        are normally distributed.

        :param level: The VaR level as a float
        :param interpolation: type of interpolation in quantile function (default value in quantile is linear)
        """
        return float(-np.sqrt(self.periods_in_a_year) *
                     self.var_down_func(level, interpolation=interpolation) / ss.norm.ppf(level))

    def vol_from_var_func(self, level: float = 0.95, months_from_last: int = None,
                          from_date: dt.date = None, to_date: dt.date = None, interpolation: str = 'lower',
                          drift_adjust: bool = False, periods_in_a_year_fixed: int = None) -> float:
        """
        Implied annualized volatility from the Downside VaR using the assumption that returns
        are normally distributed.

        :param level: The VaR level as a float
        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        :param interpolation: type of interpolation in quantile function (default value in quantile is linear)
        :param drift_adjust:
        :param periods_in_a_year_fixed:
        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[earlier:later].count(numeric_only=True)
            time_factor = how_many / fraction
        if drift_adjust:
            return float((-np.sqrt(time_factor) / ss.norm.ppf(level)) *
                         (self.var_down_func(level, months_from_last, from_date, to_date, interpolation) -
                          self.tsdf.loc[earlier:later].pct_change().sum() /
                          len(self.tsdf.loc[earlier:later].pct_change())))
        else:
            return float(-np.sqrt(time_factor) *
                         self.var_down_func(level, months_from_last, from_date, to_date,
                                            interpolation) / ss.norm.ppf(level))

    def target_weight_from_var(self, target_vol: float = 0.175, min_leverage_local: float = 0.0,
                               max_leverage_local: float = 99999.0, level: float = 0.95, months_from_last: int = None,
                               from_date: dt.date = None, to_date: dt.date = None, interpolation: str = 'lower',
                               drift_adjust: bool = False, periods_in_a_year_fixed: int = None) -> float:
        """
        A position target weight from the ratio between a VaR implied volatility and a given target volatility.
        :param target_vol:
        :param min_leverage_local:
        :param max_leverage_local:
        :param level: The VaR level as a float
        :param months_from_last: number of months offset as positive integer. Overrides use of from_date and to_date
        :param from_date: Specific from date
        :param to_date: Specific to date
        :param interpolation: type of interpolation in quantile function (default value in quantile is linear)
        :param drift_adjust:
        :param periods_in_a_year_fixed:
        """
        return max(min_leverage_local,
                   min(target_vol / self.vol_from_var_func(level=level,
                                                           months_from_last=months_from_last,
                                                           from_date=from_date,
                                                           to_date=to_date,
                                                           interpolation=interpolation,
                                                           drift_adjust=drift_adjust,
                                                           periods_in_a_year_fixed=periods_in_a_year_fixed),
                       max_leverage_local))

    def value_to_ret(self, logret: bool = False):
        """
        Function converts a valueseries into a returnseries.
        Log return is the equivalent of LN(value[t] / value[t-1]) in MS excel.

        :param logret: Boolean set to True for log return and False for simple return.
        """
        if logret:
            self.tsdf = np.log(self.tsdf).diff()
        else:
            self.tsdf = self.tsdf.pct_change()
        self.tsdf.iloc[0] = 0
        self.valuetype = 'Return(Total)'
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        return self

    def value_to_diff(self, periods: int = 1):
        """
        Function converts a valueseries to a series of its 1 period differences

        :param periods:
        """
        self.tsdf = self.tsdf.diff(periods=periods)
        self.tsdf.iloc[0] = 0
        self.valuetype = 'Return(Total)'
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        return self

    def value_to_log(self, reverse: bool = False):
        """
        Function converts a valueseries into logarithmic returns equivalent to LN(value[t] / value[t=0]) in MS excel.

        :param reverse:
        """
        if reverse:
            self.tsdf = np.exp(self.tsdf)
            self.valuetype = 'Price(Close)'
            self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        else:
            self.tsdf = np.log(self.tsdf / self.tsdf.iloc[0])
            self.valuetype = 'Return(Total)'
            self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        return self

    def to_cumret(self, div_by_first: bool = True, logret: bool = False):
        """
        Function converts a total return timeseries into a cumulative series.

        :param div_by_first:
        :param logret: Boolean set to True for log return and False for simple return.
        """
        if not any([True if x == 'Return(Total)' else False for x in self.tsdf.columns.get_level_values(1).values]):
            self.value_to_ret(logret=logret)
        self.tsdf = self.tsdf.add(1.0)
        self.tsdf = self.tsdf.cumprod(axis=0)
        if div_by_first:
            self.tsdf = self.tsdf / self.tsdf.iloc[0]
        self.valuetype = 'Price(Close)'
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        return self

    def resample(self, freq: str = 'BM'):
        """
        Function resamples timeseries frequency.
        :param freq: https://pandas.pydata.org/pandas-docs/version/0.22.0/timeseries.html#offset-aliases
        """
        self.tsdf = self.tsdf.resample(freq).last()
        return self

    def to_drawdown_series(self):
        """
        Converts the series (self.tsdf) into a drawdown series
        """
        self.tsdf = drawdown_series(self.tsdf)
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], ['Drawdowns']])
        self.tsdf.index = pd.to_datetime(self.tsdf.index)
        return self

    def rolling_vol(self, observations: int = 21, periods_in_a_year_fixed: int = None) -> pd.Series:
        """
        Calculates rolling annualised volatilities.

        :param observations: Number of observations in the overlapping window.
        :param periods_in_a_year_fixed:
        """
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            time_factor = self.periods_in_a_year
        df = self.tsdf.pct_change().copy()
        voldf = df.rolling(observations, min_periods=observations).std() * np.sqrt(time_factor)
        voldf.dropna(inplace=True)
        voldf.columns = pd.MultiIndex.from_product([[self.label], ['Rolling volatility']])
        return voldf

    def rolling_return(self, observations: int = 21) -> pd.DataFrame:
        """
        Calculates sum of the returns in a rolling window.

        :param observations: Number of observations in the overlapping window.
        """
        retdf = self.tsdf.pct_change().rolling(observations, min_periods=observations).sum()
        retdf.columns = pd.MultiIndex.from_product([[self.label], ['Rolling returns']])
        return retdf.dropna()

    def rolling_cvar_down(self, level: float = 0.95, observations: int = 252) -> pd.DataFrame:
        """
        Calculates rolling annualized downside CVaR.

        :param observations: Number of observations in the overlapping window.
        :param level: The sought CVaR level as a float
        """
        cvardf = self.tsdf.rolling(observations, min_periods=observations).apply(
            lambda x: cvar_down(x, level=level))
        cvardf = cvardf.dropna()
        cvardf.columns = pd.MultiIndex.from_product([[self.label], ['Rolling CVaR']])
        return cvardf

    def rolling_var_down(self, level: float = 0.95, observations: int = 252,
                         interpolation: str = 'lower') -> pd.DataFrame:
        """
        Calculates rolling annualized downside VaR.

        :param level: The sought VaR level as a float
        :param observations: Number of observations in the overlapping window.
        :param interpolation: type of interpolation in quantile function (default value in quantile is linear)
        """
        vardf = self.tsdf.rolling(observations, min_periods=observations).apply(
            lambda x: var_down(x, level=level, interpolation=interpolation))
        vardf = vardf.dropna()
        vardf.columns = pd.MultiIndex.from_product([[self.label], ['Rolling VaR']])
        return vardf

    def value_nan_handle(self, method: str = 'fill'):
        """
        Method handles NaN in valueseries.

        :param method: Method used to handle NaN. Either fill with last known (default) or drop.
        """
        assert method in ['fill', 'drop'], 'Method must be either fill or drop passed as string.'
        if method == 'fill':
            self.tsdf.fillna(method='pad', inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def return_nan_handle(self, method: str = 'fill'):
        """
        Method handles NaN in returnseries.

        :param method: Method used to handle NaN. Either fill with last known (default) or drop.
        """
        assert method in ['fill', 'drop'], 'Method must be either fill or drop passed as string.'
        if method == 'fill':
            self.tsdf.fillna(value=0.0, inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def running_adjustment(self, adjustment: float, days_in_year: int = 365):
        """
        Method adds (+) or subtracts (-) a fee from the timeseries return.

        :param adjustment: Fee to add or subtract
        :param days_in_year:
        """
        if any([True if x == 'Return(Total)' else False for x in self.tsdf.columns.get_level_values(1).values]):
            ra_df = self.tsdf.copy()
        else:
            ra_df = self.tsdf.pct_change().copy()
        ra_df.dropna(inplace=True)
        prev = self.first_idx
        dates: list = [prev]
        values: list = [float(self.tsdf.iloc[0])]
        for idx, row in ra_df.iterrows():
            idx = dt.datetime.strptime(str(idx), '%Y-%m-%d %H:%M:%S').date()
            dates.append(idx)
            values.append(values[-1] * (1 + float(row) + adjustment * (idx - prev).days / days_in_year))
            prev = idx
        self.tsdf = pd.DataFrame(data=values, index=dates)
        self.valuetype = 'Price(Close)'
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        self.tsdf.index = pd.to_datetime(self.tsdf.index)
        return self

    def set_new_label(self, lvl_zero: str = None, lvl_one: str = None, delete_lvl_one: bool = False):
        """
        Method allows manuel setting the columns of the tsdf Pandas Dataframe associated with the timeseries

        :param lvl_zero: New level zero label
        :param lvl_one: New level one label
        :param delete_lvl_one: Boolean. If True the level one label is deleted.
        """
        if lvl_zero is None and lvl_one is None:
            self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        elif lvl_zero is not None and lvl_one is None:
            self.tsdf.columns = pd.MultiIndex.from_product([[lvl_zero], [self.valuetype]])
            self.label = lvl_zero
        elif lvl_zero is None and lvl_one is not None:
            self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [lvl_one]])
            self.valuetype = lvl_one
        else:
            self.tsdf.columns = pd.MultiIndex.from_product([[lvl_zero], [lvl_one]])
            self.label, self.valuetype = lvl_zero, lvl_one
        if delete_lvl_one:
            self.tsdf.columns = self.tsdf.columns.droplevel(level=1)
        return self

    def plot_series(self, mode: str = 'lines', tick_fmt: str = None, directory: str = None,
                    size_array: list = None, auto_open: bool = True, add_logo: bool = True) -> (dict, str):
        """
        Function to draw a Plotly graph with lines in Captor style.

        :param mode: The type of scatter to use, lines, markers or lines+markers.
        :param tick_fmt: None, '%', '.1%' depending on number of decimals to show.
        :param directory: Directory where Plotly html file is saved.
        :param size_array: The values will set bubble sizes.
        :param auto_open: Determines whether or not to open a browser window with the plot.
        :param add_logo: If True a Captor logo is added to the plot.

        To scale the bubble size, use the attribute sizeref.
        We recommend using the following formula to calculate a sizeref value:
        sizeref = 2. * max(array of size values) / (desired maximum marker size ** 2)
        """
        if not directory:
            directory = os.path.join(str(Path.home()), 'Documents')
        filename = self.label.replace('/', '').replace('#', '').replace(' ', '').upper()
        plotfile = os.path.join(os.path.abspath(directory), '{}.html'.format(filename))

        assert mode in ['lines', 'markers', 'both'], 'Style must be specified as lines, markers or both.'
        if mode == 'both':
            mode = 'lines+markers'

        values = [float(x) for x in self.tsdf.iloc[:, 0].tolist()]

        if size_array:
            sizer = 2. * max(size_array) / (90. ** 2)
            text_array = [f'{x:.2%}' for x in size_array]
        else:
            sizer = None
            text_array = None

        data = [go.Scatter(x=self.tsdf.index,
                           y=values,
                           hovertemplate='%{y}<br>%{x|%Y-%m-%d}',
                           line=dict(width=2.5,
                                     color='rgb(33, 134, 197)',
                                     dash='solid'),
                           marker=dict(size=size_array,
                                       sizemode='area',
                                       sizeref=sizer,
                                       sizemin=4),
                           text=text_array,
                           mode=mode,
                           name=self.label)]

        fig, logo = load_plotly_dict()
        fig['data'] = data
        figure = go.Figure(fig)
        figure.update_layout(yaxis=dict(tickformat=tick_fmt))
        if add_logo:
            figure.add_layout_image(logo)
        plot(figure, filename=plotfile, auto_open=auto_open, link_text='', include_plotlyjs='cdn')

        return fig, plotfile


def timeseries_chain(front, back, old_fee: float = 0.0):
    """

    :param front: Earlier series to chain with.
    :param back: Later series to chain with.
    :param old_fee: Fee to apply to earlier series.
    """
    old = front.from_deepcopy()
    old.running_adjustment(old_fee)
    new = back.from_deepcopy()

    dates = [x.strftime('%Y-%m-%d') for x in old.tsdf.index if x < new.first_idx]
    values = np.array([float(x) for x in old.tsdf.values][:len(dates)])
    values = list(values * float(new.tsdf.iloc[0]) / float(old.tsdf.loc[new.first_idx]))

    dates.extend([x.strftime('%Y-%m-%d') for x in new.tsdf.index])
    values.extend([float(x) for x in new.tsdf.values])

    new_dict = dict(new.__dict__)
    cleaner_list = ['label', 'tsdf']
    for item in cleaner_list:
        new_dict.pop(item)
    new_dict.update(dates=dates, values=values)
    return type(back)(new_dict)
