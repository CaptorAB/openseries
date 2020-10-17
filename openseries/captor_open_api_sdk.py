# -*- coding: utf-8 -*-
import datetime as dt
import pandas as pd
import requests


class CaptorOpenApiService(object):

    def __init__(self, base_url: str = 'https://apiv2.captor.se/public/api'):
        """
        :param base_url:
        """
        self.headers = {'accept': 'application/json'}
        self.base_url = base_url

    def __repr__(self):

        return '{}(base_url={})'.format(self.__class__.__name__, self.base_url)

    def __str__(self):

        return '{}(base_url={})'.format(self.__class__.__name__, self.base_url)

    def get_timeseries(self, timeseries_id: str, url: str = '/opentimeseries') -> dict:
        """

        :param timeseries_id: str
        :param url: str
        :return: dict
        """
        response = requests.get(url=self.base_url + f'{url}/{timeseries_id}', headers=self.headers)

        if response.status_code // 100 != 2:
            raise Exception(f'{response.status_code}, {response.text}')

        return response.json()

    def get_fundinfo(self, isins: list, report_date: dt.date = None, url: str = '/fundinfo') -> dict:
        """

        :param isins: list
        :param report_date:
        :param url: str
        :return: dict
        """
        params = {'isins': isins}
        if report_date:
            params.update({'reportDate': report_date.strftime('%Y-%m-%d')})
        response = requests.get(url=self.base_url + url, params=params, headers=self.headers)

        if response.status_code // 100 != 2:
            raise Exception(f'{response.status_code}, {response.text}')

        return response.json()

    def get_nav(self, isin: str, url: str = '/nav') -> dict:
        """

        :param isin: str
        :param url: str
        :return: dict
        """
        response = requests.get(url=self.base_url + url, headers=self.headers)

        if response.status_code // 100 != 2:
            raise Exception(f'{response.status_code}, {response.text}')

        output = {}
        result = response.json()
        for res in result:
            if res['isin'] == isin:
                output.update(res)

        return output

    def get_nav_to_dataframe(self, isin: str) -> pd.DataFrame:
        """

        :param isin: str
        :return: pd.DataFrame
        """
        data = self.get_nav(isin=isin)
        return pd.DataFrame(data=data['navPerUnit'], index=data['dates'],
                            columns=[f'{data["longName"]}, {data["isin"]}'])
