# -*- coding: utf-8 -*-
import datetime as dt
import io
import pandas as pd
from pandas.testing import assert_frame_equal
import sys
from requests.exceptions import Timeout
import unittest
from unittest.mock import Mock, patch

from openseries.frenkla_open_api_sdk import FrenklaOpenApiService

requests = Mock()


class TestFrenklaOpenApiService(unittest.TestCase):
    @patch("openseries.frenkla_open_api_sdk.requests")
    def test_openapi_timeout(self, mock_requests):

        mock_requests.get.side_effect = Timeout

        with self.assertRaises(Timeout):
            sevice = FrenklaOpenApiService()
            isin_code = "SE0009807308"
            _ = sevice.get_nav(isin=isin_code)
            mock_requests.get.assert_called_once()

    @patch("openseries.frenkla_open_api_sdk.requests.get")
    def test_openapi_get_nav_status_code(self, mock_get):

        mock_get.return_value.status_code = 400

        sevice = FrenklaOpenApiService()
        isin_code = "SE0009807308"

        with self.assertRaises(Exception) as e_status:
            _ = sevice.get_nav(isin=isin_code)
            mock_get.get.assert_called_once()

        self.assertIsInstance(e_status.exception, Exception)

    @patch("openseries.frenkla_open_api_sdk.requests.get")
    def test_openapi_get_timeseries_status_code(self, mock_get):

        mock_get.return_value.status_code = 404

        sevice = FrenklaOpenApiService()
        ts_id = "62d06f9d753964781e81f185"

        with self.assertRaises(Exception) as e_status:
            _ = sevice.get_timeseries(timeseries_id=ts_id)
            mock_get.get.assert_called_once()

        self.assertIsInstance(e_status.exception, Exception)

    @patch("openseries.frenkla_open_api_sdk.requests.get")
    def test_openapi_get_fundinfo_status_code(self, mock_get):

        mock_get.return_value.status_code = 400

        sevice = FrenklaOpenApiService()
        isin_code = "SE0009807308"

        with self.assertRaises(Exception) as e_status:
            _ = sevice.get_fundinfo(isins=[isin_code])
            mock_get.get.assert_called_once()

        self.assertIsInstance(e_status.exception, Exception)

    def test_frenklaopenapiservice_repr(self):

        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        service = FrenklaOpenApiService()
        r = "FrenklaOpenApiService(" "base_url=https://api.frenkla.com/public/api/)\n"
        print(service)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        self.assertEqual(r, output)

    def test_frenklaopenapiservice_get_timeseries(self):

        sevice = FrenklaOpenApiService()
        ts_id = "62d06f9d753964781e81f185"
        series = sevice.get_timeseries(timeseries_id=ts_id)

        self.assertEqual(ts_id, series["id"])

        with self.assertRaises(Exception) as e_unique:
            sevice.get_timeseries(timeseries_id="")

        self.assertEqual(int(str(e_unique.exception)[:3]), 404)

    def test_frenklaopenapiservice_get_fundinfo(self):

        sevice = FrenklaOpenApiService()
        isin_code = "SE0009807308"
        fundinfo = sevice.get_fundinfo(isins=[isin_code])

        self.assertEqual(isin_code, fundinfo[0]["classes"][0]["isin"])

        fundinfo_date = sevice.get_fundinfo(
            isins=[isin_code], report_date=dt.date(2022, 6, 30)
        )

        self.assertEqual(isin_code, fundinfo_date[0]["classes"][0]["isin"])

        with self.assertRaises(Exception) as e_unique:
            isin_cde = ""
            _ = sevice.get_fundinfo(isins=[isin_cde])

        self.assertEqual(int(str(e_unique.exception)[:3]), 400)

    def test_frenklaopenapiservice_get_nav(self):

        sevice = FrenklaOpenApiService()
        isin_code = "SE0009807308"
        series = sevice.get_nav(isin=isin_code)

        self.assertEqual(isin_code, series["isin"])

        with self.assertRaises(Exception) as e_unique:
            isin_cde = ""
            _ = sevice.get_nav(isin=isin_cde)

        self.assertEqual(
            f"Request for NAV series using ISIN {isin_cde} returned no data.",
            e_unique.exception.args[0],
        )

    def test_frenklaopenapiservice_get_nav_to_dataframe(self):

        sevice = FrenklaOpenApiService()
        isin_code = "SE0009807308"
        df = sevice.get_nav_to_dataframe(isin=isin_code).head()
        ddf = pd.DataFrame(
            data=[100.0000, 100.0978, 100.2821, 100.1741, 100.4561],
            index=pd.DatetimeIndex(
                [
                    "2017-05-29",
                    "2017-05-30",
                    "2017-05-31",
                    "2017-06-01",
                    "2017-06-02",
                ]
            ),
            columns=["Captor Iris Bond, SE0009807308"],
        )
        assert_frame_equal(df, ddf)

        with self.assertRaises(Exception) as e_unique:
            isin_cde = ""
            _ = sevice.get_nav_to_dataframe(isin=isin_cde).head()

        self.assertEqual(
            f"Request for NAV series using ISIN {isin_cde} returned no data.",
            e_unique.exception.args[0],
        )
