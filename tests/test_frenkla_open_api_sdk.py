# -*- coding: utf-8 -*-
from openseries.frenkla_open_api_sdk import FrenklaOpenApiService
from requests.exceptions import Timeout
import unittest
from unittest.mock import Mock, patch

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
