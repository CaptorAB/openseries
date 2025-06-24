"""Test suite for the openseries/report.py module.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import datetime as dt
from json import loads
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from pandas import Series

from openseries.frame import OpenFrame
from openseries.load_plotly import load_plotly_dict
from openseries.report import calendar_period_returns, report_html
from openseries.series import OpenTimeSeries

from .test_common_sim import CommonTestCase


class ReportTestError(Exception):
    """Custom exception used for signaling test failures."""


class TestReport(CommonTestCase):  # type: ignore[misc]
    """class to run tests on the module report.py."""

    def test_calendar_period_returns(self: TestReport) -> None:
        """Test calendar_period_returns function."""
        frame = self.randomframe.from_deepcopy()

        expected_one = [
            "0.00355807",
            "-0.00609952",
            "0.00402555",
            "0.00309766",
            "0.00051440",
        ]

        returns = calendar_period_returns(data=frame, relabel=True)
        last_year = [f"{nbr:.8f}" for nbr in returns.loc[2019]]

        if last_year != expected_one:
            msg = f"calendar_period_returns not working as intended:\n{last_year}"
            raise ReportTestError(msg)

        frame.to_cumret()

        expected_two = [
            "-0.15435597",
            "-0.26899063",
            "-2.15829921",
            "-2.06087664",
            "-1.63741658",
        ]

        returns = calendar_period_returns(data=frame, relabel=False)
        last_year = [f"{nbr:.8f}" for nbr in returns.loc[dt.date(2019, 6, 28)]]

        if last_year != expected_two:
            msg = f"calendar_period_returns not working as intended:\n{last_year}"
            raise ReportTestError(msg)

        returns = calendar_period_returns(data=frame, relabel=True, freq="BQE")
        last_quarter = [f"{nbr:.8f}" for nbr in returns.loc["Q2 2019"]]

        expected_three = [
            "-2.51705728",
            "-0.94753886",
            "2.51347355",
            "0.13058329",
            "-0.91336477",
        ]

        if last_quarter != expected_three:
            msg = f"calendar_period_returns not working as intended:\n{last_quarter}"
            raise ReportTestError(msg)

    def test_report_html(self: TestReport) -> None:
        """Test report_html function."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        figure, _ = report_html(
            data=plotframe, auto_open=False, output_type="div", vertical_legend=True
        )
        fig_json = loads(cast("str", figure.to_json()))
        bar_x_axis_item = "'dtype': 'i2'"

        if bar_x_axis_item not in str(fig_json):
            msg = "report_html bar_freq argument not setup correctly."
            raise ReportTestError(msg)

        rawdata = [x.strftime("%Y-%m-%d") for x in plotframe.tsdf.index[1:5]]
        if rawdata != fig_json["data"][0]["x"][1:5]:
            msg = "Unaligned data between original and data in Figure."
            raise ReportTestError(msg)

        figure, _ = report_html(
            data=plotframe, auto_open=False, output_type="div", vertical_legend=False
        )
        fig_json = loads(cast("str", figure.to_json()))

        rawdata = [x.strftime("%Y-%m-%d") for x in plotframe.tsdf.index[1:5]]
        if rawdata != fig_json["data"][0]["x"][1:5]:
            msg = "Unaligned data between original and data in Figure."
            raise ReportTestError(msg)

        figure, _ = report_html(
            data=plotframe, auto_open=False, output_type="div", title="test_title"
        )
        fig_json = loads(cast("str", figure.to_json()))

        if "test_title" not in str(fig_json):
            msg = "report_html title argument not setup correctly."
            raise ReportTestError(msg)

        _, logo = load_plotly_dict()

        figure_bqe, _ = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            bar_freq="BQE",
        )
        fig_bqe_json = loads(cast("str", figure_bqe.to_json()))

        bar_x_axis_item_bqe = "'x': ['Q3 2009'"
        if bar_x_axis_item_bqe not in str(fig_bqe_json):
            msg = "report_html bar_freq argument not setup correctly."
            raise ReportTestError(msg)

        figure_bme, _ = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            bar_freq="BME",
        )
        fig_bme_json = loads(cast("str", figure_bme.to_json()))

        bar_x_axis_item_bme = "'x': ['Jul 09'"
        if bar_x_axis_item_bme not in str(fig_bme_json):
            msg = "report_html bar_freq argument not setup correctly."
            raise ReportTestError(msg)

        fig_logo, _ = report_html(
            data=plotframe,
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        fig_logo_json = loads(cast("str", fig_logo.to_json()))

        if logo == {}:
            if fig_logo_json["layout"]["images"][0] != logo:
                msg = "report_html add_logo argument not setup correctly"
                raise ReportTestError(msg)
        elif fig_logo_json["layout"]["images"][0]["source"] != logo["source"]:
            msg = "report_html add_logo argument not setup correctly"
            raise ReportTestError(msg)

        fig_nologo, _ = report_html(
            data=plotframe,
            auto_open=False,
            add_logo=False,
            output_type="div",
        )
        fig_nologo_json = loads(cast("str", fig_nologo.to_json()))
        if fig_nologo_json["layout"].get("images", None):
            msg = "report_html add_logo argument not setup correctly"
            raise ReportTestError(msg)

    def test_report_html_filefolders(self: TestReport) -> None:
        """Test report_html method with different file folder options."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        directory = Path(__file__).parent
        _, figfile = report_html(data=plotframe, auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        if figfile[:5] == "<div>":
            msg = "report_html method not working as intended"
            raise ReportTestError(msg)

        _, divstring = report_html(data=plotframe, auto_open=False, output_type="div")
        if divstring[:5] != "<div>" or divstring[-6:] != "</div>":
            msg = "Html div section not created"
            raise ReportTestError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = True
            mockhomefig, _ = report_html(
                data=plotframe,
                auto_open=False,
                output_type="div",
            )
            mockhomefig_json = loads(cast("str", mockhomefig.to_json()))

        if mockhomefig_json["data"][0]["name"] != "Asset_0":
            msg = "report_html method not working as intended"
            raise ReportTestError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = False
            _, mockfile = report_html(
                data=plotframe,
                filename="seriesfile.html",
                auto_open=False,
            )
            mockfilepath = Path(mockfile).resolve()

        if mockfilepath.parts[-2:] != ("tests", "seriesfile.html"):
            msg = "report_html method not working as intended"
            raise ReportTestError(msg)

        mockfilepath.unlink()

    def test_report_html_shortdata(self: TestReport) -> None:
        """Test report_html function with short data."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        org_length = 2512
        if org_length != frame.length:
            msg = "report_html shortdata test not working."
            raise ReportTestError(msg)

        figure, _ = report_html(data=frame, auto_open=False, output_type="div")
        fig_json = loads(cast("str", figure.to_json()))
        labels = "".join(
            next(
                (
                    item["cells"]["values"][0]
                    for item in fig_json["data"]
                    if "cells" in item
                ),
                None,  # type: ignore[arg-type]
            )
        )

        if "Return (simple)" in labels:
            msg = "report_html shortdata test not working."
            raise ReportTestError(msg)

        frame.trunc_frame(start_cut=dt.date(2019, 4, 30))

        new_length = 40
        if new_length != frame.length:
            msg = f"report_html shortdata test not working:{frame.length}"
            raise ReportTestError(msg)

        figure, _ = report_html(data=frame, auto_open=False, output_type="div")
        fig_json = loads(cast("str", figure.to_json()))
        labels = "".join(
            next(
                (
                    item["cells"]["values"][0]
                    for item in fig_json["data"]
                    if "cells" in item
                ),
                None,  # type: ignore[arg-type]
            )
        )

        if "Return (simple)" not in labels:
            msg = "report_html shortdata test not working."
            raise ReportTestError(msg)

    def test_tracking_error_func_mocked(self: TestReport) -> None:
        """Test report_html function with tracking_error_func mocking."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        mocked_te = Series(
            data=[0.16118588, 0.13254767, 0.11576115, None, 0.11871340],
            index=frame.tsdf.columns,
            name="Tracking Errors vs Asset_4",
            dtype="float64",
        )

        with patch.object(OpenFrame, "tracking_error_func", return_value=mocked_te):
            figure, _ = report_html(data=frame, auto_open=False, output_type="div")
            fig_json = loads(cast("str", figure.to_json()))

            rawdata = [x.strftime("%Y-%m-%d") for x in frame.tsdf.index[1:5]]
            if rawdata != fig_json["data"][0]["x"][1:5]:
                msg = "Mocked tracking_error_func test not working as intended."
                raise ReportTestError(msg)

    def test_capture_ratio_func_mocked(self: TestReport) -> None:
        """Test report_html function with capture_ratio_func mocking."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        mocked_cr = Series(
            data=[0.16118588, 0.13254767, 0.11576115, None, 0.11871340],
            index=frame.tsdf.columns,
            name="Up-Down Capture Ratios vs Asset_4",
            dtype="float64",
        )

        with patch.object(OpenFrame, "capture_ratio_func", return_value=mocked_cr):
            figure, _ = report_html(data=frame, auto_open=False, output_type="div")
            fig_json = loads(cast("str", figure.to_json()))

            rawdata = [x.strftime("%Y-%m-%d") for x in frame.tsdf.index[1:5]]
            if rawdata != fig_json["data"][0]["x"][1:5]:
                msg = "Mocked capture_ratio_func test not working as intended."
                raise ReportTestError(msg)

    def test_capture_ratio_func_zerodiv(self: TestReport) -> None:
        """Test report_html function with capture_ratio_func zerodiv mocking."""
        frame = OpenFrame(
            constituents=[
                OpenTimeSeries.from_arrays(
                    name=f"Asset_{nbr}",
                    dates=["2023-01-01", "2023-01-02", "2023-01-03"],
                    values=[1.0, 1.0, 1.0],
                )
                for nbr in range(5)
            ]
        )

        with pytest.raises(
            expected_exception=ZeroDivisionError, match=r"float division by zero"
        ):
            figure, _ = report_html(data=frame, auto_open=False, output_type="div")
