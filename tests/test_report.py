"""Test suite for the openseries/report.py module."""

from __future__ import annotations

import datetime as dt
from json import loads
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from pandas import Series
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from openseries.frame import OpenFrame
from openseries.load_plotly import load_plotly_dict
from openseries.report import (
    _configure_figure_layout,
    calendar_period_returns,
    report_html,
)
from openseries.series import OpenTimeSeries

if TYPE_CHECKING:  # pragma: no cover
    from openseries.simulation import ReturnSimulation


class ReportTestError(Exception):
    """Custom exception used for signaling test failures."""


class TestReport:
    """class to run tests on the module report.py."""

    seed: int
    seriesim: ReturnSimulation
    randomframe: OpenFrame
    randomseries: OpenTimeSeries
    random_properties: dict[str, dt.date | int | float]

    def test_calendar_period_returns(self: TestReport) -> None:
        """Test calendar_period_returns function."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        expected_one = [
            "0.62145461",
            "4.75300303",
            "-0.38834851",
            "1.23234213",
            "-0.95914766",
        ]

        returns = calendar_period_returns(data=frame, relabel=True)
        last_year = [f"{nbr:.8f}" for nbr in returns.loc[2019]]

        if last_year != expected_one:
            msg = f"calendar_period_returns not working as intended:\n{last_year}"
            raise ReportTestError(msg)

        frame.to_cumret()

        returns = calendar_period_returns(data=frame, relabel=False)
        last_year = [f"{nbr:.8f}" for nbr in returns.loc[dt.date(2019, 6, 28)]]

        returns = calendar_period_returns(data=frame, relabel=True, freq="BQE")
        last_quarter = [f"{nbr:.8f}" for nbr in returns.loc["Q2 2019"]]

        expected_two = [
            "-0.13684769",
            "-1.71819468",
            "-2.47673386",
            "-235.56746401",
            "-1.07977144",
        ]

        if last_quarter != expected_two:
            msg = f"calendar_period_returns not working as intended:\n{last_quarter}"
            raise ReportTestError(msg)

    def _verify_report_data_alignment(
        self: TestReport,
        plotframe: OpenFrame,
        fig_json: dict[str, Any],
    ) -> None:
        """Verify report data alignment.

        Args:
            plotframe: Frame to check.
            fig_json: Figure JSON data.

        Raises:
            ReportTestError: If data is not aligned.
        """
        rawdata = [x.strftime("%Y-%m-%d") for x in plotframe.tsdf.index[1:5]]
        if rawdata != fig_json["data"][0]["x"][1:5]:
            msg = "Unaligned data between original and data in Figure."
            raise ReportTestError(msg)

    def _verify_report_bar_freq(
        self: TestReport,
        fig_json: dict[str, Any],
        expected_item: str,
    ) -> None:
        """Verify report bar frequency.

        Args:
            fig_json: Figure JSON data.
            expected_item: Expected string in figure.

        Raises:
            ReportTestError: If bar frequency is not correct.
        """
        if expected_item not in str(fig_json):
            msg = "report_html bar_freq argument not setup correctly."
            raise ReportTestError(msg)

    def _verify_report_logo(
        self: TestReport,
        fig_json: dict[str, Any],
        logo: dict[str, Any],
    ) -> None:
        """Verify report logo.

        Args:
            fig_json: Figure JSON data.
            logo: Logo dictionary.

        Raises:
            ReportTestError: If logo is not correct.
        """
        if logo == {}:
            if fig_json["layout"]["images"][0] != logo:
                msg = "report_html add_logo argument not setup correctly"
                raise ReportTestError(msg)
        elif fig_json["layout"]["images"][0]["source"] != logo["source"]:
            msg = "report_html add_logo argument not setup correctly"
            raise ReportTestError(msg)

    def test_report_html(self: TestReport) -> None:
        """Test report_html function."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        figure, _ = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            vertical_legend=True,
        )
        fig_json = loads(cast("str", figure.to_json()))
        self._verify_report_bar_freq(fig_json, "'dtype': 'i2'")
        self._verify_report_data_alignment(plotframe, fig_json)

        figure, _ = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            vertical_legend=False,
        )
        fig_json = loads(cast("str", figure.to_json()))
        self._verify_report_data_alignment(plotframe, fig_json)

        figure, _ = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            title="test_title",
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
        self._verify_report_bar_freq(fig_bqe_json, "'x': ['Q3 2009'")

        figure_bme, _ = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            bar_freq="BME",
        )
        fig_bme_json = loads(cast("str", figure_bme.to_json()))
        self._verify_report_bar_freq(fig_bme_json, "'x': ['Aug 09'")

        fig_logo, _ = report_html(
            data=plotframe,
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        fig_logo_json = loads(cast("str", fig_logo.to_json()))
        self._verify_report_logo(fig_logo_json, logo)

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
        if not divstring.startswith("<div") or not divstring.endswith("</script>"):
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

        try:
            if mockfilepath.parts[-2:] != ("tests", "seriesfile.html"):
                msg = "report_html method not working as intended"
                raise ReportTestError(msg)
        finally:
            if mockfilepath.exists():
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
            ),
        )

        if "Return (simple)" in labels:
            msg = "report_html shortdata test not working."
            raise ReportTestError(msg)

        frame.trunc_frame(start_cut=dt.date(2019, 4, 30))

        new_length = 41
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
            ),
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
            ],
        )

        with pytest.raises(
            expected_exception=ZeroDivisionError,
            match=r"division by zero",
        ):
            _, _ = report_html(data=frame, auto_open=False, output_type="div")

    def test_tracking_error_hasnans(self: TestReport) -> None:
        """Test report_html function with tracking_error_func returning NaN values."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        mocked_te = Series(
            data=[np.nan, np.nan, np.nan, np.nan, np.nan],
            index=frame.tsdf.columns,
            name="Tracking Errors vs Asset_4",
            dtype="float64",
        )

        with patch.object(OpenFrame, "tracking_error_func", return_value=mocked_te):
            figure, _ = report_html(data=frame, auto_open=False, output_type="div")
            fig_json = loads(cast("str", figure.to_json()))

            rawdata = [x.strftime("%Y-%m-%d") for x in frame.tsdf.index[1:5]]
            if rawdata != fig_json["data"][0]["x"][1:5]:
                msg = "Tracking error with NaN values test not working as intended."
                raise ReportTestError(msg)

    def test_capture_ratio_hasnans(self: TestReport) -> None:
        """Test report_html function with capture_ratio_func returning NaN values."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        mocked_cr = Series(
            data=[np.nan, np.nan, np.nan, np.nan, np.nan],
            index=frame.tsdf.columns,
            name="Up-Down Capture Ratios vs Asset_4",
            dtype="float64",
        )

        with patch.object(OpenFrame, "capture_ratio_func", return_value=mocked_cr):
            figure, _ = report_html(data=frame, auto_open=False, output_type="div")
            fig_json = loads(cast("str", figure.to_json()))

            rawdata = [x.strftime("%Y-%m-%d") for x in frame.tsdf.index[1:5]]
            if rawdata != fig_json["data"][0]["x"][1:5]:
                msg = "Capture ratio with NaN values test not working as intended."
                raise ReportTestError(msg)

    def test_configure_figure_layout_with_table_min_height(self: TestReport) -> None:
        """Test _configure_figure_layout with table_min_height and total_min_height."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        figure = make_subplots(
            rows=2,
            cols=1,
            specs=[
                [{"type": "xy"}],
                [{"type": "xy"}],
            ],
        )

        _configure_figure_layout(
            figure=figure,
            copied=frame,
            add_logo=False,
            vertical_legend=False,
            title=None,
            mobile=True,
            total_min_height=950,
            table_min_height=200,
        )

        fig_json = loads(cast("str", figure.to_json()))
        layout = fig_json["layout"]

        if "yaxis" not in layout or "domain" not in layout["yaxis"]:
            msg = (
                "_configure_figure_layout with table_min_height "
                "not working as intended."
            )
            raise ReportTestError(msg)

    def test_report_html_auto_open_file(self: TestReport) -> None:
        """Test report_html with auto_open=True and output_type='file'."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        directory = Path(__file__).parent
        with patch("webbrowser.open") as mock_open:
            _, figfile = report_html(
                data=frame,
                auto_open=True,
                output_type="file",
                directory=directory,
            )
            plotfile = Path(figfile).resolve()

            if not plotfile.exists():
                msg = "html file not created"
                raise FileNotFoundError(msg)

            mock_open.assert_called_once()
            if str(plotfile.resolve()) not in str(mock_open.call_args[0][0]):
                msg = "webbrowser.open not called with correct file path"
                raise ReportTestError(msg)

            plotfile.unlink()
