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

from openseries.frame import OpenFrame
from openseries.load_plotly import load_plotly_dict
from openseries.report import (
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
        self._verify_report_data_alignment(plotframe, fig_json)

        figure, _ = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            vertical_legend=False,
        )
        fig_json = loads(cast("str", figure.to_json()))
        self._verify_report_data_alignment(plotframe, fig_json)

        figure, html_output = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            title="test_title",
        )
        fig_json = loads(cast("str", figure.to_json()))

        title_layout = fig_json.get("layout", {}).get("title", {})
        title_in_json = "test_title" in str(title_layout)
        title_in_html = "test_title" in html_output
        if not title_in_json and not title_in_html:
            msg = "report_html title argument not setup correctly."
            raise ReportTestError(msg)

        _, logo = load_plotly_dict()

        _, html_bqe = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            bar_freq="BQE",
        )
        if "Q3 2009" not in html_bqe:
            msg = "report_html bar_freq BQE argument not setup correctly."
            raise ReportTestError(msg)

        _, html_bme = report_html(
            data=plotframe,
            auto_open=False,
            output_type="div",
            bar_freq="BME",
        )
        if "Aug 09" not in html_bme:
            msg = "report_html bar_freq BME argument not setup correctly."
            raise ReportTestError(msg)

        _, html_logo = report_html(
            data=plotframe,
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        logo_source = logo.get("source", "")
        logo_str = logo_source if isinstance(logo_source, str) else ""
        if logo_str and logo_str not in html_logo:
            msg = "report_html add_logo argument not setup correctly"
            raise ReportTestError(msg)

        _, html_nologo = report_html(
            data=plotframe,
            auto_open=False,
            add_logo=False,
            output_type="div",
        )
        if logo_str and logo_str in html_nologo:
            msg = "report_html add_logo argument not setup correctly"
            raise ReportTestError(msg)

    def _verify_file_created(self: TestReport, filepath: str) -> Path:
        """Verify that a file was created and return its Path object."""
        plotfile = Path(filepath).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)
        if filepath[:5] == "<div>":
            msg = "report_html method not working as intended"
            raise ReportTestError(msg)
        return plotfile

    def test_report_html_filefolders(self: TestReport) -> None:
        """Test report_html method with different file folder options."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        directory = Path(__file__).parent
        test_filename1 = "test_report_html_filefolders_1.html"
        _, filepath = report_html(
            data=plotframe,
            auto_open=False,
            directory=directory,
            filename=test_filename1,
        )
        plotfile = self._verify_file_created(filepath)

        test_filename2 = "test_report_output.html"
        _, filepath2 = report_html(
            data=plotframe,
            auto_open=False,
            directory=directory,
            filename=test_filename2,
        )
        plotfile2 = self._verify_file_created(filepath2)

        _, divstring = report_html(data=plotframe, auto_open=False, output_type="div")
        starts_ok = divstring.lower().startswith("<!doctype html>")
        ends_ok = "</html>" in divstring
        if not starts_ok or not ends_ok:
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
            # When Documents doesn't exist, it should use the calling script's
            # directory. The file should be created somewhere (not in Documents)
            if mockfilepath.name != "seriesfile.html":
                msg = "report_html method not working as intended"
                raise ReportTestError(msg)
        finally:
            for file_to_cleanup in [mockfilepath, plotfile, plotfile2]:
                if file_to_cleanup.exists():
                    file_to_cleanup.unlink()

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
        table_data = next(
            (
                item["cells"]["values"][0]
                for item in fig_json["data"]
                if "cells" in item
            ),
            None,
        )
        labels = "".join(table_data) if table_data else ""

        if "Return (simple)" in labels:
            msg = "report_html shortdata test not working."
            raise ReportTestError(msg)

        frame.trunc_frame(start_cut=dt.date(2019, 4, 30))

        new_length = 41
        if new_length != frame.length:
            msg = f"report_html shortdata test not working:{frame.length}"
            raise ReportTestError(msg)

        # Ensure yearfrac is <= 1.0 for short data by truncating to a very short period
        # Truncate to just a few days to ensure yearfrac <= 1.0
        if frame.yearfrac > 1.0:
            frame.trunc_frame(start_cut=dt.date(2019, 6, 1))

        # Verify yearfrac is now <= 1.0
        if frame.yearfrac > 1.0:
            msg = (
                f"report_html shortdata test: yearfrac still > 1.0 "
                f"after truncation: {frame.yearfrac}"
            )
            raise ReportTestError(msg)

        figure, html_output = report_html(
            data=frame, auto_open=False, output_type="div"
        )
        fig_json = loads(cast("str", figure.to_json()))
        table_data = next(
            (
                item["cells"]["values"][0]
                for item in fig_json["data"]
                if "cells" in item
            ),
            None,
        )
        labels = "".join(table_data) if table_data else ""

        # Check both the figure JSON and HTML output for "Return (simple)"
        # Labels are formatted with HTML tags, so check for the text content
        in_labels = "Return (simple)" in labels or "<b>Return (simple)</b>" in labels
        in_html = "Return (simple)" in html_output
        if not in_labels and not in_html:
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

    def test_report_html_auto_open_file(
        self: TestReport,
        *,
        manual: bool = False,  # noqa: PT028
    ) -> None:
        """Test report_html with auto_open=True and output_type='file'."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        if manual:
            report_html(
                data=frame,
                auto_open=True,
                filename="tst.html",
                title="Nice Longish Title To Check",
            )
            narrow = frame.from_deepcopy()
            narrow.delete_timeseries(lvl_zero_item="Asset_1")
            narrow.delete_timeseries(lvl_zero_item="Asset_2")
            report_html(
                data=narrow,
                auto_open=True,
                filename="tst_narrow.html",
                title="Nice Longish Title To Check",
            )

        directory = Path(__file__).parent
        test_filename = "test_report_html_auto_open_file.html"

        with patch("webbrowser.open") as mock_open:
            _, filepath = report_html(
                data=frame,
                auto_open=True,
                directory=directory,
                filename=test_filename,
                output_type="file",
            )
            plotfile = Path(filepath).resolve()
            if not plotfile.exists():
                msg = "html file not created"
                raise FileNotFoundError(msg)
            if filepath[:5] == "<div>":
                msg = "report_html method not returning file path as intended"
                raise ReportTestError(msg)

            mock_open.assert_called_once()
            call_arg = str(mock_open.call_args[0][0])
            if not call_arg.startswith("file://"):
                msg = "webbrowser.open not called with file:// URL"
                raise ReportTestError(msg)

            with plotfile.open(encoding="utf-8") as f:
                file_content = f.read()
                if not file_content.lower().startswith("<!doctype html>"):
                    msg = "HTML file does not contain full HTML document"
                    raise ReportTestError(msg)

            if plotfile.exists():
                plotfile.unlink()

    def test_report_html_logo_exceptions(self: TestReport) -> None:
        """Test report_html with logo exceptions."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        # Test with invalid logo type to trigger exception handling
        with patch("openseries.report.load_plotly_dict") as mock_load:
            mock_load.return_value = ({}, None)  # Invalid logo type
            _, html = report_html(
                data=frame,
                auto_open=False,
                output_type="div",
                add_logo=True,
            )
            # Should return "CAPTOR" when logo has no source
            if "CAPTOR" not in html:
                msg = "report_html logo exception handling not working correctly."
                raise ReportTestError(msg)

    def test_report_html_plotly_script(self: TestReport) -> None:
        """Test report_html with different include_plotlyjs values."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        # Test with include_plotlyjs=False (should return empty string)
        _, html_false = report_html(
            data=frame,
            auto_open=False,
            output_type="div",
            include_plotlyjs=False,
        )
        if "plotly-2.35.2.min.js" in html_false:
            msg = "report_html include_plotlyjs=False not working correctly."
            raise ReportTestError(msg)

        # Test with include_plotlyjs=True (should return empty string)
        _, html_true = report_html(
            data=frame,
            auto_open=False,
            output_type="div",
            include_plotlyjs=True,
        )
        if "plotly-2.35.2.min.js" in html_true:
            msg = "report_html include_plotlyjs=True not working correctly."
            raise ReportTestError(msg)

    def test_report_html_browser_error(self: TestReport) -> None:
        """Test report_html with browser open error."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        directory = Path(__file__).parent
        test_filename = "test_report_html_browser_error.html"

        # Mock webbrowser.open to raise OSError
        with patch("webbrowser.open", side_effect=OSError("Browser error")):
            _, filepath = report_html(
                data=frame,
                auto_open=True,
                directory=directory,
                filename=test_filename,
            )
            plotfile = Path(filepath).resolve()
            if not plotfile.exists():
                msg = "html file not created when browser open fails"
                raise FileNotFoundError(msg)

            # Clean up
            if plotfile.exists():
                plotfile.unlink()
