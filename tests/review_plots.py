"""Script to manually review plot outputs in browser.

This script generates sample data and opens HTML reports/plots in the browser
for manual review. Run with: python -m tests.review_plots
"""

from __future__ import annotations

import datetime as dt

from openseries.frame import OpenFrame
from openseries.owntypes import ValueType
from openseries.report import report_html
from openseries.series import OpenTimeSeries
from openseries.simulation import ReturnSimulation


def main() -> None:
    """Generate and open sample plots/reports in browser."""
    seed = 71
    seriesim = ReturnSimulation.from_lognormal(
        number_of_sims=5,
        trading_days=1252,
        mean_annual_return=0.05,
        mean_annual_vol=0.1,
        trading_days_in_year=252,
        seed=seed,
    )
    end_date = dt.date(2019, 6, 30)
    randomframe = OpenFrame(
        constituents=[
            OpenTimeSeries.from_df(
                dframe=seriesim.to_dataframe(name="Asset", end=end_date),
                column_nmbr=serie,
                valuetype=ValueType.RTRN,
            ).to_cumret()
            for serie in range(seriesim.number_of_sims)
        ],
    )

    _, _ = report_html(
        data=randomframe,
        auto_open=True,
        filename="sample_report.html",
        title="Sample Report",
    )

    _, _ = randomframe.plot_series(
        auto_open=True,
        filename="sample_series.html",
        title="Sample Series Plot",
    )

    _, _ = randomframe.plot_histogram(
        auto_open=True,
        filename="sample_histogram.html",
        title="Sample Histogram",
    )

    randomframe.resample_to_business_period_ends(freq="BQE")
    randomframe.value_to_ret()
    _, _ = randomframe.plot_bars(
        auto_open=True,
        filename="sample_bars.html",
        title="Sample Bars Plot",
    )


if __name__ == "__main__":
    main()
