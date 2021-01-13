# -*- coding: utf-8 -*-
from openseries.series import OpenTimeSeries
from openseries.frame import OpenFrame


if __name__ == "__main__":

    bonds = OpenTimeSeries.from_open_nav(isin="SE0009807308")
    index = OpenTimeSeries.from_open_api(
        timeseries_id="5813595971051506189ba416"
    )
    frame = (
        OpenFrame([bonds, index]).trunc_frame().value_nan_handle().to_cumret()
    )
    frame.plot_series()
