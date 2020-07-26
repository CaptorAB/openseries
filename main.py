# -*- coding: utf-8 -*-
from OpenSeries.series import OpenTimeSeries
from OpenSeries.frame import OpenFrame


if __name__ == '__main__':

    capirisc = 'SE0009807308'
    scillac = 'SE0010494849'
    bonds = OpenTimeSeries.from_open_nav(isin=capirisc)
    equities = OpenTimeSeries.from_open_nav(isin=scillac)
    basket = OpenFrame([bonds, equities], weights=[0.6, 0.4]).trunc_frame().value_nan_handle().to_cumret()
    portfolio = OpenTimeSeries.from_df(basket.make_portfolio('porfolio'))
    basket.add_timeseries(portfolio)
    basket.plot_series(tick_fmt='.1%')

    # Quandl examples
    series1 = OpenTimeSeries.from_quandl(database_code='CHRIS', dataset_code='EUREX_FMWO1', field='Settle')

    series2 = OpenTimeSeries.from_quandl(database_code='NASDAQOMX', dataset_code='NOMXN120SEKNI')
    series3 = OpenTimeSeries.from_quandl(database_code='ML', dataset_code='BBTRI')
    frame = OpenFrame([series1, series2, series3]).trunc_frame().value_nan_handle().to_cumret()
    frame.plot_series(tick_fmt='.1%')
