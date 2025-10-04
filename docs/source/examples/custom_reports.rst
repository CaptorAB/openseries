Reporting
=========

This example demonstrates how to create analysis reports using openseries and the built-in report functionality.

Using the Built-in HTML Report
-------------------------------

.. code-block:: python

   import yfinance as yf
   from openseries import OpenTimeSeries, OpenFrame, report_html
   import pandas as pd

   # Load sample data for comparison
   tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
   names = ["Apple", "Microsoft", "Google", "S&P 500"]

   series_list = []
   for ticker, name in zip(tickers, names):
       data = yf.Ticker(ticker).history(period="3y")
       series = OpenTimeSeries.from_df(dframe=data['Close'], name=name)
       series.set_new_label(lvl_zero=name)
       series_list.append(series)

   # Create frame for report
   comparison_frame = OpenFrame(constituents=series_list)

   # Generate HTML report
   # The last asset in the frame is used as the benchmark
   report_html(
       frame=comparison_frame,
       output_type="file",
       file_name="stock_comparison_report.html"
   )

   print("HTML report generated: stock_comparison_report.html")
