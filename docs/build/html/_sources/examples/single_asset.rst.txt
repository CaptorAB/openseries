Single Asset Analysis
=====================

This example demonstrates comprehensive analysis of a single financial asset using openseries.

Basic Setup
-----------

.. code-block:: python

   import yfinance as yf
   from openseries import OpenTimeSeries
   import numpy as np

   # Download Apple stock data
   ticker = yf.Ticker("AAPL")
   data = ticker.history(period="5y")

   # Create OpenTimeSeries
   apple = OpenTimeSeries.from_df(
       dframe=data['Close']
   )

   # Set descriptive label
   apple.set_new_label(lvl_zero="Apple Inc. (AAPL)")

   print(f"Loaded {apple.length} observations")
   print(f"Date range: {apple.first_idx} to {apple.last_idx}")

Performance Analysis
--------------------

.. code-block:: python

   # Basic performance metrics
   print("=== PERFORMANCE METRICS ===")
   print(f"Total Return: {apple.value_ret:.2%}")
   print(f"Annualized Return: {apple.geo_ret:.2%}")
   print(f"Annualized Volatility: {apple.vol:.2%}")
   print(f"Sharpe Ratio: {apple.ret_vol_ratio:.2f}")

   # Get all metrics at once
   all_metrics = apple.all_properties()
   print("\n=== ALL METRICS ===")
   print(all_metrics)

Risk Analysis
-------------

.. code-block:: python

   # Risk metrics
   print("=== RISK ANALYSIS ===")
   print(f"Maximum Drawdown: {apple.max_drawdown:.2%}")
   print(f"Max Drawdown Date: {apple.max_drawdown_date}")
   print(f"95% VaR (daily): {apple.var_down:.2%}")
   print(f"95% CVaR (daily): {apple.cvar_down:.2%}")
   print(f"Worst Single Day: {apple.worst:.2%}")
   print(f"Sortino Ratio: {apple.sortino_ratio:.2f}")

Time Series Transformations
---------------------------

.. code-block:: python

   # Convert to returns (modifies original)
   apple.value_to_ret()
   print(f"Returns series length: {apple.length}")

   # Create drawdown series (modifies original)
   apple.to_drawdown_series()

   # Convert to log returns (modifies original)
   apple.value_to_log()

   # Resample to monthly (modifies original)
   apple.resample_to_business_period_ends(freq="BME")
   print(f"Monthly data points: {apple.length}")

Rolling Analysis
----------------

.. code-block:: python

   # Rolling volatility (1-year window)
   rolling_vol = apple.rolling_vol(observations=252)
   print(f"Current 1Y volatility: {rolling_vol.iloc[-1, 0]:.2%}")
   print(f"Average 1Y volatility: {rolling_vol.mean().iloc[0]:.2%}")

   # Rolling returns (30-day)
   rolling_returns = apple.rolling_return(observations=30)

   # Rolling VaR
   rolling_var = apple.rolling_var_down(observations=252)

Visualization
-------------

.. code-block:: python

   # Plot price series
   fig, _ = apple.plot_series()

   # Plot returns histogram
   fig, _ = apple_returns.plot_histogram()

   # Plot drawdown series
   fig, _ = apple_drawdowns.plot_series()

Calendar Analysis
-----------------

.. code-block:: python

   # Annual returns by calendar year
   years = [2019, 2020, 2021, 2022, 2023, 2024]

   print("=== CALENDAR YEAR RETURNS ===")
   for year in years:
       # This may fail if no data exists for the year
       year_return = apple.value_ret_calendar_period(year=year)
       print(f"{year}: {year_return:.2%}")

Export Results
--------------

.. code-block:: python

   # Export to Excel
   apple.to_xlsx("apple_analysis.xlsx")

   # Export metrics to CSV
   all_metrics.to_csv("apple_metrics.csv")

   # Export to JSON
   apple.to_json("apple_data.json")

Complete Analysis Workflow
----------------------------

Here's how to perform comprehensive single asset analysis using openseries methods directly:

.. code-block:: python

   import yfinance as yf
   from openseries import OpenTimeSeries

   # Example: Analyze Apple stock using openseries methods
   ticker_symbol = "AAPL"

   # Download data using openseries methods
   ticker = yf.Ticker(ticker_symbol)
   data = ticker.history(period="5y")

   # Create series using openseries from_df method
   series = OpenTimeSeries.from_df(
       dframe=data['Close'],
       name=ticker_symbol
   )

   # Analysis using openseries properties and methods
   print(f"=== {ticker_symbol} ANALYSIS ===")
   print(f"Period: {series.first_idx} to {series.last_idx}")
   print(f"Observations: {series.length}")

   # Key metrics using openseries properties
   metrics = {
       'Total Return': f"{series.value_ret:.2%}",
       'Annual Return': f"{series.geo_ret:.2%}",
       'Volatility': f"{series.vol:.2%}",
       'Sharpe Ratio': f"{series.ret_vol_ratio:.2f}",
       'Max Drawdown': f"{series.max_drawdown:.2%}",
       '95% VaR': f"{series.var_down:.2%}",
       'Skewness': f"{series.skew:.2f}",
       'Kurtosis': f"{series.kurtosis:.2f}"
   }

   for metric, value in metrics.items():
       print(f"{metric}: {value}")

   # Export results using openseries to_xlsx method
   filename = f"{ticker_symbol.lower()}_analysis.xlsx"
   series.to_xlsx(filename)
   print(f"\nResults exported to {filename}")

   # Example: Analyze multiple assets
   tickers = ["AAPL", "TSLA", "MSFT"]
   for ticker_symbol in tickers:
       ticker = yf.Ticker(ticker_symbol)
       data = ticker.history(period="2y")
       series = OpenTimeSeries.from_df(dframe=data['Close'])
       series.set_new_label(lvl_zero=ticker_symbol)

       print(f"\n{ticker_symbol}:")
       print(f"  Return: {series.geo_ret:.2%}")
       print(f"  Volatility: {series.vol:.2%}")
       print(f"  Sharpe: {series.ret_vol_ratio:.2f}")
