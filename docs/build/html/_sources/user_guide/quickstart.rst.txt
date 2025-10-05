Quick Start Guide
=================

This guide will get you up and running with openseries in just a few minutes.

Your First OpenTimeSeries
--------------------------

Let's start by creating and analyzing a simple time series:

.. code-block:: python

   from openseries import OpenTimeSeries
   import pandas as pd
   import numpy as np

   # Create sample data - 252 trading days of returns
   np.random.seed(42)  # For reproducible results
   dates = pd.date_range('2023-01-01', periods=252, freq='B')  # Business days
   returns = np.random.normal(0.0008, 0.015, 252)  # Daily returns
   prices = 100 * np.cumprod(1 + returns)  # Convert to price series

   # Create OpenTimeSeries from arrays
   series = OpenTimeSeries.from_arrays(
       dates=[d.strftime('%Y-%m-%d') for d in dates],
       values=prices.tolist(),
       name="Sample Stock"
   )

   # Display basic information
   print(f"Series: {series.name}")
   print(f"Start date: {series.first_idx}")
   print(f"End date: {series.last_idx}")
   print(f"Number of observations: {series.length}")

Loading Data from pandas
-------------------------

More commonly, you'll load data from a pandas DataFrame:

.. code-block:: python

   import yfinance as yf  # pip install yfinance
   from openseries import OpenTimeSeries

   # Download S&P 500 data
   ticker = yf.Ticker("^GSPC")
   data = ticker.history(period="2y")

   # Create OpenTimeSeries from the Close prices
   sp500 = OpenTimeSeries.from_df(
       dframe=data['Close']
   )

   # Set a more descriptive label
   sp500.set_new_label(lvl_zero="S&P 500 Index")

   print(f"Loaded {sp500.length} observations")
   print(f"Date range: {sp500.first_idx} to {sp500.last_idx}")

Basic Financial Metrics
------------------------

openseries provides a comprehensive set of financial metrics:

.. code-block:: python

   # Key performance metrics
   print(f"Total Return: {sp500.value_ret:.2%}")
   print(f"Annualized Return (CAGR): {sp500.geo_ret:.2%}")
   print(f"Annualized Volatility: {sp500.vol:.2%}")
   print(f"Sharpe Ratio: {sp500.ret_vol_ratio:.2f}")
   print(f"Maximum Drawdown: {sp500.max_drawdown:.2%}")

   # Risk metrics
   print(f"95% VaR (daily): {sp500.var_down:.2%}")
   print(f"95% CVaR (daily): {sp500.cvar_down:.2%}")
   print(f"Sortino Ratio: {sp500.sortino_ratio:.2f}")

   # Distribution statistics
   print(f"Skewness: {sp500.skew:.2f}")
   print(f"Kurtosis: {sp500.kurtosis:.2f}")
   print(f"Positive Days: {sp500.positive_share:.1%}")

Get All Metrics at Once
~~~~~~~~~~~~~~~~~~~~~~~

Use the ``all_properties`` attribute to get a comprehensive overview:

.. code-block:: python

   # Get all metrics in a DataFrame
   metrics = sp500.all_properties()
   print(metrics)

Creating Visualizations
-----------------------

openseries integrates with Plotly for interactive visualizations:

.. code-block:: python

   # Plot the price series
   fig, _ = sp500.plot_series()
   # This opens an interactive plot in your browser

   # Plot returns histogram
   returns_series = sp500.value_to_ret()
   fig, _ = returns_series.plot_histogram()

   # Plot drawdown series
   drawdown_series = sp500.to_drawdown_series()
   fig, _ = drawdown_series.plot_series()

Working with Multiple Assets (OpenFrame)
-----------------------------------------

For multi-asset analysis, use the OpenFrame class:

.. code-block:: python

   from openseries import OpenFrame
   import yfinance as yf

   # Download data for multiple assets
   tickers = ["^GSPC", "^IXIC", "^RUT"]  # S&P 500, NASDAQ, Russell 2000
   names = ["S&P 500", "NASDAQ", "Russell 2000"]

   series_list = []
   for ticker, name in zip(tickers, names):
       data = yf.Ticker(ticker).history(period="2y")
       series = OpenTimeSeries.from_df(dframe=data['Close'])
       series.set_new_label(lvl_zero=name)
       series_list.append(series)

   # Create OpenFrame
   frame = OpenFrame(constituents=series_list)

   # Get metrics for all series
   all_metrics = frame.all_properties()
   print(all_metrics)

   # Calculate correlations
   correlations = frame.correl_matrix
   print("\nCorrelation Matrix:")
   print(correlations)

Portfolio Analysis
------------------

Create and analyze portfolios:

.. code-block:: python

   # Create equal-weighted portfolio
   frame.weights = [1/3, 1/3, 1/3]
   portfolio_df = frame.make_portfolio(name="Equal Weight Portfolio")
   portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)

   print(f"Portfolio Return: {portfolio.geo_ret:.2%}")
   print(f"Portfolio Volatility: {portfolio.vol:.2%}")
   print(f"Portfolio Sharpe: {portfolio.ret_vol_ratio:.2f}")

   # Compare with individual assets
   frame.add_timeseries(portfolio)
   comparison = frame.all_properties()
   print(comparison)

Data Transformations
--------------------

openseries provides various data transformation methods:

.. code-block:: python

   # Convert prices to returns
   returns = sp500.value_to_ret()
   print(f"Returns series length: {returns.length}")

   # Convert to log returns
   log_returns = sp500.value_to_log()

   # Calculate rolling statistics
   rolling_vol = sp500.rolling_vol(observations=30)  # 30-day rolling volatility
   rolling_ret = sp500.rolling_return(window=30)  # 30-day rolling returns

   # Resample to monthly data
   monthly = sp500.resample_to_business_period_ends(freq="BME")
   print(f"Monthly data points: {monthly.length}")

Exporting Results
-----------------

Save your analysis results:

.. code-block:: python

   # Export to Excel
   sp500.to_xlsx("sp500_analysis.xlsx")

   # Export to JSON
   sp500.to_json(filename="sp500_data.json", what_output="tsdf")

   # Export metrics to CSV
   metrics.to_csv("sp500_metrics.csv")

Working with Business Days
--------------------------

openseries handles business day calendars automatically:

.. code-block:: python

   # Align to Swedish business days
   series_swe = sp500.align_index_to_local_cdays(countries="SE")

   # Use multiple countries
   series_multi = sp500.align_index_to_local_cdays(countries=["US", "GB"])

   # Handle missing values
   clean_series = sp500.value_nan_handle()  # Forward fill NaN values

Next Steps
----------

Now that you've learned the basics, explore:

1. **Tutorials** - Detailed examples for specific use cases
2. **API Reference** - Complete documentation of all methods and properties
3. **Examples** - Real-world analysis scenarios

Key Concepts to Remember
------------------------

- **OpenTimeSeries**: For single asset analysis
- **OpenFrame**: For multi-asset and portfolio analysis
- **ValueType**: Enum to identify data types (prices, returns, etc.)
- **Business day handling**: Automatic alignment to trading calendars
- **Interactive plotting**: Built-in Plotly integration
- **Type safety**: Pydantic-based validation ensures data integrity

Common Patterns
---------------

Here are some common usage patterns:

.. code-block:: python

   # Pattern 1: Load, analyze, visualize
   series = OpenTimeSeries.from_df(dframe=data['Close'])
   series.set_new_label(lvl_zero="Asset")
   metrics = series.all_properties()
   series.plot_series()

   # Pattern 2: Multi-asset comparison
   frame = OpenFrame(constituents=[series1, series2, series3])
   comparison = frame.all_properties()
   correlations = frame.correl_matrix

   # Pattern 3: Portfolio construction
   frame.weights = [0.4, 0.3, 0.3]
   portfolio_df = frame.make_portfolio(name="Custom Portfolio")
   portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)
   frame.add_timeseries(portfolio)

   # Pattern 4: Risk analysis
   drawdowns = series.to_drawdown_series()
   var_95 = series.var_down
   rolling_risk = series.rolling_vol(observations=252)

This should give you a solid foundation to start using openseries for your financial analysis needs!
