Data Handling
=============

This guide covers data loading, validation, transformation, and management in openseries.

Loading Data
------------

From pandas DataFrame/Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most common way to load data is from pandas objects:

.. code-block:: python

   import pandas as pd
   from openseries import OpenTimeSeries

   # From pandas Series with DatetimeIndex
   data = pd.Series([100, 101, 99, 102],
                   index=pd.date_range('2023-01-01', periods=4))
   series = OpenTimeSeries.from_df(dframe=data)
   series.set_new_label(lvl_zero="Sample")

   # From pandas DataFrame column
   df = pd.DataFrame({
       'Date': pd.date_range('2023-01-01', periods=4),
       'Close': [100, 101, 99, 102],
       'Volume': [1000, 1100, 900, 1200]
   })
   df.set_index('Date', inplace=True)
   series = OpenTimeSeries.from_df(dframe=df['Close'])
   series.set_new_label(lvl_zero="Stock")

From Arrays
~~~~~~~~~~~

For custom data or when working with lists:

.. code-block:: python

   # From date strings and values
   dates = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
   values = [100.0, 101.0, 99.0, 102.0]

   series = OpenTimeSeries.from_arrays(
       dates=dates,
       values=values,
       name="Custom Data"
   )

From Fixed Rate
~~~~~~~~~~~~~~~

Generate synthetic data from a fixed rate:

.. code-block:: python

   from datetime import date

   # Create 252 trading days at 5% annual rate
   series = OpenTimeSeries.from_fixed_rate(
       rate=0.05,
       days=252,
       end_date=date(2023, 12, 31),
       name="5% Fixed Rate"
   )

Data Validation
---------------

Date Format Validation
~~~~~~~~~~~~~~~~~~~~~~

openseries enforces strict date formats:

.. code-block:: python

   # Valid date formats
   valid_dates = ['2023-01-01', '2023-12-31', '2024-02-29']  # ISO format

   # Invalid formats will raise ValidationError
   # This will fail with a validation error
   invalid_series = OpenTimeSeries.from_arrays(
       dates=['01/01/2023', '2023-1-1'],  # Wrong format
       values=[100, 101]
   )

Value Validation
~~~~~~~~~~~~~~~~

Values must be numeric and finite:

.. code-block:: python

   import numpy as np

   # Valid values
   valid_values = [100.0, 101.5, 99.25, 102.75]

   # Handle NaN values appropriately
   values_with_nan = [100.0, np.nan, 99.0, 102.0]
   series = OpenTimeSeries.from_arrays(
       dates=['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
       values=values_with_nan,
       name="Data with NaN"
   )

   # Clean NaN values (modifies original)
   series.value_nan_handle()  # Forward fill

Length Consistency
~~~~~~~~~~~~~~~~~~

Dates and values must have the same length:

.. code-block:: python

   # This will raise an error
   # This will fail with a length mismatch error
   invalid_series = OpenTimeSeries.from_arrays(
       dates=['2023-01-01', '2023-01-02'],
       values=[100.0, 101.0, 102.0]  # Different length
   )

Data Transformations
--------------------

Price and Return Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Assume we have a price series
   prices = OpenTimeSeries.from_arrays(
       dates=['2023-01-01', '2023-01-02', '2023-01-03'],
       values=[100.0, 102.0, 99.0],
       name="Stock Price"
   )

   # Convert to simple returns (modifies original)
   prices.value_to_ret()
   print(f"Returns: {prices.values}")  # [0.02, -0.0294...]

   # Convert to log returns (modifies original)
   prices.value_to_log()

   # Convert returns back to cumulative values (modifies original)
   prices.to_cumret()

   # Convert to differences (absolute changes) (modifies original)
   prices.value_to_diff()

Resampling
~~~~~~~~~~

Change the frequency of your data:

.. code-block:: python

   # Daily to monthly (business month end) (modifies original)
   series.resample_to_business_period_ends(freq="BME")

   # Daily to quarterly (modifies original)
   series.resample_to_business_period_ends(freq="BQE")

   # Daily to annual (modifies original)
   series.resample_to_business_period_ends(freq="BYE")

   # Custom resampling with pandas frequency strings (modifies original)
   series.resample(freq="W")

   # Resample with specific method (modifies original)
   series.resample(freq="W", method="mean")

Business Day Alignment
~~~~~~~~~~~~~~~~~~~~~~

Align data to business day calendars:

.. code-block:: python

   # Align to US business days (modifies original)
   series.align_index_to_local_cdays(countries="US")

   # Align to multiple countries (intersection) (modifies original)
   series.align_index_to_local_cdays(countries=["US", "GB", "JP"])

   # Align to specific market calendar (modifies original)
   series.align_index_to_local_cdays(markets="NYSE")

Handling Missing Data
---------------------

NaN Handling Strategies
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   # Create series with missing values
   dates = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
   values = [100.0, np.nan, 102.0, np.nan]

   series_with_nan = OpenTimeSeries.from_arrays(
       dates=dates, values=values, name="With NaN"
   )

   # Forward fill missing values (for price series) (modifies original)
   series_with_nan.value_nan_handle()

   # For return series, replace NaN with 0.0 (modifies original)
   series_with_nan.value_to_ret()
   series_with_nan.return_nan_handle()

Dropping Missing Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Remove NaN values entirely (modifies original)
   series_with_nan.value_nan_handle(method="drop")

Working with Multiple Assets
-----------------------------

Creating OpenFrame
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openseries import OpenFrame

   # Create multiple series
   series1 = OpenTimeSeries.from_arrays(
       dates=['2023-01-01', '2023-01-02', '2023-01-03'],
       values=[100, 102, 99], name="Asset A"
   )

   series2 = OpenTimeSeries.from_arrays(
       dates=['2023-01-01', '2023-01-02', '2023-01-03'],
       values=[50, 51, 49], name="Asset B"
   )

   # Create frame
   frame = OpenFrame(constituents=[series1, series2])

Handling Different Date Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenFrame automatically handles series with different date ranges:

.. code-block:: python

   # Series with different start/end dates
   early_series = OpenTimeSeries.from_arrays(
       dates=['2022-12-01', '2023-01-01', '2023-01-02'],
       values=[95, 100, 102], name="Early Start"
   )

   late_series = OpenTimeSeries.from_arrays(
       dates=['2023-01-02', '2023-01-03', '2023-01-04'],
       values=[51, 49, 52], name="Late Start"
   )

   # Frame will align to common date range
   frame = OpenFrame(constituents=[early_series, late_series])
   print(f"Frame date range: {frame.first_idx} to {frame.last_idx}")

Adding and Removing Series
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add a new series
   new_series = OpenTimeSeries.from_arrays(
       dates=['2023-01-01', '2023-01-02', '2023-01-03'],
       values=[200, 205, 198], name="Asset C"
   )
   frame.add_timeseries(new_series)

   # Remove a series by index
   frame.delete_timeseries(item_idx=0)

Data Export and Import
----------------------

Excel Export
~~~~~~~~~~~~

.. code-block:: python

   # Export single series
   series.to_xlsx(filename="single_series.xlsx")

   # Export frame (multiple series)
   frame.to_xlsx(filename="multiple_series.xlsx")

   # Export with custom sheet title
   series.to_xlsx(
       filename="formatted_export.xlsx",
       sheet_title="Analysis"
   )

JSON Export
~~~~~~~~~~~

.. code-block:: python

   # Export series values only
   series.to_json(what_output="values", filename="series_values.json")

   # Export full dataframe structure
   series.to_json(what_output="tsdf", filename="series_dataframe.json")

   # Export frame data
   frame.to_json(what_output="values", filename="frame_values.json")

Working with Real Data Sources
-------------------------------

Yahoo Finance Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import yfinance as yf

   # Single asset
   ticker = yf.Ticker("AAPL")
   data = ticker.history(period="2y")

   apple = OpenTimeSeries.from_df(
       dframe=data['Close'],
       name="Apple Inc."
   )

   # Multiple assets
   tickers = ["AAPL", "GOOGL", "MSFT"]
   series_list = []

   for ticker_symbol in tickers:
       ticker = yf.Ticker(ticker_symbol)
       data = ticker.history(period="1y")
       series = OpenTimeSeries.from_df(
           dframe=data['Close'],
           name=ticker_symbol
       )
       series_list.append(series)

   tech_frame = OpenFrame(constituents=series_list)

CSV Data
~~~~~~~~

.. code-block:: python

   # Load from CSV
   df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)

   series = OpenTimeSeries.from_df(
       dframe=df['Close'],
       name="Stock from CSV"
   )

Data Quality Checks
-------------------

Validation Methods
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check for data quality issues
   print(f"Series length: {series.length}")
   print(f"Date range: {series.first_idx} to {series.last_idx}")
   print(f"Span of days: {series.span_of_days}")

   # Check for gaps in data
   expected_length = (series.last_idx - series.first_idx).days + 1
   actual_length = series.length

   if expected_length != actual_length:
       print(f"Data gaps detected: expected {expected_length}, got {actual_length}")

Outlier Detection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert to returns for outlier analysis (modifies original)
   series.value_to_ret()

   # Calculate z-scores
   returns_df = series.tsdf
   mean_return = returns_df.mean().iloc[0]
   std_return = returns_df.std().iloc[0]

   z_scores = (returns_df - mean_return) / std_return
   outliers = z_scores[abs(z_scores) > 3].dropna()

   print(f"Found {len(outliers)} outliers (|z| > 3)")

Performance Considerations
--------------------------

Memory Usage
~~~~~~~~~~~~

.. code-block:: python

   # For large datasets, consider resampling
   large_series = series  # Assume this is large daily data

   # Reduce to monthly for analysis (modifies original)
   large_series.resample_to_business_period_ends(freq="BME")

   # Use monthly for computationally intensive operations
   monthly_metrics = large_series.all_properties

Efficient Data Loading
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # When loading multiple assets, batch the operations
   tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

   # Download all at once
   data = yf.download(tickers, period="2y")['Close']

   # Create series efficiently
   series_list = []
   for ticker in tickers:
       series = OpenTimeSeries.from_df(
           dframe=data[ticker].dropna(),
           name=ticker
       )
       series_list.append(series)

   frame = OpenFrame(constituents=series_list)

This comprehensive guide should help you handle various data scenarios effectively with openseries.
