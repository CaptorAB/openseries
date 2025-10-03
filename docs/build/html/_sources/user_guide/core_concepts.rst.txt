Core Concepts
=============

This section explains the fundamental concepts and design principles behind openseries.

Architecture Overview
----------------------

openseries is built around two main classes that inherit from Pydantic's BaseModel:

- **OpenTimeSeries**: Manages individual financial time series
- **OpenFrame**: Manages collections of OpenTimeSeries objects

Both classes provide:

- **Type safety** through Pydantic validation
- **Immutable data** - original data is preserved
- **Consistent API** - similar methods across both classes
- **Financial focus** - methods designed for financial analysis

The OpenTimeSeries Class
-------------------------

Core Properties
~~~~~~~~~~~~~~~

Every OpenTimeSeries has these fundamental properties:

.. code-block:: python

   from openseries import OpenTimeSeries
   import pandas as pd
   import numpy as np

   # Create a sample series
   dates = pd.date_range('2023-01-01', periods=100, freq='B')
   values = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))

   series = OpenTimeSeries.from_arrays(
       dates=[d.strftime('%Y-%m-%d') for d in dates],
       values=values.tolist(),
       name="Sample Asset"
   )

   # Core properties
   print(f"Name: {series.name}")
   print(f"Length: {series.length}")
   print(f"First date: {series.first_idx}")
   print(f"Last date: {series.last_idx}")
   print(f"Value type: {series.valuetype}")

Data Immutability
~~~~~~~~~~~~~~~~~

The original data is never modified:

.. code-block:: python

   # Original data is preserved
   original_dates = series.dates      # List of date strings
   original_values = series.values    # List of float values

   # Working data is in the tsdf DataFrame
   working_data = series.tsdf         # pandas DataFrame

   # Transformations create new objects
   returns = series.value_to_ret()    # New OpenTimeSeries
   print(f"Original length: {series.length}")
   print(f"Returns length: {returns.length}")  # Usually length - 1

Value Types
~~~~~~~~~~~

The ValueType enum identifies what the series represents:

.. code-block:: python

   from openseries import ValueType

   # Common value types
   print(ValueType.PRICE)      # "Price(Close)"
   print(ValueType.RTRN)       # "Return(Total)"
   print(ValueType.ROLLVOL)    # "Rolling volatility"

   # Check series type
   print(f"Series type: {series.valuetype}")

   # Type changes with transformations
   returns = series.value_to_ret()
   print(f"Returns type: {returns.valuetype}")

The OpenFrame Class
--------------------

Managing Multiple Series
~~~~~~~~~~~~~~~~~~~~~~~~

OpenFrame manages collections of OpenTimeSeries:

.. code-block:: python

   from openseries import OpenFrame

   # Create multiple series (example with synthetic data)
   series_list = []
   for i, name in enumerate(["Asset A", "Asset B", "Asset C"]):
       values = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
       series = OpenTimeSeries.from_arrays(
           dates=[d.strftime('%Y-%m-%d') for d in dates],
           values=values.tolist(),
           name=name
       )
       series_list.append(series)

   # Create OpenFrame
   frame = OpenFrame(constituents=series_list)

   # Frame properties
   print(f"Number of series: {frame.item_count}")
   print(f"Column names: {frame.columns_lvl_zero}")
   print(f"Common length: {frame.length}")

Automatic Alignment
~~~~~~~~~~~~~~~~~~~

OpenFrame automatically aligns series to common dates:

.. code-block:: python

   # Series with different date ranges are aligned
   print("Individual series lengths:")
   for series in frame.constituents:
       print(f"  {series.name}: {series.length}")

   print(f"Frame length (aligned): {frame.length}")

   # Access aligned data
   aligned_data = frame.tsdf
   print(f"Aligned DataFrame shape: {aligned_data.shape}")

Financial Calculations
----------------------

Return Calculations
~~~~~~~~~~~~~~~~~~~

openseries uses standard financial formulas:

.. code-block:: python

   # Simple returns: (P_t / P_{t-1}) - 1
   simple_returns = series.value_to_ret()

   # Log returns: ln(P_t / P_{t-1})
   log_returns = series.value_to_log()

   # Cumulative returns: rebasing to start at 1.0
   cumulative = series.to_cumret()

Annualization
~~~~~~~~~~~~~

Metrics are annualized using the actual number of observations per year:

.. code-block:: python

   # Automatic calculation of periods per year
   print(f"Periods per year: {series.periods_in_a_year:.1f}")

   # Annualized return (geometric mean)
   annual_return = series.geo_ret
   print(f"Annualized return: {annual_return:.2%}")

   # Annualized volatility
   annual_vol = series.vol
   print(f"Annualized volatility: {annual_vol:.2%}")

Risk Metrics
~~~~~~~~~~~~

Risk calculations follow industry standards:

.. code-block:: python

   # Value at Risk (95% confidence)
   var_95 = series.var_down
   print(f"95% VaR: {var_95:.2%}")

   # Conditional Value at Risk (Expected Shortfall)
   cvar_95 = series.cvar_down
   print(f"95% CVaR: {cvar_95:.2%}")

   # Maximum Drawdown
   max_dd = series.max_drawdown
   print(f"Maximum Drawdown: {max_dd:.2%}")

   # Sortino Ratio (downside deviation)
   sortino = series.sortino_ratio
   print(f"Sortino Ratio: {sortino:.2f}")

Date Handling
-------------

Business Day Calendars
~~~~~~~~~~~~~~~~~~~~~~~

openseries integrates with business day calendars:

.. code-block:: python

   # Align to specific country's business days
   us_aligned = series.align_index_to_local_cdays(countries="US")

   # Multiple countries (intersection of business days)
   multi_country = series.align_index_to_local_cdays(countries=["US", "GB"])

   # Custom markets using pandas-market-calendars
   nyse_aligned = series.align_index_to_local_cdays(markets="NYSE")

Resampling
~~~~~~~~~~

Convert between different frequencies:

.. code-block:: python

   # Resample to month-end
   monthly = series.resample_to_business_period_ends(freq="BME")

   # Resample to quarter-end
   quarterly = series.resample_to_business_period_ends(freq="BQE")

   # Custom resampling
   weekly = series.resample(freq="W")

Data Validation
---------------

Type Safety
~~~~~~~~~~~

Pydantic ensures data integrity:

.. code-block:: python

   # Dates must be valid ISO format strings
   try:
       invalid_series = OpenTimeSeries.from_arrays(
           dates=["invalid-date"],
           values=[100.0]
       )
   except ValueError as e:
       print(f"Validation error: {e}")

   # Values must be numeric
   try:
       invalid_series = OpenTimeSeries.from_arrays(
           dates=["2023-01-01"],
           values=["not a number"]
       )
   except ValueError as e:
       print(f"Validation error: {e}")

Consistency Checks
~~~~~~~~~~~~~~~~~~

The library performs consistency checks:

.. code-block:: python

   # Dates and values must have same length
   # Mixed value types in OpenFrame are detected
   # Date alignment issues are caught

Method Categories
-----------------

openseries methods fall into several categories:

Properties vs Methods
~~~~~~~~~~~~~~~~~~~~~

- **Properties**: Return calculated values (e.g., ``series.vol``)
- **Methods**: Perform operations or take parameters (e.g., ``series.vol_func()``)

.. code-block:: python

   # Property - uses full series
   volatility = series.vol

   # Method - can specify date range
   recent_vol = series.vol_func(months_from_last=12)

Transformation Methods
~~~~~~~~~~~~~~~~~~~~~~

Methods that return new objects:

.. code-block:: python

   # Data transformations
   returns = series.value_to_ret()        # Prices to returns
   drawdowns = series.to_drawdown_series() # Drawdown series
   cumulative = series.to_cumret()        # Cumulative returns

   # Time transformations
   monthly = series.resample_to_business_period_ends(freq="BME")
   aligned = series.align_index_to_local_cdays(countries="US")

Analysis Methods
~~~~~~~~~~~~~~~~

Methods that return calculated values:

.. code-block:: python

   # Rolling calculations
   rolling_vol = series.rolling_vol(window=30)
   rolling_corr = frame.rolling_corr(window=60)

   # Statistical analysis
   beta = frame.beta()
   tracking_error = frame.tracking_error_func()

Export Methods
~~~~~~~~~~~~~~

Methods for saving results:

.. code-block:: python

   # File exports
   series.to_xlsx("analysis.xlsx")
   series.to_json("data.json")

   # Visualization
   fig, _ = series.plot_series()
   fig, _ = series.plot_histogram()

Best Practices
--------------

Data Loading
~~~~~~~~~~~~

.. code-block:: python

   # Prefer from_df for pandas data
   series = OpenTimeSeries.from_df(dataframe['Close'], name="Asset")

   # Use from_arrays for custom data
   series = OpenTimeSeries.from_arrays(dates=date_list, values=value_list)

   # Always set meaningful names
   series.set_new_label(lvl_zero="Descriptive Name")

Analysis Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Load and validate data
   series = OpenTimeSeries.from_df(data['Close'], name="Asset")

   # 2. Basic analysis
   metrics = series.all_properties

   # 3. Specific calculations
   drawdowns = series.to_drawdown_series()
   rolling_metrics = series.rolling_vol(window=252)

   # 4. Visualization
   series.plot_series()

   # 5. Export results
   series.to_xlsx("analysis.xlsx")

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Original data is preserved - use deepcopy if needed
   series_copy = OpenTimeSeries.from_deepcopy(series)

   # Large datasets - consider resampling
   monthly_data = series.resample_to_business_period_ends(freq="BME")

   # Clean up intermediate results
   del intermediate_series

Understanding these core concepts will help you use openseries effectively and build more sophisticated financial analysis workflows.
