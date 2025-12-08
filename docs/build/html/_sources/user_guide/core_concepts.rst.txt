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

    # Create a sample series using openseries simulation
    from openseries import ReturnSimulation, ValueType
    import datetime as dt

    simulation = ReturnSimulation.from_lognormal(
         number_of_sims=1,
         trading_days=100,
         mean_annual_return=0.25,  # ~0.001 daily
         mean_annual_vol=0.32,     # ~0.02 daily
         trading_days_in_year=252,
         seed=42
    )

    series = OpenTimeSeries.from_df(
         dframe=simulation.to_dataframe(name="Sample Asset", end=dt.date(2023, 12, 31)),
         valuetype=ValueType.RTRN
    ).to_cumret()  # Convert returns to cumulative prices

    # Core properties
    print(f"Name: {series.label}")
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

    # Transformations modify the original object (method chaining)
    series.value_to_ret()    # Modifies original series
    print(f"Series length: {series.length}")  # Usually length - 1

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
    series.value_to_ret()  # Modifies original
    print(f"Returns type: {series.valuetype}")

The OpenFrame Class
--------------------

Managing Multiple Series
~~~~~~~~~~~~~~~~~~~~~~~~

OpenFrame manages collections of OpenTimeSeries:

.. code-block:: python

    from openseries import OpenFrame

    # Create multiple series using openseries simulation
    simulation = ReturnSimulation.from_lognormal(
         number_of_sims=3,
         trading_days=100,
         mean_annual_return=0.25,  # ~0.001 daily
         mean_annual_vol=0.32,     # ~0.02 daily
         trading_days_in_year=252,
         seed=42
    )

    # Create OpenFrame with multiple series from simulation
    frame = OpenFrame(
         constituents=[
              OpenTimeSeries.from_df(
                    dframe=simulation.to_dataframe(name="Asset", end=dt.date(2023, 12, 31)),
                    column_nmbr=serie,
                    valuetype=ValueType.RTRN,
              ).to_cumret()  # Convert returns to cumulative prices
              for serie in range(simulation.number_of_sims)
         ]
    )

    # Frame properties
    print(f"Number of series: {frame.item_count}")
    print(f"Column names: {frame.columns_lvl_zero}")
    print(f"Common length: {frame.length}")

Data Alignment
~~~~~~~~~~~~~~~

OpenFrame concatenates series data but does **not** automatically align them.
The library provides explicit methods for alignment that require user choice:

.. code-block:: python

    # Series with different date ranges are concatenated (not aligned)
    print("Individual series lengths:")
    print(frame.lengths_of_items)

    print(f"Frame length (concatenated): {frame.length}")

    # Explicit alignment methods require user choice:

    # 1. Truncate to common date range
    frame.trunc_frame()

    # 2. Align to business day calendar (modifies original)
    frame.align_index_to_local_cdays(countries="US")

    # 3. Handle missing values (modifies original)
    frame.value_nan_handle(method="fill")

    # 4. Merge with explicit join strategy
    frame.merge_series(how="inner")
    frame.merge_series(how="outer")

Financial Calculations
----------------------

Return Calculations
~~~~~~~~~~~~~~~~~~~

openseries uses standard financial formulas:

.. code-block:: python

    # Simple returns: (P_t / P_{t-1}) - 1
    series.value_to_ret()  # Modifies original

    # Log returns: ln(P_t / P_{t-1})
    series.value_to_log()  # Modifies original

    # Cumulative returns: rebasing to start at 1.0 (modifies original)
    series.to_cumret()

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

    # Align to specific country's business days (modifies original)
    series.align_index_to_local_cdays(countries="US")

    # Multiple countries (intersection of business days) (modifies original)
    series.align_index_to_local_cdays(countries=["US", "GB"])

    # Custom markets using pandas-market-calendars (modifies original)
    series.align_index_to_local_cdays(markets="NYSE")

Resampling
~~~~~~~~~~

Convert between different frequencies:

.. code-block:: python

    # Resample to month-end (modifies original)
    series.resample_to_business_period_ends(freq="BME")

    # Resample to quarter-end (modifies original)
    series.resample_to_business_period_ends(freq="BQE")

    # Custom resampling (modifies original)
    series.resample(freq="W")

Data Validation
---------------

Type Safety
~~~~~~~~~~~

Pydantic ensures data integrity:

.. code-block:: python

    # Dates must be valid ISO format strings
    # This will fail with a validation error
    invalid_series = OpenTimeSeries.from_arrays(
         dates=["invalid-date"],
         values=[100.0]
    )

    # Values must be numeric
    # This will fail with a validation error
    invalid_series = OpenTimeSeries.from_arrays(
         dates=["2023-01-01"],
         values=["not a number"]
    )

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

Methods that modify the original object (return self for chaining):

.. code-block:: python

    # Data transformations (modify original)
    series.value_to_ret()        # Prices to returns
    series.to_drawdown_series()  # Drawdown series
    series.to_cumret()           # Cumulative returns

    # Time transformations (modify original)
    series.resample_to_business_period_ends(freq="BME")
    series.align_index_to_local_cdays(countries="US")

Methods that return new objects:

.. code-block:: python

    # Analysis methods (return new objects)
    rolling_vol = series.rolling_vol(observations=30)
    rolling_ret = series.rolling_return(observations=30)

Analysis Methods
~~~~~~~~~~~~~~~~

Methods that return calculated values:

.. code-block:: python

    # Rolling calculations
    rolling_vol = series.rolling_vol(observations=30)
    rolling_corr = frame.rolling_corr(observations=60)

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
    series.plot_series()
    series.plot_histogram()

Best Practices
--------------

Data Loading
~~~~~~~~~~~~

.. code-block:: python

    # Prefer from_df for pandas data
    series = OpenTimeSeries.from_df(dframe=dataframe['Close'])
    series.set_new_label(lvl_zero="Asset")

    # Use from_arrays for custom data
    series = OpenTimeSeries.from_arrays(dates=date_list, values=value_list)

    # Always set meaningful names
    series.set_new_label(lvl_zero="Descriptive Name")

Analysis Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # 1. Load and validate data
    series = OpenTimeSeries.from_df(dframe=data['Close'])
    series.set_new_label(lvl_zero="Asset")

    # 2. Basic analysis
    metrics = series.all_properties()

    # 3. Specific calculations
    series.to_drawdown_series()  # Convert to drawdown (modifies original)
    rolling_metrics = series.rolling_vol(observations=252)  # Returns DataFrame

    # 4. Visualization
    series.plot_series()

    # 5. Export results
    series.to_xlsx(fiilename="analysis.xlsx")

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Original data is preserved - use deepcopy if needed
    series_copy = OpenTimeSeries.from_deepcopy(series)

    # Large datasets - consider resampling (modifies original)
    series.resample_to_business_period_ends(freq="BME")

    # Clean up intermediate results
    del intermediate_series

Portfolio Construction
~~~~~~~~~~~~~~~~~~~~~~

OpenFrame provides several built-in weight strategies for portfolio construction:

.. code-block:: python

    from openseries.owntypes import MaxDiversificationNaNError, MaxDiversificationNegativeWeightsError

    # Available weight strategies
    strategies = {
         'eq_weights': 'Equal weights for all assets',
         'inv_vol': 'Inverse volatility weighting (risk parity)',
         'max_div': 'Maximum diversification optimization',
         'min_vol_overweight': 'Minimum volatility overweight strategy'
    }

    # Example with error handling
    # This may fail with MaxDiversificationNaNError or MaxDiversificationNegativeWeightsError
    portfolio_df = frame.make_portfolio(name="Max Div", weight_strat="max_div")

Understanding these core concepts will help you use openseries effectively and build more sophisticated financial analysis workflows.
