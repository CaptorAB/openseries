API Consistency Notes
======================

This section provides important information about API consistency and common issues when using openseries.

Important API Notes
-------------------

ValueType Specification
~~~~~~~~~~~~~~~~~~~~~~~~

When creating `OpenTimeSeries` objects from price data, `ValueType.PRICE` is the default, so you don't need to specify it explicitly:

.. code-block:: python

   # Correct way (ValueType.PRICE is default)
   series = OpenTimeSeries.from_df(dframe=data['Close'])
   series.set_new_label(lvl_zero="Asset Name")

   # This also works but is unnecessary
   series = OpenTimeSeries.from_df(dframe=data['Close'], valuetype=ValueType.PRICE)
   series.set_new_label(lvl_zero="Asset Name")

Destructive Methods
~~~~~~~~~~~~~~~~~~~

Some methods modify the original series in place. Always restore the original series after calling these methods:

.. code-block:: python

   # This modifies the original series
   returns = series.value_to_ret()

   # Restore original series for further analysis
   series = OpenTimeSeries.from_df(dframe=data['Close'])
   series.set_new_label(lvl_zero="Asset Name")

Method Parameter Names
~~~~~~~~~~~~~~~~~~~~~~

Pay attention to parameter names as they may differ from what you might expect:

- `rolling_return(observations=30)` not `rolling_return(window=30)`
- `rolling_vol(observations=252)` not `rolling_vol(window=252)`
- `rolling_corr(observations=60)` not `rolling_corr(window=60)`
- `rolling_var_down(observations=252)` not `rolling_var_down(window=252)`

Function Parameter Names
~~~~~~~~~~~~~~~~~~~~~~~~

Some functions use different parameter names:

- `efficient_frontier(eframe=frame, ...)` not `efficient_frontier(frame=frame, ...)`
- `simulate_portfolios(simframe=frame, ...)` not `simulate_portfolios(frame=frame, ...)`

Properties vs Methods
~~~~~~~~~~~~~~~~~~~~~

Some items are properties, not methods:

- `frame.correl_matrix` not `frame.correl_matrix()`
- `series.vol` not `series.vol()`

Portfolio Creation
~~~~~~~~~~~~~~~~~~

When creating portfolios, use weight_strat parameter for built-in strategies:

.. code-block:: python

   # Correct way - use weight_strat parameter
   portfolio_df = frame.make_portfolio(name="My Portfolio", weight_strat="eq_weights")

   # For custom weights, set weights first then call make_portfolio
   frame.weights = [0.5, 0.3, 0.2]
   portfolio_df = frame.make_portfolio(name="Custom Portfolio")

Metric Names in DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~

When accessing metrics from `all_properties()`, use the actual column names:

.. code-block:: python

   # Correct metric names
   metrics.loc[['Geometric return', 'Volatility', 'Return vol ratio', 'Max drawdown']]

   # Not these (they don't exist)
   metrics.loc[['geo_ret', 'vol', 'ret_vol_ratio', 'max_drawdown']]

Function Return Values
~~~~~~~~~~~~~~~~~~~~~~

Some functions return tuples that need to be unpacked:

.. code-block:: python

   # efficient_frontier returns a tuple
   frontier_df, simulated_df, optimal_portfolio = efficient_frontier(eframe=frame, ...)

   # simulate_portfolios returns a DataFrame
   simulation_results = simulate_portfolios(simframe=frame, ...)

Common Issues and Solutions
---------------------------

Issue: "Do not run resample_to_business_period_ends on return series"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: `ValueType.PRICE` is the default, so you don't need to specify it explicitly.

Issue: "TypeError: 'DataFrame' object is not callable"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Use properties without parentheses: `frame.correl_matrix` not `frame.correl_matrix()`.

Issue: "TypeError: unsupported format string passed to Series.__format__"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Access scalar values using `.iloc[0]` before formatting:

.. code-block:: python

   # Correct
   print(f"VaR: {var_series.iloc[0]:.2%}")

   # Incorrect
   print(f"VaR: {var_series:.2%}")

Issue: "KeyError" when accessing metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Use the actual metric names from `all_properties()`. Check the output to see the exact names.

Best Practices
--------------

1. **ValueType is optional**: `ValueType.PRICE` is the default for `from_df()`
2. **Handle destructive methods**: Restore original series after calling `value_to_ret()`
3. **Check parameter names**: Use `observations` not `window` for rolling methods
4. **Use correct function parameters**: `eframe` and `simframe` for optimization functions
5. **Set weights before portfolio creation**: Use `frame.weights = [...]` before `make_portfolio()`
6. **Verify metric names**: Check `all_properties()` output for exact column names
7. **Unpack return values**: Handle tuples returned by `efficient_frontier()`

These notes will help you avoid common pitfalls and use openseries more effectively.
