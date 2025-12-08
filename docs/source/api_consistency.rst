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

Method Chaining vs Object Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods that return `self` are designed for method chaining and modify the original object in place. They do NOT create new objects:

.. code-block:: python

    # CORRECT: Method chaining (modifies original)
    series.value_to_ret().plot_histogram()

    # CORRECT: Sequential operations (modifies original)
    series.value_to_ret()  # Convert to returns
    series.plot_histogram()  # Plot the returns

    # INCORRECT: This doesn't create a new object
    # returns_series = series.value_to_ret()  # Wrong pattern!

    # To create a new object, use from_deepcopy()
    returns_series = series.from_deepcopy()
    returns_series.value_to_ret()  # Now you have both original and returns

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

When creating portfolios, the weight_strat parameter can be used for built-in strategies:

.. code-block:: python

    # Using the weight_strat parameter
    portfolio_df = frame.make_portfolio(name="My Portfolio", weight_strat="eq_weights")

    # Or set custom weights
    weights = [0.5, 0.3, 0.2]
    frame.weights = weights
    portfolio_df = frame.make_portfolio(name="Custom Portfolio")

Metric Names in DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~

The `all_properties()` method has two modes:

1. **Without arguments**: Returns all properties with "tidied" display names
2. **With `properties` argument**: Returns only specified properties using their internal names

.. code-block:: python

    # Get all properties with display names
    all_metrics = frame.all_properties()
    # Access using display names
    key_metrics = all_metrics.loc[['Geometric return', 'Volatility', 'Return vol ratio', 'Max drawdown']]

    # Get only specific properties using internal names
    specific_metrics = frame.all_properties(properties=['geo_ret', 'vol', 'ret_vol_ratio', 'max_drawdown'])
    # No need to filter - only requested properties are returned

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

Best Practices
--------------

1. **ValueType is optional**: `ValueType.PRICE` is the default for `from_df()`
2. **Understand method chaining**: Methods returning `self` modify the original object, use `from_deepcopy()` to create new objects
3. **Check parameter names**: Use `observations` not `window` for rolling methods
4. **Use correct function parameters**: `eframe` and `simframe` for optimization functions
5. **Set weights before portfolio creation**: Use `frame.weights = [...]` before `make_portfolio()`
6. **Use properties parameter**: Pass specific properties to `all_properties(properties=[...])` instead of filtering post-call
7. **Unpack return values**: Handle tuples returned by `efficient_frontier()`

These notes will help you avoid common pitfalls and use openseries more effectively.
