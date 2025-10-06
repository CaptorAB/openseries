Basic Financial Analysis
========================

This tutorial demonstrates how to perform fundamental financial analysis using openseries with real market data.

Setting Up
----------

First, let's import the necessary libraries and download some data:

.. code-block:: python

   import yfinance as yf
   import pandas as pd
   import numpy as np
   from openseries import OpenTimeSeries, OpenFrame
   from datetime import date, datetime

   # Download S&P 500 data for the last 5 years
   ticker = yf.Ticker("^GSPC")
   data = ticker.history(period="5y")

   # Create OpenTimeSeries
   sp500 = OpenTimeSeries.from_df(
       dframe=data['Close']
   )

   # Set a descriptive label
   sp500.set_new_label(lvl_zero="S&P 500 Index")

   print(f"Loaded {sp500.length} observations")
   print(f"Date range: {sp500.first_idx} to {sp500.last_idx}")

Basic Performance Metrics
--------------------------

Let's calculate the fundamental performance metrics:

.. code-block:: python

   # Total return over the period
   total_return = sp500.value_ret
   print(f"Total Return: {total_return:.2%}")

   # Annualized return (CAGR)
   annual_return = sp500.geo_ret
   print(f"Annualized Return (CAGR): {annual_return:.2%}")

   # Arithmetic mean return
   arithmetic_return = sp500.arithmetic_ret
   print(f"Arithmetic Mean Return: {arithmetic_return:.2%}")

   # Time period analysis
   print(f"Investment period: {sp500.yearfrac:.2f} years")
   print(f"Number of observations: {sp500.length}")
   print(f"Periods per year: {sp500.periods_in_a_year:.1f}")

Risk Analysis
-------------

Now let's examine the risk characteristics:

.. code-block:: python

   # Volatility (annualized standard deviation)
   volatility = sp500.vol
   print(f"Annualized Volatility: {volatility:.2%}")

   # Downside deviation (volatility of negative returns only)
   downside_vol = sp500.downside_deviation
   print(f"Downside Deviation: {downside_vol:.2%}")

   # Value at Risk (95% confidence level)
   var_95 = sp500.var_down
   print(f"95% Value at Risk (daily): {var_95:.2%}")

   # Conditional Value at Risk (Expected Shortfall)
   cvar_95 = sp500.cvar_down
   print(f"95% CVaR (daily): {cvar_95:.2%}")

   # Maximum single-day loss
   worst_day = sp500.worst
   print(f"Worst single day: {worst_day:.2%}")

Risk-Adjusted Returns
---------------------

Calculate risk-adjusted performance metrics:

.. code-block:: python

   # Sharpe Ratio (return per unit of total risk)
   sharpe_ratio = sp500.ret_vol_ratio
   print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

   # Sortino Ratio (return per unit of downside risk)
   sortino_ratio = sp500.sortino_ratio
   print(f"Sortino Ratio: {sortino_ratio:.2f}")

   # Kappa-3 Ratio (penalizes larger downside deviations more)
   kappa3_ratio = sp500.kappa3_ratio
   print(f"Kappa-3 Ratio: {kappa3_ratio:.2f}")

   # Omega Ratio
   omega_ratio = sp500.omega_ratio
   print(f"Omega Ratio: {omega_ratio:.2f}")

Drawdown Analysis
-----------------

Analyze drawdowns to understand downside risk:

.. code-block:: python

   # Maximum drawdown
   max_drawdown = sp500.max_drawdown
   max_dd_date = sp500.max_drawdown_date
   print(f"Maximum Drawdown: {max_drawdown:.2%}")
   print(f"Max Drawdown Date: {max_dd_date}")

   # Create drawdown series for visualization (modifies original)
   sp500.to_drawdown_series()

   # Plot drawdowns
   sp500.plot_series()
   # This will open an interactive plot in your browser

   # Worst calendar year drawdown
   worst_year_dd = sp500.max_drawdown_cal_year
   print(f"Worst Calendar Year Drawdown: {worst_year_dd:.2%}")

Distribution Analysis
---------------------

Examine the return distribution characteristics:

.. code-block:: python

   # Convert to returns for distribution analysis (modifies original)
   sp500.value_to_ret()

   # Note: value_to_ret() modifies the original series in place
   # Restore the original series for further analysis
   sp500 = OpenTimeSeries.from_df(dframe=data['Close'])
   sp500.set_new_label(lvl_zero="S&P 500 Index")

   # Skewness (asymmetry of the distribution)
   skewness = sp500.skew
   print(f"Skewness: {skewness:.2f}")
   if skewness < 0:
       print("  → Negative skew: more extreme negative returns")
   elif skewness > 0:
       print("  → Positive skew: more extreme positive returns")

   # Kurtosis (tail heaviness)
   kurtosis = sp500.kurtosis
   print(f"Kurtosis: {kurtosis:.2f}")
   if kurtosis > 3:
       print("  → Fat tails: more extreme returns than normal distribution")

   # Percentage of positive days
   positive_share = sp500.positive_share
   print(f"Positive Days: {positive_share:.1%}")

   # Current Z-score (how unusual is the last return?)
   z_score = sp500.z_score
   print(f"Last Return Z-score: {z_score:.2f}")

Monthly and Annual Analysis
---------------------------

Break down performance by different time periods:

.. code-block:: python

   # Resample to monthly data (modifies original)
   sp500.resample_to_business_period_ends(freq="BME")
   print(f"Monthly observations: {sp500.length}")

   # Monthly metrics
   monthly_return = sp500.geo_ret
   monthly_vol = sp500.vol
   print(f"Monthly Return (annualized): {monthly_return:.2%}")
   print(f"Monthly Volatility (annualized): {monthly_vol:.2%}")

   # Worst month
   worst_month = sp500.worst_month
   print(f"Worst Month: {worst_month:.2%}")

   # Annual data (modifies original)
   sp500.resample_to_business_period_ends(freq="BYE")
   print(f"Annual observations: {sp500.length}")

Calendar Year Returns
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate calendar year returns
   years = range(2019, 2025)  # Adjust based on your data range

   for year in years:
       # This may fail if no data exists for the year
       year_return = sp500.value_ret_calendar_period(year=year)
       print(f"{year}: {year_return:.2%}")

Rolling Analysis
----------------

Analyze how metrics change over time:

.. code-block:: python

   # 252-day (1-year) rolling volatility
   rolling_vol = sp500.rolling_vol(observations=252)
   print(f"Rolling volatility calculated for {len(rolling_vol)} periods")

   # 30-day rolling returns
   rolling_returns = sp500.rolling_return(observations=30)

   # Plot rolling volatility
   # Convert to OpenTimeSeries for plotting
   vol_dates = rolling_vol.index.strftime('%Y-%m-%d').tolist()
   vol_values = rolling_vol.iloc[:, 0].tolist()

   vol_series = OpenTimeSeries.from_arrays(
       dates=vol_dates,
       values=vol_values,
       name="Rolling Volatility"
   )

   vol_series.plot_series()

Comprehensive Report
--------------------

Get all metrics at once:

.. code-block:: python

   # Generate comprehensive metrics report
   all_metrics = sp500.all_properties()
   print("\n=== COMPREHENSIVE ANALYSIS REPORT ===")
   print(all_metrics)

   # Save to Excel for further analysis
   sp500.to_xlsx("sp500_analysis.xlsx")
   all_metrics.to_excel("sp500_metrics.xlsx")

Visualization
-------------

Create various visualizations:

.. code-block:: python

   # Price chart
   sp500.plot_series()

   # Returns bar plot and histogram
   returns = sp500.from_deepcopy()
   returns.value_to_ret()
   returns.plot_bars()
   returns.plot_histogram()

   # Drawdown chart
   sp500.to_drawdown_series()
   sp500.plot_series()

Comparison with Benchmark
-------------------------

Let's compare with a bond index:

.. code-block:: python

   # Download bond data (10-year Treasury)
   bond_ticker = yf.Ticker("^TNX")
   bond_data = bond_ticker.history(period="5y")

   # Create bond series (using yield data)
   bonds = OpenTimeSeries.from_df(
       dframe=bond_data['Close']
   )
   bonds.set_new_label(lvl_zero="10Y Treasury Yield")

   # Create frame for comparison
   comparison_frame = OpenFrame(constituents=[sp500, bonds])

   # Compare metrics
   comparison_metrics = comparison_frame.all_properties()
   print("\n=== ASSET COMPARISON ===")
   print(comparison_metrics)

   # Calculate correlation
   correlation_matrix = comparison_frame.correl_matrix
   print("\n=== CORRELATION MATRIX ===")
   print(correlation_matrix)

Advanced Risk Metrics
---------------------

Calculate some advanced risk measures:

.. code-block:: python

   # VaR at different confidence levels
   var_90 = sp500.var_down_func(level=0.90)
   var_95 = sp500.var_down_func(level=0.95)
   var_99 = sp500.var_down_func(level=0.99)

   print(f"90% VaR: {var_90:.2%}")
   print(f"95% VaR: {var_95:.2%}")
   print(f"99% VaR: {var_99:.2%}")

   # CVaR at different confidence levels
   cvar_90 = sp500.cvar_down_func(level=0.90)
   cvar_95 = sp500.cvar_down_func(level=0.95)
   cvar_99 = sp500.cvar_down_func(level=0.99)

   print(f"90% CVaR: {cvar_90:.2%}")
   print(f"95% CVaR: {cvar_95:.2%}")
   print(f"99% CVaR: {cvar_99:.2%}")

   # Implied volatility from VaR (assuming normal distribution)
   vol_from_var = sp500.vol_from_var
   print(f"Volatility implied from VaR: {vol_from_var:.2%}")
   print(f"Actual volatility: {sp500.vol:.2%}")

Summary and Interpretation
--------------------------

.. code-block:: python

   print("\n=== INVESTMENT SUMMARY ===")
   print(f"Asset: {sp500.label}")
   print(f"Period: {sp500.first_idx} to {sp500.last_idx}")
   print(f"Total Return: {sp500.value_ret:.2%}")
   print(f"Annualized Return: {sp500.geo_ret:.2%}")
   print(f"Annualized Volatility: {sp500.vol:.2%}")
   print(f"Sharpe Ratio: {sp500.ret_vol_ratio:.2f}")
   print(f"Maximum Drawdown: {sp500.max_drawdown:.2%}")
   print(f"95% VaR (daily): {sp500.var_down:.2%}")

   # Risk assessment
   if sp500.ret_vol_ratio > 1.0:
       print("✓ Good risk-adjusted returns (Sharpe > 1.0)")
   else:
       print("⚠ Moderate risk-adjusted returns (Sharpe < 1.0)")

   if abs(sp500.max_drawdown) < 0.20:
       print("✓ Moderate maximum drawdown (< 20%)")
   else:
       print("⚠ Significant maximum drawdown (> 20%)")

This tutorial provides a comprehensive foundation for financial analysis using openseries. You can adapt these techniques for any financial time series data.
