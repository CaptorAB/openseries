Advanced Features
=================

This tutorial covers advanced openseries features including custom analysis, integration with other libraries, and extending functionality.

Factor Analysis and Regression
------------------------------

Multi-Factor Model Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load factor data (Fama-French factors would be ideal, using proxies here)
   factor_tickers = {
       "^GSPC": "Market",
       "^RUT": "Small Cap",  # Size factor proxy
       "EFA": "International",  # International factor
       "TLT": "Bonds"  # Interest rate factor
   }

   # Load factor data
   factor_series = []
   for ticker, name in factor_tickers.items():
       try:
           data = yf.Ticker(ticker).history(period="3y")
           series = OpenTimeSeries.from_df(dframe=data['Close'], name=name)
           factor_series.append(series)
       except:
           print(f"Failed to load {name}")

   # Create factor frame
   factors = OpenFrame(constituents=factor_series)

   # Load individual stock for analysis
   stock_data = yf.Ticker("AAPL").history(period="3y")
   apple = OpenTimeSeries.from_df(dframe=stock_data['Close'], name="Apple")

   # Add stock to factor frame for regression
   analysis_frame = OpenFrame(constituents=factor_series + [apple])

   # Perform multi-factor regression
   try:
       regression_results = analysis_frame.multi_factor_linear_regression(
           dependent_variable_idx=-1  # Apple is the last series (dependent variable)
       )

       print("\n=== MULTI-FACTOR REGRESSION RESULTS ===")
       print("Regression Summary:")
       print(regression_results['summary'])

       print("\nFactor Loadings (Betas):")
       for i, factor_name in enumerate([s.name for s in factor_series]):
           beta = regression_results['coefficients'][i+1]  # Skip intercept
           print(f"  {factor_name}: {beta:.4f}")

       print(f"\nR-squared: {regression_results['r_squared']:.4f}")
       print(f"Adjusted R-squared: {regression_results['adj_r_squared']:.4f}")

   except Exception as e:
       print(f"Regression analysis failed: {e}")

Rolling Factor Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Rolling beta analysis with market
   market_series = factor_series[0]  # S&P 500
   stock_vs_market = OpenFrame(constituents=[apple, market_series])

   # Calculate rolling beta
   rolling_beta = stock_vs_market.rolling_beta(window=252)  # 1-year rolling

   print(f"\n=== ROLLING BETA ANALYSIS ===")
   print(f"Current Beta: {rolling_beta.iloc[-1, 0]:.3f}")
   print(f"Average Beta: {rolling_beta.mean().iloc[0]:.3f}")
   print(f"Beta Range: {rolling_beta.min().iloc[0]:.3f} to {rolling_beta.max().iloc[0]:.3f}")
   print(f"Beta Volatility: {rolling_beta.std().iloc[0]:.3f}")

   # Rolling correlation
   rolling_corr = stock_vs_market.rolling_corr(window=252)

   print(f"\n=== ROLLING CORRELATION ANALYSIS ===")
   print(f"Current Correlation: {rolling_corr.iloc[-1, 0]:.3f}")
   print(f"Average Correlation: {rolling_corr.mean().iloc[0]:.3f}")
   print(f"Correlation Range: {rolling_corr.min().iloc[0]:.3f} to {rolling_corr.max().iloc[0]:.3f}")


This tutorial demonstrates how to extend openseries with advanced functionality for sophisticated financial analysis workflows.
