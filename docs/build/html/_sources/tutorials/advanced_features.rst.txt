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

Advanced Portfolio Techniques
-----------------------------

Black-Litterman Model Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def black_litterman_weights(returns_df, market_caps, risk_aversion=3, tau=0.025):
       """
       Simplified Black-Litterman model implementation
       """
       # Calculate market-implied returns
       cov_matrix = returns_df.cov() * 252  # Annualized
       market_weights = np.array(market_caps) / sum(market_caps)

       # Market-implied equilibrium returns
       pi = risk_aversion * np.dot(cov_matrix, market_weights)

       # Without views, BL reduces to market weights
       # In practice, you would incorporate investor views here

       # Calculate BL weights (simplified - no views)
       bl_weights = market_weights  # Without views, equals market weights

       return bl_weights, pi

   # Example with our portfolio assets
   if 'portfolio_assets' in locals():
       # Get returns data
       returns_data = []
       for series in portfolio_assets.constituents:
           returns = series.value_to_ret()
           returns_data.append(returns.tsdf.iloc[:, 0])

       returns_df = pd.concat(returns_data, axis=1)
       returns_df.columns = [s.name for s in portfolio_assets.constituents]

       # Simulate market caps (in practice, use real market cap data)
       market_caps = [1000, 800, 900, 300, 500, 400, 200, 100]  # Billions

       bl_weights, implied_returns = black_litterman_weights(returns_df, market_caps)

       print("\n=== BLACK-LITTERMAN MODEL ===")
       print("Market-implied returns:")
       for i, (asset, ret) in enumerate(zip(returns_df.columns, implied_returns)):
           print(f"  {asset}: {ret:.2%}")

       print("\nBlack-Litterman weights:")
       for i, (asset, weight) in enumerate(zip(returns_df.columns, bl_weights)):
           print(f"  {asset}: {weight:.1%}")

Risk Parity Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def risk_parity_weights(cov_matrix, max_iterations=1000, tolerance=1e-8):
       """
       Calculate risk parity weights using iterative algorithm
       """
       n = len(cov_matrix)
       weights = np.ones(n) / n  # Start with equal weights

       for iteration in range(max_iterations):
           # Calculate risk contributions
           portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
           marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
           risk_contrib = weights * marginal_contrib

           # Target risk contribution (equal for all assets)
           target_risk = portfolio_vol / n

           # Update weights
           weights = weights * (target_risk / risk_contrib)
           weights = weights / weights.sum()  # Normalize

           # Check convergence
           risk_contrib_new = weights * np.dot(cov_matrix, weights) / portfolio_vol
           if np.max(np.abs(risk_contrib_new - target_risk)) < tolerance:
               break

       return weights

   # Calculate risk parity weights
   if 'returns_df' in locals():
       cov_matrix = returns_df.cov().values * 252  # Annualized
       rp_weights = risk_parity_weights(cov_matrix)

       print("\n=== RISK PARITY WEIGHTS ===")
       for i, (asset, weight) in enumerate(zip(returns_df.columns, rp_weights)):
           print(f"  {asset}: {weight:.1%}")

       # Create risk parity portfolio
       if 'portfolio_assets' in locals():
           rp_portfolio = portfolio_assets.make_portfolio(
               weights=rp_weights.tolist(),
               name="Risk Parity Portfolio"
           )

           print(f"\nRisk Parity Portfolio Metrics:")
           print(f"  Return: {rp_portfolio.geo_ret:.2%}")
           print(f"  Volatility: {rp_portfolio.vol:.2%}")
           print(f"  Sharpe: {rp_portfolio.ret_vol_ratio:.3f}")





This tutorial demonstrates how to extend openseries with advanced functionality for sophisticated financial analysis workflows.
