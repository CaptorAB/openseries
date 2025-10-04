Portfolio Analysis
==================

This tutorial demonstrates how to construct and analyze portfolios using openseries, including optimization techniques and performance attribution.

Setting Up the Data
--------------------

Let's start by downloading data for a diversified set of assets:

.. code-block:: python

   import yfinance as yf
   import pandas as pd
   import numpy as np
   from openseries import OpenTimeSeries, OpenFrame
   from openseries import efficient_frontier, simulate_portfolios

   # Define our universe of assets
   tickers = {
       "^GSPC": "S&P 500",
       "EFA": "EAFE International",
       "EEM": "Emerging Markets",
       "AGG": "US Aggregate Bonds",
       "VNQ": "US REITs",
       "GLD": "Gold",
       "DBC": "Commodities"
   }

   # Download 5 years of data
   series_list = []
   for ticker, name in tickers.items():
       try:
           data = yf.Ticker(ticker).history(period="5y")
           series = OpenTimeSeries.from_df(
               dframe=data['Close'],
               name=name
           )
           series.set_new_label(lvl_zero=name)
           series_list.append(series)
           print(f"Loaded {name}: {series.length} observations")
       except Exception as e:
           print(f"Failed to load {name}: {e}")

   # Create OpenFrame
   assets = OpenFrame(constituents=series_list)
   print(f"\nCreated frame with {assets.item_count} assets")
   print(f"Common date range: {assets.first_idx} to {assets.last_idx}")

Asset Analysis
--------------

First, let's analyze the individual assets:

.. code-block:: python

   # Get metrics for all assets
   asset_metrics = assets.all_properties()
   print("=== INDIVIDUAL ASSET METRICS ===")
   print(asset_metrics)

   # Key metrics comparison
   returns = asset_metrics.loc['geo_ret'] * 100
   volatilities = asset_metrics.loc['vol'] * 100
   sharpe_ratios = asset_metrics.loc['ret_vol_ratio']
   max_drawdowns = asset_metrics.loc['max_drawdown'] * 100

   print("\n=== ASSET COMPARISON ===")
   comparison_df = pd.DataFrame({
       'Annual Return (%)': returns,
       'Volatility (%)': volatilities,
       'Sharpe Ratio': sharpe_ratios,
       'Max Drawdown (%)': max_drawdowns
   })
   print(comparison_df.round(2))

Correlation Analysis
--------------------

Understanding correlations is crucial for portfolio construction:

.. code-block:: python

   # Calculate correlation matrix
   correlation_matrix = assets.correl_matrix
   print("\n=== CORRELATION MATRIX ===")
   print(correlation_matrix.round(3))

   # Identify highly correlated pairs
   print("\n=== HIGHLY CORRELATED PAIRS (>0.7) ===")
   for i in range(len(correlation_matrix.columns)):
       for j in range(i+1, len(correlation_matrix.columns)):
           corr = correlation_matrix.iloc[i, j]
           if abs(corr) > 0.7:
               asset1 = correlation_matrix.columns[i]
               asset2 = correlation_matrix.columns[j]
               print(f"{asset1} - {asset2}: {corr:.3f}")

   # Average correlation with other assets
   avg_correlations = correlation_matrix.mean()
   print("\n=== AVERAGE CORRELATIONS ===")
   for asset, avg_corr in avg_correlations.items():
       print(f"{asset}: {avg_corr:.3f}")

Simple Portfolio Construction
-----------------------------

Let's start with basic portfolio construction methods:

Equal Weight Portfolio
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create equal-weighted portfolio using native weight_strat
   portfolio_df = assets.make_portfolio(name="Equal Weight Portfolio", weight_strat="eq_weights")
   equal_weight_portfolio = OpenTimeSeries.from_df(dframe=portfolio_df.iloc[:, 0])

   print(f"Equal Weight Portfolio Return: {equal_weight_portfolio.geo_ret:.2%}")
   print(f"Equal Weight Portfolio Volatility: {equal_weight_portfolio.vol:.2%}")
   print(f"Equal Weight Portfolio Sharpe: {equal_weight_portfolio.ret_vol_ratio:.2f}")

Market Cap Weighted Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate market cap weights (in practice, you'd use actual market caps)
   # Larger weights for larger markets
   market_cap_weights = [0.50, 0.15, 0.10, 0.15, 0.05, 0.03, 0.02]  # Must sum to 1

   assets.weights = market_cap_weights
   portfolio_df = assets.make_portfolio(name="Market Cap Weighted")
   market_cap_portfolio = OpenTimeSeries.from_df(dframe=portfolio_df.iloc[:, 0])

   print(f"Market Cap Portfolio Return: {market_cap_portfolio.geo_ret:.2%}")
   print(f"Market Cap Portfolio Volatility: {market_cap_portfolio.vol:.2%}")
   print(f"Market Cap Portfolio Sharpe: {market_cap_portfolio.ret_vol_ratio:.2f}")

Risk Parity Portfolio
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use native inverse volatility weighting (risk parity)
   portfolio_df = assets.make_portfolio(name="Risk Parity", weight_strat="inv_vol")
   risk_parity_portfolio = OpenTimeSeries.from_df(dframe=portfolio_df.iloc[:, 0])

   print(f"Risk Parity Portfolio Return: {risk_parity_portfolio.geo_ret:.2%}")
   print(f"Risk Parity Portfolio Volatility: {risk_parity_portfolio.vol:.2%}")
   print(f"Risk Parity Portfolio Sharpe: {risk_parity_portfolio.ret_vol_ratio:.2f}")

Portfolio Optimization
----------------------

Now let's use openseries' optimization tools:

Efficient Frontier
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate efficient frontier
   try:
       frontier_results = efficient_frontier(
           frame=assets,
           num_portfolios=50,
           max_weight=0.4,  # Maximum 40% in any single asset
           min_weight=0.0   # No short selling
       )

       print("Efficient frontier calculated successfully")
       print(f"Number of portfolios: {len(frontier_results['returns'])}")

       # Find maximum Sharpe ratio portfolio
       sharpe_ratios = np.array(frontier_results['returns']) / np.array(frontier_results['volatilities'])
       max_sharpe_idx = np.argmax(sharpe_ratios)

       print(f"\n=== MAXIMUM SHARPE RATIO PORTFOLIO ===")
       print(f"Expected Return: {frontier_results['returns'][max_sharpe_idx]:.2%}")
       print(f"Volatility: {frontier_results['volatilities'][max_sharpe_idx]:.2%}")
       print(f"Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.2f}")

       # Get optimal weights
       optimal_weights = frontier_results['weights'][max_sharpe_idx]
       print("\nOptimal Weights:")
       for i, weight in enumerate(optimal_weights):
           asset_name = assets.constituents[i].name
           print(f"  {asset_name}: {weight:.1%}")

   except Exception as e:
       print(f"Optimization failed: {e}")

Monte Carlo Portfolio Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate random portfolios
   try:
       simulation_results = simulate_portfolios(
           frame=assets,
           num_portfolios=10000,
           max_weight=0.5,
           min_weight=0.0
       )

       print(f"\nSimulated {len(simulation_results['returns'])} random portfolios")

       # Find best performing portfolios
       sim_sharpe_ratios = np.array(simulation_results['returns']) / np.array(simulation_results['volatilities'])

       # Top 5 Sharpe ratios
       top_indices = np.argsort(sim_sharpe_ratios)[-5:]

       print("\n=== TOP 5 SIMULATED PORTFOLIOS ===")
       for i, idx in enumerate(reversed(top_indices)):
           print(f"\nRank {i+1}:")
           print(f"  Return: {simulation_results['returns'][idx]:.2%}")
           print(f"  Volatility: {simulation_results['volatilities'][idx]:.2%}")
           print(f"  Sharpe: {sim_sharpe_ratios[idx]:.2f}")

   except Exception as e:
       print(f"Simulation failed: {e}")

Portfolio Comparison
--------------------

Let's compare all our portfolios:

.. code-block:: python

   # Add all portfolios to a comparison frame
   portfolios = [equal_weight_portfolio, market_cap_portfolio, risk_parity_portfolio]

   # Add individual assets for comparison
   all_series = assets.constituents + portfolios
   comparison_frame = OpenFrame(constituents=all_series)

   # Get comprehensive metrics
   portfolio_metrics = comparison_frame.all_properties()

   # Focus on key metrics
   key_metrics = portfolio_metrics.loc[['geo_ret', 'vol', 'ret_vol_ratio', 'max_drawdown']]
   key_metrics.index = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']

   print("\n=== PORTFOLIO COMPARISON ===")
   print((key_metrics * 100).round(2))  # Convert to percentages

Risk Attribution
----------------

Analyze the risk contribution of each asset:

.. code-block:: python

   # Calculate portfolio statistics for equal weight portfolio
   returns_data = []
   for series in assets.constituents:
       returns = series.value_to_ret()
       returns_data.append(returns.tsdf.iloc[:, 0])

   # Create returns matrix
   returns_matrix = pd.concat(returns_data, axis=1)
   returns_matrix.columns = [series.name for series in assets.constituents]

   # Calculate covariance matrix (annualized)
   cov_matrix = returns_matrix.cov() * 252  # Assuming daily data

   # Portfolio weights (equal weight)
   weights = np.array(equal_weights)

   # Portfolio variance
   portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
   portfolio_volatility = np.sqrt(portfolio_variance)

   # Marginal contribution to risk
   marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility

   # Component contribution to risk
   component_contrib = weights * marginal_contrib

   # Percentage contribution
   percent_contrib = component_contrib / portfolio_volatility

   print("\n=== RISK ATTRIBUTION (Equal Weight Portfolio) ===")
   risk_attribution = pd.DataFrame({
       'Weight': weights,
       'Marginal Contrib': marginal_contrib,
       'Component Contrib': component_contrib,
       'Percent Contrib': percent_contrib
   }, index=[series.name for series in assets.constituents])

   print(risk_attribution.round(4))

Performance Attribution
-----------------------

Analyze performance contribution over time:

.. code-block:: python

   # Calculate individual asset returns
   asset_returns = []
   for series in assets.constituents:
       returns = series.value_to_ret()
       asset_returns.append(returns.tsdf.iloc[:, 0])

   returns_df = pd.concat(asset_returns, axis=1)
   returns_df.columns = [series.name for series in assets.constituents]

   # Calculate weighted returns (equal weight portfolio)
   weighted_returns = returns_df * equal_weights

   # Cumulative contribution
   cumulative_contrib = (1 + weighted_returns).cumprod()

   print("\n=== PERFORMANCE ATTRIBUTION ===")
   print("Final cumulative contribution by asset:")
   final_contrib = cumulative_contrib.iloc[-1]
   for asset, contrib in final_contrib.items():
       print(f"  {asset}: {contrib:.3f}")

Rolling Portfolio Analysis
--------------------------

Analyze how portfolio characteristics change over time:

.. code-block:: python

   # Rolling correlation with market (S&P 500)
   market_proxy = assets.constituents[0]  # Assuming first asset is S&P 500

   # Create frame with portfolio and market
   portfolio_vs_market = OpenFrame(constituents=[equal_weight_portfolio, market_proxy])

   # Calculate rolling correlation
   rolling_corr = portfolio_vs_market.rolling_corr(window=252)  # 1-year rolling

   print(f"\nRolling correlation calculated for {len(rolling_corr)} periods")
   print(f"Average correlation: {rolling_corr.mean().iloc[0]:.3f}")
   print(f"Correlation range: {rolling_corr.min().iloc[0]:.3f} to {rolling_corr.max().iloc[0]:.3f}")

   # Rolling portfolio volatility
   portfolio_rolling_vol = equal_weight_portfolio.rolling_vol(observations=252)

   print(f"\nRolling volatility statistics:")
   print(f"Average volatility: {portfolio_rolling_vol.mean().iloc[0]:.2%}")
   print(f"Volatility range: {portfolio_rolling_vol.min().iloc[0]:.2%} to {portfolio_rolling_vol.max().iloc[0]:.2%}")

Rebalancing Analysis
--------------------

Analyze the impact of rebalancing frequency:

.. code-block:: python

   # Simulate different rebalancing frequencies
   # This is a simplified example - in practice you'd implement full rebalancing logic

   # Monthly rebalancing
   monthly_data = assets.resample_to_business_period_ends(freq="BME")
   monthly_data.weights = equal_weights
   monthly_portfolio_df = monthly_data.make_portfolio(name="Monthly Rebalanced")
   monthly_portfolio = OpenTimeSeries.from_df(dframe=monthly_portfolio_df.iloc[:, 0])

   # Quarterly rebalancing
   quarterly_data = assets.resample_to_business_period_ends(freq="BQE")
   quarterly_data.weights = equal_weights
   quarterly_portfolio_df = quarterly_data.make_portfolio(name="Quarterly Rebalanced")
   quarterly_portfolio = OpenTimeSeries.from_df(dframe=quarterly_portfolio_df.iloc[:, 0])

   # Compare rebalancing frequencies
   rebalancing_comparison = OpenFrame(constituents=[
       equal_weight_portfolio,  # Daily rebalanced (theoretical)
       monthly_portfolio,
       quarterly_portfolio
   ])

   rebal_metrics = rebalancing_comparison.all_properties()
   print("\n=== REBALANCING FREQUENCY COMPARISON ===")
   print(rebal_metrics.loc[['geo_ret', 'vol', 'ret_vol_ratio']].round(4))

Stress Testing
--------------

Test portfolio performance during market stress:

.. code-block:: python

   # Identify worst periods for the market
   market_returns = market_proxy.value_to_ret()
   market_returns_df = market_returns.tsdf

   # Find worst 5% of days
   worst_days_threshold = market_returns_df.quantile(0.05).iloc[0]
   worst_days = market_returns_df[market_returns_df.iloc[:, 0] <= worst_days_threshold]

   print(f"\n=== STRESS TEST RESULTS ===")
   print(f"Market stress threshold: {worst_days_threshold:.2%}")
   print(f"Number of stress days: {len(worst_days)}")

   # Portfolio performance during stress
   portfolio_returns = equal_weight_portfolio.value_to_ret()
   portfolio_returns_df = portfolio_returns.tsdf

   # Align dates and calculate portfolio performance during market stress
   stress_dates = worst_days.index
   portfolio_stress_returns = portfolio_returns_df.loc[stress_dates]

   print(f"Portfolio average return during stress: {portfolio_stress_returns.mean().iloc[0]:.2%}")
   print(f"Portfolio worst day during stress: {portfolio_stress_returns.min().iloc[0]:.2%}")

Summary Report
--------------

Generate a comprehensive portfolio analysis report:

.. code-block:: python

   print("\n" + "="*60)
   print("PORTFOLIO ANALYSIS SUMMARY REPORT")
   print("="*60)

   print(f"\nAnalysis Period: {assets.first_idx} to {assets.last_idx}")
   print(f"Number of Assets: {assets.item_count}")
   print(f"Asset Universe: {', '.join([s.name for s in assets.constituents])}")

   print(f"\n--- EQUAL WEIGHT PORTFOLIO PERFORMANCE ---")
   print(f"Total Return: {equal_weight_portfolio.value_ret:.2%}")
   print(f"Annualized Return: {equal_weight_portfolio.geo_ret:.2%}")
   print(f"Annualized Volatility: {equal_weight_portfolio.vol:.2%}")
   print(f"Sharpe Ratio: {equal_weight_portfolio.ret_vol_ratio:.2f}")
   print(f"Maximum Drawdown: {equal_weight_portfolio.max_drawdown:.2%}")
   print(f"95% VaR (daily): {equal_weight_portfolio.var_down:.2%}")

   print(f"\n--- PORTFOLIO CHARACTERISTICS ---")
   avg_correlation = correlation_matrix.mean().mean()
   print(f"Average Asset Correlation: {avg_correlation:.3f}")
   print(f"Portfolio Diversification Benefit: {(asset_metrics.loc['vol'].mean() - equal_weight_portfolio.vol):.2%}")

   # Export results
   portfolio_metrics.to_excel("portfolio_analysis.xlsx")
   correlation_matrix.to_excel("correlation_matrix.xlsx")

   print(f"\nResults exported to Excel files")
   print("Analysis complete!")

This tutorial provides a comprehensive framework for portfolio analysis using openseries. You can extend these techniques for more sophisticated portfolio management strategies.
