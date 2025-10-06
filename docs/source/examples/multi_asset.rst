Multi-Asset Analysis
====================

This example shows how to analyze multiple assets simultaneously using OpenFrame.

Setting Up Multi-Asset Analysis
--------------------------------

.. code-block:: python

   import yfinance as yf
   from openseries import OpenTimeSeries, OpenFrame
   import pandas as pd
   import numpy as np

   # Define asset universe
   assets = {
       "AAPL": "Apple Inc.",
       "GOOGL": "Alphabet Inc.",
       "MSFT": "Microsoft Corp.",
       "AMZN": "Amazon.com Inc.",
       "TSLA": "Tesla Inc.",
       "NVDA": "NVIDIA Corp.",
       "META": "Meta Platforms Inc.",
       "NFLX": "Netflix Inc."
   }

   # Download data for all assets
   series_list = []
   for ticker, name in assets.items():
       try:
           data = yf.Ticker(ticker).history(period="3y")
           series = OpenTimeSeries.from_df(
               dframe=data['Close']
           )
           series.set_new_label(lvl_zero=name)
           series_list.append(series)
           print(f"Loaded {name}: {series.length} observations")
       except Exception as e:
           print(f"Failed to load {name}: {e}")

   # Create OpenFrame
   tech_stocks = OpenFrame(constituents=series_list)
   print(f"\nCreated frame with {tech_stocks.item_count} assets")
   print(f"Common period: {tech_stocks.first_idx} to {tech_stocks.last_idx}")

Comparative Analysis
--------------------

.. code-block:: python

   # Get metrics for all assets
   all_metrics = tech_stocks.all_properties()
   print("=== COMPARATIVE METRICS ===")
   print(all_metrics)

   # Focus on key metrics
   key_metrics = all_metrics.loc[['Geometric return', 'Volatility', 'Return vol ratio', 'Max drawdown']]
   key_metrics.index = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']

   # Convert to percentages for better readability
   percentage_metrics = key_metrics.copy()
   percentage_metrics.loc[['Annual Return', 'Volatility', 'Max Drawdown']] *= 100

   print("\n=== KEY METRICS COMPARISON ===")
   print(percentage_metrics.round(2))

Ranking Analysis
----------------

.. code-block:: python

   # Rank assets by different criteria
   rankings = pd.DataFrame(index=all_metrics.columns)

   # Rank by return (higher is better)
   rankings['Return Rank'] = all_metrics.loc['Geometric return'].rank(ascending=False)

   # Rank by volatility (lower is better)
   rankings['Volatility Rank'] = all_metrics.loc['Volatility'].rank(ascending=True)

   # Rank by Sharpe ratio (higher is better)
   rankings['Sharpe Rank'] = all_metrics.loc['Return vol ratio'].rank(ascending=False)

   # Rank by max drawdown (higher/less negative is better)
   rankings['Drawdown Rank'] = all_metrics.loc['Max drawdown'].rank(ascending=False)

   # Overall rank (average of all ranks)
   rankings['Overall Rank'] = rankings.mean(axis=1)
   rankings = rankings.sort_values('Overall Rank')

   print("\n=== ASSET RANKINGS ===")
   print(rankings.round(1))

Correlation Analysis
--------------------

.. code-block:: python

   # Calculate correlation matrix
   correlation_matrix = tech_stocks.correl_matrix()
   print("\n=== CORRELATION MATRIX ===")
   print(correlation_matrix.round(3))

   # Find most and least correlated pairs
   corr_pairs = []
   for i in range(len(correlation_matrix.columns)):
       for j in range(i+1, len(correlation_matrix.columns)):
           asset1 = correlation_matrix.columns[i]
           asset2 = correlation_matrix.columns[j]
           corr = correlation_matrix.iloc[i, j]
           corr_pairs.append((asset1, asset2, corr))

   # Sort by correlation
   corr_pairs.sort(key=lambda x: x[2], reverse=True)

   print("\n=== HIGHEST CORRELATIONS ===")
   for asset1, asset2, corr in corr_pairs[:5]:
       print(f"{asset1} - {asset2}: {corr:.3f}")

   print("\n=== LOWEST CORRELATIONS ===")
   for asset1, asset2, corr in corr_pairs[-5:]:
       print(f"{asset1} - {asset2}: {corr:.3f}")

Risk-Return Analysis
--------------------

.. code-block:: python

   # Create risk-return scatter data
   returns = all_metrics.loc['Geometric return'] * 100
   volatilities = all_metrics.loc['Volatility'] * 100
   sharpe_ratios = all_metrics.loc['Return vol ratio']

   risk_return_df = pd.DataFrame({
       'Asset': returns.index,
       'Return (%)': returns.values,
       'Volatility (%)': volatilities.values,
       'Sharpe Ratio': sharpe_ratios.values
   })

   print("\n=== RISK-RETURN ANALYSIS ===")
   print(risk_return_df.round(2))

   # Identify efficient assets (high return per unit risk)
   efficient_threshold = sharpe_ratios.quantile(0.75)
   efficient_assets = sharpe_ratios[sharpe_ratios >= efficient_threshold]

   print(f"\n=== MOST EFFICIENT ASSETS (Sharpe >= {efficient_threshold:.2f}) ===")
   for asset, sharpe in efficient_assets.sort_values(ascending=False).items():
       print(f"{asset}: {sharpe:.2f}")

Sector/Style Analysis
---------------------

.. code-block:: python

   # Group assets by characteristics (example grouping)
   asset_groups = {
       'Mega Cap': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.'],
       'Growth': ['Tesla Inc.', 'NVIDIA Corp.', 'Netflix Inc.'],
       'Social Media': ['Meta Platforms Inc.']
   }

   print("\n=== GROUP ANALYSIS ===")
   for group_name, group_assets in asset_groups.items():
       # Filter assets that exist in our data
       group_series = [s for s in tech_stocks.constituents if s.name in group_assets]

       if group_series:
           group_frame = OpenFrame(constituents=group_series)
           group_metrics = group_frame.all_properties()

           avg_return = group_metrics.loc['Geometric return'].mean()
           avg_vol = group_metrics.loc['Volatility'].mean()
           avg_sharpe = group_metrics.loc['Return vol ratio'].mean()

           print(f"\n{group_name} ({len(group_series)} assets):")
           print(f"  Average Return: {avg_return:.2%}")
           print(f"  Average Volatility: {avg_vol:.2%}")
           print(f"  Average Sharpe: {avg_sharpe:.2f}")

Time Series Analysis
--------------------

.. code-block:: python

   # Rolling correlation analysis
   # Pick two assets for detailed analysis
   apple = next(s for s in tech_stocks.constituents if "Apple" in s.name)
   microsoft = next(s for s in tech_stocks.constituents if "Microsoft" in s.name)

   pair_frame = OpenFrame(constituents=[apple, microsoft])
   rolling_corr = pair_frame.rolling_corr(window=252)  # 1-year rolling

   print(f"\n=== ROLLING CORRELATION: {apple.name} vs {microsoft.name} ===")
   print(f"Current correlation: {rolling_corr.iloc[-1, 0]:.3f}")
   print(f"Average correlation: {rolling_corr.mean().iloc[0]:.3f}")
   print(f"Correlation range: {rolling_corr.min().iloc[0]:.3f} to {rolling_corr.max().iloc[0]:.3f}")

Performance Attribution
-----------------------

.. code-block:: python

   # Create equal-weighted portfolio for attribution
   portfolio_df = tech_stocks.make_portfolio(name="Tech Portfolio", weight_strat="eq_weights")
   portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)

   print(f"\n=== PORTFOLIO vs INDIVIDUAL ASSETS ===")
   print(f"Portfolio Return: {portfolio.geo_ret:.2%}")
   print(f"Portfolio Volatility: {portfolio.vol:.2%}")
   print(f"Portfolio Sharpe: {portfolio.ret_vol_ratio:.2f}")

   # Compare with individual assets using OpenFrame
   asset_metrics = tech_stocks.all_properties()
   individual_returns = asset_metrics.loc['Geometric return'].values
   individual_vols = asset_metrics.loc['Volatility'].values

   print(f"\nDiversification benefit:")
   equal_weights = [1/tech_stocks.item_count] * tech_stocks.item_count
   print(f"  Weighted avg return: {np.average(individual_returns, weights=equal_weights):.2%}")
   print(f"  Portfolio return: {portfolio.geo_ret:.2%}")
   print(f"  Weighted avg volatility: {np.average(individual_vols, weights=equal_weights):.2%}")
   print(f"  Portfolio volatility: {portfolio.vol:.2%}")
   print(f"  Volatility reduction: {(np.average(individual_vols, weights=equal_weights) - portfolio.vol):.2%}")

Stress Testing
--------------

.. code-block:: python

   # Identify worst market days
   market_proxy = tech_stocks.constituents[0]  # Use first asset as market proxy
   market_returns = market_proxy.value_to_ret()
   market_data = market_returns.tsdf
   # Find worst 5% of days
   worst_threshold = market_data.quantile(0.05)
   worst_days = market_data[market_data <= worst_threshold]

   print(f"\n=== STRESS TEST ANALYSIS ===")
   print(f"Market stress threshold: {worst_threshold:.2%}")
   print(f"Number of stress days: {len(worst_days)}")

   # Analyze each asset's performance during stress
   print("\nAsset performance during market stress:")
   for series in tech_stocks.constituents:
       asset_returns = series.value_to_ret()
       asset_data = asset_returns.tsdf
       # Get returns on stress days
       stress_returns = asset_data.loc[worst_days.index]
       avg_stress_return = stress_returns.mean()

       print(f"  {series.name}: {avg_stress_return:.2%}")

Export Multi-Asset Results
--------------------------

.. code-block:: python

   # Export comprehensive analysis
   with pd.ExcelWriter('multi_asset_analysis.xlsx') as writer:
       # All metrics
       all_metrics.to_excel(writer, sheet_name='All Metrics')

       # Rankings
       rankings.to_excel(writer, sheet_name='Rankings')

       # Correlations
       correlation_matrix.to_excel(writer, sheet_name='Correlations')

       # Risk-return data
       risk_return_df.to_excel(writer, sheet_name='Risk Return', index=False)

       # Individual series data
       tech_stocks.to_xlsx(writer, sheet_name='Price Data')

   print("\nMulti-asset analysis exported to 'multi_asset_analysis.xlsx'")

Complete Multi-Asset Analysis Workflow
---------------------------------------

Here's how to perform a complete multi-asset analysis using openseries methods directly:

.. code-block:: python

   # Example: Analyze tech stocks using openseries methods
   tech_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

   # Load data using openseries methods
   series_list = []
   for ticker in tech_tickers:
       try:
           data = yf.Ticker(ticker).history(period="3y")
           series = OpenTimeSeries.from_df(dframe=data['Close'])
           series.set_new_label(lvl_zero=ticker)
           series_list.append(series)
       except:
           print(f"Failed to load {ticker}")

   if not series_list:
       print("No data loaded")
   else:
       # Create frame using openseries
       frame = OpenFrame(constituents=series_list)

       # Analysis using openseries properties and methods
       print(f"=== MULTI-ASSET ANALYSIS ===")
       print(f"Assets: {frame.item_count}")
       print(f"Period: {frame.first_idx} to {frame.last_idx}")

       # Key metrics using openseries all_properties method
       metrics = frame.all_properties()
       key_metrics = metrics.loc[['Geometric return', 'Volatility', 'Return vol ratio', 'Max drawdown']]

       print("\nKey Metrics:")
       print((key_metrics * 100).round(2))  # Convert to percentages

       # Correlations using openseries correl_matrix method
       correlations = frame.correl_matrix()
       avg_correlation = correlations.mean().mean()
       print(f"\nAverage correlation: {avg_correlation:.3f}")

       # Create portfolio using openseries make_portfolio method
       portfolio_df = frame.make_portfolio(name="Equal Weight", weight_strat="eq_weights")
       portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)

       print(f"\nEqual-weight portfolio:")
       print(f"  Return: {portfolio.geo_ret:.2%}")
       print(f"  Volatility: {portfolio.vol:.2%}")
       print(f"  Sharpe: {portfolio.ret_vol_ratio:.2f}")
