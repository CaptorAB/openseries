Multi-Asset Analysis
====================

This example shows how to analyze multiple assets simultaneously using OpenFrame.

Setting Up Multi-Asset Analysis
--------------------------------

.. code-block:: python

    import yfinance as yf
    from openseries import OpenTimeSeries, OpenFrame

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
         # This may fail if the ticker is invalid or data unavailable
         data = yf.Ticker(ticker).history(period="3y")
         series = OpenTimeSeries.from_df(
              dframe=data['Close']
         )
         series.set_new_label(lvl_zero=name)
         series_list.append(series)
         print(f"Loaded {name}: {series.length} observations")

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

    # Rank assets by different criteria using openseries metrics
    # Get key metrics for ranking
    returns = all_metrics.loc['Geometric return']
    volatilities = all_metrics.loc['Volatility']
    sharpe_ratios = all_metrics.loc['Return vol ratio']
    drawdowns = all_metrics.loc['Max drawdown']

    print("\n=== ASSET RANKINGS ===")
    print("Ranked by Return (highest first):")
    for i, (asset, ret) in enumerate(returns.sort_values(ascending=False).items(), 1):
        print(f"  {i}. {asset}: {ret:.2%}")

    print("\nRanked by Volatility (lowest first):")
    for i, (asset, vol) in enumerate(volatilities.sort_values(ascending=True).items(), 1):
        print(f"  {i}. {asset}: {vol:.2%}")

    print("\nRanked by Sharpe Ratio (highest first):")
    for i, (asset, sharpe) in enumerate(sharpe_ratios.sort_values(ascending=False).items(), 1):
        print(f"  {i}. {asset}: {sharpe:.2f}")

    print("\nRanked by Max Drawdown (least negative first):")
    for i, (asset, dd) in enumerate(drawdowns.sort_values(ascending=False).items(), 1):
        print(f"  {i}. {asset}: {dd:.2%}")

Correlation Analysis
--------------------

.. code-block:: python

    # Calculate correlation matrix
    correlation_matrix = tech_stocks.correl_matrix
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

    # Analyze risk-return using openseries metrics
    returns = all_metrics.loc['Geometric return']
    volatilities = all_metrics.loc['Volatility']
    sharpe_ratios = all_metrics.loc['Return vol ratio']

    print("\n=== RISK-RETURN ANALYSIS ===")
    for asset in returns.index:
        ret_pct = returns[asset] * 100
        vol_pct = volatilities[asset] * 100
        sharpe = sharpe_ratios[asset]
        print(f"{asset}: Return={ret_pct:.2f}%, Volatility={vol_pct:.2f}%, Sharpe={sharpe:.2f}")

    # Identify efficient assets (high return per unit risk)
    # Calculate 75th percentile threshold manually
    sorted_sharpes = sorted(sharpe_ratios.values, reverse=True)
    threshold_idx = int(len(sorted_sharpes) * 0.25)
    efficient_threshold = sorted_sharpes[threshold_idx] if threshold_idx < len(sorted_sharpes) else sorted_sharpes[-1]

    print(f"\n=== MOST EFFICIENT ASSETS (Sharpe >= {efficient_threshold:.2f}) ===")
    for asset, sharpe in sharpe_ratios.items():
        if sharpe >= efficient_threshold:
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
         group_series = [s for s in tech_stocks.constituents if s.label in group_assets]

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
    apple = next(s for s in tech_stocks.constituents if "Apple" in s.label)
    microsoft = next(s for s in tech_stocks.constituents if "Microsoft" in s.label)

    pair_frame = OpenFrame(constituents=[apple, microsoft])
    rolling_corr = pair_frame.rolling_corr(observations=252)  # 1-year rolling

    print(f"\n=== ROLLING CORRELATION: {apple.label} vs {microsoft.label} ===")
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
    # Calculate weighted average manually
    weighted_avg_return = sum(ret * w for ret, w in zip(individual_returns, equal_weights))
    weighted_avg_vol = sum(vol * w for vol, w in zip(individual_vols, equal_weights))
    print(f"  Weighted avg return: {weighted_avg_return:.2%}")
    print(f"  Portfolio return: {portfolio.geo_ret:.2%}")
    print(f"  Weighted avg volatility: {weighted_avg_vol:.2%}")
    print(f"  Portfolio volatility: {portfolio.vol:.2%}")
    print(f"  Volatility reduction: {(weighted_avg_vol - portfolio.vol):.2%}")

Stress Testing
--------------

.. code-block:: python

    # Identify worst market days (modifies original)
    market_proxy = tech_stocks.constituents[0]  # Use first asset as market proxy
    market_proxy.value_to_ret()
    market_data = market_proxy.tsdf
    # Find worst 5% of days
    worst_threshold = market_data.quantile(0.05)
    worst_days = market_data[market_data <= worst_threshold]

    print(f"\n=== STRESS TEST ANALYSIS ===")
    print(f"Market stress threshold: {worst_threshold:.2%}")
    print(f"Number of stress days: {len(worst_days)}")

    # Analyze each asset's performance during stress
    print("\nAsset performance during market stress:")
    for series in tech_stocks.constituents:
         series.value_to_ret()  # Modifies original
         asset_data = series.tsdf
         # Get returns on stress days
         stress_returns = asset_data.loc[worst_days.index]
         avg_stress_return = stress_returns.mean()

         print(f"  {series.label}: {avg_stress_return:.2%}")

Export Multi-Asset Results
--------------------------

.. code-block:: python

    # Export using openseries native methods
    # Export frame data
    tech_stocks.to_xlsx('multi_asset_analysis.xlsx')

    # Note: For comprehensive Excel export with multiple sheets,
    # you can use the DataFrame returned by all_properties() and correl_matrix
    # which are pandas DataFrames and support to_excel() method
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
         # This may fail if the ticker is invalid or data unavailable
         data = yf.Ticker(ticker).history(period="3y")
         series = OpenTimeSeries.from_df(dframe=data['Close'])
         series.set_new_label(lvl_zero=ticker)
         series_list.append(series)

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
         key_metrics = frame.all_properties(
              properties=['geo_ret', 'vol', 'ret_vol_ratio', 'max_drawdown']
         )

         print("\nKey Metrics:")
         print((key_metrics * 100).round(2))  # Convert to percentages

         # Correlations using openseries correl_matrix property
         correlations = frame.correl_matrix
         avg_correlation = correlations.mean().mean()
         print(f"\nAverage correlation: {avg_correlation:.3f}")

         # Create portfolio using openseries make_portfolio method
         portfolio_df = frame.make_portfolio(name="Equal Weight", weight_strat="eq_weights")
         portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)

         print(f"\nEqual-weight portfolio:")
         print(f"  Return: {portfolio.geo_ret:.2%}")
         print(f"  Volatility: {portfolio.vol:.2%}")
         print(f"  Sharpe: {portfolio.ret_vol_ratio:.2f}")
