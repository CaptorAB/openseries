Portfolio Analysis
==================

This tutorial demonstrates how to construct and analyze portfolios using openseries, including optimization techniques and performance attribution.

Setting Up the Data
--------------------

Let's start by downloading data for a diversified set of assets:

.. code-block:: python

    import yfinance as yf
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
         # This may fail if the ticker is invalid or data unavailable
         data = yf.Ticker(ticker).history(period="5y")
         series = OpenTimeSeries.from_df(
              dframe=data['Close']
         )
         series.set_new_label(lvl_zero=name)
         series_list.append(series)
         print(f"Loaded {name}: {series.length} observations")

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
    returns = asset_metrics.loc['Geometric return']
    volatilities = asset_metrics.loc['Volatility']
    sharpe_ratios = asset_metrics.loc['Return vol ratio']
    max_drawdowns = asset_metrics.loc['Max drawdown']

    print("\n=== ASSET COMPARISON ===")
    for asset in returns.index:
        print(f"{asset}:")
        print(f"  Annual Return: {returns[asset]:.2%}")
        print(f"  Volatility: {volatilities[asset]:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratios[asset]:.2f}")
        print(f"  Max Drawdown: {max_drawdowns[asset]:.2%}")

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
    equal_weight_portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)

    print(f"Equal Weight Portfolio Return: {equal_weight_portfolio.geo_ret:.2%}")
    print(f"Equal Weight Portfolio Volatility: {equal_weight_portfolio.vol:.2%}")
    print(f"Equal Weight Portfolio Sharpe: {equal_weight_portfolio.ret_vol_ratio:.2f}")

Custom Weight Portfolio
~~~~~~~~~~~~~~~~~~~~~~~

You can also specify custom weights for portfolio construction:

.. code-block:: python

    # Define custom weights (must sum to 1)
    custom_weights = [0.50, 0.15, 0.10, 0.15, 0.05, 0.03, 0.02]

    assets.weights = custom_weights
    portfolio_df = assets.make_portfolio(name="Custom Weighted")
    custom_portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)

    print(f"Custom Portfolio Return: {custom_portfolio.geo_ret:.2%}")
    print(f"Custom Portfolio Volatility: {custom_portfolio.vol:.2%}")
    print(f"Custom Portfolio Sharpe: {custom_portfolio.ret_vol_ratio:.2f}")

Risk Parity Portfolio
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use native inverse volatility weighting (risk parity)
    portfolio_df = assets.make_portfolio(name="Risk Parity", weight_strat="inv_vol")
    risk_parity_portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)

    print(f"Risk Parity Portfolio Return: {risk_parity_portfolio.geo_ret:.2%}")
    print(f"Risk Parity Portfolio Volatility: {risk_parity_portfolio.vol:.2%}")
    print(f"Risk Parity Portfolio Sharpe: {risk_parity_portfolio.ret_vol_ratio:.2f}")

Advanced Weight Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenSeries provides additional weight strategies beyond basic equal weighting and risk parity:

Maximum Diversification Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The maximum diversification strategy optimizes the correlation structure to maximize portfolio diversification:

.. code-block:: python

    from openseries.owntypes import MaxDiversificationNaNError, MaxDiversificationNegativeWeightsError

    # This may fail with MaxDiversificationNaNError or MaxDiversificationNegativeWeightsError
    max_div_portfolio_df = assets.make_portfolio(
         name="Maximum Diversification",
         weight_strat="max_div"
    )
    max_div_portfolio = OpenTimeSeries.from_df(dframe=max_div_portfolio_df)

    print(f"Max Diversification Return: {max_div_portfolio.geo_ret:.2%}")
    print(f"Max Diversification Volatility: {max_div_portfolio.vol:.2%}")
    print(f"Max Diversification Sharpe: {max_div_portfolio.ret_vol_ratio:.2f}")

Minimum Volatility Overweight Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The minimum volatility overweight strategy overweights the least volatile asset:

.. code-block:: python

    # This may fail with various exceptions
    min_vol_portfolio_df = assets.make_portfolio(
         name="Min Vol Overweight",
         weight_strat="min_vol_overweight"
    )
    min_vol_portfolio = OpenTimeSeries.from_df(dframe=min_vol_portfolio_df)

    print(f"Min Vol Overweight Return: {min_vol_portfolio.geo_ret:.2%}")
    print(f"Min Vol Overweight Volatility: {min_vol_portfolio.vol:.2%}")
    print(f"Min Vol Overweight Sharpe: {min_vol_portfolio.ret_vol_ratio:.2f}")

Strategy Comparison with Error Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When comparing multiple strategies, it's important to handle potential failures gracefully:

.. code-block:: python

    strategies = {
         'Equal Weight': 'eq_weights',
         'Risk Parity': 'inv_vol',
         'Max Diversification': 'max_div',
         'Min Vol Overweight': 'min_vol_overweight'
    }

    results = {}
    for name, strategy in strategies.items():
         # This may fail with MaxDiversificationNaNError, MaxDiversificationNegativeWeightsError, or other exceptions
         portfolio_df = assets.make_portfolio(name=name, weight_strat=strategy)
         portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)
         results[name] = {
              'Return': portfolio.geo_ret,
              'Volatility': portfolio.vol,
              'Sharpe': portfolio.ret_vol_ratio
         }

    if results:
         print("\n=== STRATEGY COMPARISON ===")
         for strategy_name, metrics in results.items():
             print(f"\n{strategy_name}:")
             print(f"  Return: {metrics['return']*100:.2f}%")
             print(f"  Volatility: {metrics['volatility']*100:.2f}%")
             print(f"  Sharpe: {metrics['sharpe']:.2f}")
             print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")

Portfolio Optimization
----------------------

Now let's use openseries' optimization tools:

Efficient Frontier
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate efficient frontier
    # This may fail with various exceptions
    frontier_df, simulated_df, optimal_portfolio = efficient_frontier(
         eframe=assets,
         num_ports=50,
         seed=42
    )

    print("Efficient frontier calculated successfully")
    print(f"Number of frontier points: {len(frontier_df)}")
    print(f"Number of simulated portfolios: {len(simulated_df)}")

    # Find maximum Sharpe ratio portfolio
    sharpe_ratios = frontier_df['ret'] / frontier_df['stdev']
    max_sharpe_idx = sharpe_ratios.idxmax()

    print(f"\n=== MAXIMUM SHARPE RATIO PORTFOLIO ===")
    print(f"Expected Return: {frontier_df.iloc[max_sharpe_idx]['ret']:.2%}")
    print(f"Volatility: {frontier_df.iloc[max_sharpe_idx]['stdev']:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratios.iloc[max_sharpe_idx]:.2f}")

    # Get optimal weights
    optimal_weights = optimal_portfolio[-len(assets.constituents):]
    print("\nOptimal Weights:")
    for i, weight in enumerate(optimal_weights):
         asset_name = assets.constituents[i].label
         print(f"  {asset_name}: {weight:.1%}")

Monte Carlo Portfolio Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Simulate random portfolios
    # This may fail with various exceptions
    simulation_results = simulate_portfolios(
         simframe=assets,
         num_ports=10000,
         seed=42
    )

    print(f"\nSimulated {len(simulation_results)} random portfolios")

    # Find best performing portfolios
    sim_sharpe_ratios = simulation_results['ret'] / simulation_results['stdev']

    # Top 5 Sharpe ratios
    sorted_indices = sorted(range(len(sim_sharpe_ratios)), key=lambda i: sim_sharpe_ratios.iloc[i], reverse=True)
    top_indices = sorted_indices[:5]

    print("\n=== TOP 5 SIMULATED PORTFOLIOS ===")
    for i, idx in enumerate(top_indices, 1):
         print(f"\nRank {i}:")
         print(f"  Return: {simulation_results.iloc[idx]['ret']:.2%}")
         print(f"  Volatility: {simulation_results.iloc[idx]['stdev']:.2%}")
         print(f"  Sharpe: {sim_sharpe_ratios.iloc[idx]:.2f}")

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
    key_metrics = portfolio_metrics.loc[['Geometric return', 'Volatility', 'Return vol ratio', 'Max drawdown']]
    key_metrics.index = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']

    print("\n=== PORTFOLIO COMPARISON ===")
    print((key_metrics * 100).round(2))  # Convert to percentages

Risk Attribution
----------------

Analyze the risk contribution of each asset:

.. code-block:: python

    # Calculate portfolio statistics using openseries methods
    # Create equal weight portfolio
    equal_weight_portfolio_df = assets.make_portfolio(name="Equal Weight", weight_strat="eq_weights")
    equal_weight_portfolio = OpenTimeSeries.from_df(dframe=equal_weight_portfolio_df)

    print("\n=== RISK ATTRIBUTION (Equal Weight Portfolio) ===")
    print(f"Portfolio Volatility: {equal_weight_portfolio.vol:.4f}")
    print(f"Portfolio Return: {equal_weight_portfolio.geo_ret:.4f}")

    # Individual asset contributions can be analyzed using openseries properties
    for i, series in enumerate(assets.constituents):
        weight = equal_weights[i]
        asset_vol = series.vol
        print(f"\n{series.label}:")
        print(f"  Weight: {weight:.4f}")
        print(f"  Individual Volatility: {asset_vol:.4f}")
        print(f"  Weighted Contribution: {weight * asset_vol:.4f}")

Performance Attribution
-----------------------

Analyze performance contribution over time:

.. code-block:: python

    # Calculate performance attribution using openseries
    # Individual asset performance is available through openseries properties
    print("\n=== PERFORMANCE ATTRIBUTION ===")
    for i, series in enumerate(assets.constituents):
        weight = equal_weights[i]
        asset_return = series.geo_ret
        contribution = weight * asset_return
        print(f"{series.label}:")
        print(f"  Weight: {weight:.2%}")
        print(f"  Return: {asset_return:.2%}")
        print(f"  Contribution: {contribution:.2%}")

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
    rolling_corr = portfolio_vs_market.rolling_corr(observations=252)  # 1-year rolling

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

Analyze the impact of rebalancing frequency using the realistic `rebalanced_portfolio` method:

.. code-block:: python

    # Compare different rebalancing frequencies using realistic simulation
    frequencies = [1, 21, 63]  # Daily, monthly, quarterly
    frequency_names = ["Daily", "Monthly", "Quarterly"]

    rebalanced_portfolios = []

    for freq, name in zip(frequencies, frequency_names):
         portfolio = assets.rebalanced_portfolio(
              name=f"{name} Rebalanced",
              frequency=freq,
              bal_weights=equal_weights
         )
         rebalanced_portfolios.append(portfolio.constituents[-1])

    # Compare with theoretical portfolio
    assets.weights = equal_weights
    theoretical_portfolio_df = assets.make_portfolio(name="Theoretical")
    theoretical_portfolio = OpenTimeSeries.from_df(dframe=theoretical_portfolio_df)

    # Create comprehensive comparison
    all_portfolios = [theoretical_portfolio] + rebalanced_portfolios
    comparison_frame = OpenFrame(constituents=all_portfolios)
    comparison_metrics = comparison_frame.all_properties()

    print("\n=== REALISTIC REBALANCING COMPARISON ===")
    print("Strategy | Return | Volatility | Sharpe | Max DD")
    print("-" * 50)

    for series in all_portfolios:
         ret = comparison_metrics.loc['Geometric return', series.label].iloc[0] * 100
         vol = comparison_metrics.loc['Volatility', series.label].iloc[0] * 100
         sharpe = comparison_metrics.loc['Return vol ratio', series.label].iloc[0]
         max_dd = comparison_metrics.loc['Max drawdown', series.label].iloc[0] * 100

         print(f"{series.label:>15} | {ret:6.2f}% | {vol:10.2f}% | {sharpe:6.2f} | {max_dd:6.2f}%")

    # Analyze transaction costs
    print(f"\n=== TRANSACTION COST ANALYSIS ===")
    for freq, name in zip(frequencies, frequency_names):
         detailed_portfolio = assets.rebalanced_portfolio(
              name=f"{name} Detailed",
              frequency=freq,
              bal_weights=equal_weights,
              drop_extras=False  # Get detailed trading data
         )

         # Count rebalancing events
         rebalancing_days = 0
         for series in detailed_portfolio.constituents:
              if "buysell_qty" in series.label:
                    # Count days with non-zero trading
                    trading_days = (series.tsdf != 0).any(axis=1).sum()
                    rebalancing_days = max(rebalancing_days, trading_days)

         print(f"{name:>15}: {rebalancing_days} rebalancing events")

Stress Testing
--------------

Test portfolio performance during market stress:

.. code-block:: python

    # Identify worst periods for the market (modifies original)
    market_proxy.value_to_ret()
    market_returns_df = market_proxy.tsdf

    # Find worst 5% of days
    worst_days_threshold = market_returns_df.quantile(0.05).iloc[0]
    worst_days = market_returns_df[market_returns_df <= worst_days_threshold]

    print(f"\n=== STRESS TEST RESULTS ===")
    print(f"Market stress threshold: {worst_days_threshold:.2%}")
    print(f"Number of stress days: {len(worst_days)}")

    # Portfolio performance during stress (modifies original)
    equal_weight_portfolio.value_to_ret()
    portfolio_returns_df = equal_weight_portfolio.tsdf

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
    print(f"Asset Universe: {', '.join([s.label for s in assets.constituents])}")

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
    print(f"Portfolio Diversification Benefit: {(asset_metrics.loc['Volatility'].mean() - equal_weight_portfolio.vol):.2%}")

    # Export results
    portfolio_metrics.to_excel("portfolio_analysis.xlsx")
    correlation_matrix.to_excel("correlation_matrix.xlsx")

    print(f"\nResults exported to Excel files")
    print("Analysis complete!")

This tutorial provides a comprehensive framework for portfolio analysis using openseries. You can extend these techniques for more sophisticated portfolio management strategies.
