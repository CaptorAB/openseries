Rebalanced Portfolio Simulation
=================================

This example demonstrates how to use the `rebalanced_portfolio` method to simulate realistic portfolio trading strategies with periodic rebalancing, cash management, and transaction cost tracking.

Understanding Rebalanced Portfolio Simulation
----------------------------------------------

The `rebalanced_portfolio` method simulates actual trading mechanics rather than theoretical portfolio calculations. It:

- **Simulates real trading**: Executes buy/sell transactions on rebalancing days
- **Manages cash separately**: Tracks cash position with interest accrual
- **Tracks transaction costs**: Implicitly accounts for trading friction
- **Provides detailed data**: Returns position-level information for analysis

Basic Rebalanced Portfolio Setup
---------------------------------

Let's start with a simple example using simulated data:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from openseries import OpenTimeSeries, OpenFrame, ValueType
   from openseries.simulation import ReturnSimulation
   import datetime as dt

   # Create simulated asset data
   seed = 42
   end_date = dt.date(2023, 12, 31)

   # Generate correlated returns for 3 assets
   seriesim = ReturnSimulation.from_merton_jump_gbm(
       number_of_sims=3,
       trading_days=1000,
       mean_annual_return=0.08,
       mean_annual_vol=0.15,
       jumps_lamda=0.05,
       jumps_sigma=0.2,
       jumps_mu=-0.1,
       trading_days_in_year=252,
       seed=seed,
   )

   # Create individual asset series
   assets = []
   asset_names = ["Equity Fund", "Bond Fund", "Commodity Fund"]

   for i, name in enumerate(asset_names):
       series = OpenTimeSeries.from_df(
           dframe=seriesim.to_dataframe(name=name, end=end_date),
           valuetype=ValueType.RTRN,
       ).to_cumret()
       series.set_new_label(lvl_zero=name)
       assets.append(series)

   # Create investment universe
   investment_universe = OpenFrame(constituents=assets)
   investment_universe.weights = [0.6, 0.3, 0.1]  # 60% equity, 30% bonds, 10% commodities

   print(f"Created investment universe with {investment_universe.item_count} assets")
   print(f"Analysis period: {investment_universe.first_idx} to {investment_universe.last_idx}")
   print(f"Target weights: {investment_universe.weights}")

Daily Rebalancing vs Theoretical Portfolio
------------------------------------------

Let's compare daily rebalancing with the theoretical portfolio calculation:

.. code-block:: python

   # Create theoretical portfolio (no rebalancing friction)
   theoretical_portfolio_df = investment_universe.make_portfolio(
       name="Theoretical Portfolio"
   )
   theoretical_portfolio = OpenTimeSeries.from_df(dframe=theoretical_portfolio_df)

   # Create daily rebalanced portfolio (simulates actual trading)
   daily_rebalanced = investment_universe.rebalanced_portfolio(
       name="Daily Rebalanced",
       frequency=1  # Rebalance every day
   )

   # Extract portfolio series for comparison
   theoretical_series = theoretical_portfolio
   daily_rebalanced_series = daily_rebalanced.constituents[-1]

   print("=== PORTFOLIO COMPARISON ===")
   print(f"Theoretical Portfolio:")
   print(f"  Total Return: {theoretical_series.value_ret:.2%}")
   print(f"  Annualized Return: {theoretical_series.geo_ret:.2%}")
   print(f"  Volatility: {theoretical_series.vol:.2%}")
   print(f"  Sharpe Ratio: {theoretical_series.ret_vol_ratio:.2f}")

   print(f"\nDaily Rebalanced Portfolio:")
   print(f"  Total Return: {daily_rebalanced_series.value_ret:.2%}")
   print(f"  Annualized Return: {daily_rebalanced_series.geo_ret:.2%}")
   print(f"  Volatility: {daily_rebalanced_series.vol:.2%}")
   print(f"  Sharpe Ratio: {daily_rebalanced_series.ret_vol_ratio:.2f}")

   # Calculate difference
   return_diff = daily_rebalanced_series.geo_ret - theoretical_series.geo_ret
   vol_diff = daily_rebalanced_series.vol - theoretical_series.vol

   print(f"\nDifference (Rebalanced - Theoretical):")
   print(f"  Return Difference: {return_diff:+.2%}")
   print(f"  Volatility Difference: {vol_diff:+.2%}")

Different Rebalancing Frequencies
---------------------------------

Now let's compare different rebalancing frequencies:

.. code-block:: python

   # Test different rebalancing frequencies
   frequencies = [1, 5, 21, 63]  # Daily, weekly, monthly, quarterly
   frequency_names = ["Daily", "Weekly", "Monthly", "Quarterly"]

   portfolios = []

   for freq, name in zip(frequencies, frequency_names):
       portfolio = investment_universe.rebalanced_portfolio(
           name=f"{name} Rebalanced",
           frequency=freq
       )
       portfolios.append(portfolio.constituents[-1])  # Get portfolio series

   # Create comparison frame
   comparison_frame = OpenFrame(constituents=portfolios)
   metrics = comparison_frame.all_properties()

   print("\n=== REBALANCING FREQUENCY COMPARISON ===")
   print("Frequency | Return | Volatility | Sharpe | Max DD")
   print("-" * 50)

   for i, name in enumerate(frequency_names):
       ret = metrics.loc['Geometric return', portfolios[i].name].iloc[0] * 100
       vol = metrics.loc['Volatility', portfolios[i].name].iloc[0] * 100
       sharpe = metrics.loc['Return vol ratio', portfolios[i].name].iloc[0]
       max_dd = metrics.loc['Max drawdown', portfolios[i].name].iloc[0] * 100

       print(f"{name:>9} | {ret:6.2f}% | {vol:10.2f}% | {sharpe:6.2f} | {max_dd:6.2f}%")

Detailed Portfolio Analysis
----------------------------

Let's examine the detailed trading data by setting `drop_extras=False`:

.. code-block:: python

   # Get detailed trading data
   detailed_portfolio = investment_universe.rebalanced_portfolio(
       name="Detailed Analysis",
       frequency=21,  # Monthly rebalancing
       drop_extras=False  # Return all trading details
   )

   print(f"\nDetailed portfolio contains {detailed_portfolio.item_count} series")
   print("Available data series:")
   for series in detailed_portfolio.constituents:
       print(f"  - {series.name}")

   # Extract key trading metrics
   portfolio_twr = None
   cash_position = None

   for series in detailed_portfolio.constituents:
       if "Detailed Analysis, twr" in series.name:
           portfolio_twr = series
       elif "cash, twr" in series.name:
           cash_position = series

   if portfolio_twr and cash_position:
       print(f"\n=== TRADING ANALYSIS ===")
       print(f"Portfolio TWR (final): {portfolio_twr.tsdf.iloc[-1, 0]:.4f}")
       print(f"Cash TWR (final): {cash_position.tsdf.iloc[-1, 0]:.4f}")

       # Calculate cash as percentage of portfolio
       cash_pct = cash_position.tsdf.iloc[-1, 0] / portfolio_twr.tsdf.iloc[-1, 0] * 100
       print(f"Cash as % of portfolio: {cash_pct:.2f}%")

Equal Weight vs Custom Weight Strategies
----------------------------------------

Compare equal weight strategy with custom weights:

.. code-block:: python

   # Equal weight strategy
   equal_weight_portfolio = investment_universe.rebalanced_portfolio(
       name="Equal Weight Strategy",
       frequency=21,
       equal_weights=True  # Use equal weights
   )

   # Custom weight strategy
   custom_weights = [0.7, 0.2, 0.1]  # 70% equity, 20% bonds, 10% commodities
   custom_weight_portfolio = investment_universe.rebalanced_portfolio(
       name="Custom Weight Strategy",
       frequency=21,
       bal_weights=custom_weights
   )

   # Compare strategies
   strategies = [
       equal_weight_portfolio.constituents[-1],
       custom_weight_portfolio.constituents[-1]
   ]

   strategy_frame = OpenFrame(constituents=strategies)
   strategy_metrics = strategy_frame.all_properties()

   print("\n=== STRATEGY COMPARISON ===")
   print("Strategy | Return | Volatility | Sharpe | Max DD")
   print("-" * 50)

   for strategy in strategies:
       ret = strategy_metrics.loc['Geometric return', strategy.name].iloc[0] * 100
       vol = strategy_metrics.loc['Volatility', strategy.name].iloc[0] * 100
       sharpe = strategy_metrics.loc['Return vol ratio', strategy.name].iloc[0]
       max_dd = strategy_metrics.loc['Max drawdown', strategy.name].iloc[0] * 100

       print(f"{strategy.name:>15} | {ret:6.2f}% | {vol:10.2f}% | {sharpe:6.2f} | {max_dd:6.2f}%")

Cash Management Analysis
------------------------

Let's examine how cash is managed in the rebalanced portfolio:

.. code-block:: python

   # Create portfolio with cash analysis
   cash_analysis = investment_universe.rebalanced_portfolio(
       name="Cash Analysis",
       frequency=21,
       drop_extras=False
   )

   # Extract cash-related series
   cash_series = {}
   for series in cash_analysis.constituents:
       if "cash" in series.name.lower():
           series_type = series.name.split(", ")[1] if ", " in series.name else series.name
           cash_series[series_type] = series

   print("\n=== CASH MANAGEMENT ANALYSIS ===")
   print("Available cash data:")
   for data_type, series in cash_series.items():
       print(f"  - {data_type}: {len(series.tsdf)} observations")

   # Analyze cash position over time
   if "position" in cash_series:
       cash_positions = cash_series["position"].tsdf
       print(f"\nCash position statistics:")
       print(f"  Average cash position: {cash_positions.mean().iloc[0]:.4f}")
       print(f"  Maximum cash position: {cash_positions.max().iloc[0]:.4f}")
       print(f"  Minimum cash position: {cash_positions.min().iloc[0]:.4f}")
       print(f"  Final cash position: {cash_positions.iloc[-1, 0]:.4f}")

Subset Portfolio Analysis
-------------------------

Analyze performance with a subset of assets:

.. code-block:: python

   # Create portfolio with only equity and bonds (exclude commodities)
   subset_portfolio = investment_universe.rebalanced_portfolio(
       name="Equity-Bond Portfolio",
       items=["Equity Fund", "Bond Fund"],  # Only use these assets
       bal_weights=[0.7, 0.3],  # 70% equity, 30% bonds
       frequency=21
   )

   # Compare with full universe
   full_portfolio = investment_universe.rebalanced_portfolio(
       name="Full Universe Portfolio",
       frequency=21
   )

   # Performance comparison
   comparison_series = [
       subset_portfolio.constituents[-1],
       full_portfolio.constituents[-1]
   ]

   comparison_frame = OpenFrame(constituents=comparison_series)
   comparison_metrics = comparison_frame.all_properties()

   print("\n=== SUBSET vs FULL UNIVERSE ===")
   print("Portfolio | Return | Volatility | Sharpe | Max DD")
   print("-" * 50)

   for series in comparison_series:
       ret = comparison_metrics.loc['Geometric return', series.name].iloc[0] * 100
       vol = comparison_metrics.loc['Volatility', series.name].iloc[0] * 100
       sharpe = comparison_metrics.loc['Return vol ratio', series.name].iloc[0]
       max_dd = comparison_metrics.loc['Max drawdown', series.name].iloc[0] * 100

       print(f"{series.name:>20} | {ret:6.2f}% | {vol:10.2f}% | {sharpe:6.2f} | {max_dd:6.2f}%")

Transaction Cost Analysis
-------------------------

Analyze the implicit transaction costs from rebalancing:

.. code-block:: python

   # Get detailed transaction data
   transaction_data = investment_universe.rebalanced_portfolio(
       name="Transaction Analysis",
       frequency=21,
       drop_extras=False
   )

   # Extract transaction-related series
   transaction_series = {}
   for series in transaction_data.constituents:
       if "buysell_qty" in series.name or "settle" in series.name:
           transaction_series[series.name] = series

   print("\n=== TRANSACTION ANALYSIS ===")
   print("Transaction data available:")
   for name, series in transaction_series.items():
       print(f"  - {name}: {len(series.tsdf)} observations")

   # Calculate total trading activity
   total_trades = 0
   for name, series in transaction_series.items():
       if "buysell_qty" in name:
           # Sum absolute trading quantities
           total_trades += series.tsdf.abs().sum().iloc[0]

   print(f"\nTotal trading activity: {total_trades:.2f}")
   print("(Sum of absolute buy/sell quantities across all assets)")

Performance Attribution
------------------------

Analyze the contribution of each asset to portfolio performance:

.. code-block:: python

   # Get individual asset performance from rebalanced portfolio
   asset_performance = investment_universe.rebalanced_portfolio(
       name="Asset Performance Analysis",
       frequency=21
   )

   print("\n=== ASSET PERFORMANCE ATTRIBUTION ===")
   print("Asset | Final TWR | Contribution")
   print("-" * 40)

   # Calculate weighted contribution
   target_weights = investment_universe.weights

   for i, series in enumerate(asset_performance.constituents[:-1]):  # Exclude portfolio series
       final_twr = series.tsdf.iloc[-1, 0]
       weight = target_weights[i]
       contribution = final_twr * weight

       print(f"{series.name:>15} | {final_twr:8.4f} | {contribution:8.4f}")

   # Portfolio total
   portfolio_series = asset_performance.constituents[-1]
   portfolio_twr = portfolio_series.tsdf.iloc[-1, 0]
   print(f"{'Portfolio Total':>15} | {portfolio_twr:8.4f} | {portfolio_twr:8.4f}")

Real-World Application Example
-------------------------------

Here's a practical example using real market data:

.. code-block:: python

   import yfinance as yf

   # Download real market data
   tickers = ["SPY", "TLT", "GLD"]  # S&P 500, Long-term Treasury, Gold
   names = ["S&P 500", "US Treasury", "Gold"]

   real_assets = []
   for ticker, name in zip(tickers, names):
       try:
           data = yf.Ticker(ticker).history(period="3y")
           series = OpenTimeSeries.from_df(
               dframe=data['Close'],
               name=name
           )
           real_assets.append(series)
           print(f"Loaded {name}: {series.length} observations")
       except Exception as e:
           print(f"Failed to load {name}: {e}")

   if len(real_assets) >= 2:
       # Create real-world portfolio
       real_universe = OpenFrame(constituents=real_assets)
       real_universe.weights = [0.6, 0.3, 0.1]  # 60% stocks, 30% bonds, 10% gold

       # Monthly rebalanced portfolio
       real_portfolio = real_universe.rebalanced_portfolio(
           name="Real-World Portfolio",
           frequency=21,  # Approximately monthly
       )

       portfolio_series = real_portfolio.constituents[-1]

       print(f"\n=== REAL-WORLD PORTFOLIO RESULTS ===")
       print(f"Analysis period: {real_universe.first_idx} to {real_universe.last_idx}")
       print(f"Total return: {portfolio_series.value_ret:.2%}")
       print(f"Annualized return: {portfolio_series.geo_ret:.2%}")
       print(f"Volatility: {portfolio_series.vol:.2%}")
       print(f"Sharpe ratio: {portfolio_series.ret_vol_ratio:.2f}")
       print(f"Maximum drawdown: {portfolio_series.max_drawdown:.2%}")

Summary and Best Practices
--------------------------

Key takeaways for using `rebalanced_portfolio`:

1. **Realistic Simulation**: Unlike `make_portfolio`, this method simulates actual trading with transaction costs and cash management.

2. **Rebalancing Frequency**: Higher frequency (lower number) means more trading but closer to target weights. Consider transaction costs vs. tracking error.

3. **Cash Management**: The method automatically handles cash positions and can include cash interest if a cash index is provided.

4. **Detailed Analysis**: Use `drop_extras=False` to get comprehensive trading data for analysis.

5. **Performance Attribution**: Individual asset series show the actual performance of each position in the portfolio.

6. **Transaction Costs**: The method implicitly accounts for trading friction through settlement tracking.

This simulation approach provides a more realistic view of portfolio performance compared to theoretical calculations, making it valuable for backtesting and strategy evaluation.
