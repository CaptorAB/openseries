Portfolio Optimization
======================

This example demonstrates various portfolio optimization techniques using openseries, including both theoretical approaches and real-world applications with actual fund data.

Basic Portfolio Optimization Setup
-----------------------------------

.. code-block:: python

   import yfinance as yf
   import pandas as pd
   import numpy as np
   from openseries import OpenTimeSeries, OpenFrame
   from openseries import efficient_frontier, simulate_portfolios

   # Define investment universe
   universe = {
       "VTI": "Total Stock Market",
       "VEA": "Developed Markets",
       "VWO": "Emerging Markets",
       "BND": "Total Bond Market",
       "VNQ": "Real Estate",
       "VDE": "Energy",
       "VGT": "Technology",
       "VHT": "Healthcare"
   }

   # Load data
   assets = []
   for ticker, name in universe.items():
       try:
           data = yf.Ticker(ticker).history(period="5y")
           series = OpenTimeSeries.from_df(dframe=data['Close'])
           series.set_new_label(lvl_zero=name)
           assets.append(series)
           print(f"Loaded {name}")
       except Exception as e:
           print(f"Failed to load {name}: {e}")

   # Create investment universe frame
   investment_universe = OpenFrame(constituents=assets)
   print(f"\nInvestment universe: {investment_universe.item_count} assets")
   print(f"Period: {investment_universe.first_idx} to {investment_universe.last_idx}")

Mean-Variance Optimization
--------------------------

.. code-block:: python

   # Calculate efficient frontier
   try:
       frontier_df, simulated_df, optimal_portfolio = efficient_frontier(
           eframe=investment_universe,
           num_ports=100,
           seed=42
       )

       print("=== EFFICIENT FRONTIER RESULTS ===")
       print(f"Generated {len(frontier_df)} efficient portfolios")
       print(f"Simulated {len(simulated_df)} random portfolios")

       # Find key portfolios
       returns = frontier_df['ret'].values
       volatilities = frontier_df['stdev'].values
       sharpe_ratios = returns / volatilities

       # Maximum Sharpe ratio portfolio
       max_sharpe_idx = np.argmax(sharpe_ratios)
       max_sharpe_weights = optimal_portfolio[-len(investment_universe.constituents):]

       print(f"\n=== MAXIMUM SHARPE RATIO PORTFOLIO ===")
       print(f"Expected Return: {returns[max_sharpe_idx]:.2%}")
       print(f"Volatility: {volatilities[max_sharpe_idx]:.2%}")
       print(f"Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.2f}")

       print("\nOptimal Weights:")
       for i, weight in enumerate(max_sharpe_weights):
           asset_name = investment_universe.constituents[i].name
           if weight > 0.01:  # Only show weights > 1%
               print(f"  {asset_name}: {weight:.1%}")

       # Minimum volatility portfolio
       min_vol_idx = np.argmin(volatilities)
       min_vol_weights = frontier_results['weights'][min_vol_idx]

       print(f"\n=== MINIMUM VOLATILITY PORTFOLIO ===")
       print(f"Expected Return: {returns[min_vol_idx]:.2%}")
       print(f"Volatility: {volatilities[min_vol_idx]:.2%}")
       print(f"Sharpe Ratio: {sharpe_ratios[min_vol_idx]:.2f}")

       print("\nMinimum Volatility Weights:")
       for i, weight in enumerate(min_vol_weights):
           asset_name = investment_universe.constituents[i].name
           if weight > 0.01:
               print(f"  {asset_name}: {weight:.1%}")

   except Exception as e:
       print(f"Efficient frontier calculation failed: {e}")

Monte Carlo Portfolio Simulation
--------------------------------

.. code-block:: python

   # Generate random portfolios
   try:
       simulation_results = simulate_portfolios(
           simframe=investment_universe,
           num_ports=50000,
           seed=42
       )

       print(f"\n=== MONTE CARLO SIMULATION ===")
       print(f"Simulated {len(simulation_results)} random portfolios")

       sim_returns = simulation_results['ret'].values
       sim_volatilities = simulation_results['stdev'].values
       sim_sharpe_ratios = sim_returns / sim_volatilities

       # Statistics of simulated portfolios
       print(f"\nSimulation Statistics:")
       print(f"Return range: {sim_returns.min():.2%} to {sim_returns.max():.2%}")
       print(f"Volatility range: {sim_volatilities.min():.2%} to {sim_volatilities.max():.2%}")
       print(f"Sharpe range: {sim_sharpe_ratios.min():.2f} to {sim_sharpe_ratios.max():.2f}")

       # Best portfolios from simulation
       top_sharpe_indices = np.argsort(sim_sharpe_ratios)[-5:]

       print(f"\n=== TOP 5 SIMULATED PORTFOLIOS ===")
       for i, idx in enumerate(reversed(top_sharpe_indices)):
           print(f"\nRank {i+1}:")
           print(f"  Return: {sim_returns[idx]:.2%}")
           print(f"  Volatility: {sim_volatilities[idx]:.2%}")
           print(f"  Sharpe: {sim_sharpe_ratios[idx]:.2f}")

           weights = simulation_results['weights'][idx]
           print("  Weights:")
           for j, weight in enumerate(weights):
               if weight > 0.05:  # Only show weights > 5%
                   asset_name = investment_universe.constituents[j].name
                   print(f"    {asset_name}: {weight:.1%}")

   except Exception as e:
       print(f"Portfolio simulation failed: {e}")

Risk-Based Portfolio Strategies
-------------------------------

Equal Weight Portfolio
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Equal weight portfolio using native weight_strat
   equal_weight_portfolio_df = investment_universe.make_portfolio(
       name="Equal Weight",
       weight_strat="eq_weights"
   )
   equal_weight_portfolio = OpenTimeSeries.from_df(dframe=equal_weight_portfolio_df)

   print(f"\n=== EQUAL WEIGHT PORTFOLIO ===")
   print(f"Return: {equal_weight_portfolio.geo_ret:.2%}")
   print(f"Volatility: {equal_weight_portfolio.vol:.2%}")
   print(f"Sharpe: {equal_weight_portfolio.ret_vol_ratio:.2f}")

Inverse Volatility Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Inverse volatility weighting using native weight_strat
   inv_vol_portfolio_df = investment_universe.make_portfolio(
       name="Inverse Volatility",
       weight_strat="inv_vol"
   )
   inv_vol_portfolio = OpenTimeSeries.from_df(dframe=inv_vol_portfolio_df)

   print(f"\n=== INVERSE VOLATILITY PORTFOLIO ===")
   print(f"Return: {inv_vol_portfolio.geo_ret:.2%}")
   print(f"Volatility: {inv_vol_portfolio.vol:.2%}")
   print(f"Sharpe: {inv_vol_portfolio.ret_vol_ratio:.2f}")


Maximum Diversification Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The maximum diversification strategy aims to maximize portfolio diversification by optimizing the correlation structure. This strategy can encounter numerical issues in certain scenarios:

.. code-block:: python

   # Maximum diversification portfolio using native weight_strat
   try:
       max_div_portfolio_df = investment_universe.make_portfolio(
           name="Maximum Diversification",
           weight_strat="max_div"
       )
       max_div_portfolio = OpenTimeSeries.from_df(dframe=max_div_portfolio_df)

       print(f"\n=== MAXIMUM DIVERSIFICATION PORTFOLIO ===")
       print(f"Return: {max_div_portfolio.geo_ret:.2%}")
       print(f"Volatility: {max_div_portfolio.vol:.2%}")
       print(f"Sharpe: {max_div_portfolio.ret_vol_ratio:.2f}")
   except MaxDiversificationNaNError as e:
       print(f"Maximum diversification failed due to numerical issues: {e}")
       print("Consider using a different weight strategy or checking your data quality")
   except MaxDiversificationNegativeWeightsError as e:
       print(f"Maximum diversification produced negative weights: {e}")
       print("This strategy may not be suitable for your data - consider using 'eq_weights' or 'inv_vol'")

Target Risk Portfolio
---------------------

.. code-block:: python

   # Target risk portfolio using native weight_strat
   target_vol_portfolio_df = investment_universe.make_portfolio(
       name="Target Risk",
       weight_strat="target_risk"
   )
   target_vol_portfolio = OpenTimeSeries.from_df(dframe=target_vol_portfolio_df)

   print(f"\n=== TARGET RISK PORTFOLIO ===")
   print(f"Return: {target_vol_portfolio.geo_ret:.2%}")
   print(f"Volatility: {target_vol_portfolio.vol:.2%}")
   print(f"Sharpe: {target_vol_portfolio.ret_vol_ratio:.2f}")

Portfolio Comparison
--------------------

.. code-block:: python

   # Compare all portfolio strategies
   portfolios = [
       equal_weight_portfolio,
       inv_vol_portfolio,
       max_div_portfolio,
       target_vol_portfolio
   ]

   # Add optimized portfolios if available
   if 'max_sharpe_weights' in locals():
       max_sharpe_portfolio_df = investment_universe.make_portfolio(
           weights=max_sharpe_weights.tolist(),
           name="Max Sharpe (Optimized)"
       )
       max_sharpe_portfolio = OpenTimeSeries.from_df(dframe=max_sharpe_portfolio_df)
       portfolios.append(max_sharpe_portfolio)

   if 'min_vol_weights' in locals():
       min_vol_portfolio_df = investment_universe.make_portfolio(
           weights=min_vol_weights.tolist(),
           name="Min Vol (Optimized)"
       )
       min_vol_portfolio = OpenTimeSeries.from_df(dframe=min_vol_portfolio_df)
       portfolios.append(min_vol_portfolio)

   # Create comparison frame
   comparison_frame = OpenFrame(constituents=portfolios)
   comparison_metrics = comparison_frame.all_properties()

   # Display key metrics
   key_metrics = comparison_metrics.loc[['geo_ret', 'vol', 'ret_vol_ratio', 'max_drawdown']]
   key_metrics.index = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']

   print(f"\n=== PORTFOLIO STRATEGY COMPARISON ===")
   print((key_metrics * 100).round(2))  # Convert to percentages

Weight Strategy Details
~~~~~~~~~~~~~~~~~~~~~~~

The openseries library provides several built-in weight strategies for portfolio construction:

**Equal Weights (``eq_weights``)**
   - Assigns equal weight to all assets
   - Most robust strategy, always works
   - Good baseline for comparison

**Inverse Volatility (``inv_vol``)**
   - Weights assets inversely to their volatility
   - Lower volatility assets get higher weights
   - Generally stable and reliable

**Maximum Diversification (``max_div``)**
   - Optimizes correlation structure for maximum diversification
   - Can encounter numerical issues with certain data patterns
   - May produce negative weights in some scenarios
   - Raises ``MaxDiversificationNaNError`` for numerical issues
   - Raises ``MaxDiversificationNegativeWeightsError`` for negative weights

**Target Risk (``target_risk``)**
   - Targets a specific portfolio volatility level
   - Requires additional parameters for target volatility

**Exception Handling**
   When using the maximum diversification strategy, it's recommended to handle potential exceptions:

   .. code-block:: python

      from openseries.owntypes import MaxDiversificationNaNError, MaxDiversificationNegativeWeightsError

      try:
          portfolio_df = frame.make_portfolio(name="Max Div", weight_strat="max_div")
      except MaxDiversificationNaNError:
          print("Numerical issues detected - using equal weights instead")
          portfolio_df = frame.make_portfolio(name="Equal Weight", weight_strat="eq_weights")
      except MaxDiversificationNegativeWeightsError:
          print("Negative weights detected - using inverse volatility instead")
          portfolio_df = frame.make_portfolio(name="Inv Vol", weight_strat="inv_vol")

Backtesting Framework
---------------------

.. code-block:: python

   # Define strategies to backtest using native weight_strat
   strategies = {
       'Equal Weight': 'eq_weights',
       'Inverse Volatility': 'inv_vol',
       'Max Diversification': 'max_div',
       'Target Risk': 'target_risk'
   }

   # Run backtest using native strategies
   backtest_results = {}
   for strategy_name, weight_strat in strategies.items():
       try:
           portfolio_df = investment_universe.make_portfolio(
               name=strategy_name,
               weight_strat=weight_strat
           )
           portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)
           backtest_results[strategy_name] = {
               'return': portfolio.geo_ret,
               'volatility': portfolio.vol,
               'sharpe': portfolio.ret_vol_ratio,
               'max_drawdown': portfolio.max_drawdown,
               'calmar': portfolio.geo_ret / abs(portfolio.max_drawdown) if portfolio.max_drawdown != 0 else np.nan
           }
       except (MaxDiversificationNaNError, MaxDiversificationNegativeWeightsError) as e:
           print(f"Skipping {strategy_name}: {e}")
           continue

   backtest_results = pd.DataFrame(backtest_results).T

   print(f"\n=== BACKTEST RESULTS ===")
   print(backtest_results.round(4))

   # Rank strategies
   backtest_results['Rank'] = backtest_results['sharpe'].rank(ascending=False)
   best_strategy = backtest_results.sort_values('Rank').index[0]

   print(f"\nBest performing strategy: {best_strategy}")
   print(f"Sharpe ratio: {backtest_results.loc[best_strategy, 'sharpe']:.3f}")

Export Optimization Results
---------------------------

.. code-block:: python

   # Export comprehensive optimization results
   with pd.ExcelWriter('portfolio_optimization_results.xlsx') as writer:

       # Portfolio comparison
       comparison_metrics.to_excel(writer, sheet_name='Portfolio Comparison')

       # Individual asset metrics
       asset_metrics = investment_universe.all_properties()
       asset_metrics.to_excel(writer, sheet_name='Asset Metrics')

       # Correlation matrix
       correlation_matrix = investment_universe.correl_matrix()
       correlation_matrix.to_excel(writer, sheet_name='Correlations')

       # Backtest results
       backtest_results.to_excel(writer, sheet_name='Backtest Results')

       # Efficient frontier data (if available)
       if 'frontier_results' in locals():
           frontier_df = pd.DataFrame({
               'Return': frontier_results['returns'],
               'Volatility': frontier_results['volatilities'],
               'Sharpe': np.array(frontier_results['returns']) / np.array(frontier_results['volatilities'])
           })
           frontier_df.to_excel(writer, sheet_name='Efficient Frontier', index=False)

   print("\nOptimization results exported to 'portfolio_optimization_results.xlsx'")

Real-World Fund Portfolio Optimization
---------------------------------------

This section demonstrates portfolio optimization using actual fund data from professional fund managers, showing how optimization techniques apply in practice.

Using Real Fund Data for Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to work with real fund data using openseries methods directly:

.. code-block:: python

   from requests import get as requests_get
   from openseries import (
       OpenTimeSeries, OpenFrame, ValueType,
       efficient_frontier, prepare_plot_data, sharpeplot,
       load_plotly_dict, get_previous_business_day_before_today
   )

   # Define fund universe for optimization
   fund_universe_isins = [
       "SE0015243886",  # Global High Yield
       "SE0011337195",  # Global Equity
       "SE0011670843",  # Global Bond
       "SE0017832280",  # Alternative Strategy
       "SE0017832330",  # Multi-Asset Strategy
   ]

   # Load fund data using openseries methods
   response = requests_get(url="https://api.captor.se/public/api/nav", timeout=10)
   response.raise_for_status()

   series_list = []
   result = response.json()

   for data in result:
       if data["isin"] in fund_universe_isins:
           series = OpenTimeSeries.from_arrays(
               name=data["longName"],
               isin=data["isin"],
               baseccy=data["currency"],
               dates=data["dates"],
               values=data["navPerUnit"],
               valuetype=ValueType.PRICE,
           )
           series_list.append(series)

   # Create fund universe using openseries OpenFrame
   fund_universe = OpenFrame(constituents=series_list)

   # Process data using openseries methods
   fund_universe = fund_universe.value_nan_handle().trunc_frame().to_cumret()

   print(f"Fund universe created with {fund_universe.item_count} funds")
   print(f"Analysis period: {fund_universe.first_idx} to {fund_universe.last_idx}")

Advanced Optimization with Real Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Set optimization parameters
   simulations = 10000
   frontier_points = 50
   seed = 55

   # Create current portfolio (equal weights)
   current_portfolio_df = fund_universe.make_portfolio(
       name="Current Portfolio",
       weight_strat="eq_weights",
   )
   current_portfolio = OpenTimeSeries.from_df(dframe=current_portfolio_df)

   # Calculate efficient frontier
   frontier, simulated_portfolios, optimal_portfolio = efficient_frontier(
       eframe=fund_universe,
       num_ports=simulations,
       seed=seed,
       frontier_points=frontier_points,
   )

   # Prepare visualization data
   plot_data = prepare_plot_data(
       assets=fund_universe,
       current=current_portfolio,
       optimized=optimal_portfolio,
   )

   # Load plotly configuration
   figdict, _ = load_plotly_dict()

   # Create efficient frontier plot
   optimization_plot, _ = sharpeplot(
       sim_frame=simulated_portfolios,
       line_frame=frontier,
       point_frame=plot_data,
       point_frame_mode="markers+text",
       title="Real Fund Portfolio Optimization",
       add_logo=False,
       auto_open=False,
       output_type="div",
   )
   optimization_plot = optimization_plot.update_layout(width=1200, height=700)

   # Display the optimization results
   optimization_plot.show(config=figdict["config"])

Performance Comparison Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare different portfolio strategies
   strategies = {}

   # Equal weight portfolio
   equal_weight_portfolio_df = fund_universe.make_portfolio(
       name="Equal Weight", weight_strat="eq_weights"
   )
   equal_weight_portfolio = OpenTimeSeries.from_df(dframe=equal_weight_portfolio_df)
   strategies['Equal Weight'] = equal_weight_portfolio

   # Optimal portfolio from efficient frontier
   optimal_portfolio_df = fund_universe.make_portfolio(
       weights=optimal_portfolio.weights, name="Optimal Portfolio"
   )
   optimal_portfolio_series = OpenTimeSeries.from_df(dframe=optimal_portfolio_df)
   strategies['Optimal Portfolio'] = optimal_portfolio_series

   # Create comparison frame
   comparison_frame = OpenFrame(constituents=list(strategies.values()))
   comparison_metrics = comparison_frame.all_properties()

   # Display key metrics
   key_metrics = comparison_metrics.loc[['geo_ret', 'vol', 'ret_vol_ratio', 'max_drawdown']]
   key_metrics.index = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']

   print("=== PORTFOLIO STRATEGY COMPARISON ===")
   print((key_metrics * 100).round(2))

   # Calculate improvement metrics
   improvement = {
       'Return Improvement': (optimal_portfolio_series.geo_ret - equal_weight_portfolio.geo_ret) * 100,
       'Volatility Change': (optimal_portfolio_series.vol - equal_weight_portfolio.vol) * 100,
       'Sharpe Improvement': optimal_portfolio_series.ret_vol_ratio - equal_weight_portfolio.ret_vol_ratio,
   }

   print("\n=== OPTIMIZATION IMPROVEMENTS ===")
   for metric, value in improvement.items():
       print(f"{metric}: {value:+.2f}")

Complete Optimization Workflow
------------------------------

Here's how to perform portfolio optimization using openseries methods directly:

.. code-block:: python

   # Example: Optimize ETF portfolio using openseries methods
   etf_tickers = ["VTI", "VEA", "VWO", "BND", "VNQ"]

   # Load data using openseries methods
   assets = []
   for ticker in etf_tickers:
       try:
           data = yf.Ticker(ticker).history(period="5y")
           series = OpenTimeSeries.from_df(dframe=data['Close'])
           series.set_new_label(lvl_zero=ticker)
           assets.append(series)
       except:
           print(f"Failed to load {ticker}")

   if len(assets) < 2:
       print("Need at least 2 assets for optimization")
   else:
       frame = OpenFrame(constituents=assets)

       # Use openseries native weight strategies
       strategies = {
           'Equal Weight': 'eq_weights',
           'Inverse Volatility': 'inv_vol',
           'Max Diversification': 'max_div',
           'Target Risk': 'target_risk'
       }

       # Create portfolios using openseries make_portfolio method
       results = {}
       for name, weight_strat in strategies.items():
           try:
               portfolio_df = frame.make_portfolio(name=name, weight_strat=weight_strat)
               portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)
               results[name] = {
                   'Return': portfolio.geo_ret,
                   'Volatility': portfolio.vol,
                   'Sharpe': portfolio.ret_vol_ratio,
                   'Max Drawdown': portfolio.max_drawdown
               }
           except (MaxDiversificationNaNError, MaxDiversificationNegativeWeightsError) as e:
               print(f"Skipping {name}: {e}")
               continue

       results_df = pd.DataFrame(results).T

       print("=== PORTFOLIO OPTIMIZATION RESULTS ===")
       print((results_df * 100).round(2))
