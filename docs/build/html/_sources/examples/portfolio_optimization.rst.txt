Portfolio Optimization
======================

This example demonstrates various portfolio optimization techniques using openseries.

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
           series = OpenTimeSeries.from_df(dframe=data['Close'], name=name)
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
       frontier_results = efficient_frontier(
           frame=investment_universe,
           num_portfolios=100,
           max_weight=0.40,  # Maximum 40% in any asset
           min_weight=0.05   # Minimum 5% in each asset
       )

       print("=== EFFICIENT FRONTIER RESULTS ===")
       print(f"Generated {len(frontier_results['returns'])} efficient portfolios")

       # Find key portfolios
       returns = np.array(frontier_results['returns'])
       volatilities = np.array(frontier_results['volatilities'])
       sharpe_ratios = returns / volatilities

       # Maximum Sharpe ratio portfolio
       max_sharpe_idx = np.argmax(sharpe_ratios)
       max_sharpe_weights = frontier_results['weights'][max_sharpe_idx]

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
           frame=investment_universe,
           num_portfolios=50000,
           max_weight=0.50,
           min_weight=0.0
       )

       print(f"\n=== MONTE CARLO SIMULATION ===")
       print(f"Simulated {len(simulation_results['returns'])} random portfolios")

       sim_returns = np.array(simulation_results['returns'])
       sim_volatilities = np.array(simulation_results['volatilities'])
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

   # Equal weight portfolio
   n_assets = investment_universe.item_count
   equal_weights = [1/n_assets] * n_assets

   equal_weight_portfolio = investment_universe.make_portfolio(
       weights=equal_weights,
       name="Equal Weight"
   )

   print(f"\n=== EQUAL WEIGHT PORTFOLIO ===")
   print(f"Return: {equal_weight_portfolio.geo_ret:.2%}")
   print(f"Volatility: {equal_weight_portfolio.vol:.2%}")
   print(f"Sharpe: {equal_weight_portfolio.ret_vol_ratio:.2f}")

Inverse Volatility Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Inverse volatility weighting using OpenFrame
   asset_metrics = investment_universe.all_properties()
   asset_volatilities = asset_metrics.loc['vol'].values
   inv_vol_weights = [1/vol for vol in asset_volatilities]
   total_inv_vol = sum(inv_vol_weights)
   inv_vol_weights = [w/total_inv_vol for w in inv_vol_weights]

   inv_vol_portfolio = investment_universe.make_portfolio(
       weights=inv_vol_weights,
       name="Inverse Volatility"
   )

   print(f"\n=== INVERSE VOLATILITY PORTFOLIO ===")
   print(f"Return: {inv_vol_portfolio.geo_ret:.2%}")
   print(f"Volatility: {inv_vol_portfolio.vol:.2%}")
   print(f"Sharpe: {inv_vol_portfolio.ret_vol_ratio:.2f}")

   print("\nInverse Volatility Weights:")
   for i, weight in enumerate(inv_vol_weights):
       asset_name = investment_universe.constituents[i].name
       print(f"  {asset_name}: {weight:.1%}")

Maximum Diversification Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def max_diversification_weights(returns_df):
       """Calculate maximum diversification portfolio weights"""
       # Calculate correlation matrix and volatilities
       corr_matrix = returns_df.corr()
       volatilities = returns_df.std() * np.sqrt(252)  # Annualized

       # Maximum diversification ratio = weighted average vol / portfolio vol
       # This is a simplified approximation
       inv_corr_sum = np.sum(np.linalg.inv(corr_matrix.values), axis=1)
       weights = inv_corr_sum / np.sum(inv_corr_sum)

       return weights

   # Get returns data for calculation
   returns_data = []
   for asset in investment_universe.constituents:
       returns = asset.value_to_ret()
       returns_data.append(returns.tsdf.iloc[:, 0])

   returns_df = pd.concat(returns_data, axis=1)
   returns_df.columns = [asset.name for asset in investment_universe.constituents]

   # Calculate max diversification weights
   max_div_weights = max_diversification_weights(returns_df)

   max_div_portfolio = investment_universe.make_portfolio(
       weights=max_div_weights.tolist(),
       name="Maximum Diversification"
   )

   print(f"\n=== MAXIMUM DIVERSIFICATION PORTFOLIO ===")
   print(f"Return: {max_div_portfolio.geo_ret:.2%}")
   print(f"Volatility: {max_div_portfolio.vol:.2%}")
   print(f"Sharpe: {max_div_portfolio.ret_vol_ratio:.2f}")

Target Risk Portfolio
---------------------

.. code-block:: python

   def target_risk_portfolio(frame, target_volatility=0.10):
       """Create portfolio targeting specific volatility level"""

       # Start with minimum volatility portfolio weights using OpenFrame
       asset_metrics = frame.all_properties()
       asset_vols = asset_metrics.loc['vol'].values
       min_vol_asset_idx = np.argmin(asset_vols)

       # Create weights that target the desired volatility
       # Simple approach: blend minimum vol asset with equal weight
       min_vol_weight = 0.6  # 60% in minimum volatility asset
       remaining_weight = 0.4
       n_other_assets = frame.item_count - 1

       weights = [remaining_weight / n_other_assets] * frame.item_count
       weights[min_vol_asset_idx] = min_vol_weight

       return weights

   # Create 10% volatility target portfolio
   target_vol_weights = target_risk_portfolio(investment_universe, target_volatility=0.10)

   target_vol_portfolio = investment_universe.make_portfolio(
       weights=target_vol_weights,
       name="10% Target Volatility"
   )

   print(f"\n=== TARGET VOLATILITY PORTFOLIO (10%) ===")
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
       max_sharpe_portfolio = investment_universe.make_portfolio(
           weights=max_sharpe_weights.tolist(),
           name="Max Sharpe (Optimized)"
       )
       portfolios.append(max_sharpe_portfolio)

   if 'min_vol_weights' in locals():
       min_vol_portfolio = investment_universe.make_portfolio(
           weights=min_vol_weights.tolist(),
           name="Min Vol (Optimized)"
       )
       portfolios.append(min_vol_portfolio)

   # Create comparison frame
   comparison_frame = OpenFrame(constituents=portfolios)
   comparison_metrics = comparison_frame.all_properties()

   # Display key metrics
   key_metrics = comparison_metrics.loc[['geo_ret', 'vol', 'ret_vol_ratio', 'max_drawdown']]
   key_metrics.index = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']

   print(f"\n=== PORTFOLIO STRATEGY COMPARISON ===")
   print((key_metrics * 100).round(2))  # Convert to percentages

Backtesting Framework
---------------------

.. code-block:: python

   def backtest_portfolio_strategies(frame, strategies, rebalance_freq='BME'):
       """Simple backtesting framework for portfolio strategies"""

       results = {}

       for strategy_name, weights in strategies.items():
           # Create portfolio
           portfolio = frame.make_portfolio(weights=weights, name=strategy_name)

           # Calculate metrics
           results[strategy_name] = {
               'return': portfolio.geo_ret,
               'volatility': portfolio.vol,
               'sharpe': portfolio.ret_vol_ratio,
               'max_drawdown': portfolio.max_drawdown,
               'calmar': portfolio.geo_ret / abs(portfolio.max_drawdown) if portfolio.max_drawdown != 0 else np.nan
           }

       return pd.DataFrame(results).T

   # Define strategies to backtest
   strategies = {
       'Equal Weight': equal_weights,
       'Inverse Volatility': inv_vol_weights,
       'Max Diversification': max_div_weights.tolist(),
       'Target Volatility': target_vol_weights
   }

   # Run backtest
   backtest_results = backtest_portfolio_strategies(investment_universe, strategies)

   print(f"\n=== BACKTEST RESULTS ===")
   print(backtest_results.round(4))

   # Rank strategies
   backtest_results['Rank'] = backtest_results['sharpe'].rank(ascending=False)
   best_strategy = backtest_results.sort_values('Rank').index[0]

   print(f"\nBest performing strategy: {best_strategy}")
   print(f"Sharpe ratio: {backtest_results.loc[best_strategy, 'sharpe']:.3f}")

Risk Budgeting
--------------

.. code-block:: python

   def risk_budget_portfolio(returns_df, risk_budgets):
       """Create portfolio based on risk budgets"""

       # Calculate covariance matrix
       cov_matrix = returns_df.cov() * 252  # Annualized

       # This is a simplified risk budgeting approach
       # In practice, you would use iterative optimization

       # Start with risk budget proportions as initial weights
       weights = np.array(risk_budgets) / np.sum(risk_budgets)

       # Simple adjustment based on volatilities
       volatilities = np.sqrt(np.diag(cov_matrix))
       adjusted_weights = weights / volatilities
       adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

       return adjusted_weights

   # Define risk budgets (must sum to 1)
   risk_budgets = [0.20, 0.15, 0.10, 0.25, 0.10, 0.05, 0.10, 0.05]  # Equal to number of assets

   risk_budget_weights = risk_budget_portfolio(returns_df, risk_budgets)

   risk_budget_portfolio_obj = investment_universe.make_portfolio(
       weights=risk_budget_weights.tolist(),
       name="Risk Budget"
   )

   print(f"\n=== RISK BUDGET PORTFOLIO ===")
   print(f"Return: {risk_budget_portfolio_obj.geo_ret:.2%}")
   print(f"Volatility: {risk_budget_portfolio_obj.vol:.2%}")
   print(f"Sharpe: {risk_budget_portfolio_obj.ret_vol_ratio:.2f}")

   print("\nRisk Budget Weights:")
   for i, (weight, budget) in enumerate(zip(risk_budget_weights, risk_budgets)):
       asset_name = investment_universe.constituents[i].name
       print(f"  {asset_name}: {weight:.1%} (budget: {budget:.1%})")

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

Complete Optimization Function
------------------------------

.. code-block:: python

   def comprehensive_portfolio_optimization(tickers, period="5y"):
       """Complete portfolio optimization workflow"""

       # Load data
       assets = []
       for ticker in tickers:
           try:
               data = yf.Ticker(ticker).history(period=period)
               series = OpenTimeSeries.from_df(dframe=data['Close'], name=ticker)
               assets.append(series)
           except:
               print(f"Failed to load {ticker}")

       if len(assets) < 2:
           print("Need at least 2 assets for optimization")
           return None

       frame = OpenFrame(constituents=assets)

       # Basic strategies
       n = frame.item_count
       equal_weights = [1/n] * n

       # Asset volatilities for inverse vol weighting using OpenFrame
       asset_metrics = frame.all_properties()
       vols = asset_metrics.loc['vol'].values
       inv_vol_weights = [1/vol for vol in vols]
       inv_vol_weights = [w/sum(inv_vol_weights) for w in inv_vol_weights]

       strategies = {
           'Equal Weight': equal_weights,
           'Inverse Volatility': inv_vol_weights
       }

       # Create portfolios
       results = {}
       for name, weights in strategies.items():
           portfolio = frame.make_portfolio(weights=weights, name=name)
           results[name] = {
               'Return': portfolio.geo_ret,
               'Volatility': portfolio.vol,
               'Sharpe': portfolio.ret_vol_ratio,
               'Max Drawdown': portfolio.max_drawdown
           }

       results_df = pd.DataFrame(results).T

       print("=== PORTFOLIO OPTIMIZATION RESULTS ===")
       print((results_df * 100).round(2))

       return frame, results_df

   # Example usage
   etf_tickers = ["VTI", "VEA", "VWO", "BND", "VNQ"]
   optimization_results = comprehensive_portfolio_optimization(etf_tickers)
