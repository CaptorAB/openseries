Risk Management
===============

This tutorial demonstrates comprehensive risk management techniques using openseries, including VaR calculations, stress testing, and risk monitoring.

Setting Up Risk Analysis
-------------------------

Let's start with a portfolio of assets for risk analysis:

.. code-block:: python

   import yfinance as yf
   import pandas as pd
   import numpy as np
   from openseries import OpenTimeSeries, OpenFrame
   from datetime import datetime, timedelta
   import warnings
   warnings.filterwarnings('ignore')

   # Download data for a mixed portfolio
   tickers = {
       "AAPL": "Apple Inc.",
       "GOOGL": "Alphabet Inc.",
       "MSFT": "Microsoft Corp.",
       "TSLA": "Tesla Inc.",
       "SPY": "SPDR S&P 500 ETF",
       "QQQ": "Invesco QQQ Trust",
       "TLT": "iShares 20+ Year Treasury",
       "GLD": "SPDR Gold Shares"
   }

   # Download 3 years of data
   series_list = []
   for ticker, name in tickers.items():
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

   # Create portfolio frame
   portfolio_assets = OpenFrame(constituents=series_list)

   # Create equal-weighted portfolio
   n_assets = portfolio_assets.item_count

   # Set weights on the frame first
   portfolio_df = portfolio_assets.make_portfolio(
       name="Diversified Portfolio",
       weight_strat="eq_weights"
   )
   portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)

   print(f"\nPortfolio created with {n_assets} assets")
   print(f"Date range: {portfolio.first_idx} to {portfolio.last_idx}")

Basic Risk Metrics
------------------

Start with fundamental risk measurements:

.. code-block:: python

   print("=== BASIC RISK METRICS ===")

   # Volatility measures
   print(f"Annualized Volatility: {portfolio.vol:.2%}")
   print(f"Downside Deviation: {portfolio.downside_deviation:.2%}")

   # Return distribution
   print(f"Skewness: {portfolio.skew:.3f}")
   print(f"Kurtosis: {portfolio.kurtosis:.3f}")

   # Tail risk
   print(f"Worst Single Day: {portfolio.worst:.2%}")
   print(f"Worst Month: {portfolio.worst_month:.2%}")

   # Drawdown analysis
   print(f"Maximum Drawdown: {portfolio.max_drawdown:.2%}")
   print(f"Max Drawdown Date: {portfolio.max_drawdown_date}")

Value at Risk (VaR) Analysis
-----------------------------

Calculate VaR at different confidence levels:

.. code-block:: python

   print("\n=== VALUE AT RISK ANALYSIS ===")

   # VaR at different confidence levels
   confidence_levels = [0.90, 0.95, 0.99]

   for level in confidence_levels:
       var_value = portfolio.var_down_func(level=level)
       print(f"{level*100:.0f}% VaR (daily): {var_value:.2%}")

   # Convert daily VaR to different time horizons
   # Assuming normal distribution and independence
   daily_var_95 = portfolio.var_down_func(level=0.95)

   print(f"\n=== VaR TIME HORIZONS (95% confidence) ===")
   print(f"1-day VaR: {daily_var_95:.2%}")
   print(f"1-week VaR: {daily_var_95 * np.sqrt(5):.2%}")
   print(f"1-month VaR: {daily_var_95 * np.sqrt(22):.2%}")
   print(f"1-year VaR: {daily_var_95 * np.sqrt(252):.2%}")

Conditional Value at Risk (CVaR)
--------------------------------

Analyze expected shortfall beyond VaR:

.. code-block:: python

   print("\n=== CONDITIONAL VALUE AT RISK (CVaR) ===")

   for level in confidence_levels:
       cvar_value = portfolio.cvar_down_func(level=level)
       var_value = portfolio.var_down_func(level=level)

       print(f"{level*100:.0f}% CVaR: {cvar_value:.2%} (VaR: {var_value:.2%})")
       print(f"  Expected loss beyond VaR: {cvar_value - var_value:.2%}")

Rolling Risk Analysis
---------------------

Monitor how risk changes over time:

.. code-block:: python

   # Calculate rolling risk metrics
   window = 252  # 1-year rolling window

   print(f"\n=== ROLLING RISK ANALYSIS ({window}-day window) ===")

   # Rolling volatility
   rolling_vol = portfolio.rolling_vol(observations=window)
   print(f"Rolling Volatility - Current: {rolling_vol.iloc[-1, 0]:.2%}")
   print(f"Rolling Volatility - Average: {rolling_vol.mean().iloc[0]:.2%}")
   print(f"Rolling Volatility - Range: {rolling_vol.min().iloc[0]:.2%} to {rolling_vol.max().iloc[0]:.2%}")

   # Rolling VaR
   rolling_var = portfolio.rolling_var_down(observations=window)
   print(f"Rolling VaR (95%) - Current: {rolling_var.iloc[-1, 0]:.2%}")
   print(f"Rolling VaR (95%) - Average: {rolling_var.mean().iloc[0]:.2%}")

   # Rolling CVaR
   rolling_cvar = portfolio.rolling_cvar_down(observations=window)
   print(f"Rolling CVaR (95%) - Current: {rolling_cvar.iloc[-1, 0]:.2%}")
   print(f"Rolling CVaR (95%) - Average: {rolling_cvar.mean().iloc[0]:.2%}")

Stress Testing
--------------

Test portfolio performance under extreme scenarios:

Historical Stress Testing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print("\n=== HISTORICAL STRESS TESTING ===")

   # Convert to returns for analysis
   portfolio_returns = portfolio.value_to_ret()
   returns_data = portfolio_returns.tsdf

   # Note: value_to_ret() modifies the original series in place
   # Restore the original portfolio for further analysis
   portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)

   # Identify worst periods
   worst_1_percent = returns_data.quantile(0.01).iloc[0]
   worst_5_percent = returns_data.quantile(0.05).iloc[0]

   print(f"Worst 1% threshold: {worst_1_percent:.2%}")
   print(f"Worst 5% threshold: {worst_5_percent:.2%}")

   # Count extreme events
   extreme_events_1pct = (returns_data <= worst_1_percent).sum().iloc[0]
   extreme_events_5pct = (returns_data <= worst_5_percent).sum().iloc[0]

   print(f"Days with returns <= 1% threshold: {extreme_events_1pct}")
   print(f"Days with returns <= 5% threshold: {extreme_events_5pct}")

   # Worst consecutive days - simplified approach
   print(f"\nWorst 5 single days:")
   returns_series = returns_data.iloc[:, 0]  # Get the first (and only) column
   worst_5_days = returns_series.nsmallest(5)
   for i, (date, return_val) in enumerate(worst_5_days.items()):
       print(f"  {i+1}. {date.strftime('%Y-%m-%d')}: {return_val:.2%}")

Scenario Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   print("\n=== SCENARIO ANALYSIS ===")

   # Define stress scenarios (percentage moves in underlying assets)
   scenarios = {
       "Market Crash": [-0.20, -0.25, -0.22, -0.30, -0.18, -0.20, 0.05, 0.10],
       "Tech Selloff": [-0.35, -0.40, -0.30, -0.45, -0.10, -0.15, 0.02, 0.03],
       "Interest Rate Shock": [-0.10, -0.12, -0.08, -0.15, -0.05, -0.08, -0.15, 0.01],
       "Flight to Quality": [0.05, 0.02, 0.08, -0.10, 0.10, 0.12, 0.20, 0.15]
   }

   print("Portfolio impact under stress scenarios:")
   for scenario_name, asset_moves in scenarios.items():
       # Calculate portfolio impact
       portfolio_impact = sum(w * move for w, move in zip(equal_weights, asset_moves))
       print(f"  {scenario_name}: {portfolio_impact:.2%}")

Monte Carlo Risk Simulation
---------------------------

Use Monte Carlo methods for risk assessment:

.. code-block:: python

   print("\n=== MONTE CARLO RISK SIMULATION ===")

   # Import the simulate_portfolios function
   from openseries.portfoliotools import simulate_portfolios

   # Monte Carlo simulation using native function
   num_simulations = 10000
   seed = 42  # For reproducible results

   # Generate simulated portfolios using the native function
   simulated_portfolios = simulate_portfolios(
       simframe=portfolio_assets,
       num_ports=num_simulations,
       seed=seed
   )

   # Extract portfolio metrics from simulation
   portfolio_returns = simulated_portfolios['ret']
   portfolio_volatilities = simulated_portfolios['stdev']
   portfolio_sharpes = simulated_portfolios['sharpe']

   # Calculate risk metrics from simulation
   sim_var_95 = np.percentile(portfolio_returns, 5)
   sim_cvar_95 = portfolio_returns[portfolio_returns <= sim_var_95].mean()

   print(f"Monte Carlo Results ({num_simulations:,} simulations):")
   print(f"Expected Return: {portfolio_returns.mean():.2%}")
   print(f"Average Volatility: {portfolio_volatilities.mean():.2%}")
   print(f"95% VaR: {sim_var_95:.2%}")
   print(f"95% CVaR: {sim_cvar_95:.2%}")
   print(f"Worst Case (0.1%): {np.percentile(portfolio_returns, 0.1):.2%}")
   print(f"Best Case (99.9%): {np.percentile(portfolio_returns, 99.9):.2%}")
   print(f"Average Sharpe Ratio: {portfolio_sharpes.mean():.3f}")

   # Show distribution of portfolio characteristics
   print(f"\nPortfolio Distribution:")
   print(f"Return Range: {portfolio_returns.min():.2%} to {portfolio_returns.max():.2%}")
   print(f"Volatility Range: {portfolio_volatilities.min():.2%} to {portfolio_volatilities.max():.2%}")
   print(f"Sharpe Range: {portfolio_sharpes.min():.3f} to {portfolio_sharpes.max():.3f}")

Risk Decomposition
------------------

Analyze risk contribution by asset:

.. code-block:: python

   print("\n=== RISK DECOMPOSITION ===")

   # Calculate individual asset volatilities using OpenFrame
   asset_metrics = portfolio_assets.all_properties()
   asset_vols = asset_metrics.loc['Volatility'].values

   # Portfolio volatility
   portfolio_vol = portfolio.vol

   # Calculate correlation matrix
   correlation_matrix = portfolio_assets.correl_matrix()

   # Risk contribution analysis
   weights = np.array(equal_weights)
   vols = np.array(asset_vols)
   corr_matrix = correlation_matrix.values

   # Portfolio variance
   portfolio_variance = np.dot(weights.T, np.dot(np.outer(vols, vols) * corr_matrix, weights))

   # Marginal contribution to risk
   marginal_contrib = np.dot(np.outer(vols, vols) * corr_matrix, weights) / np.sqrt(portfolio_variance)

   # Component contribution to risk
   component_contrib = weights * marginal_contrib

   # Percentage contribution
   percent_contrib = component_contrib / np.sqrt(portfolio_variance)

   print("Risk Contribution Analysis:")
   risk_decomp = pd.DataFrame({
       'Asset': [series.name for series in portfolio_assets.constituents],
       'Weight': weights,
       'Individual Vol': vols,
       'Marginal Contrib': marginal_contrib,
       'Component Contrib': component_contrib,
       'Risk Contrib %': percent_contrib * 100
   })

   print(risk_decomp.round(4))

   # Verify risk contributions sum to portfolio volatility
   print(f"\nVerification:")
   print(f"Sum of component contributions: {component_contrib.sum():.4f}")
   print(f"Portfolio volatility: {portfolio_vol:.4f}")

Risk-Adjusted Performance
-------------------------

Evaluate risk-adjusted returns:

.. code-block:: python

   print("\n=== RISK-ADJUSTED PERFORMANCE ===")

   # Sharpe ratio
   print(f"Sharpe Ratio: {portfolio.ret_vol_ratio:.3f}")

   # Sortino ratio (downside risk only)
   print(f"Sortino Ratio: {portfolio.sortino_ratio:.3f}")

   # Kappa-3 ratio (higher-order downside risk)
   print(f"Kappa-3 Ratio: {portfolio.kappa3_ratio:.3f}")

   # Omega ratio
   print(f"Omega Ratio: {portfolio.omega_ratio:.3f}")

   # Compare with individual assets
   print(f"\n=== RISK-ADJUSTED COMPARISON ===")
   all_assets = portfolio_assets.constituents + [portfolio]
   comparison_frame = OpenFrame(constituents=all_assets)

   risk_adj_metrics = comparison_frame.all_properties().loc[
       ['ret_vol_ratio', 'sortino_ratio', 'kappa3_ratio', 'omega_ratio']
   ]

   print(risk_adj_metrics.round(3))

Risk Monitoring Dashboard
-------------------------

Create a comprehensive risk monitoring summary using openseries properties and methods:

.. code-block:: python

   print("\n" + "="*60)
   print("RISK MONITORING DASHBOARD")
   print("="*60)

   # Current date and lookback period
   current_date = portfolio.last_idx
   lookback_date = portfolio.first_idx

   print(f"Portfolio: {portfolio.name}")
   print(f"Current Date: {current_date}")
   print(f"Analysis Period: {lookback_date} to {current_date}")
   print(f"Observations: {portfolio.length}")

   # Risk metrics using openseries properties
   print(f"\n--- CURRENT RISK METRICS ---")
   print(f"Volatility (annualized): {portfolio.vol:.2%}")
   print(f"Downside Deviation: {portfolio.downside_deviation:.2%}")
   print(f"95% VaR (daily): {portfolio.var_down:.2%}")
   print(f"95% CVaR (daily): {portfolio.cvar_down:.2%}")
   print(f"Maximum Drawdown: {portfolio.max_drawdown:.2%}")

   # Performance metrics using openseries properties
   print(f"\n--- PERFORMANCE METRICS ---")
   print(f"Total Return: {portfolio.value_ret:.2%}")
   print(f"Annualized Return: {portfolio.geo_ret:.2%}")
   print(f"Sharpe Ratio: {portfolio.ret_vol_ratio:.3f}")
   print(f"Sortino Ratio: {portfolio.sortino_ratio:.3f}")

   # Distribution characteristics using openseries properties
   print(f"\n--- RETURN DISTRIBUTION ---")
   print(f"Skewness: {portfolio.skew:.3f}")
   print(f"Kurtosis: {portfolio.kurtosis:.3f}")
   print(f"Positive Days: {portfolio.positive_share:.1%}")

   # Recent performance using openseries properties
   recent_return = portfolio.z_score
   print(f"\n--- RECENT ACTIVITY ---")
   print(f"Last Return Z-Score: {recent_return:.2f}")

   if abs(recent_return) > 2:
       print("  ⚠️  ALERT: Recent return is unusual (|z| > 2)")
   elif abs(recent_return) > 3:
       print("  🚨 WARNING: Recent return is extreme (|z| > 3)")
   else:
       print("  ✅ Recent return is within normal range")

   # Risk alerts based on openseries metrics
   print(f"\n--- RISK ALERTS ---")
   alerts = []

   if portfolio.vol > 0.25:
       alerts.append("High volatility (>25%)")

   if abs(portfolio.max_drawdown) > 0.20:
       alerts.append("Large maximum drawdown (>20%)")

   if portfolio.ret_vol_ratio < 0.5:
       alerts.append("Low Sharpe ratio (<0.5)")

   if portfolio.skew < -1:
       alerts.append("Highly negative skew (<-1)")

   if portfolio.kurtosis > 5:
       alerts.append("High kurtosis (>5) - fat tails")

   if alerts:
       for alert in alerts:
           print(f"  ⚠️  {alert}")
   else:
       print("  ✅ No risk alerts")

Risk Limits and Controls
------------------------

Implement risk limit monitoring:

.. code-block:: python

   print("\n=== RISK LIMITS MONITORING ===")

   # Define risk limits
   risk_limits = {
       'max_volatility': 0.20,      # 20% annual volatility
       'max_var_daily': -0.03,      # 3% daily VaR
       'max_drawdown': -0.15,       # 15% maximum drawdown
       'min_sharpe': 0.5,           # Minimum Sharpe ratio
       'max_concentration': 0.30    # Maximum single asset weight
   }

   # Check current metrics against limits
   current_metrics = {
       'volatility': portfolio.vol,
       'var_daily': portfolio.var_down,
       'drawdown': portfolio.max_drawdown,
       'sharpe': portfolio.ret_vol_ratio,
       'max_weight': max(equal_weights)
   }

   print("Risk Limit Monitoring:")
   print("-" * 40)

   # Volatility check
   if current_metrics['volatility'] > risk_limits['max_volatility']:
       print(f"❌ BREACH: Volatility {current_metrics['volatility']:.2%} > {risk_limits['max_volatility']:.2%}")
   else:
       print(f"✅ OK: Volatility {current_metrics['volatility']:.2%} <= {risk_limits['max_volatility']:.2%}")

   # VaR check
   if current_metrics['var_daily'] < risk_limits['max_var_daily']:
       print(f"❌ BREACH: VaR {current_metrics['var_daily']:.2%} < {risk_limits['max_var_daily']:.2%}")
   else:
       print(f"✅ OK: VaR {current_metrics['var_daily']:.2%} >= {risk_limits['max_var_daily']:.2%}")

   # Drawdown check
   if current_metrics['drawdown'] < risk_limits['max_drawdown']:
       print(f"❌ BREACH: Drawdown {current_metrics['drawdown']:.2%} < {risk_limits['max_drawdown']:.2%}")
   else:
       print(f"✅ OK: Drawdown {current_metrics['drawdown']:.2%} >= {risk_limits['max_drawdown']:.2%}")

   # Sharpe ratio check
   if current_metrics['sharpe'] < risk_limits['min_sharpe']:
       print(f"❌ BREACH: Sharpe {current_metrics['sharpe']:.3f} < {risk_limits['min_sharpe']:.3f}")
   else:
       print(f"✅ OK: Sharpe {current_metrics['sharpe']:.3f} >= {risk_limits['min_sharpe']:.3f}")

   # Concentration check
   if current_metrics['max_weight'] > risk_limits['max_concentration']:
       print(f"❌ BREACH: Max weight {current_metrics['max_weight']:.2%} > {risk_limits['max_concentration']:.2%}")
   else:
       print(f"✅ OK: Max weight {current_metrics['max_weight']:.2%} <= {risk_limits['max_concentration']:.2%}")

Export Risk Report
------------------

Save comprehensive risk analysis:

.. code-block:: python

   # Create comprehensive risk report
   risk_report = pd.DataFrame({
       'Metric': [
           'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
           'Sortino Ratio', 'Maximum Drawdown', '95% VaR (daily)',
           '95% CVaR (daily)', 'Skewness', 'Kurtosis', 'Positive Days %'
       ],
       'Value': [
           f"{portfolio.geo_ret:.2%}",
           f"{portfolio.vol:.2%}",
           f"{portfolio.ret_vol_ratio:.3f}",
           f"{portfolio.sortino_ratio:.3f}",
           f"{portfolio.max_drawdown:.2%}",
           f"{portfolio.var_down:.2%}",
           f"{portfolio.cvar_down:.2%}",
           f"{portfolio.skew:.3f}",
           f"{portfolio.kurtosis:.3f}",
           f"{portfolio.positive_share:.1%}"
       ]
   })

   # Export to Excel
   with pd.ExcelWriter('risk_analysis_report.xlsx') as writer:
       risk_report.to_excel(writer, sheet_name='Risk Metrics', index=False)
       risk_decomp.to_excel(writer, sheet_name='Risk Decomposition', index=False)
       correlation_matrix.to_excel(writer, sheet_name='Correlations')

       # Add rolling metrics if available
       if 'rolling_vol' in locals():
           rolling_vol.to_excel(writer, sheet_name='Rolling Volatility')
       if 'rolling_var' in locals():
           rolling_var.to_excel(writer, sheet_name='Rolling VaR')

   print(f"\nRisk analysis report exported to 'risk_analysis_report.xlsx'")
   print("Risk management analysis complete!")

This comprehensive risk management tutorial provides the foundation for implementing robust risk controls and monitoring systems using openseries.
