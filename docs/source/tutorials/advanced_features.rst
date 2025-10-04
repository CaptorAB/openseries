Advanced Features
=================

This tutorial covers advanced openseries features including custom analysis, integration with other libraries, and extending functionality.

Custom Analysis Functions
--------------------------

Creating Custom Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

You can extend openseries with custom analysis functions:

.. code-block:: python

   import yfinance as yf
   import pandas as pd
   import numpy as np
   from openseries import OpenTimeSeries, OpenFrame
   from scipy import stats
   import matplotlib.pyplot as plt

   # Load sample data
   ticker = yf.Ticker("^GSPC")
   data = ticker.history(period="5y")
   sp500 = OpenTimeSeries.from_df(dframe=data['Close'], name="S&P 500")

   def custom_risk_metrics(series):
       """Calculate custom risk metrics not available in openseries"""

       # Convert to returns
       returns = series.value_to_ret()
       returns_data = returns.tsdf.iloc[:, 0].dropna()

       # Custom metrics
       metrics = {}

       # Ulcer Index (alternative to standard deviation)
       drawdowns = series.to_drawdown_series()
       dd_data = drawdowns.tsdf.iloc[:, 0]
       ulcer_index = np.sqrt(np.mean(dd_data ** 2))
       metrics['ulcer_index'] = ulcer_index

       # Calmar Ratio (annual return / max drawdown)
       calmar_ratio = series.geo_ret / abs(series.max_drawdown)
       metrics['calmar_ratio'] = calmar_ratio

       # Sterling Ratio (annual return / average drawdown)
       avg_drawdown = abs(dd_data.mean())
       sterling_ratio = series.geo_ret / avg_drawdown if avg_drawdown > 0 else np.nan
       metrics['sterling_ratio'] = sterling_ratio

       # Pain Index (average drawdown)
       pain_index = abs(dd_data.mean())
       metrics['pain_index'] = pain_index

       # Tail Ratio (95th percentile / 5th percentile)
       tail_ratio = returns_data.quantile(0.95) / abs(returns_data.quantile(0.05))
       metrics['tail_ratio'] = tail_ratio

       # Gain-to-Pain Ratio
       positive_returns = returns_data[returns_data > 0].sum()
       negative_returns = abs(returns_data[returns_data < 0].sum())
       gain_to_pain = positive_returns / negative_returns if negative_returns > 0 else np.nan
       metrics['gain_to_pain_ratio'] = gain_to_pain

       return metrics

   # Calculate custom metrics
   custom_metrics = custom_risk_metrics(sp500)

   print("=== CUSTOM RISK METRICS ===")
   for metric, value in custom_metrics.items():
       print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

Advanced Statistical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def advanced_distribution_analysis(series):
       """Perform advanced statistical analysis on returns"""

       returns = series.value_to_ret()
       returns_data = returns.tsdf.iloc[:, 0].dropna()

       results = {}

       # Jarque-Bera test for normality
       jb_stat, jb_pvalue = stats.jarque_bera(returns_data)
       results['jarque_bera_stat'] = jb_stat
       results['jarque_bera_pvalue'] = jb_pvalue
       results['is_normal'] = jb_pvalue > 0.05

       # Ljung-Box test for autocorrelation
       from statsmodels.stats.diagnostic import acorr_ljungbox
       lb_result = acorr_ljungbox(returns_data, lags=10, return_df=True)
       results['ljung_box_pvalue'] = lb_result['lb_pvalue'].iloc[-1]
       results['has_autocorrelation'] = results['ljung_box_pvalue'] < 0.05

       # ARCH test for heteroscedasticity
       from statsmodels.stats.diagnostic import het_arch
       arch_stat, arch_pvalue, _, _ = het_arch(returns_data, nlags=5)
       results['arch_test_pvalue'] = arch_pvalue
       results['has_arch_effects'] = arch_pvalue < 0.05

       # Hurst exponent (measure of long-term memory)
       def hurst_exponent(ts):
           """Calculate Hurst exponent"""
           lags = range(2, 100)
           tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
           poly = np.polyfit(np.log(lags), np.log(tau), 1)
           return poly[0] * 2.0

       results['hurst_exponent'] = hurst_exponent(returns_data.values)

       return results

   # Perform advanced analysis
   advanced_stats = advanced_distribution_analysis(sp500)

   print("\n=== ADVANCED STATISTICAL ANALYSIS ===")
   print(f"Jarque-Bera p-value: {advanced_stats['jarque_bera_pvalue']:.6f}")
   print(f"Returns are normal: {advanced_stats['is_normal']}")
   print(f"Ljung-Box p-value: {advanced_stats['ljung_box_pvalue']:.6f}")
   print(f"Has autocorrelation: {advanced_stats['has_autocorrelation']}")
   print(f"ARCH test p-value: {advanced_stats['arch_test_pvalue']:.6f}")
   print(f"Has ARCH effects: {advanced_stats['has_arch_effects']}")
   print(f"Hurst exponent: {advanced_stats['hurst_exponent']:.4f}")

   # Interpret Hurst exponent
   if advanced_stats['hurst_exponent'] > 0.5:
       print("  → Persistent/trending behavior")
   elif advanced_stats['hurst_exponent'] < 0.5:
       print("  → Mean-reverting behavior")
   else:
       print("  → Random walk behavior")

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

Advanced Visualization
----------------------

Custom Plotting Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_risk_return_scatter(frame):
       """Create risk-return scatter plot"""
       import plotly.graph_objects as go
       import plotly.express as px

       # Get metrics for all assets
       metrics = frame.all_properties()
       returns = metrics.loc['geo_ret'] * 100
       volatilities = metrics.loc['vol'] * 100
       sharpe_ratios = metrics.loc['ret_vol_ratio']

       # Create scatter plot
       fig = go.Figure()

       fig.add_trace(go.Scatter(
           x=volatilities,
           y=returns,
           mode='markers+text',
           text=returns.index,
           textposition="top center",
           marker=dict(
               size=sharpe_ratios * 20,  # Size based on Sharpe ratio
               color=sharpe_ratios,
               colorscale='Viridis',
               showscale=True,
               colorbar=dict(title="Sharpe Ratio")
           ),
           name='Assets'
       ))

       fig.update_layout(
           title='Risk-Return Analysis',
           xaxis_title='Volatility (%)',
           yaxis_title='Annual Return (%)',
           hovermode='closest'
       )

       return fig

   # Create risk-return plot
   if 'portfolio_assets' in locals():
       risk_return_fig = create_risk_return_scatter(portfolio_assets)
       # risk_return_fig.show()  # Uncomment to display

Performance Attribution
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def performance_attribution_chart(frame, weights):
       """Create performance attribution waterfall chart"""
       import plotly.graph_objects as go

       # Calculate individual contributions using OpenFrame
       asset_metrics = frame.all_properties()
       asset_returns = asset_metrics.loc['geo_ret'].values
       contributions = [weight * ret for weight, ret in zip(weights, asset_returns)]

       # Create waterfall chart
       fig = go.Figure(go.Waterfall(
           name="Performance Attribution",
           orientation="v",
           measure=["relative"] * len(contributions) + ["total"],
           x=[series.name for series in frame.constituents] + ["Portfolio"],
           textposition="outside",
           text=[f"{contrib:.2%}" for contrib in contributions] + [f"{sum(contributions):.2%}"],
           y=contributions + [sum(contributions)],
           connector={"line": {"color": "rgb(63, 63, 63)"}},
       ))

       fig.update_layout(
           title="Portfolio Performance Attribution",
           showlegend=True,
           yaxis_title="Contribution to Return (%)"
       )

       return fig

   # Create attribution chart
   if 'portfolio_assets' in locals() and 'equal_weights' in locals():
       attribution_fig = performance_attribution_chart(portfolio_assets, equal_weights)
       # attribution_fig.show()  # Uncomment to display

Integration with External Libraries
-----------------------------------

QuantLib Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example of integrating with QuantLib for advanced calculations
   # Note: Requires 'pip install QuantLib-Python'

   try:
       import QuantLib as ql

       def calculate_option_greeks(spot_price, strike, risk_free_rate, volatility, time_to_expiry):
           """Calculate option Greeks using QuantLib"""

           # Set up the option
           payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
           exercise = ql.EuropeanExercise(ql.Date.todaysDate() + int(time_to_expiry * 365))
           option = ql.VanillaOption(payoff, exercise)

           # Set up the Black-Scholes process
           spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
           flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql.Date.todaysDate(), risk_free_rate, ql.Actual365Fixed()))
           flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ql.Date.todaysDate(), ql.NullCalendar(), volatility, ql.Actual365Fixed()))

           bs_process = ql.BlackScholesProcess(spot_handle, flat_ts, flat_vol_ts)

           # Set up the pricing engine
           engine = ql.AnalyticEuropeanEngine(bs_process)
           option.setPricingEngine(engine)

           # Calculate Greeks
           greeks = {
               'price': option.NPV(),
               'delta': option.delta(),
               'gamma': option.gamma(),
               'theta': option.theta(),
               'vega': option.vega(),
               'rho': option.rho()
           }

           return greeks

       # Example usage with current market data
       current_price = sp500.tsdf.iloc[-1, 0]
       implied_vol = sp500.vol  # Use historical vol as proxy for implied vol

       greeks = calculate_option_greeks(
           spot_price=current_price,
           strike=current_price,  # At-the-money
           risk_free_rate=0.05,   # 5% risk-free rate
           volatility=implied_vol,
           time_to_expiry=0.25    # 3 months
       )

       print("\n=== OPTION GREEKS (ATM, 3M expiry) ===")
       for greek, value in greeks.items():
           print(f"{greek.capitalize()}: {value:.4f}")

   except ImportError:
       print("QuantLib not available. Install with: pip install QuantLib-Python")

Machine Learning Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example of using machine learning for return prediction
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error, r2_score

   def ml_return_prediction(series, lookback_window=20, forecast_horizon=5):
       """Use machine learning to predict future returns"""

       # Convert to returns
       returns = series.value_to_ret()
       returns_data = returns.tsdf.iloc[:, 0].dropna()

       # Create features (lagged returns and technical indicators)
       features = []
       targets = []

       for i in range(lookback_window, len(returns_data) - forecast_horizon):
           # Features: past returns and simple moving averages
           past_returns = returns_data.iloc[i-lookback_window:i].values
           sma_5 = returns_data.iloc[i-5:i].mean()
           sma_20 = returns_data.iloc[i-20:i].mean()
           volatility = returns_data.iloc[i-20:i].std()

           feature_vector = list(past_returns) + [sma_5, sma_20, volatility]
           features.append(feature_vector)

           # Target: future return
           future_return = returns_data.iloc[i:i+forecast_horizon].mean()
           targets.append(future_return)

       features = np.array(features)
       targets = np.array(targets)

       # Split data
       X_train, X_test, y_train, y_test = train_test_split(
           features, targets, test_size=0.2, random_state=42
       )

       # Train model
       model = RandomForestRegressor(n_estimators=100, random_state=42)
       model.fit(X_train, y_train)

       # Make predictions
       y_pred = model.predict(X_test)

       # Evaluate
       mse = mean_squared_error(y_test, y_pred)
       r2 = r2_score(y_test, y_pred)

       results = {
           'model': model,
           'mse': mse,
           'r2': r2,
           'feature_importance': model.feature_importances_
       }

       return results

   # Apply ML prediction
   try:
       ml_results = ml_return_prediction(sp500)

       print("\n=== MACHINE LEARNING PREDICTION RESULTS ===")
       print(f"Mean Squared Error: {ml_results['mse']:.6f}")
       print(f"R-squared Score: {ml_results['r2']:.4f}")
       print(f"Top 3 Important Features:")

       importance_indices = np.argsort(ml_results['feature_importance'])[-3:]
       for i, idx in enumerate(reversed(importance_indices)):
           print(f"  {i+1}. Feature {idx}: {ml_results['feature_importance'][idx]:.4f}")

   except Exception as e:
       print(f"ML analysis failed: {e}")

Custom Data Sources
-------------------

Creating Custom Data Loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomDataLoader:
       """Custom data loader for various data sources"""

       @staticmethod
       def from_csv_file(filepath, date_column='Date', value_column='Close', name='Custom Series'):
           """Load data from CSV file"""
           df = pd.read_csv(filepath)
           df[date_column] = pd.to_datetime(df[date_column])
           df.set_index(date_column, inplace=True)

           return OpenTimeSeries.from_df(dframe=df[value_column], name=name)

       @staticmethod
       def from_database(connection_string, query, date_column='date', value_column='price', name='DB Series'):
           """Load data from database"""
           import sqlite3

           conn = sqlite3.connect(connection_string)
           df = pd.read_sql_query(query, conn, index_col=date_column, parse_dates=[date_column])
           conn.close()

           return OpenTimeSeries.from_df(dframe=df[value_column], name=name)

       @staticmethod
       def from_api(url, headers=None, date_field='date', value_field='close', name='API Series'):
           """Load data from REST API"""
           import requests

           response = requests.get(url, headers=headers or {})
           data = response.json()

           dates = [item[date_field] for item in data]
           values = [float(item[value_field]) for item in data]

           return OpenTimeSeries.from_arrays(dates=dates, values=values, name=name)

   # Example usage (commented out as it requires actual data sources)
   # custom_series = CustomDataLoader.from_csv_file('data.csv', name='Custom Data')

Performance Optimization
------------------------

Efficient Data Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def optimize_large_dataset_processing(series_list, chunk_size=1000):
       """Optimize processing of large datasets"""

       # Process in chunks to manage memory
       results = []

       for i in range(0, len(series_list), chunk_size):
           chunk = series_list[i:i+chunk_size]

           # Process chunk
           chunk_frame = OpenFrame(constituents=chunk)
           chunk_metrics = chunk_frame.all_properties()

           results.append(chunk_metrics)

           # Clear memory
           del chunk_frame, chunk_metrics

       # Combine results
       final_results = pd.concat(results, axis=1)
       return final_results

   # Parallel processing example
   from concurrent.futures import ProcessPoolExecutor

   def parallel_metric_calculation(series):
       """Calculate metrics for a single series (for parallel processing)"""
       return {
           'name': series.name,
           'return': series.geo_ret,
           'volatility': series.vol,
           'sharpe': series.ret_vol_ratio,
           'max_drawdown': series.max_drawdown
       }

   def process_series_parallel(series_list, max_workers=4):
       """Process multiple series in parallel"""

       with ProcessPoolExecutor(max_workers=max_workers) as executor:
           results = list(executor.map(parallel_metric_calculation, series_list))

       return pd.DataFrame(results).set_index('name')

   # Example usage
   if 'portfolio_assets' in locals():
       print("\n=== PERFORMANCE OPTIMIZATION EXAMPLE ===")

       # Sequential processing
       import time
       start_time = time.time()
       sequential_results = portfolio_assets.all_properties()
       sequential_time = time.time() - start_time

       print(f"Sequential processing time: {sequential_time:.4f} seconds")

       # Note: Parallel processing would be beneficial for larger datasets
       print("Parallel processing is most beneficial with 100+ assets")

Export and Reporting
--------------------

Advanced Report Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_comprehensive_report(portfolio, filename='comprehensive_report.html'):
       """Generate a comprehensive HTML report"""

       html_content = f"""
       <!DOCTYPE html>
       <html>
       <head>
           <title>Portfolio Analysis Report</title>
           <style>
               body {{ font-family: Arial, sans-serif; margin: 40px; }}
               .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
               .metric {{ margin: 10px 0; }}
               .section {{ margin: 30px 0; }}
               table {{ border-collapse: collapse; width: 100%; }}
               th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
               th {{ background-color: #f2f2f2; }}
           </style>
       </head>
       <body>
           <div class="header">
               <h1>Portfolio Analysis Report</h1>
               <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
               <p>Portfolio: {portfolio.name}</p>
               <p>Period: {portfolio.first_idx} to {portfolio.last_idx}</p>
           </div>

           <div class="section">
               <h2>Performance Summary</h2>
               <div class="metric">Total Return: {portfolio.value_ret:.2%}</div>
               <div class="metric">Annualized Return: {portfolio.geo_ret:.2%}</div>
               <div class="metric">Annualized Volatility: {portfolio.vol:.2%}</div>
               <div class="metric">Sharpe Ratio: {portfolio.ret_vol_ratio:.3f}</div>
               <div class="metric">Maximum Drawdown: {portfolio.max_drawdown:.2%}</div>
           </div>

           <div class="section">
               <h2>Risk Metrics</h2>
               <div class="metric">95% VaR (daily): {portfolio.var_down:.2%}</div>
               <div class="metric">95% CVaR (daily): {portfolio.cvar_down:.2%}</div>
               <div class="metric">Sortino Ratio: {portfolio.sortino_ratio:.3f}</div>
               <div class="metric">Skewness: {portfolio.skew:.3f}</div>
               <div class="metric">Kurtosis: {portfolio.kurtosis:.3f}</div>
           </div>
       </body>
       </html>
       """

       with open(filename, 'w') as f:
           f.write(html_content)

       print(f"Comprehensive report saved to {filename}")

   # Generate report
   if 'portfolio' in locals():
       generate_comprehensive_report(portfolio)

   print("\n=== ADVANCED FEATURES TUTORIAL COMPLETE ===")
   print("You've learned about:")
   print("• Custom metrics and analysis functions")
   print("• Advanced statistical analysis")
   print("• Factor models and regression")
   print("• Advanced portfolio techniques")
   print("• Custom visualization")
   print("• External library integration")
   print("• Performance optimization")
   print("• Advanced reporting")

This tutorial demonstrates how to extend openseries with advanced functionality for sophisticated financial analysis workflows.
