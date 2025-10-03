Custom Reports
==============

This example demonstrates how to create custom analysis reports using openseries and the built-in report functionality.

Using the Built-in HTML Report
-------------------------------

.. code-block:: python

   import yfinance as yf
   from openseries import OpenTimeSeries, OpenFrame, report_html
   import pandas as pd

   # Load sample data for comparison
   tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
   names = ["Apple", "Microsoft", "Google", "S&P 500"]

   series_list = []
   for ticker, name in zip(tickers, names):
       data = yf.Ticker(ticker).history(period="3y")
       series = OpenTimeSeries.from_df(dframe=data['Close'], name=name)
       series.set_new_label(lvl_zero=name)
       series_list.append(series)

   # Create frame for report
   comparison_frame = OpenFrame(constituents=series_list)

   # Generate HTML report
   # The last asset in the frame is used as the benchmark
   report_html(
       frame=comparison_frame,
       output_type="file",
       file_name="stock_comparison_report.html"
   )

   print("HTML report generated: stock_comparison_report.html")

Creating Custom Analysis Reports
---------------------------------

Custom Performance Report
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_performance_report(series, filename="performance_report.html"):
       """Create a custom performance analysis report"""

       # Calculate comprehensive metrics
       metrics = series.all_properties

       # Additional custom calculations
       returns = series.value_to_ret()
       returns_data = returns.tsdf.iloc[:, 0]

       # Monthly statistics
       monthly_series = series.resample_to_business_period_ends(freq="BME")
       monthly_returns = monthly_series.value_to_ret()
       monthly_data = monthly_returns.tsdf.iloc[:, 0]

       # Create HTML content
       html_content = f"""
       <!DOCTYPE html>
       <html>
       <head>
           <title>Performance Report - {series.name}</title>
           <style>
               body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
               .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
               .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
               .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
               .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
               .metric-title {{ font-weight: bold; color: #495057; margin-bottom: 10px; }}
               .metric-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
               .section {{ margin: 30px 0; }}
               .section-title {{ font-size: 1.3em; font-weight: bold; margin-bottom: 15px; color: #495057; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
               table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
               th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
               th {{ background-color: #f8f9fa; font-weight: bold; }}
               .positive {{ color: #28a745; }}
               .negative {{ color: #dc3545; }}
               .neutral {{ color: #6c757d; }}
           </style>
       </head>
       <body>
           <div class="container">
               <div class="header">
                   <h1>Performance Analysis Report</h1>
                   <h2>{series.name}</h2>
                   <p>Analysis Period: {series.first_idx} to {series.last_idx}</p>
                   <p>Total Observations: {series.length:,}</p>
               </div>

               <div class="section">
                   <div class="section-title">Key Performance Metrics</div>
                   <div class="metric-grid">
                       <div class="metric-card">
                           <div class="metric-title">Total Return</div>
                           <div class="metric-value {'positive' if series.value_ret > 0 else 'negative'}">{series.value_ret:.2%}</div>
                       </div>
                       <div class="metric-card">
                           <div class="metric-title">Annualized Return</div>
                           <div class="metric-value {'positive' if series.geo_ret > 0 else 'negative'}">{series.geo_ret:.2%}</div>
                       </div>
                       <div class="metric-card">
                           <div class="metric-title">Volatility</div>
                           <div class="metric-value">{series.vol:.2%}</div>
                       </div>
                       <div class="metric-card">
                           <div class="metric-title">Sharpe Ratio</div>
                           <div class="metric-value {'positive' if series.ret_vol_ratio > 0 else 'negative'}">{series.ret_vol_ratio:.3f}</div>
                       </div>
                       <div class="metric-card">
                           <div class="metric-title">Maximum Drawdown</div>
                           <div class="metric-value negative">{series.max_drawdown:.2%}</div>
                       </div>
                       <div class="metric-card">
                           <div class="metric-title">Sortino Ratio</div>
                           <div class="metric-value {'positive' if series.sortino_ratio > 0 else 'negative'}">{series.sortino_ratio:.3f}</div>
                       </div>
                   </div>
               </div>

               <div class="section">
                   <div class="section-title">Risk Metrics</div>
                   <table>
                       <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
                       <tr><td>95% VaR (daily)</td><td class="negative">{series.var_down:.2%}</td><td>Expected worst daily loss (95% confidence)</td></tr>
                       <tr><td>95% CVaR (daily)</td><td class="negative">{series.cvar_down:.2%}</td><td>Average loss beyond VaR</td></tr>
                       <tr><td>Downside Deviation</td><td>{series.downside_deviation:.2%}</td><td>Volatility of negative returns only</td></tr>
                       <tr><td>Worst Single Day</td><td class="negative">{series.worst:.2%}</td><td>Largest single-day loss</td></tr>
                       <tr><td>Positive Days</td><td class="{'positive' if series.positive_share > 0.5 else 'negative'}">{series.positive_share:.1%}</td><td>Percentage of positive return days</td></tr>
                   </table>
               </div>

               <div class="section">
                   <div class="section-title">Distribution Analysis</div>
                   <table>
                       <tr><th>Statistic</th><th>Value</th><th>Interpretation</th></tr>
                       <tr><td>Skewness</td><td class="{'negative' if series.skew < 0 else 'positive' if series.skew > 0 else 'neutral'}">{series.skew:.3f}</td><td>{'Negative skew - more extreme losses' if series.skew < -0.5 else 'Positive skew - more extreme gains' if series.skew > 0.5 else 'Approximately symmetric'}</td></tr>
                       <tr><td>Kurtosis</td><td class="{'negative' if series.kurtosis > 3 else 'neutral'}">{series.kurtosis:.3f}</td><td>{'Fat tails - more extreme events than normal' if series.kurtosis > 3 else 'Normal tail behavior'}</td></tr>
                       <tr><td>Z-Score (last return)</td><td class="{'negative' if abs(series.z_score) > 2 else 'neutral'}">{series.z_score:.2f}</td><td>{'Unusual recent return' if abs(series.z_score) > 2 else 'Normal recent return'}</td></tr>
                   </table>
               </div>

               <div class="section">
                   <div class="section-title">Monthly Performance Summary</div>
                   <table>
                       <tr><th>Metric</th><th>Value</th></tr>
                       <tr><td>Number of Months</td><td>{len(monthly_data)}</td></tr>
                       <tr><td>Positive Months</td><td>{(monthly_data > 0).sum()} ({(monthly_data > 0).mean():.1%})</td></tr>
                       <tr><td>Average Monthly Return</td><td class="{'positive' if monthly_data.mean() > 0 else 'negative'}">{monthly_data.mean():.2%}</td></tr>
                       <tr><td>Best Month</td><td class="positive">{monthly_data.max():.2%}</td></tr>
                       <tr><td>Worst Month</td><td class="negative">{monthly_data.min():.2%}</td></tr>
                       <tr><td>Monthly Volatility</td><td>{monthly_data.std():.2%}</td></tr>
                   </table>
               </div>

               <div class="section">
                   <div class="section-title">Investment Summary</div>
                   <p>Based on the analysis of <strong>{series.name}</strong> over the period from {series.first_idx} to {series.last_idx}:</p>
                   <ul>
                       <li>The investment {'generated' if series.value_ret > 0 else 'lost'} a total return of <strong>{series.value_ret:.2%}</strong></li>
                       <li>Annualized return was <strong>{series.geo_ret:.2%}</strong> with volatility of <strong>{series.vol:.2%}</strong></li>
                       <li>Risk-adjusted performance (Sharpe ratio) was <strong>{series.ret_vol_ratio:.3f}</strong></li>
                       <li>Maximum drawdown reached <strong>{series.max_drawdown:.2%}</strong> on {series.max_drawdown_date}</li>
                       <li>The investment had positive returns on <strong>{series.positive_share:.1%}</strong> of trading days</li>
                   </ul>
               </div>
           </div>
       </body>
       </html>
       """

       # Write to file
       with open(filename, 'w', encoding='utf-8') as f:
           f.write(html_content)

       print(f"Custom performance report saved to {filename}")

   # Generate custom report for Apple
   apple = series_list[0]  # Apple from our earlier example
   create_performance_report(apple, "apple_custom_report.html")

Portfolio Comparison Report
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_portfolio_comparison_report(frame, filename="portfolio_comparison.html"):
       """Create a comprehensive portfolio comparison report"""

       # Get metrics for all assets
       all_metrics = frame.all_properties

       # Calculate correlations
       correlations = frame.correl_matrix()

       # Create HTML content
       html_content = f"""
       <!DOCTYPE html>
       <html>
       <head>
           <title>Portfolio Comparison Report</title>
           <style>
               body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
               .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
               .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
               .section {{ margin: 30px 0; }}
               .section-title {{ font-size: 1.3em; font-weight: bold; margin-bottom: 15px; color: #495057; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
               table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
               th, td {{ padding: 8px; text-align: right; border-bottom: 1px solid #dee2e6; }}
               th {{ background-color: #f8f9fa; font-weight: bold; position: sticky; top: 0; }}
               .asset-name {{ text-align: left; font-weight: bold; }}
               .best {{ background-color: #d4edda; color: #155724; }}
               .worst {{ background-color: #f8d7da; color: #721c24; }}
               .correlation-matrix {{ font-size: 0.8em; }}
               .correlation-matrix td {{ text-align: center; }}
               .high-corr {{ background-color: #fff3cd; }}
               .low-corr {{ background-color: #d1ecf1; }}
           </style>
       </head>
       <body>
           <div class="container">
               <div class="header">
                   <h1>Portfolio Comparison Report</h1>
                   <p>Analysis Period: {frame.first_idx} to {frame.last_idx}</p>
                   <p>Number of Assets: {frame.item_count}</p>
               </div>

               <div class="section">
                   <div class="section-title">Performance Comparison</div>
                   <table>
                       <thead>
                           <tr>
                               <th class="asset-name">Asset</th>
                               <th>Total Return</th>
                               <th>Annual Return</th>
                               <th>Volatility</th>
                               <th>Sharpe Ratio</th>
                               <th>Max Drawdown</th>
                               <th>Sortino Ratio</th>
                           </tr>
                       </thead>
                       <tbody>
       """

       # Add performance data for each asset
       for asset_name in all_metrics.columns:
           total_ret = all_metrics.loc['value_ret', asset_name]
           annual_ret = all_metrics.loc['geo_ret', asset_name]
           volatility = all_metrics.loc['vol', asset_name]
           sharpe = all_metrics.loc['ret_vol_ratio', asset_name]
           max_dd = all_metrics.loc['max_drawdown', asset_name]
           sortino = all_metrics.loc['sortino_ratio', asset_name]

           html_content += f"""
                           <tr>
                               <td class="asset-name">{asset_name}</td>
                               <td>{total_ret:.2%}</td>
                               <td>{annual_ret:.2%}</td>
                               <td>{volatility:.2%}</td>
                               <td>{sharpe:.3f}</td>
                               <td>{max_dd:.2%}</td>
                               <td>{sortino:.3f}</td>
                           </tr>
           """

       html_content += """
                       </tbody>
                   </table>
               </div>

               <div class="section">
                   <div class="section-title">Risk Metrics Comparison</div>
                   <table>
                       <thead>
                           <tr>
                               <th class="asset-name">Asset</th>
                               <th>95% VaR</th>
                               <th>95% CVaR</th>
                               <th>Downside Dev</th>
                               <th>Skewness</th>
                               <th>Kurtosis</th>
                               <th>Positive Days</th>
                           </tr>
                       </thead>
                       <tbody>
       """

       # Add risk data for each asset
       for asset_name in all_metrics.columns:
           var_95 = all_metrics.loc['var_down', asset_name]
           cvar_95 = all_metrics.loc['cvar_down', asset_name]
           downside_dev = all_metrics.loc['downside_deviation', asset_name]
           skewness = all_metrics.loc['skew', asset_name]
           kurtosis = all_metrics.loc['kurtosis', asset_name]
           positive_share = all_metrics.loc['positive_share', asset_name]

           html_content += f"""
                           <tr>
                               <td class="asset-name">{asset_name}</td>
                               <td>{var_95:.2%}</td>
                               <td>{cvar_95:.2%}</td>
                               <td>{downside_dev:.2%}</td>
                               <td>{skewness:.3f}</td>
                               <td>{kurtosis:.3f}</td>
                               <td>{positive_share:.1%}</td>
                           </tr>
           """

       html_content += """
                       </tbody>
                   </table>
               </div>

               <div class="section">
                   <div class="section-title">Correlation Matrix</div>
                   <table class="correlation-matrix">
                       <thead>
                           <tr>
                               <th class="asset-name">Asset</th>
       """

       # Add correlation matrix headers
       for asset_name in correlations.columns:
           html_content += f"<th>{asset_name[:10]}</th>"  # Truncate long names

       html_content += """
                           </tr>
                       </thead>
                       <tbody>
       """

       # Add correlation matrix data
       for i, asset_name in enumerate(correlations.index):
           html_content += f'<tr><td class="asset-name">{asset_name}</td>'
           for j, corr_value in enumerate(correlations.iloc[i]):
               css_class = ""
               if i != j:  # Don't highlight diagonal
                   if corr_value > 0.7:
                       css_class = "high-corr"
                   elif corr_value < 0.3:
                       css_class = "low-corr"
               html_content += f'<td class="{css_class}">{corr_value:.3f}</td>'
           html_content += '</tr>'

       html_content += """
                       </tbody>
                   </table>
               </div>

               <div class="section">
                   <div class="section-title">Summary and Recommendations</div>
       """

       # Add summary analysis
       best_return = all_metrics.loc['geo_ret'].idxmax()
       best_sharpe = all_metrics.loc['ret_vol_ratio'].idxmax()
       lowest_vol = all_metrics.loc['vol'].idxmin()
       lowest_dd = all_metrics.loc['max_drawdown'].idxmax()  # Highest (least negative) drawdown

       html_content += f"""
                   <h3>Key Findings:</h3>
                   <ul>
                       <li><strong>Best Performer:</strong> {best_return} with {all_metrics.loc['geo_ret', best_return]:.2%} annual return</li>
                       <li><strong>Best Risk-Adjusted Return:</strong> {best_sharpe} with Sharpe ratio of {all_metrics.loc['ret_vol_ratio', best_sharpe]:.3f}</li>
                       <li><strong>Lowest Volatility:</strong> {lowest_vol} with {all_metrics.loc['vol', lowest_vol]:.2%} volatility</li>
                       <li><strong>Smallest Drawdown:</strong> {lowest_dd} with {all_metrics.loc['max_drawdown', lowest_dd]:.2%} maximum drawdown</li>
                   </ul>

                   <h3>Correlation Insights:</h3>
                   <ul>
       """

       # Add correlation insights
       high_corr_pairs = []
       low_corr_pairs = []

       for i in range(len(correlations.columns)):
           for j in range(i+1, len(correlations.columns)):
               corr_val = correlations.iloc[i, j]
               asset1 = correlations.columns[i]
               asset2 = correlations.columns[j]

               if corr_val > 0.8:
                   high_corr_pairs.append((asset1, asset2, corr_val))
               elif corr_val < 0.2:
                   low_corr_pairs.append((asset1, asset2, corr_val))

       if high_corr_pairs:
           html_content += "<li><strong>Highly Correlated Pairs:</strong><ul>"
           for asset1, asset2, corr in high_corr_pairs[:3]:  # Show top 3
               html_content += f"<li>{asset1} and {asset2}: {corr:.3f}</li>"
           html_content += "</ul></li>"

       if low_corr_pairs:
           html_content += "<li><strong>Low Correlation Pairs (Good for Diversification):</strong><ul>"
           for asset1, asset2, corr in low_corr_pairs[:3]:  # Show top 3
               html_content += f"<li>{asset1} and {asset2}: {corr:.3f}</li>"
           html_content += "</ul></li>"

       html_content += """
                   </ul>
               </div>
           </div>
       </body>
       </html>
       """

       # Write to file
       with open(filename, 'w', encoding='utf-8') as f:
           f.write(html_content)

       print(f"Portfolio comparison report saved to {filename}")

   # Generate comparison report
   create_portfolio_comparison_report(comparison_frame, "portfolio_comparison_report.html")

Risk Assessment Report
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_risk_assessment_report(series, filename="risk_assessment.html"):
       """Create a detailed risk assessment report"""

       # Calculate additional risk metrics
       returns = series.value_to_ret()
       returns_data = returns.tsdf.iloc[:, 0].dropna()

       # Rolling metrics
       rolling_vol = series.rolling_vol(window=252)
       rolling_var = series.rolling_var_down(window=252)

       # Stress test scenarios
       worst_5pct = returns_data.quantile(0.05)
       worst_1pct = returns_data.quantile(0.01)

       html_content = f"""
       <!DOCTYPE html>
       <html>
       <head>
           <title>Risk Assessment Report - {series.name}</title>
           <style>
               body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
               .container {{ max-width: 1000px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }}
               .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background-color: #dc3545; color: white; border-radius: 8px; }}
               .risk-level {{ padding: 10px; border-radius: 5px; margin: 10px 0; font-weight: bold; text-align: center; }}
               .low-risk {{ background-color: #d4edda; color: #155724; }}
               .medium-risk {{ background-color: #fff3cd; color: #856404; }}
               .high-risk {{ background-color: #f8d7da; color: #721c24; }}
               .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
               .metric-box {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; }}
               table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
               th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #dee2e6; }}
               th {{ background-color: #f8f9fa; }}
               .section {{ margin: 25px 0; }}
               .section-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 15px; color: #495057; }}
           </style>
       </head>
       <body>
           <div class="container">
               <div class="header">
                   <h1>Risk Assessment Report</h1>
                   <h2>{series.name}</h2>
                   <p>Assessment Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
               </div>
       """

       # Risk level assessment
       risk_score = 0
       if series.vol > 0.25: risk_score += 2
       elif series.vol > 0.15: risk_score += 1

       if abs(series.max_drawdown) > 0.30: risk_score += 2
       elif abs(series.max_drawdown) > 0.20: risk_score += 1

       if series.ret_vol_ratio < 0.5: risk_score += 1

       if risk_score >= 4:
           risk_level = "HIGH RISK"
           risk_class = "high-risk"
       elif risk_score >= 2:
           risk_level = "MEDIUM RISK"
           risk_class = "medium-risk"
       else:
           risk_level = "LOW RISK"
           risk_class = "low-risk"

       html_content += f"""
               <div class="risk-level {risk_class}">
                   OVERALL RISK ASSESSMENT: {risk_level}
               </div>

               <div class="section">
                   <div class="section-title">Key Risk Metrics</div>
                   <div class="metric-grid">
                       <div class="metric-box">
                           <strong>Volatility</strong><br>
                           {series.vol:.2%}
                       </div>
                       <div class="metric-box">
                           <strong>Maximum Drawdown</strong><br>
                           {series.max_drawdown:.2%}
                       </div>
                       <div class="metric-box">
                           <strong>95% VaR (Daily)</strong><br>
                           {series.var_down:.2%}
                       </div>
                       <div class="metric-box">
                           <strong>95% CVaR (Daily)</strong><br>
                           {series.cvar_down:.2%}
                       </div>
                   </div>
               </div>

               <div class="section">
                   <div class="section-title">Stress Test Results</div>
                   <table>
                       <tr><th>Scenario</th><th>Threshold</th><th>Frequency</th><th>Impact</th></tr>
                       <tr><td>Worst 5% Days</td><td>{worst_5pct:.2%}</td><td>{(returns_data <= worst_5pct).sum()} days</td><td>Moderate stress</td></tr>
                       <tr><td>Worst 1% Days</td><td>{worst_1pct:.2%}</td><td>{(returns_data <= worst_1pct).sum()} days</td><td>Severe stress</td></tr>
                       <tr><td>Maximum Single Day Loss</td><td>{series.worst:.2%}</td><td>1 day</td><td>Extreme event</td></tr>
                   </table>
               </div>

               <div class="section">
                   <div class="section-title">Risk Monitoring Alerts</div>
                   <ul>
       """

       # Add risk alerts
       alerts = []
       if series.vol > 0.30:
           alerts.append("⚠️ ALERT: Very high volatility (>30%)")
       if abs(series.max_drawdown) > 0.40:
           alerts.append("🚨 WARNING: Extreme drawdown (>40%)")
       if series.ret_vol_ratio < 0.3:
           alerts.append("⚠️ ALERT: Poor risk-adjusted returns (Sharpe < 0.3)")
       if abs(series.skew) > 2:
           alerts.append("⚠️ ALERT: Highly skewed return distribution")
       if series.kurtosis > 6:
           alerts.append("⚠️ ALERT: Fat-tailed distribution (high kurtosis)")

       if alerts:
           for alert in alerts:
               html_content += f"<li>{alert}</li>"
       else:
           html_content += "<li>✅ No immediate risk alerts</li>"

       html_content += f"""
                   </ul>
               </div>

               <div class="section">
                   <div class="section-title">Risk Management Recommendations</div>
                   <ul>
       """

       # Add recommendations based on risk profile
       if risk_score >= 4:
           html_content += """
                       <li>Consider reducing position size due to high risk profile</li>
                       <li>Implement strict stop-loss levels</li>
                       <li>Monitor daily for unusual price movements</li>
                       <li>Consider hedging strategies</li>
           """
       elif risk_score >= 2:
           html_content += """
                       <li>Maintain moderate position sizing</li>
                       <li>Set appropriate stop-loss levels</li>
                       <li>Monitor weekly for risk changes</li>
                       <li>Consider diversification benefits</li>
           """
       else:
           html_content += """
                       <li>Suitable for larger position sizes</li>
                       <li>Standard risk monitoring procedures</li>
                       <li>Good candidate for core portfolio holding</li>
           """

       html_content += """
                   </ul>
               </div>
           </div>
       </body>
       </html>
       """

       with open(filename, 'w', encoding='utf-8') as f:
           f.write(html_content)

       print(f"Risk assessment report saved to {filename}")

   # Generate risk assessment for Apple
   create_risk_assessment_report(apple, "apple_risk_assessment.html")

Executive Summary Report
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_executive_summary(frame, filename="executive_summary.html"):
       """Create an executive summary report for portfolio"""

       # Create equal-weighted portfolio for summary
       n_assets = frame.item_count
       equal_weights = [1/n_assets] * n_assets
       portfolio = frame.make_portfolio(weights=equal_weights, name="Portfolio")

       # Get individual asset metrics
       asset_metrics = frame.all_properties

       html_content = f"""
       <!DOCTYPE html>
       <html>
       <head>
           <title>Executive Summary</title>
           <style>
               body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }}
               .container {{ max-width: 800px; margin: 0 auto; }}
               .header {{ text-align: center; margin-bottom: 30px; padding: 30px; background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); color: white; border-radius: 10px; }}
               .summary-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #007bff; }}
               .key-metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }}
               .metric {{ text-align: center; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
               .metric-value {{ font-size: 1.8em; font-weight: bold; color: #007bff; }}
               .metric-label {{ color: #6c757d; margin-top: 5px; }}
               .section {{ margin: 25px 0; }}
               .section-title {{ font-size: 1.3em; font-weight: bold; margin-bottom: 15px; color: #2c3e50; }}
           </style>
       </head>
       <body>
           <div class="container">
               <div class="header">
                   <h1>Investment Portfolio</h1>
                   <h2>Executive Summary</h2>
                   <p>{pd.Timestamp.now().strftime('%B %Y')}</p>
               </div>

               <div class="summary-box">
                   <h3>Portfolio Overview</h3>
                   <p>This report summarizes the performance of a diversified portfolio containing {frame.item_count} assets
                   over the period from {frame.first_idx} to {frame.last_idx}. The portfolio demonstrates
                   {'strong' if portfolio.ret_vol_ratio > 1.0 else 'moderate' if portfolio.ret_vol_ratio > 0.5 else 'weak'}
                   risk-adjusted performance with a Sharpe ratio of {portfolio.ret_vol_ratio:.2f}.</p>
               </div>

               <div class="section">
                   <div class="section-title">Key Performance Indicators</div>
                   <div class="key-metrics">
                       <div class="metric">
                           <div class="metric-value">{portfolio.geo_ret:.1%}</div>
                           <div class="metric-label">Annual Return</div>
                       </div>
                       <div class="metric">
                           <div class="metric-value">{portfolio.vol:.1%}</div>
                           <div class="metric-label">Volatility</div>
                       </div>
                       <div class="metric">
                           <div class="metric-value">{portfolio.ret_vol_ratio:.2f}</div>
                           <div class="metric-label">Sharpe Ratio</div>
                       </div>
                       <div class="metric">
                           <div class="metric-value">{portfolio.max_drawdown:.1%}</div>
                           <div class="metric-label">Max Drawdown</div>
                       </div>
                   </div>
               </div>

               <div class="section">
                   <div class="section-title">Investment Highlights</div>
                   <ul>
                       <li>Portfolio generated <strong>{portfolio.value_ret:.1%}</strong> total return over the analysis period</li>
                       <li>Annualized return of <strong>{portfolio.geo_ret:.1%}</strong> with <strong>{portfolio.vol:.1%}</strong> volatility</li>
                       <li>Maximum drawdown was limited to <strong>{portfolio.max_drawdown:.1%}</strong></li>
                       <li>Risk-adjusted performance (Sharpe ratio) of <strong>{portfolio.ret_vol_ratio:.2f}</strong></li>
                   </ul>
               </div>

               <div class="section">
                   <div class="section-title">Risk Assessment</div>
                   <p>The portfolio exhibits <strong>{'low' if portfolio.vol < 0.15 else 'moderate' if portfolio.vol < 0.25 else 'high'}</strong>
                   volatility at {portfolio.vol:.1%} annually. Daily Value at Risk (95% confidence) is {portfolio.var_down:.2%},
                   indicating potential daily losses could reach this level in 1 out of 20 trading days.</p>
               </div>

               <div class="section">
                   <div class="section-title">Recommendations</div>
       """

       # Add recommendations based on performance
       if portfolio.ret_vol_ratio > 1.0:
           html_content += "<p><strong>POSITIVE:</strong> Strong risk-adjusted performance suggests effective portfolio construction.</p>"
       elif portfolio.ret_vol_ratio > 0.5:
           html_content += "<p><strong>NEUTRAL:</strong> Moderate risk-adjusted performance. Consider optimization opportunities.</p>"
       else:
           html_content += "<p><strong>CAUTION:</strong> Weak risk-adjusted performance. Portfolio review recommended.</p>"

       if abs(portfolio.max_drawdown) > 0.25:
           html_content += "<p><strong>RISK:</strong> Significant drawdown experienced. Consider risk management enhancements.</p>"

       html_content += """
               </div>
           </div>
       </body>
       </html>
       """

       with open(filename, 'w', encoding='utf-8') as f:
           f.write(html_content)

       print(f"Executive summary saved to {filename}")

   # Generate executive summary
   create_executive_summary(comparison_frame, "executive_summary.html")

Batch Report Generation
-----------------------

.. code-block:: python

   def generate_all_reports(frame, base_filename="analysis"):
       """Generate all types of reports for a given frame"""

       print("Generating comprehensive report suite...")

       # Built-in HTML report
       report_html(frame, output_type="file", file_name=f"{base_filename}_builtin.html")

       # Custom reports for individual assets
       for i, series in enumerate(frame.constituents):
           asset_name = series.name.replace(" ", "_").replace(".", "")
           create_performance_report(series, f"{base_filename}_{asset_name}_performance.html")
           create_risk_assessment_report(series, f"{base_filename}_{asset_name}_risk.html")

       # Portfolio-level reports
       create_portfolio_comparison_report(frame, f"{base_filename}_comparison.html")
       create_executive_summary(frame, f"{base_filename}_executive_summary.html")

       print(f"All reports generated with base filename: {base_filename}")

   # Generate all reports for our comparison frame
   generate_all_reports(comparison_frame, "stock_analysis")

   print("\n=== CUSTOM REPORTS TUTORIAL COMPLETE ===")
   print("Generated reports:")
   print("• Built-in HTML report using report_html()")
   print("• Custom performance reports for individual assets")
   print("• Portfolio comparison report")
   print("• Risk assessment reports")
   print("• Executive summary report")
   print("• Batch report generation function")
