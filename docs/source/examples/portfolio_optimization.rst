Portfolio Optimization
======================

This example demonstrates various portfolio optimization techniques using openseries, including both theoretical approaches and real-world applications with actual fund data.

Basic Portfolio Optimization Setup
-----------------------------------

.. code-block:: python

    import yfinance as yf
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
         # This may fail if the ticker is invalid or data unavailable
         data = yf.Ticker(ticker).history(period="5y")
         series = OpenTimeSeries.from_df(dframe=data['Close'])
         series.set_new_label(lvl_zero=name)
         assets.append(series)
         print(f"Loaded {name}")

    # Create investment universe frame
    investment_universe = OpenFrame(constituents=assets)
    print(f"\nInvestment universe: {investment_universe.item_count} assets")
    print(f"Period: {investment_universe.first_idx} to {investment_universe.last_idx}")

Mean-Variance Optimization
--------------------------

.. code-block:: python

    # Calculate efficient frontier
    # This may fail with various exceptions
    frontier_df, simulated_df, optimal_portfolio = efficient_frontier(
         eframe=investment_universe,
         num_ports=100,
         seed=42
    )

    print("=== EFFICIENT FRONTIER RESULTS ===")
    print(f"Generated {len(frontier_df)} efficient portfolios")
    print(f"Simulated {len(simulated_df)} random portfolios")

    # Find key portfolios
    returns = frontier_df['ret']
    volatilities = frontier_df['stdev']
    sharpe_ratios = returns / volatilities

    # Maximum Sharpe ratio portfolio
    max_sharpe_idx = sharpe_ratios.idxmax()
    max_sharpe_weights = optimal_portfolio[-len(investment_universe.constituents):]

    print(f"\n=== MAXIMUM SHARPE RATIO PORTFOLIO ===")
    print(f"Expected Return: {frontier_df.iloc[max_sharpe_idx]['ret']:.2%}")
    print(f"Volatility: {frontier_df.iloc[max_sharpe_idx]['stdev']:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratios.iloc[max_sharpe_idx]:.2f}")

    print("\nOptimal Weights:")
    for i, weight in enumerate(max_sharpe_weights):
         asset_name = investment_universe.constituents[i].label
         if weight > 0.01:  # Only show weights > 1%
              print(f"  {asset_name}: {weight:.1%}")

    # Minimum volatility portfolio
    min_vol_idx = volatilities.idxmin()
    min_vol_weights = frontier_df.iloc[min_vol_idx][investment_universe.columns_lvl_zero].values

    print(f"\n=== MINIMUM VOLATILITY PORTFOLIO ===")
    print(f"Expected Return: {min_vol_row['ret']:.2%}")
    print(f"Volatility: {min_vol_row['stdev']:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratios.iloc[min_vol_idx]:.2f}")

    print("\nMinimum Volatility Weights:")
    for col in investment_universe.columns_lvl_zero:
         weight = min_vol_row[col]
         if weight > 0.01:
              print(f"  {col}: {weight:.1%}")

Monte Carlo Portfolio Simulation
--------------------------------

.. code-block:: python

    # Generate random portfolios
    # This may fail with various exceptions
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
    sorted_indices = sorted(range(len(sim_sharpe_ratios)), key=lambda i: sim_sharpe_ratios.iloc[i], reverse=True)
    top_sharpe_indices = sorted_indices[:5]

    print(f"\n=== TOP 5 SIMULATED PORTFOLIOS ===")
    for i, idx in enumerate(reversed(top_sharpe_indices)):
         print(f"\nRank {i+1}:")
         print(f"  Return: {sim_returns[idx]:.2%}")
         print(f"  Volatility: {sim_volatilities[idx]:.2%}")
         print(f"  Sharpe: {sim_sharpe_ratios[idx]:.2f}")

         weights = simulation_results.iloc[idx][investment_universe.columns_lvl_zero].values
         print("  Weights:")
         for j, weight in enumerate(weights):
              if weight > 0.05:  # Only show weights > 5%
                    asset_name = investment_universe.constituents[j].label
                    print(f"    {asset_name}: {weight:.1%}")

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
    # This may fail with MaxDiversificationNaNError or MaxDiversificationNegativeWeightsError
    max_div_portfolio_df = investment_universe.make_portfolio(
         name="Maximum Diversification",
         weight_strat="max_div"
    )
    max_div_portfolio = OpenTimeSeries.from_df(dframe=max_div_portfolio_df)

    print(f"\n=== MAXIMUM DIVERSIFICATION PORTFOLIO ===")
    print(f"Return: {max_div_portfolio.geo_ret:.2%}")
    print(f"Volatility: {max_div_portfolio.vol:.2%}")
    print(f"Sharpe: {max_div_portfolio.ret_vol_ratio:.2f}")

Minimum Volatility Overweight Portfolio
----------------------------------------

.. code-block:: python

    # Minimum volatility overweight portfolio using native weight_strat
    min_vol_portfolio_df = investment_universe.make_portfolio(
         name="Min Vol Overweight",
         weight_strat="min_vol_overweight"
    )
    min_vol_portfolio = OpenTimeSeries.from_df(dframe=min_vol_portfolio_df)

    print(f"\n=== MINIMUM VOLATILITY OVERWEIGHT PORTFOLIO ===")
    print(f"Return: {min_vol_portfolio.geo_ret:.2%}")
    print(f"Volatility: {min_vol_portfolio.vol:.2%}")
    print(f"Sharpe: {min_vol_portfolio.ret_vol_ratio:.2f}")

Portfolio Comparison
--------------------

.. code-block:: python

    # Compare all portfolio strategies
    portfolios = [
         equal_weight_portfolio,
         inv_vol_portfolio,
         max_div_portfolio,
         min_vol_portfolio
    ]

    # Add optimized portfolios if available
    if 'max_sharpe_weights' in locals():
         investment_universe.weights = max_sharpe_weights.tolist()
         max_sharpe_portfolio_df = investment_universe.make_portfolio(
              name="Max Sharpe (Optimized)"
         )
         max_sharpe_portfolio = OpenTimeSeries.from_df(dframe=max_sharpe_portfolio_df)
         portfolios.append(max_sharpe_portfolio)

    if 'min_vol_weights' in locals():
         investment_universe.weights = min_vol_weights.tolist()
         min_vol_portfolio_df = investment_universe.make_portfolio(
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

**Minimum Volatility Overweight (``min_vol_overweight``)**
   - Overweights the least volatile asset (60% weight)
   - Distributes remaining 40% equally among other assets
   - Based on the low volatility anomaly

**Exception Handling**
   When using the maximum diversification strategy, it's recommended to handle potential exceptions:

   .. code-block:: python

      from openseries.owntypes import (
          MaxDiversificationNaNError,
          MaxDiversificationNegativeWeightsError
      )

      # This may fail with MaxDiversificationNaNError or MaxDiversificationNegativeWeightsError
      portfolio_df = frame.make_portfolio(name="Max Div", weight_strat="max_div")

Backtesting Framework
---------------------

.. code-block:: python

    # Define strategies to backtest using native weight_strat
    strategies = {
         'Equal Weight': 'eq_weights',
         'Inverse Volatility': 'inv_vol',
         'Max Diversification': 'max_div',
         'Min Vol Overweight': 'min_vol_overweight'
    }

    # Run backtest using native strategies
    backtest_results = {}
    for strategy_name, weight_strat in strategies.items():
         # This may fail with MaxDiversificationNaNError, MaxDiversificationNegativeWeightsError, or other exceptions
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
              'calmar': portfolio.geo_ret / abs(portfolio.max_drawdown) if portfolio.max_drawdown != 0 else float('nan')
         }

    print(f"\n=== BACKTEST RESULTS ===")
    for strategy_name, metrics in backtest_results.items():
        print(f"\n{strategy_name}:")
        print(f"  Return: {metrics['return']:.4f}")
        print(f"  Volatility: {metrics['volatility']:.4f}")
        print(f"  Sharpe: {metrics['sharpe']:.4f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"  Calmar: {metrics['calmar']:.4f}")

    # Rank strategies
    sorted_strategies = sorted(backtest_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    best_strategy = sorted_strategies[0][0]

    print(f"\nBest performing strategy: {best_strategy}")
    print(f"Sharpe ratio: {sorted_strategies[0][1]['sharpe']:.3f}")

Export Optimization Results
---------------------------

.. code-block:: python

    # Export using openseries native methods
    # Export frame data
    investment_universe.to_xlsx('portfolio_optimization_results.xlsx')

    # Note: For comprehensive Excel export with multiple sheets,
    # the DataFrames returned by all_properties() and correl_matrix
    # are pandas DataFrames and support to_excel() method
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
    fund_universe.weights = optimal_portfolio[-fund_universe.item_count:].tolist()
    optimal_portfolio_df = fund_universe.make_portfolio(name="Optimal Portfolio")
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
         # This may fail if the ticker is invalid or data unavailable
         data = yf.Ticker(ticker).history(period="5y")
         series = OpenTimeSeries.from_df(dframe=data['Close'])
         series.set_new_label(lvl_zero=ticker)
         assets.append(series)

    if len(assets) < 2:
         print("Need at least 2 assets for optimization")
    else:
         frame = OpenFrame(constituents=assets)

         # Use openseries native weight strategies
         strategies = {
              'Equal Weight': 'eq_weights',
              'Inverse Volatility': 'inv_vol',
              'Max Diversification': 'max_div',
              'Min Vol Overweight': 'min_vol_overweight'
         }

         # Create portfolios using openseries make_portfolio method
         results = {}
         for name, weight_strat in strategies.items():
              # This may fail with MaxDiversificationNaNError, MaxDiversificationNegativeWeightsError, or other exceptions
              portfolio_df = frame.make_portfolio(name=name, weight_strat=weight_strat)
              portfolio = OpenTimeSeries.from_df(dframe=portfolio_df)
              results[name] = {
                    'Return': portfolio.geo_ret,
                    'Volatility': portfolio.vol,
                    'Sharpe': portfolio.ret_vol_ratio,
                    'Max Drawdown': portfolio.max_drawdown
              }

         print("=== PORTFOLIO OPTIMIZATION RESULTS ===")
         for name, metrics in results.items():
             print(f"\n{name}:")
             print(f"  Return: {metrics['Return']*100:.2f}%")
             print(f"  Volatility: {metrics['Volatility']*100:.2f}%")
             print(f"  Sharpe: {metrics['Sharpe']:.2f}")
             print(f"  Max Drawdown: {metrics['Max Drawdown']*100:.2f}%")
