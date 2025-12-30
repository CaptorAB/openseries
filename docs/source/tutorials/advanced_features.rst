Advanced Features
=================

This tutorial covers advanced openseries features including custom analysis, integration with other libraries, and extending functionality.

Factor Analysis and Regression
------------------------------

Multi-Factor Model Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import yfinance as yf
    from openseries import OpenTimeSeries, OpenFrame

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
         # This may fail if the ticker is invalid or data unavailable
         data = yf.Ticker(ticker).history(period="3y")
         series = OpenTimeSeries.from_df(dframe=data['Close'])
         series.set_new_label(lvl_zero=name)
         factor_series.append(series)

    # Create factor frame
    factors = OpenFrame(constituents=factor_series)

    # Load individual stock for analysis
    stock_data = yf.Ticker("AAPL").history(period="3y")
    apple = OpenTimeSeries.from_df(dframe=stock_data['Close'])
    apple.set_new_label(lvl_zero="Apple")

    # Add stock to factor frame for regression
    analysis_frame = OpenFrame(constituents=factor_series + [apple])

    # Perform multi-factor regression
    # This may fail with various exceptions
    regression_results = analysis_frame.multi_factor_linear_regression(
         dependent_variable_idx=-1  # Apple is the last series (dependent variable)
    )

    print("\n=== MULTI-FACTOR REGRESSION RESULTS ===")
    print("Regression Summary:")
    print(regression_results['summary'])

    print("\nFactor Loadings (Betas):")
    for i, factor_name in enumerate([s.label for s in factor_series]):
         beta = regression_results['coefficients'][i+1]  # Skip intercept
         print(f"  {factor_name}: {beta:.4f}")

    print(f"\nR-squared: {regression_results['r_squared']:.4f}")
    print(f"Adjusted R-squared: {regression_results['adj_r_squared']:.4f}")

Rolling Factor Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Rolling beta analysis with market
    market_series = factor_series[0]  # S&P 500
    stock_vs_market = OpenFrame(constituents=[apple, market_series])

    # Calculate rolling beta
    rolling_beta = stock_vs_market.rolling_beta(observations=252)  # 1-year rolling

    print(f"\n=== ROLLING BETA ANALYSIS ===")
    print(f"Current Beta: {rolling_beta.iloc[-1, 0]:.3f}")
    print(f"Average Beta: {rolling_beta.mean().iloc[0]:.3f}")
    print(f"Beta Range: {rolling_beta.min().iloc[0]:.3f} to {rolling_beta.max().iloc[0]:.3f}")
    print(f"Beta Volatility: {rolling_beta.std().iloc[0]:.3f}")

    # Rolling correlation
    rolling_corr = stock_vs_market.rolling_corr(observations=252)

    print(f"\n=== ROLLING CORRELATION ANALYSIS ===")
    print(f"Current Correlation: {rolling_corr.iloc[-1, 0]:.3f}")
    print(f"Average Correlation: {rolling_corr.mean().iloc[0]:.3f}")
    print(f"Correlation Range: {rolling_corr.min().iloc[0]:.3f} to {rolling_corr.max().iloc[0]:.3f}")

Exporting Custom Plotly Figures
---------------------------------

The ``export_plotly_figure`` function allows you to export any Plotly figure to a mobile-responsive HTML file. This is useful when you create custom visualizations using Plotly's graph objects that aren't directly available through openseries plotting methods.

Creating Custom Plots
~~~~~~~~~~~~~~~~~~~~~

You can create any Plotly figure and export it using the same responsive HTML format that openseries uses internally:

.. code-block:: python

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from openseries import export_plotly_figure
    from pathlib import Path

    # Create a custom subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Chart', 'Volume', 'Returns Distribution', 'Drawdown'),
        specs=[[{"secondary_y": True}, {"type": "bar"}],
              [{"type": "histogram"}, {"type": "scatter"}]]
    )

    # Add traces (example data)
    fig.add_trace(
        go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], name="Price"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=[1, 2, 3, 4], y=[100, 200, 150, 300], name="Volume"),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=[0.01, -0.02, 0.015, -0.01, 0.02], name="Returns"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=[1, 2, 3, 4], y=[0, -0.05, -0.03, -0.08], name="Drawdown"),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(height=800, title_text="Custom Multi-Panel Dashboard")

    # Export to responsive HTML
    output_path = export_plotly_figure(
        figure=fig,
        fig_config={"responsive": True},
        output_type="file",
        filename="custom_dashboard.html",
        include_plotlyjs="cdn",
        plotfile=Path("output/custom_dashboard.html"),
        title="Custom Financial Dashboard",
        auto_open=True,
    )

    print(f"Dashboard saved to: {output_path}")

Using with Plotly Express
~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use ``export_plotly_figure`` with Plotly Express figures:

.. code-block:: python

    import plotly.express as px
    import pandas as pd
    from openseries import export_plotly_figure
    from pathlib import Path

    # Create sample data
    df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=100),
        'Asset_A': 100 + pd.Series(range(100)).cumsum() * 0.1,
        'Asset_B': 100 + pd.Series(range(100)).cumsum() * 0.15,
    })

    # Create a Plotly Express figure
    fig = px.line(
        df, x='Date', y=['Asset_A', 'Asset_B'],
        title='Asset Comparison',
        labels={'value': 'Price', 'variable': 'Asset'}
    )

    # Export with responsive HTML
    export_plotly_figure(
        figure=fig,
        fig_config={"responsive": True, "displayModeBar": True},
        output_type="file",
        filename="asset_comparison.html",
        include_plotlyjs="cdn",
        plotfile=Path("output/asset_comparison.html"),
        title="Asset Price Comparison",
        auto_open=False,
    )

Inline HTML Output
~~~~~~~~~~~~~~~~~~

For embedding in web applications or reports, you can generate inline HTML divs:

.. code-block:: python

    import plotly.graph_objects as go
    from openseries import export_plotly_figure

    # Create a simple figure
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13]))

    # Generate inline HTML div
    html_div = export_plotly_figure(
        figure=fig,
        fig_config={},
        output_type="div",
        filename="my_plot.html",
        include_plotlyjs="cdn",
        plotfile=Path("dummy.html"),  # Ignored for div output
    )

    # html_div can now be embedded in HTML documents
    print(html_div[:100])  # Preview the HTML

Benefits of export_plotly_figure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``export_plotly_figure`` function provides several advantages over Plotly's default HTML export:

- **Mobile Responsive**: Automatically adapts to different screen sizes and device orientations
- **Optimized Viewport**: Proper viewport settings for mobile devices
- **Auto-Resize**: JavaScript handles window resizing and orientation changes
- **Consistent Styling**: Uses the same responsive CSS as openseries internal plots
- **Optional Title Container**: Can include a title and logo in a responsive header

This makes it ideal for creating dashboards and reports that need to work well on both desktop and mobile devices.


This tutorial demonstrates how to extend openseries with advanced functionality for sophisticated financial analysis workflows.
