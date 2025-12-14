Reporting
=========

This example demonstrates how to create analysis reports using openseries and the built-in report functionality.

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
         series = OpenTimeSeries.from_df(dframe=data['Close'])
         series.set_new_label(lvl_zero=name)
         series_list.append(series)

    # Create frame for report
    comparison_frame = OpenFrame(constituents=series_list)

    # Generate HTML report
    # The last asset in the frame is used as the benchmark
    figure, filepath = report_html(
         data=comparison_frame,
         output_type="file",
         filename="stock_comparison_report.html"
    )

    # filepath contains the path to the saved HTML file
    print(f"Report saved to: {filepath}")

    # The figure object can be used for further customization if needed
    # figure.show()  # Display the figure interactively

Embedding Reports in Existing HTML Pages
-----------------------------------------

When you need to embed a report in an existing HTML page, use ``output_type="div"``:

.. code-block:: python

    # Generate HTML div section for embedding
    figure, html_div = report_html(
         data=comparison_frame,
         output_type="div"
    )

    # html_div contains the responsive HTML div section
    # that can be embedded in your existing HTML page
    # It includes both desktop and mobile layouts with CSS and JavaScript

    # Example: Save to a custom HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>My Custom Report</title>
    </head>
    <body>
        <h1>Portfolio Analysis Report</h1>
        {html_div}
    </body>
    </html>
    """

    with open("custom_report.html", "w", encoding="utf-8") as f:
        f.write(html_template)
