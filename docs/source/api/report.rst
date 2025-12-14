Report Generation
=================

.. currentmodule:: openseries.report

HTML Report Function
--------------------

.. autofunction:: report_html
   :no-index:

The ``report_html`` function creates comprehensive HTML reports for financial analysis, comparing multiple assets and providing detailed performance metrics, charts, and risk analysis.

The generated report includes:

- **Interactive line charts** showing cumulative returns over time for all assets
- **Bar charts** displaying period returns (annual, quarterly, or monthly depending on data length)
- **Performance metrics table** including:
  - Return metrics (CAGR or simple return, Year-to-Date, Month-to-Date)
  - Risk metrics (Volatility, Sharpe Ratio, Sortino Ratio)
  - Relative performance metrics (Jensen's Alpha, Information Ratio, Tracking Error, Index Beta)
  - Capture Ratio (for periods longer than one year)
  - Worst period returns
  - Comparison period dates

**Important Notes:**

- The last asset in the ``OpenFrame`` is used as the benchmark for relative performance metrics
  (Jensen's Alpha, Information Ratio, Tracking Error, Index Beta, and Capture Ratio)
- For periods shorter than one year, the report uses simple returns instead of CAGR
- For periods shorter than a quarter, bar charts show daily returns instead of period returns
- Capture Ratio is only included for periods longer than one year

Responsive Design
-----------------

The report features **responsive design** with separate layouts optimized for desktop and mobile devices:

- **Desktop layout**: Charts and tables are displayed side-by-side in a 2x2 grid layout. The table
  is rendered as an interactive Plotly table integrated with the charts.

- **Mobile layout**: Content is stacked vertically for better viewing on smaller screens. The table
  is rendered as a standard HTML table below the charts for better mobile compatibility.

The HTML output automatically adapts to screen size and device capabilities using CSS media queries
and JavaScript detection. The layout switches at a breakpoint of 960px width or when touch capabilities
are detected.

Return Values
-------------

The function returns a tuple containing:

- **Plotly Figure**: The desktop version of the figure object (can be used for further customization
  or interactive display)

- **String output**: The type depends on the ``output_type`` parameter:

  - When ``output_type="file"`` (default): Returns the file path string to the saved HTML file.
    The file contains a complete HTML document (with DOCTYPE, html, head, and body tags)
    that can be opened directly in a web browser. If ``auto_open=True``, the file will
    automatically open in the default web browser.

  - When ``output_type="div"``: Returns a string containing the responsive HTML div section
    (includes both desktop and mobile layouts with CSS and JavaScript) that can be embedded
    in an existing HTML page. The ``filename`` parameter is optional when using this mode and
    is only used to generate unique div IDs for the embedded content.
