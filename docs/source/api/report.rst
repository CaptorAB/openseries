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

The report features **responsive design** with separate layouts optimized for desktop and mobile devices. On desktop, charts and tables are displayed side-by-side, while mobile layouts stack content vertically for better viewing on smaller screens. The HTML output automatically adapts to screen size and device capabilities.
