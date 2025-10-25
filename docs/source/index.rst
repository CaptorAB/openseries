openseries Documentation
========================

.. image:: https://img.shields.io/pypi/v/openseries.svg
   :target: https://pypi.org/project/openseries/
   :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/openseries.svg
   :target: https://anaconda.org/conda-forge/openseries
   :alt: Conda Version

.. image:: https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-blue
   :alt: Platform

.. image:: https://img.shields.io/pypi/pyversions/openseries.svg
   :target: https://www.python.org/
   :alt: Python version

.. image:: https://github.com/CaptorAB/openseries/actions/workflows/test.yml/badge.svg
   :target: https://github.com/CaptorAB/openseries/actions/workflows/test.yml
   :alt: GitHub Action Test Suite

.. image:: https://img.shields.io/codecov/c/gh/CaptorAB/openseries?logo=codecov
   :target: https://codecov.io/gh/CaptorAB/openseries/branch/master
   :alt: codecov

.. image:: https://img.shields.io/github/license/CaptorAB/openseries
   :target: https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
   :alt: GitHub License

**openseries** is a Python library for analyzing financial time series data. It provides tools to work with single assets or groups of assets, designed specifically for daily or less frequent data.

The library is built around two main classes:

- **OpenTimeSeries**: For managing and analyzing individual time series
- **OpenFrame**: For managing groups of time series and portfolio analysis

Key Features
------------

- **Financial Analysis**: Comprehensive set of financial metrics and ratios
- **Risk Management**: VaR, CVaR, drawdown analysis, and risk-adjusted returns
- **Portfolio Tools**: Portfolio optimization, rebalancing, and performance attribution
- **Visualization**: Interactive plots using Plotly
- **Data Handling**: Robust date handling and business day calendars
- **Type Safety**: Built with Pydantic for data validation and type safety

Quick Start
-----------

Install openseries using pip:

.. code-block:: bash

   pip install openseries

Or using conda:

.. code-block:: bash

   conda install -c conda-forge openseries

Here's a simple example to get you started:

.. code-block:: python

   from openseries import OpenTimeSeries
   import yfinance as yf

   # Download data
   ticker = yf.Ticker("^GSPC")
   history = ticker.history(period="5y")

   # Create OpenTimeSeries
   series = OpenTimeSeries.from_df(dframe=history.loc[:, "Close"])
   series.set_new_label(lvl_zero="S&P 500")

   # Calculate key metrics
   print(f"Annual Return: {series.geo_ret:.2%}")
   print(f"Volatility: {series.vol:.2%}")
   print(f"Sharpe Ratio: {series.ret_vol_ratio:.2f}")
   print(f"Max Drawdown: {series.max_drawdown:.2%}")

   # Create interactive plot
   series.plot_series()

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/core_concepts
   user_guide/data_handling

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_analysis
   tutorials/portfolio_analysis
   tutorials/risk_management
   tutorials/advanced_features

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/single_asset
   examples/multi_asset
   examples/portfolio_optimization
   examples/rebalanced_portfolio
   examples/custom_reports

.. toctree::
   :maxdepth: 1
   :caption: Important Notes

   api_consistency

Python Version Support
----------------------

.. warning::
   **Python 3.10 Deprecation Notice**: Python 3.10 support is deprecated and will be removed, no earlier than 2025-12-01. Please upgrade to Python ≥3.11. See `GitHub Issue #340 <https://github.com/CaptorAB/openseries/issues/340>`_ (pinned) for details.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/openseries
   api/series
   api/frame
   api/portfoliotools
   api/simulation
   api/report
   api/datefixer
   api/types

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
