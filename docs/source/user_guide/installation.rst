Installation
============

System Requirements
-------------------

openseries requires Python 3.10 or higher and is compatible with:

- **Operating Systems**: Windows, macOS, Linux
- **Python versions**: 3.10, 3.11, 3.12, 3.13

Installing openseries
---------------------

Using pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install openseries is using pip:

.. code-block:: bash

   pip install openseries

Using conda
~~~~~~~~~~~

openseries is also available on conda-forge:

.. code-block:: bash

   conda install -c conda-forge openseries

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/CaptorAB/openseries.git
   cd openseries
   pip install -e .

Dependencies
------------

openseries automatically installs the following dependencies:

Core Dependencies
~~~~~~~~~~~~~~~~~

- **pandas** (>=2.1.2,<3.0.0) - Data manipulation and analysis
- **numpy** (>=1.23.2,!=2.3.0,<3.0.0) - Numerical computing
- **pydantic** (>=2.5.2,<3.0.0) - Data validation and settings management
- **plotly** (>=5.18.0,<7.0.0) - Interactive plotting
- **scipy** (>=1.11.4,<2.0.0) - Scientific computing
- **scikit-learn** (>=1.4.0,<2.0.0) - Machine learning utilities

Financial and Date Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **exchange-calendars** (>=4.8,<6.0) - Trading calendar support
- **holidays** (>=0.30,<1.0) - Holiday calendar support
- **python-dateutil** (>=2.8.2,<4.0.0) - Date parsing utilities

File and Network Support
~~~~~~~~~~~~~~~~~~~~~~~~~

- **openpyxl** (>=3.1.2,<5.0.0) - Excel file support
- **requests** (>=2.20.0,<3.0.0) - HTTP library

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For data acquisition examples, you may want to install:

.. code-block:: bash

   pip install yfinance  # For Yahoo Finance data

Verifying Installation
----------------------

To verify that openseries is installed correctly, run:

.. code-block:: python

   import openseries
   print(openseries.__version__)

You can also run a quick test:

.. code-block:: python

   from openseries import OpenTimeSeries, ReturnSimulation, ValueType
   import datetime as dt

   # Create sample data using openseries simulation
   simulation = ReturnSimulation.from_lognormal(
       number_of_sims=1,
       trading_days=100,
       mean_annual_return=0.25,  # ~0.001 daily
       mean_annual_vol=0.32,     # ~0.02 daily
       trading_days_in_year=252,
       seed=42
   )

   # Create OpenTimeSeries
   series = OpenTimeSeries.from_df(
       dframe=simulation.to_dataframe(name="Test Series", end=dt.date(2023, 12, 31)),
       valuetype=ValueType.RTRN
   ).to_cumret()  # Convert returns to cumulative prices

   print(f"Series length: {series.length}")
   print(f"Annual return: {series.geo_ret:.2%}")

Development Installation
------------------------

If you plan to contribute to openseries or need the development dependencies:

.. code-block:: bash

   git clone https://github.com/CaptorAB/openseries.git
   cd openseries

   # Install Poetry (if not already installed)
   pip install poetry

   # Install dependencies
   poetry install

   # Activate virtual environment
   poetry shell

This will install additional development dependencies including:

- **pytest** - Testing framework
- **mypy** - Static type checking
- **ruff** - Linting and formatting
- **pre-commit** - Git hooks for code quality

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'openseries'**

Make sure openseries is installed in the correct Python environment. If using virtual environments, ensure it's activated.

**Version conflicts**

If you encounter dependency conflicts, try creating a fresh virtual environment:

.. code-block:: bash

   python -m venv openseries_env
   source openseries_env/bin/activate  # On Windows: openseries_env\Scripts\activate
   pip install openseries

**Performance issues**

For better performance with large datasets, consider installing optional accelerated packages:

.. code-block:: bash

   pip install numba  # For numerical acceleration
   pip install bottleneck  # For faster pandas operations

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/CaptorAB/openseries/issues>`_
2. Review the `Release Notes <https://github.com/CaptorAB/openseries/releases>`_
3. Create a new issue with a minimal reproducible example

Platform-Specific Notes
------------------------

Windows
~~~~~~~

On Windows, you may need to install Microsoft Visual C++ Build Tools if you encounter compilation errors with dependencies.

macOS
~~~~~

On macOS with Apple Silicon (M1/M2), all dependencies should install without issues. If you encounter problems, try using conda instead of pip.

Linux
~~~~~

Most Linux distributions should work without issues. On minimal installations, you may need to install additional system packages for some dependencies.
