<img src="https://sales.captor.se/captor_logo_sv_1600_icketransparent.png" alt="Captor
Fund Management AB"
width="81" height="100" align="left" float="right"/><br/>

<br><br>

# openseries

[![PyPI version](https://img.shields.io/pypi/v/openseries.svg)](https://pypi.org/project/openseries/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/openseries.svg)](https://anaconda.org/conda-forge/openseries)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/openseries.svg)](https://anaconda.org/conda-forge/openseries)
[![Python version](https://img.shields.io/pypi/pyversions/openseries.svg)](https://www.python.org/)
[![GitHub Action Test Suite](https://github.com/CaptorAB/openseries/actions/workflows/test.yml/badge.svg)](https://github.com/CaptorAB/openseries/actions/workflows/test.yml)
[![Coverage](https://cdn.jsdelivr.net/gh/CaptorAB/openseries@master/coverage.svg)](https://github.com/CaptorAB/openseries/actions/workflows/test.yml)
[![Styling, Linting & Type checks](https://github.com/CaptorAB/openseries/actions/workflows/check.yml/badge.svg)](https://github.com/CaptorAB/openseries/actions/workflows/check.yml)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://beta.ruff.rs/docs/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

This is a project with tools to analyze financial timeseries of a single
asset or a group of assets. It is solely made for daily or less frequent data.

<span style="font-size:2em;">[CHANGELOG](https://github.com/CaptorAB/openseries/blob/master/CHANGELOG.md)</span>


## Basic Usage

To install:

```bash
pip install openseries
```

```bash
conda install -c conda-forge openseries
```

An example of how to make use of the OpenTimeSeries is shown below. The design
aligns with how we within our fund company's code base have a subclass of the
OpenTimeSeries with class methods for our different data sources. Combined with some
additional tools it allows us to efficiently present investment cases to clients.

The code snippet can be pasted into a Python console to run it.
Install openseries and yfinance first.

```python
from openseries import OpenTimeSeries
import yfinance as yf

msft=yf.Ticker("MSFT")
history=msft.history(period="max")
series=OpenTimeSeries.from_df(history.loc[:, "Close"])
_=series.value_to_log().set_new_label("Microsoft Log Returns of Close Prices")
_,_=series.plot_series()

```

### Sample output using the OpenFrame.all_properties() method:
```
                       Scilla Global Equity C (simulation+fund) Global Low Volatility index, SEK
                                                ValueType.PRICE                  ValueType.PRICE
Total return                                           3.641282                         1.946319
Arithmetic return                                      0.096271                         0.069636
Geometric return                                       0.093057                          0.06464
Volatility                                             0.120279                         0.117866
Return vol ratio                                       0.800396                          0.59081
Downside deviation                                     0.085956                         0.086723
Sortino ratio                                          1.119993                         0.802975
Positive share                                         0.541783                         0.551996
Worst                                                 -0.071616                        -0.089415
Worst month                                           -0.122503                        -0.154485
Max drawdown                                          -0.309849                        -0.435444
Max drawdown in cal yr                                -0.309849                        -0.348681
Max drawdown dates                                   2020-03-23                       2009-03-09
CVaR 95.0%                                             -0.01793                        -0.018429
VaR 95.0%                                             -0.011365                        -0.010807
Imp vol from VaR 95%                                   0.109204                         0.103834
Z-score                                                0.587905                         0.103241
Skew                                                  -0.650782                        -0.888109
Kurtosis                                               8.511166                        17.527367
observations                                               4309                             4309
span of days                                               6301                             6301
first indices                                        2006-01-03                       2006-01-03
last indices                                         2023-04-05                       2023-04-05
```

### Usage example on Jupyter Nbviewer

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/karrmagadgeteer2/NoteBook/blob/master/openseriesnotebook.ipynb)


## Development Instructions

These instructions assume that you have a compatible Python version installed on
your machine and that you are OK to install this project in a virtual environment.

The OpenTimeSeries and OpenFrame classes are both subclasses of
the [Pydantic BaseModel](https://docs.pydantic.dev/usage/models/). Please refer to its documentation for information
on any attributes or methods inherited from this model.


### Windows Powershell

```powershell
git clone https://github.com/CaptorAB/openseries.git
cd openseries
./make.ps1 make

```

### Mac Terminal/Linux

```bash
git clone https://github.com/CaptorAB/openseries.git
cd openseries
make
source source_me
make install

```

## Testing and Linting / Type-checking

Ruff and Mypy checking is embedded in the pre-commit hook. Both
are also used in the project's GitHub workflows and are run when the `lint`
alternative is chosen in the below commands.
Any silenced error codes can be found in the
[pyproject.toml](https://github.com/CaptorAB/openseries/blob/master/pyproject.toml)
file or in in-line comments.

### Windows Powershell

```powershell
./make.ps1 test
./make.ps1 lint

```

### Mac Terminal/Linux

```bash
make test
make lint

```


## Table of Contents

- [Basic Usage](#basic-usage)
- [Development Instructions](#development-instructions)
- [Testing and Linting / Type-checking](#testing-and-linting--type-checking)
- [On some files in the project](#on-some-files-in-the-project)
- [Class methods used to construct an OpenTimeSeries](#class-methods-used-to-construct-objects)
- [OpenTimeSeries non-numerical properties](#non-numerical-or-helper-properties-that-apply-only-to-the-opentimeseries-class)
- [OpenFrame non-numerical properties](#non-numerical-or-helper-properties-that-apply-only-to-the-openframe-class)
- [Non-numerical properties for both classes](#non-numerical-or-helper-properties-that-apply-to-both-the-opentimeseries-and-the-openframe-class)
- [OpenTimeSeries only methods](#methods-that-apply-only-to-the-opentimeseries-class)
- [OpenFrame only methods](#methods-that-apply-only-to-the-openframe-class)
- [Methods for both classes](#methods-that-apply-to-both-the-opentimeseries-and-the-openframe-class)
- [Numerical properties for both classes](#numerical-properties-available-for-individual-opentimeseries-or-on-all-series-in-an-openframe)
- [Numerical methods with period arguments for both classes](#methods-below-are-identical-to-the-numerical-properties-above)

### On some files in the project

| File                                                                                                             | Description                                                                                                                                                                                                                               |
|:-----------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [series.py](https://github.com/CaptorAB/openseries/blob/master/openseries/series.py)                             | Defines the class _OpenTimeSeries_ for managing and analyzing a single timeseries. The module also defines a function `timeseries_chain` that can be used to chain two timeseries objects together.                                       |
| [frame.py](https://github.com/CaptorAB/openseries/blob/master/openseries/frame.py)                               | Defines the class _OpenFrame_ for managing a group of timeseries, and e.g. calculate a portfolio timeseries from a rebalancing strategy between timeseries. The module also defines functions to simulate, optimize, and plot portfolios. |
| [simulation.py](https://github.com/CaptorAB/openseries/blob/master/openseries/simulation.py)                     | Defines the class _ReturnSimulation_ to create simulated financial timeseries. Used in the project's test suite                                                                                                                           |

### Class methods used to construct objects.

| Method            | Applies to                    | Description                                                                                        |
|:------------------|:------------------------------|:---------------------------------------------------------------------------------------------------|
| `from_arrays`     | `OpenTimeSeries`              | Class method to create an OpenTimeSeries object from a list of date strings and a list of values.  |
| `from_df`         | `OpenTimeSeries`              | Class method to create an OpenTimeSeries object from a pandas.DataFrame or pandas.Series.          |
| `from_fixed_rate` | `OpenTimeSeries`              | Class method to create an OpenTimeSeries object from a fixed rate, number of days and an end date. |
| `from_deepcopy`   | `OpenTimeSeries`, `OpenFrame` | Creates a copy of an OpenTimeSeries object.                                                        |

### Non-numerical or "helper" properties that apply only to the [OpenTimeSeries](https://github.com/CaptorAB/openseries/blob/master/openseries/series.py) class.

| Property        | type            | Applies to       | Description                                                                                                                                  |
|:----------------|:----------------|:-----------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| `timeseries_id` | `str`           | `OpenTimeSeries` | Placeholder for database identifier for the timeseries. Can be left as empty string.                                                         |
| `instrument_id` | `str`           | `OpenTimeSeries` | Placeholder for database identifier for the instrument associated with the timeseries. Can be left as empty string.                          |
| `dates`         | `list[str]`     | `OpenTimeSeries` | Dates of the timeseries. Not edited by any method to allow reversion to original.                                                            |
| `values`        | `list[float]`   | `OpenTimeSeries` | Values of the timeseries. Not edited by any method to allow reversion to original.                                                           |
| `currency`      | `str`           | `OpenTimeSeries` | Currency of the timeseries. Only used if conversion/hedging methods are added.                                                               |
| `domestic`      | `str`           | `OpenTimeSeries` | Domestic currency of the user / investor. Only used if conversion/hedging methods are added.                                                 |
| `local_ccy`     | `bool`          | `OpenTimeSeries` | Indicates if series should be in its local currency or the domestic currency of the user. Only used if conversion/hedging methods are added. |
| `name`          | `str`           | `OpenTimeSeries` | An identifier field.                                                                                                                         |
| `isin`          | `str`           | `OpenTimeSeries` | ISIN code of the associated instrument. If any.                                                                                              |
| `label`         | `str`           | `OpenTimeSeries` | Field used in outputs. Derived from name as default.                                                                                         |
| `countries`     | `list` or `str` | `OpenTimeSeries` | (List of) country code(s) according to ISO 3166-1 alpha-2 used to generate business days.                                                    |
| `valuetype`     | `ValueType`     | `OpenTimeSeries` | Field identifies the type of values in the series. ValueType is an Enum.                                                                     |

### Non-numerical or "helper" properties that apply only to the [OpenFrame](https://github.com/CaptorAB/openseries/blob/master/openseries/frame.py) class.

| Property           | type                   | Applies to  | Description                                                              |
|:-------------------|:-----------------------|:------------|:-------------------------------------------------------------------------|
| `constituents`     | `list[OpenTimeSeries]` | `OpenFrame` | A list of the OpenTimeSeries that make up an OpenFrame.                  |
| `columns_lvl_zero` | `list`                 | `OpenFrame` | A list of the level zero column names in the OpenFrame pandas.DataFrame. |
| `columns_lvl_one`  | `list`                 | `OpenFrame` | A list of the level one column names in the OpenFrame pandas.DataFrame.  |
| `item_count`       | `int`                  | `OpenFrame` | Number of columns in the OpenFrame pandas.DataFrame.                     |
| `weights`          | `list[float]`          | `OpenFrame` | Weights used in the method `make_portfolio`.                             |
| `first_indices`    | `pandas.Series`        | `OpenFrame` | First dates of all the series in the OpenFrame.                          |
| `last_indices`     | `pandas.Series`        | `OpenFrame` | Last dates of all the series in the OpenFrame.                           |
| `lengths_of_items` | `pandas.Series`        | `OpenFrame` | Number of items in each of the series in the OpenFrame.                  |
| `span_of_days_all` | `pandas.Series`        | `OpenFrame` | Number of days from the first to the last in each of the series.         |

### Non-numerical or "helper" properties that apply to both the [OpenTimeSeries](https://github.com/CaptorAB/openseries/blob/master/openseries/series.py) and the [OpenFrame](https://github.com/CaptorAB/openseries/blob/master/openseries/frame.py) class.

| Property            | type                             | Applies to                    | Description                                                                       |
|:--------------------|:---------------------------------|:------------------------------|:----------------------------------------------------------------------------------|
| `first_idx`         | `datetime.date`                  | `OpenTimeSeries`, `OpenFrame` | First date of the series.                                                         |
| `last_idx`          | `datetime.date`                  | `OpenTimeSeries`, `OpenFrame` | Last date of the series.                                                          |
| `length`            | `int`                            | `OpenTimeSeries`, `OpenFrame` | Number of items in the series.                                                    |
| `span_of_days`      | `int`                            | `OpenTimeSeries`, `OpenFrame` | Number of days from the first to the last date in the series.                     |
| `tsdf`              | `pandas.DataFrame`               | `OpenTimeSeries`, `OpenFrame` | The Pandas DataFrame which gets edited by the class methods.                      |
| `max_drawdown_date` | `datetime.date`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Date when the maximum drawdown occurred.                                          |
| `periods_in_a_year` | `float`                          | `OpenTimeSeries`, `OpenFrame` | The number of observations in an average year for all days in the data.           |
| `yearfrac`          | `float`                          | `OpenTimeSeries`, `OpenFrame` | Length of timeseries expressed as np.float64 fraction of a year with 365.25 days. |

### Methods that apply only to the [OpenTimeSeries](https://github.com/CaptorAB/openseries/blob/master/openseries/series.py) class.

| Method                   | Applies to       | Description                                                                                                                                    |
|:-------------------------|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| `pandas_df`              | `OpenTimeSeries` | Method to create the `tsdf` pandas.DataFrame from the `dates` and `values`.                                                                    |
| `set_new_label`          | `OpenTimeSeries` | Method to change the pandas.DataFrame column MultiIndex.                                                                                       |
| `running_adjustment`     | `OpenTimeSeries` | Adjusts the series performance with a `float` factor.                                                                                          |
| `ewma_vol_func`          | `OpenTimeSeries` | Returns a `pandas.Series` with volatility based on [Exponentially Weighted Moving Average](https://www.investopedia.com/articles/07/ewma.asp). |
| `from_1d_rate_to_cumret` | `OpenTimeSeries` | Converts a series of 1-day rates into a cumulative valueseries.                                                                                |
                                                                           |

### Methods that apply only to the [OpenFrame](https://github.com/CaptorAB/openseries/blob/master/openseries/frame.py) class.

| Method                  | Applies to  | Description                                                                                                                                                                        |
|:------------------------|:------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `merge_series`          | `OpenFrame` | Merges the Pandas Dataframes of the constituent OpenTimeSeries.                                                                                                                    |
| `trunc_frame`           | `OpenFrame` | Truncates the OpenFrame to a common period.                                                                                                                                        |
| `add_timeseries`        | `OpenFrame` | Adds a given OpenTimeSeries to the OpenFrame.                                                                                                                                      |
| `delete_timeseries`     | `OpenFrame` | Deletes an OpenTimeSeries from the OpenFrame.                                                                                                                                      |
| `relative`              | `OpenFrame` | Calculates a new series that is the relative performance of two others.                                                                                                            |
| `make_portfolio`        | `OpenFrame` | Calculates a portfolio timeseries based on the series and weights. Weights can be provided as a list, or a weight strategy can be set as *equal weights* or *inverted volatility*. |
| `ord_least_squares_fit` | `OpenFrame` | Performs a regression and an [Ordinary Least Squares](https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html) fit.                                               |
| `beta`                  | `OpenFrame` | Calculates [Beta](https://www.investopedia.com/terms/b/beta.asp) of an asset relative a market.                                                                                    |
| `jensen_alpha`          | `OpenFrame` | Calculates [Jensen's Alpha](https://www.investopedia.com/terms/j/jensensmeasure.asp) of an asset relative a market.                                                                |
| `tracking_error_func`   | `OpenFrame` | Calculates the [tracking errors](https://www.investopedia.com/terms/t/trackingerror.asp) relative to a selected series in the OpenFrame.                                           |
| `info_ratio_func`       | `OpenFrame` | Calculates the [information ratios](https://www.investopedia.com/terms/i/informationratio.asp) relative to a selected series in the OpenFrame.                                     |
| `capture_ratio_func`    | `OpenFrame` | Calculates up, down and up/down [capture ratios](https://www.investopedia.com/terms/d/down-market-capture-ratio.asp) relative to a selected series.                                |
| `rolling_info_ratio`    | `OpenFrame` | Returns a pandas.DataFrame with the rolling [information ratio](https://www.investopedia.com/terms/i/informationratio.asp) between two series.                                     |
| `rolling_beta`          | `OpenFrame` | Returns a pandas.DataFrame with the rolling [Beta](https://www.investopedia.com/terms/b/beta.asp) of an asset relative a market.                                                   |
| `rolling_corr`          | `OpenFrame` | Calculates and adds a series of rolling [correlations](https://www.investopedia.com/terms/c/correlation.asp) between two other series.                                             |
| `correl_matrix`         | `OpenFrame` | Returns a `pandas.DataFrame` with a correlation matrix.                                                                                                                            |
| `ewma_risk`             | `OpenFrame` | Returns a `pandas.DataFrame` with volatility and correlation based on [Exponentially Weighted Moving Average](https://www.investopedia.com/articles/07/ewma.asp).                  |

### Methods that apply to both the [OpenTimeSeries](https://github.com/CaptorAB/openseries/blob/master/openseries/series.py) and the [OpenFrame](https://github.com/CaptorAB/openseries/blob/master/openseries/frame.py) class.

| Method                             | Applies to                    | Description                                                                                                                                              |
|:-----------------------------------|:------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `align_index_to_local_cdays`       | `OpenTimeSeries`, `OpenFrame` | Aligns the series dates to a business calendar. Defaults to Sweden.                                                                                      |
| `resample`                         | `OpenTimeSeries`, `OpenFrame` | Resamples the series to a specific frequency.                                                                                                            |
| `resample_to_business_period_ends` | `OpenTimeSeries`, `OpenFrame` | Resamples the series to month-end dates with monthly, quarterly or annual frequency.                                                                     |
| `value_nan_handle`                 | `OpenTimeSeries`, `OpenFrame` | Fills `Nan` in a value series with the preceding non-Nan value.                                                                                          |
| `return_nan_handle`                | `OpenTimeSeries`, `OpenFrame` | Replaces `Nan` in a return series with a 0.0 `float`.                                                                                                    |
| `to_cumret`                        | `OpenTimeSeries`, `OpenFrame` | Converts a return series into a value series and/or resets a value series to be rebased from 1.0.                                                        |
| `to_json`                          | `OpenTimeSeries`, `OpenFrame` | Method to export object data to a json file.                                                                                                             |
| `to_xlsx`                          | `OpenTimeSeries`, `OpenFrame` | Method to save the data in the .tsdf DataFrame to an Excel file.                                                                                         |
| `value_to_ret`                     | `OpenTimeSeries`, `OpenFrame` | Converts a value series into a percentage return series.                                                                                                 |
| `value_to_diff`                    | `OpenTimeSeries`, `OpenFrame` | Converts a value series into a series of differences.                                                                                                    |
| `value_to_log`                     | `OpenTimeSeries`, `OpenFrame` | Converts a value series into a logarithmic return series.                                                                                                |
| `value_ret_calendar_period`        | `OpenTimeSeries`, `OpenFrame` | Returns the series simple return for a specific calendar period.                                                                                         |
| `plot_series`                      | `OpenTimeSeries`, `OpenFrame` | Opens a HTML [Plotly Scatter](https://plotly.com/python/line-and-scatter/) plot of the series in a browser window.                                       |
| `plot_bars`                        | `OpenTimeSeries`, `OpenFrame` | Opens a HTML [Plotly Bar](https://plotly.com/python/bar-charts/) plot of the series in a browser window.                                                 |
| `to_drawdown_series`               | `OpenTimeSeries`, `OpenFrame` | Converts the series into drawdown series.                                                                                                                |
| `rolling_return`                   | `OpenTimeSeries`, `OpenFrame` | Returns a pandas.DataFrame with rolling returns.                                                                                                         |
| `rolling_vol`                      | `OpenTimeSeries`, `OpenFrame` | Returns a pandas.DataFrame with rolling volatilities.                                                                                                    |
| `rolling_var_down`                 | `OpenTimeSeries`, `OpenFrame` | Returns a pandas.DataFrame with rolling VaR figures.                                                                                                     |
| `rolling_cvar_down`                | `OpenTimeSeries`, `OpenFrame` | Returns a pandas.DataFrame with rolling CVaR figures.                                                                                                    |
| `calc_range`                       | `OpenTimeSeries`, `OpenFrame` | Returns the start and end dates of a range from specific period definitions. Used by the below numerical methods and not meant to be used independently. |

### Numerical properties available for individual [OpenTimeSeries](https://github.com/CaptorAB/openseries/blob/master/openseries/series.py) or on all series in an [OpenFrame](https://github.com/CaptorAB/openseries/blob/master/openseries/frame.py).

| Property                | type                     | Applies to                    | Description                                                                                                                                                                                                             |
|:------------------------|:-------------------------|:------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `all_properties`        | `pandas.DataFrame`       | `OpenTimeSeries`, `OpenFrame` | Returns most of the properties in one go.                                                                                                                                                                               |
| `arithmetic_ret`        | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Annualized arithmetic mean of returns](https://www.investopedia.com/terms/a/arithmeticmean.asp).                                                                                                                       |
| `geo_ret`               | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Compound Annual Growth Rate(CAGR)](https://www.investopedia.com/terms/c/cagr.asp), a specific implementation of geometric mean.                                                                                        |
| `value_ret`             | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Simple return from first to last observation.                                                                                                                                                                           |
| `vol`                   | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Annualized [volatility](https://www.investopedia.com/terms/v/volatility.asp). Pandas .std() is the equivalent of stdev.s([...]) in MS excel.                                                                            |
| `downside_deviation`    | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Downside deviation](https://www.investopedia.com/terms/d/downside-deviation.asp) is the volatility of all negative return observations. Minimum Accepted Return (MAR) set to zero.                                     |
| `ret_vol_ratio`         | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Ratio of arithmetic mean return and annualized volatility. It is the [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp) with the riskfree rate set to zero.                                           |
| `sortino_ratio`         | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The [Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp) is the arithmetic mean return divided by the downside deviation. This attribute assumes that the riskfree rate and the MAR are both zero.    |
| `omega_ratio`           | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The [Omega Ratio](https://en.wikipedia.org/wiki/Omega_ratio) compares returns above a certain target level (MAR) to the total downside risk below MAR.                                                                  |
| `var_down`              | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Downside 95% [Value At Risk](https://www.investopedia.com/terms/v/var.asp), "VaR". The equivalent of percentile.inc([...], 1-level) over returns in MS Excel. For other confidence levels use the corresponding method. |
| `cvar_down`             | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Downside 95% [Conditional Value At Risk](https://www.investopedia.com/terms/c/conditional_value_at_risk.asp), "CVaR". For other confidence levels use the corresponding method.                                         |
| `worst`                 | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Most negative percentage change of a single observation.                                                                                                                                                                |
| `worst_month`           | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Most negative month.                                                                                                                                                                                                    |
| `max_drawdown`          | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Maximum drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp).                                                                                                                                      |
| `max_drawdown_cal_year` | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Max drawdown in a single calendar year.                                                                                                                                                                                 |
| `positive_share`        | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The share of percentage changes that are positive.                                                                                                                                                                      |
| `vol_from_var`          | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Implied annualized volatility from the Downside VaR using the assumption that returns are normally distributed.                                                                                                         |
| `skew`                  | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Skew](https://www.investopedia.com/terms/s/skewness.asp) of the return distribution.                                                                                                                                   |
| `kurtosis`              | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Kurtosis](https://www.investopedia.com/terms/k/kurtosis.asp) of the return distribution.                                                                                                                               |
| `z_score`               | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Z-score](https://www.investopedia.com/terms/z/zscore.asp) as (last return - mean return) / standard deviation of returns.                                                                                              |

### Methods below are identical to the Numerical Properties above.

_They are simply methods that take different date or length inputs to return the
properties for subset periods._

| Method                    | type                     | Applies to                    | Description                                                                                                                                                                                                                                                    |
|:--------------------------|:-------------------------|:------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `arithmetic_ret_func`     | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Annualized arithmetic mean of returns](https://www.investopedia.com/terms/a/arithmeticmean.asp).                                                                                                                                                              |
| `geo_ret_func`            | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Compound Annual Growth Rate(CAGR)](https://www.investopedia.com/terms/c/cagr.asp), a specific implementation of geometric mean.                                                                                                                               |
| `value_ret_func`          | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Simple return from first to last observation.                                                                                                                                                                                                                  |
| `vol_func`                | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Annualized [volatility](https://www.investopedia.com/terms/v/volatility.asp). Pandas .std() is the equivalent of stdev.s([...]) in MS excel.                                                                                                                   |
| `downside_deviation_func` | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Downside deviation](https://www.investopedia.com/terms/d/downside-deviation.asp) is the volatility of all negative return observations. MAR and riskfree rate can be set.                                                                                     |
| `ret_vol_ratio_func`      | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Ratio of arithmetic mean return and annualized volatility. It is the [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp) with the riskfree rate set to zero. A riskfree rate can be set as a float or a series chosen for the frame function. |
| `sortino_ratio_func`      | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The [Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp) is the arithmetic mean return divided by the downside deviation. A riskfree rate can be set as a float or a series chosen for the frame function. MAR is set to zero.               |
| `omega_ratio_func`        | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The [Omega Ratio](https://en.wikipedia.org/wiki/Omega_ratio) compares returns above a certain target level (MAR) to the total downside risk below MAR.                                                                                                         |
| `var_down_func`           | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Downside 95% [Value At Risk](https://www.investopedia.com/terms/v/var.asp), "VaR". The equivalent of percentile.inc([...], 1-level) over returns in MS Excel. Default is 95% confidence level.                                                                 |
| `cvar_down_func`          | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Downside 95% [Conditional Value At Risk](https://www.investopedia.com/terms/c/conditional_value_at_risk.asp), "CVaR". Default is 95% confidence level.                                                                                                         |
| `worst_func`              | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Most negative percentage change for a given number of observations (default=1).                                                                                                                                                                                |
| `max_drawdown_func`       | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Maximum drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp).                                                                                                                                                                             |
| `positive_share_func`     | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The share of percentage changes that are positive.                                                                                                                                                                                                             |
| `vol_from_var_func`       | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Implied annualized volatility from the Downside VaR using the assumption that returns are normally distributed.                                                                                                                                                |
| `skew_func`               | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Skew](https://www.investopedia.com/terms/s/skewness.asp) of the return distribution.                                                                                                                                                                          |
| `kurtosis_func`           | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Kurtosis](https://www.investopedia.com/terms/k/kurtosis.asp) of the return distribution.                                                                                                                                                                      |
| `z_score_func`            | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Z-score](https://www.investopedia.com/terms/z/zscore.asp) as (last return - mean return) / standard deviation of returns.                                                                                                                                     |
| `target_weight_from_var`  | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | A position target weight from the ratio between a VaR implied volatility and a given target volatility.                                                                                                                                                        |
