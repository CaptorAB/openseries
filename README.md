<a href="https://captor.se/"><img src="https://sales.captor.se/captor_logo_sv_1600_icketransparent.png" alt="Captor Fund Management AB" width="81" height="100" align="left" float="right"/></a><br/>

<br><br>

# openseries

[![PyPI version](https://img.shields.io/pypi/v/openseries.svg)](https://pypi.org/project/openseries/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/openseries.svg)](https://anaconda.org/conda-forge/openseries)
![Platform](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-blue)
[![Python version](https://img.shields.io/pypi/pyversions/openseries.svg)](https://www.python.org/)
[![GitHub Action Test Suite](https://github.com/CaptorAB/openseries/actions/workflows/test.yml/badge.svg)](https://github.com/CaptorAB/openseries/actions/workflows/test.yml)
[![codecov](https://img.shields.io/codecov/c/gh/CaptorAB/openseries?logo=codecov)](https://codecov.io/gh/CaptorAB/openseries/branch/master)
![Documentation Status](https://readthedocs.org/projects/openseries/badge/?version=latest)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://beta.ruff.rs/docs/)
[![GitHub License](https://img.shields.io/github/license/CaptorAB/openseries)](https://github.com/CaptorAB/openseries/blob/master/LICENSE.md)
[![Code Sample](https://img.shields.io/badge/-Code%20Sample-blue)](https://nbviewer.org/github/karrmagadgeteer2/NoteBook/blob/master/openseriesnotebook.ipynb)

Tools for analyzing financial timeseries of a single asset or a group of assets. Designed for daily or less frequent data.

## Documentation

Complete documentation is available at: [https://openseries.readthedocs.io](https://openseries.readthedocs.io/)

The documentation includes:

- Quick start guide
- API reference
- Tutorials and examples
- Installation instructions

## Installation

```bash
pip install openseries
```

or:

```bash
conda install -c conda-forge openseries
```

## Quick Start

```python
from openseries import OpenTimeSeries
import yfinance as yf

move=yf.Ticker(ticker="^MOVE")
history=move.history(period="max")
series=OpenTimeSeries.from_df(dframe=history.loc[:, "Close"])
_=series.set_new_label(lvl_zero="ICE BofAML MOVE Index")
_,_=series.plot_series()
```

### Sample output using the report_html() function

<img src="https://raw.githubusercontent.com/CaptorAB/openseries/master/openseries_plot.png" alt="Two Assets Compared" width="1000" />
