# Changelog

For a long time I have not kept a log of the changes implemented in the different
versions of the openseries package. In
this file I am attempting to rectify this somewhat. However, unfortunately I do not have
the resources to issue any form
of guarantee that this log will cover all changes, and I will not attempt to go back
very far in history.

## Version [0.11.3] - 2023-03-18

Moved all defined types into own module and replaced TypedDict with Pydantic
BaseModel throughout.

## Version [0.11.2] - 2023-03-16

Mostly tooling cleanup making.

## Version [0.11.0] - 2023-03-01

Silenced float('nan') and numpy.float64('nan) differences when validating values
input against tsdf.values.

## Version [0.10.8] - 2023-02-28

Using [Pydantic](https://docs.pydantic.dev/) for data validation using Python type
annotations.

## Version [0.10.5] - 2023-02-19

Using [Poetry](https://python-poetry.org/) to build and deploy.

## Version [0.10.3] - 2023-02-12

Added an offset_business_days function in the datefixer module. It bumps/offsets with
a given number of business days, not calendar days. Also added the ability to add
custom holidays in the relevant datefixer functions.

## Version [0.10.2] - 2023-02-11

Added ability to use multiple countries calendars in combination.

## Version [0.10.1] - 2023-02-06

Clean-up of variable names, annotations and type hints with test of their alignment.

## Version [0.9.9] - 2023-02-05

Added strip_duplicates() function in series module used with new remove_duplicates
option on the OpenTimeSeries class. Also added resample_to_business_period_ends()
method on both OpenTimeSeries and OpenFrame. Finalizes issue #23.

## Version [0.9.8] - 2023-02-02

Improved OpenTimeSeries.from_fixed_rate() method, added exceptions module to provide
better user feedback and added typing to output from class methods.

## Version [0.9.6] - 2023-01-29

Added typing for self.

## Version [0.9.5] - 2023-01-28

Added a plot_bars method to both OpenTimeSeries and OpenFrame classes.

## Version [0.9.4] - 2023-01-28

Reverted Literal typing for offset argument in .resample() methods. Fixed merge_series
method on OpenFrame so that constituent DataFrames are aligned on inner merge.

## Version [0.9.3] - 2023-01-25

Renamed Numpy.busdaycalendar attribute on OpenTimeSeries class from `sweden` to
`calendar` and corrected type description in README.md. Aiming to make package
country and currency agnostic.

## Version [0.9.1] - 2023-01-12

Minimized imports

## Version [0.9.0] - 2023-01-04

Removed all connections to outside API.

## Version [0.8.4] - 2022-12-20

Changed Jensen's alpha to be based on asset and market CAGR, instead of cumulative log
return.

## Version [0.8.2] - 2022-12-19

Added an OpenTimeSeries method that converts a series of 1-day rates into a cumulative
valueseries, and added an
OpenFrame method to calculate the Jensen's alpha of an asset relative a market.

## Version [0.8.1] - 2022-11-25

Typed all Pandas.Series outputs from OpenFrame methods.

## Version [0.8.0] - 2022-11-20

Updated dependencies.

## Version [0.7.9] - 2022-10-30

Sped up holidays check, added merge_series method on OpenFrame, and added tests on
failing trunc_frame scenarios.

## Version [0.7.7] - 2022-10-24

Updated all project dependencies.

## Version [0.7.6] - 22-10-18

Updated to remove deprecation warnings related to holidays==0.16.

## Version [0.7.5] - 22-10-12

Removed sweden_holidays module and replaced it with Numpy busdaycalendar. Also added
Literal type interpolation input
for VaR related functions.

## Version [0.7.4] - 22-09-17

Opened up the output from OpenFrame.ord_least_squares_fit()

## Version [0.7.3] - 2022-09-10

Removed pandas.MultiIndex.from_product() as constructor for columns to fix hard to
replicate test failure.

## Version [0.7.2] - 2022-09-05

Added EWMA volatility and correlation.

## Version [0.7.1] - 2022-08-31

Improved OpenFrame.trunc_frame().

## Version [0.6.9] - 2022-08-30

Fixed timeseries_chain to better capture date mismatches.

## Version [0.6.8] - 2022-08-30

Added missing name of last highlighted point in plot series.

## Version [0.6.7] - 2022-08-13

Added new .png for Captor logo and resized and realigned it for plotly plots.

## Version [0.6.5] - 2022-08-08

Cleaned up setup.py and moved all build configurations to pyproject.toml.

## Version [0.6.0] - 2022-08-07

Cleaned up docstrings and removed unnecessary functions in risk.py.

## Version [0.5.9] - 2022-07-31

Reorganized and cleaned up test folder to make test rationale easier to understand.

## Version [0.5.8] - 2022-07-27

Improved test coverage to 99%.

## Version [0.5.7] - 2022-07-24

Fixed rolling correlation, added beta attribute and rolling beta for OpenFrame and
associated tests.

## Version [0.5.5] - 2022-07-17

Removed log returns everywhere. Removed keyvaluetable and reduced use of date_fix
function. Improved test coverage
further and will leave at this level for now.

## Version [0.5.2] - 2022-07-15

Fixed so that ratios based on geometric returns will use arithmetic return instead to
avoid some failures. The geo_ret
functions will now raise and exception on initial zeroes and on negative values.
Improved test coverage and also added
missing PEP604 type hints.

## Version [0.5.0] - 2022-07-12

This version can only run on Python version 3.10 due to the implementation of type hints
following [PEP 604](https://peps.python.org/pep-0604/).

## Version [0.4.1] - 2022-07-12

This version is backwards compatible only from Python version 3.8 due to the
implementation
of [PEP 589](https://peps.python.org/pep-0589/).

## Version [0.4.0] - 2022-06-30

This version is backwards compatible to Python version 3.6 and works up to version 3.10,
docstrings have been improved
and deprecation warnings fixed.

## Version [0.3.8] - 2022-05-30

This is the first draft version to work with Python version 3.10. It runs but with
several deprecation warnings
primarily from Pandas. Prior to this version openseries was not compatible with Python
version 3.10.
