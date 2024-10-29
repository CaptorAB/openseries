# Changelog

## Version [1.7.6] - 2024-10-29

- Changed worst_month() to be consistent with resample_to_business_period_ends().
- Updated Poetry to version 1.8.4.
- Miscellaneous dependency and lockfile updates.

## Version [1.7.5] - 2024-10-13

- Added pandas .ffill() as precursor to all remaining pct_change() to suppress pandas FutureWarnings in dependent projects.

## Version [1.7.4] - 2024-10-06

- Tightened existing checks to not allow mixed series types as methods input. The coverage level of the checks has not been widened.
- Added pandas .ffill() as precursor to pct_change() to suppress pandas FutureWarnings in dependent projects.
- Fixed method .resample_to_business_period_ends() so it considers renamed labels.
- Corrected warning in this changelog for release of version 1.7.0. Added 'NO'.
- Miscellaneous dependency and lockfile updates.

## Version [1.7.3] - 2024-09-17

- Consolidated all_properties() method and its string validations.
- Simplified pandas.pct_change() after new pandas type stubs and fill_method no longer required by mypy.
- Simplified dependencies after bug in statsmodels resolved in version 0.14.3.
- Miscellaneous dependency and lockfile updates

## Version [1.7.2] - 2024-08-24

- Replaced the dot operator from numpy and pandas with the Python @ operator.
- Made OpenTimeSeries Pydantic validator methods private.
- Adjustments to adhere to mypy unreachable code warnings.
- Removed table in security.md to limit maintenance.
- Removed setup_class from readme.md as it was deleted in version 1.7.1.
- Cosmetic improvements on raise exception statements.
- Miscellaneous dependency updates

## Version [1.7.1] - 2024-08-10

- Changed so that resample_to_business_period_ends method on OpenFrame now retains original stubs on all constituent OpenTimeSeries.
- Made do_resample_to_business_period_ends private.
- Cleaned up calc_range and efficient_frontier functions.
- Improved calc_range test and exception message.
- Removed classmethod setup_class on OpenTimeSeries. Replaced with Pydantic field_validator decorator for domestic (currency) field and countries field.
- Made input type more strict on OpenTimeSeries.from_df().
- Introduced branch coverage and added tests to bring coverage back to 100% from 99%.
- Miscellaneous cleanup and depedencies updated.

## Version [1.7.0] - 2024-07-27

- Changed code to enforce PEP 604 on typing. This means that the PACKAGE WILL NO LONGER WORK FOR PYTHON 3.9.
- Limited GitHub workflow build.yaml to no longer run on Python 3.9
- Adjustments to adhere to ruff TCH type checking imports.
- Introduced strict requirement that generate_calendar_date_range argument trading_days must be positive.
- Further simplified common simulation for CommonTestCase class.
- Removed unnecessary .resolve() from pathlib.Path and simplified Path.open()
- Made all Path.open() statements follow the same syntax.
- Cleaned up test setups and added tearDownClass method on common testcase class.
- Removed year from copyright notice in license.md
- Suppressed pycharm type checker on OpenFrame class.
- Miscellaneous dependency updates

## Version [1.6.0] - 2024-07-08

- Added \_\_all__ where appropriate for proper import autocompletion and hinting
- Potentially BREAKING CHANGE in constrain_optimized_portfolios and efficient_frontier. They no longer take a single upper bound as argument but instead take both lower and upper bounds for each position in a portfolio. And if given they must all be given.
- Added option to set which method is passed into the scipy.minimize function that is used when finding the efficient frontier of a portfolio.
- BREAKING CHANGE - moved functions from frame.py to portfoliotools.py to improve project structure. Users will need to change imports because of this.
- Improved speed in portfoliotools tests.
- Miscellaneous dependency updates and tweaks in project tool scripts.

## Version [1.5.7] - 2024-06-23

- Changed behaviour in get_previous_business_day_before_today() function. None input means days argument is set to zero and not raise Exception. This will eliminate scenario that I earlier believed was bug
- Simplified Github workflows install of Poetry
- Cut down functions in _risk.py module to simplify code. Moved code to inline in main modules instead
- Widened acceptance on holidays dependency while silencing its DeprecationWarning
- Widened acceptance on numpy dependency and checked that project can use numpy 2.0.0
- improved exception message on date_fixer.offset_business_days
- Removed utc timezone setting where datetime.date.today() used. Aimed for local timezone. Do not believe it will impact anything

## Version [1.5.6] - 2024-06-04

- removed warning filter for deprecation in openpyxl that is now resolved.
- limiting holidays dependency due to warning in version 0.50.
- suppress erroneous pycharm inspection on unreachable code in datefixer.py
- Added int() in range in holiday_calendar method to safeguard against bug seen but not replicated.
- Corrected description of new to_json() in README.md
- Also miscellaneous dependencies updated.

## Version [1.5.5] - 2024-05-24

- Possibly a breaking change, the .to_json() method on both main classes now require an argument to choose if the raw values or the potentially amended values from the associated tsdf DataFrame is exported.
- The Poetry version for package building has been updated from 1.8.2 to 1.8.3.
- Outside of the package I have added more to the shell scripts used in development.

## Version [1.5.4] - 2024-05-05

- Added new measure Omega Ratio, https://en.wikipedia.org/wiki/Omega_ratio. Works on both OpenTimeSeries and OpenFrame.
- Changed casing on Positive share and Max drawdown labels for consistency.
- Outside of the package I have improved the shell scripts used in development.

## Version [1.5.3] - 2024-05-01

- Changed randomizer location on ReturnSimulation. Since constructing methods are never run twice this should have no effect.
- Improved hover labels in the sharpe_plot.
- Also miscellaneous dependencies updated.

## Version [1.5.2] - 2024-04-16

- Corrected typing.Literal on weight strategies that were removed in 1.5.1
- Lower cased project name throughout to remove any confusion
- Added Self typing where missing in types.py
- Miscellaneous development dependencies update. Minimal related changes.

## Version [1.5.1] - 2024-03-30

- Removed two weight strategies in OpenFrame.make_portfolio() to remove ffn as dependency.
- Removing ffn revealed that requests as explicit dependency was missing in pyproject.toml.
- Also miscellaneous development dependencies updates

## Version [1.5.0] - 2024-03-11

- Added helper functions for additional portfolio analyses.
- The new functions cover weight simulations, efficient frontier optimizations, and plotting.
- I intend to make an example that will be added [here](https://nbviewer.org/github/karrmagadgeteer2/NoteBook/blob/master/openseriesnotebook.ipynb)
- Also miscellaneous development dependencies updates

## Version [1.4.12] - 2024-02-20

- Fixed issue that OpenFrame.merge_series() method ignored changes to labels
- Added option to set statistical method parameters for related methods
- Miscellaneous development dependencies updates and test cleanup

## Version [1.4.11] - 2024-02-13

- Fixed issue that ReturnSimulation.from_merton_jump_gbm() class method missed jump parameters
- Miscellaneous development dependencies updates

## Version [1.4.10] - 2024-01-30

- Consolidated worst_month function into _common_model.py
- Miscellaneous dependencies updates & adapting to more recent pandas versions

## Version [1.4.9] - 2023-11-28

- Fixed so that project can be run on Python 3.12

## Version [1.4.8] - 2023-11-22

- Some readability cleanup in simulation.py
- Preparations to allow publishing on anaconda.org

## Version [1.4.7] - 2023-11-15

- Some doc string cleanup. Removed sources where code no longer resembles original
- Finalized simplification of simulation.py
- Implemented Pydantic BaseModel on ReturnSimulation class
- Further consolidated date range and vol from VaR functions
- Other smaller miscellaneous

## Version [1.4.6] - 2023-11-12

- Removed drawdown_details function as it was unnecessarily complex and of little interest
- Some dev dependency updates such as ruff and mypy
- Implemented typing.Self throughout where relevant
- Consolidated date range functions from series.py and frame.py into _common_model.py
- Made many functions "private" that are more appropriate with this status

## Version [1.4.5] - 2023-11-06

- Cleaned up function for exponentially weighted moving average "EWMA" volatility
- Some dev dependency updates such as ruff and Poetry
- Major cleanup of simulation module. Removed Heston CIR & OU models as I have never checked them. Also rewrote Merton JDM from scratch to make it consisten with GBM model.
- Rewrite of simulation required update of many calculation result tests

## Version [1.4.4] - 2023-10-30

- Fixed two pct_change() methods missing fill_method.
- Allowed for setting plotly include_plotlyjs argument controlling how to include plotly.js

## Version [1.4.3] - 2023-10-30

- Fixed plot tests and fixed test that relied on internet connection
- Made plots with output type div more meaningful by explicitly returning div section string
- pd.Series typing cleanup and a bunch of other cleanups in mypy
- Added fill_method to deal with pandas FutureWarning .pct_change(fill_method=cast(str, None)). May need to be reverted when pandas is updated from 2.1.2

## Version [1.4.2] - 2023-10-28

- Fixed error in timeseries_chain introduced in version 1.4.0. Because of this versions 1.4.0 & 1.4.1 was deleted
- Removed ignore on FutureWarning from pandas.pct_change
- Also removed as I deem them unnecessary pandas.ffill() before pct_change
- More type/mypy cleanup
- Fixed file folder behaviour in plot methods and added test coverage for it
- Updated pandas again

## Version [1.3.9] - 2023-10-24

- Mainly type/mypy and docstring cleanup
- Updated numpy and pandas which may affect users downstream

## Version [1.3.8] - 2023-10-15

- tightened .to_json() and .to_xlsx() methods. File path now defaults to homepath/Documents and if it does not exist it will save in the directory of the calling function
- added handing for the evant the the remote Captor logo file is not present.
- improved scenario coverage in test suite for both of the above changes.
- fixed .relative() function that previously did not add new dataframe column as item in constituents.

## Version [1.3.7] - 2023-10-12

- makefile fix
- consolidated simulations for tests
- updated ruff for dev tools
- improved doc strings for OpenTimeSeries and OpenFrame
- added tests on package metadata
- updated mypy for dev tools
- added tick format argument to hovertemplate in plots

## Version [1.3.6] - 2023-10-01

Implemented mypy type checking more strictly. Many in-line ignores that will remain until Pandas
typing is more developed.

## Version [1.3.5] - 2023-09-28

Removed multiple inheritance on OpenTimeSeries and OpenFrame by inheriting CommonModel from the Pydantic BaseModel.
Also removed unnecessary TypeVar throughout.

## Version [1.3.4] - 2023-09-20

Simplified ReturnSimulation to make it easier to use. The change will not affect users of OpenTimeSeries and OpenFrame.

## Version [1.3.3] - 2023-09-19

Improved ruff code coverage by setting the configuration to select ALL.

## Version [1.3.2] - 2023-09-10

Found a mistake in geo_ret_func that was made when it was consolidated into the common_model.py
from frame.py and series.py. The function has worked as intended for any situations where no
arguments were provided, and therefore the geo_ret property has not been compromised.

## Version [1.3.1] - 2023-09-05

Validation of raw dates and values with a minimum of 2 items caused significant issues
downstream in our fund company because the OpenTimeSeries subclass is used as a validator
also for fetching single date price data to be used in some valuations.
Because of this the validation now only requires 1 data point.

## Version [1.3.0] - 2023-09-05

Due to test redesign validation to not allow empty dates and values arrays had stopped.
Reintroduced with this version.

## Version [1.2.9] - 2023-09-02

Very minor due to still received FutureWarning downstream. Unclear if it helps.

## Version [1.2.8] - 2023-09-02

Implemented missing adjustments to work with Pandas 2.1.0. Removed tsdf annotation from
CommonModel to avoid override in OpenFrame and OpenTimeSeries.

## Version [1.2.7] - 2023-08-30

New Pandas 2.1.0 released and in this version necessary adjustments to silence
related FutureWarnings have been made.

## Version [1.2.6] - 2023-08-30

Removed isin code validation because it created conflicting validations downstream
within our fund company. Removed python-stdnum dependency as a result.
Also continued typing cleanup.

## Version [1.2.5] - 2023-08-28

Simplified timeseries_chain function and changed variable case to resolve N815 error.

## Version [1.2.4] - 2023-08-26

Minor cleanup.

## Version [1.2.3] - 2023-08-26

Changed linter check package from Pylint to Ruff and made changes in code to
conform with standards upheld by Ruff.

## Version [1.2.2] - 2023-08-17

This version is primarily a fine tuning. I found some incorrect type hints and
doc strings in common model to correct.

## Version [1.2.1] - 2023-08-16

Removed the option of selecting a column as the riskfree asset for the Sortino
ratio and Return/vol ratios. Also moved max drawdown date to common model after
I noticed that the existing OpenFrame result was wrong. The result now has been
verified against the timeseries.

## Version [1.2.0] - 2023-08-16

A significant change that will hopefully not break anything backwards as it is a
rebuild to remove as much duplicate code as possible. This has been achieved by
introducing a new base class that OpenTimeSeries and OpenFrame inherits properties
and methods from, while still also inheriting from the Pydantic BaseModel.
Further described in [issue 41](https://github.com/CaptorAB/openseries/issues/41).

## Version [1.1.7] - 2023-08-11

Changed the Pydantic configuration which will improve popup docs in e.g. Pycharm.
Also resolved issue [#35](https://github.com/CaptorAB/openseries/issues/35) with
nested decorator warning.

## Version [1.1.6] - 2023-08-08

Replaced to_opentimeseries_openframe method on ReturnSimulation with to_dataframe.

## Version [1.1.5] - 2023-08-08

Consolidated sim_price.py and stoch_processes.py into a single simulation.py with the
class ReturnSimulation to hold all steps. Still no optimization of any kind and I still
need a second opinion on the models.
Tightening and cleaning up typing. Also removed pipe syntax for optional parameters and
replaced with typing.Optional which allows the project to work on Python 3.9.
Relatively significant adaptations made to allow update of dependency to Pydantic 2.0.
Dependency to statsmodels also updated to 0.14.0. Made a few improvements to pytest based
tests in the test suite and cleaned up the unittest based ones.

## Version [1.0.1] - 2023-08-02

Removed requests and urllib3 as dependencies as they are no longer directly required.

## Version [1.0.0] - 2023-08-01

After successfully deploying with Pandas 2.0 dependency in use at our company I decided that
it is more appropriate to now deploy a 1.0.0 version. I also intend to start deploying with
simultaneous creation of a Git tag and GitHub release that will match the version number
published on Pypi.

## Version [0.14.0] - 2023-07-21

Moved to Pandas version ^2.0.3

## Version [0.13.4] - 2023-07-19

Cleaned up dependencies and fixed methods that would not work across time zones.

## Version [0.13.3] - 2023-07-01

Improved flexibility on dependencies.

## Version [0.13.2] - 2023-06-19

Added a .to_xlsx() method to both OpenTimeSeries and OpenFrame.

## Version [0.13.1] - 2023-05-30

Changed default plot legend layout.

## Version [0.12.9] - 2023-05-13

Added a new OpenTimeSeries constructor method .from_arrays().

## Version [0.12.8] - 2023-05-09

Improved speed of EWMA risk functions.

## Version [0.12.6] - 2023-04-23

Changes made to ensure compatability with Pandas 2.0.0.

## Version [0.12.5] - 2023-04-16

Added weight setting strategies to the OpenFrame.make_portfolio() method. Using
the [ffn](https://github.com/pmorissette/ffn) package.

## Version [0.12.4] - 2023-04-09

Improved project description with links to mybinder.org and Jupyter Nbviewer.

## Version [0.12.3] - 2023-04-06

Added pylint as a code checker. Some checks silenced in `pyproject.toml`.

## Version [0.12.1] - 2023-04-03

Introduced `validate_assignment = True` which allowed the removal of several separate
validators. Unfortunately this implies that interdependent argument validations have
been removed and will be difficult to implement without significant user impact.

## Version [0.11.9] - 2023-04-02

Removed the from_frame() constructor and fixed most of the silenced mypy warnings.

## Version [0.11.8] - 2023-03-27

Fixed typing the inputs for OpenFrame methods

## Version [0.11.7] - 2023-03-25

Aligned code to pass many of the type checks performed by `mypy`. A list of error
codes are for now silenced in the project's
[pyproject.toml](https://github.com/CaptorAB/openseries/blob/master/pyproject.toml)
file.

## Version [0.11.4] - 2023-03-19

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
