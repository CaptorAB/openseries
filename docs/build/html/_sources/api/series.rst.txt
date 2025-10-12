OpenTimeSeries
==============

.. currentmodule:: openseries

.. autoclass:: OpenTimeSeries
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

The OpenTimeSeries class is the core component for analyzing individual financial time series. It provides comprehensive functionality for:

- Loading data from various sources (arrays, DataFrames, fixed rates)
- Calculating financial metrics and risk measures
- Performing time series transformations
- Creating visualizations
- Exporting results

Class Methods for Construction
------------------------------

.. automethod:: OpenTimeSeries.from_arrays
   :no-index:
.. automethod:: OpenTimeSeries.from_df
   :no-index:
.. automethod:: OpenTimeSeries.from_fixed_rate
   :no-index:
.. automethod:: OpenTimeSeries.from_deepcopy
   :no-index:

Properties
----------

Non-numerical Properties
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenTimeSeries.timeseries_id
   :no-index:
.. autoattribute:: OpenTimeSeries.instrument_id
   :no-index:
.. autoattribute:: OpenTimeSeries.dates
   :no-index:
.. autoattribute:: OpenTimeSeries.values
   :no-index:
.. autoattribute:: OpenTimeSeries.currency
   :no-index:
.. autoattribute:: OpenTimeSeries.domestic
   :no-index:
.. autoattribute:: OpenTimeSeries.local_ccy
   :no-index:
.. autoattribute:: OpenTimeSeries.name
   :no-index:
.. autoattribute:: OpenTimeSeries.isin
   :no-index:
.. autoattribute:: OpenTimeSeries.label
   :no-index:
.. autoattribute:: OpenTimeSeries.countries
   :no-index:
.. autoattribute:: OpenTimeSeries.markets
   :no-index:
.. autoattribute:: OpenTimeSeries.valuetype
   :no-index:

Common Properties
~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenTimeSeries.first_idx
   :no-index:
.. autoattribute:: OpenTimeSeries.last_idx
   :no-index:
.. autoattribute:: OpenTimeSeries.length
   :no-index:
.. autoattribute:: OpenTimeSeries.span_of_days
   :no-index:
.. autoattribute:: OpenTimeSeries.tsdf
   :no-index:
.. autoattribute:: OpenTimeSeries.max_drawdown_date
   :no-index:
.. autoattribute:: OpenTimeSeries.periods_in_a_year
   :no-index:
.. autoattribute:: OpenTimeSeries.yearfrac
   :no-index:

Financial Metrics
~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenTimeSeries.all_properties
   :no-index:
.. autoattribute:: OpenTimeSeries.arithmetic_ret
   :no-index:
.. autoattribute:: OpenTimeSeries.geo_ret
   :no-index:
.. autoattribute:: OpenTimeSeries.value_ret
   :no-index:
.. autoattribute:: OpenTimeSeries.vol
   :no-index:
.. autoattribute:: OpenTimeSeries.downside_deviation
   :no-index:
.. autoattribute:: OpenTimeSeries.ret_vol_ratio
   :no-index:
.. autoattribute:: OpenTimeSeries.sortino_ratio
   :no-index:
.. autoattribute:: OpenTimeSeries.kappa3_ratio
   :no-index:
.. autoattribute:: OpenTimeSeries.omega_ratio
   :no-index:
.. autoattribute:: OpenTimeSeries.var_down
   :no-index:
.. autoattribute:: OpenTimeSeries.cvar_down
   :no-index:
.. autoattribute:: OpenTimeSeries.worst
   :no-index:
.. autoattribute:: OpenTimeSeries.worst_month
   :no-index:
.. autoattribute:: OpenTimeSeries.max_drawdown
   :no-index:
.. autoattribute:: OpenTimeSeries.max_drawdown_cal_year
   :no-index:
.. autoattribute:: OpenTimeSeries.positive_share
   :no-index:
.. autoattribute:: OpenTimeSeries.vol_from_var
   :no-index:
.. autoattribute:: OpenTimeSeries.skew
   :no-index:
.. autoattribute:: OpenTimeSeries.kurtosis
   :no-index:
.. autoattribute:: OpenTimeSeries.z_score
   :no-index:

Methods
-------

Data Manipulation
~~~~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.pandas_df
   :no-index:
.. automethod:: OpenTimeSeries.set_new_label
   :no-index:
.. automethod:: OpenTimeSeries.running_adjustment
   :no-index:
.. automethod:: OpenTimeSeries.from_1d_rate_to_cumret
   :no-index:
.. automethod:: OpenTimeSeries.align_index_to_local_cdays
   :no-index:
.. automethod:: OpenTimeSeries.resample
   :no-index:
.. automethod:: OpenTimeSeries.resample_to_business_period_ends
   :no-index:
.. automethod:: OpenTimeSeries.value_nan_handle
   :no-index:
.. automethod:: OpenTimeSeries.return_nan_handle
   :no-index:

Transformations
~~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.to_cumret
   :no-index:
.. automethod:: OpenTimeSeries.value_to_ret
   :no-index:
.. automethod:: OpenTimeSeries.value_to_diff
   :no-index:
.. automethod:: OpenTimeSeries.value_to_log
   :no-index:
.. automethod:: OpenTimeSeries.to_drawdown_series
   :no-index:

Analysis Methods
~~~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.ewma_vol_func
   :no-index:
.. automethod:: OpenTimeSeries.value_ret_calendar_period
   :no-index:
.. automethod:: OpenTimeSeries.rolling_return
   :no-index:
.. automethod:: OpenTimeSeries.rolling_vol
   :no-index:
.. automethod:: OpenTimeSeries.rolling_var_down
   :no-index:
.. automethod:: OpenTimeSeries.rolling_cvar_down
   :no-index:
.. automethod:: OpenTimeSeries.calc_range
   :no-index:
.. automethod:: OpenTimeSeries.outliers
   :no-index:

Financial Metrics Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.arithmetic_ret_func
   :no-index:
.. automethod:: OpenTimeSeries.geo_ret_func
   :no-index:
.. automethod:: OpenTimeSeries.value_ret_func
   :no-index:
.. automethod:: OpenTimeSeries.vol_func
   :no-index:
.. automethod:: OpenTimeSeries.lower_partial_moment_func
   :no-index:
.. automethod:: OpenTimeSeries.ret_vol_ratio_func
   :no-index:
.. automethod:: OpenTimeSeries.sortino_ratio_func
   :no-index:
.. automethod:: OpenTimeSeries.omega_ratio_func
   :no-index:
.. automethod:: OpenTimeSeries.var_down_func
   :no-index:
.. automethod:: OpenTimeSeries.cvar_down_func
   :no-index:
.. automethod:: OpenTimeSeries.worst_func
   :no-index:
.. automethod:: OpenTimeSeries.max_drawdown_func
   :no-index:
.. automethod:: OpenTimeSeries.positive_share_func
   :no-index:
.. automethod:: OpenTimeSeries.vol_from_var_func
   :no-index:
.. automethod:: OpenTimeSeries.skew_func
   :no-index:
.. automethod:: OpenTimeSeries.kurtosis_func
   :no-index:
.. automethod:: OpenTimeSeries.z_score_func
   :no-index:
.. automethod:: OpenTimeSeries.target_weight_from_var
   :no-index:

Visualization
~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.plot_series
   :no-index:
.. automethod:: OpenTimeSeries.plot_bars
   :no-index:
.. automethod:: OpenTimeSeries.plot_histogram
   :no-index:

Export Methods
~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.to_json
   :no-index:
.. automethod:: OpenTimeSeries.to_xlsx
   :no-index:

Utility Functions
-----------------

.. autofunction:: timeseries_chain
   :no-index:
