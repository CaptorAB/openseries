OpenTimeSeries
==============

.. currentmodule:: openseries

.. autoclass:: OpenTimeSeries
   :members:
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
.. automethod:: OpenTimeSeries.from_df
.. automethod:: OpenTimeSeries.from_fixed_rate
.. automethod:: OpenTimeSeries.from_deepcopy

Properties
----------

Non-numerical Properties
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenTimeSeries.timeseries_id
.. autoattribute:: OpenTimeSeries.instrument_id
.. autoattribute:: OpenTimeSeries.dates
.. autoattribute:: OpenTimeSeries.values
.. autoattribute:: OpenTimeSeries.currency
.. autoattribute:: OpenTimeSeries.domestic
.. autoattribute:: OpenTimeSeries.local_ccy
.. autoattribute:: OpenTimeSeries.name
.. autoattribute:: OpenTimeSeries.isin
.. autoattribute:: OpenTimeSeries.label
.. autoattribute:: OpenTimeSeries.countries
.. autoattribute:: OpenTimeSeries.markets
.. autoattribute:: OpenTimeSeries.valuetype

Common Properties
~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenTimeSeries.first_idx
.. autoattribute:: OpenTimeSeries.last_idx
.. autoattribute:: OpenTimeSeries.length
.. autoattribute:: OpenTimeSeries.span_of_days
.. autoattribute:: OpenTimeSeries.tsdf
.. autoattribute:: OpenTimeSeries.max_drawdown_date
.. autoattribute:: OpenTimeSeries.periods_in_a_year
.. autoattribute:: OpenTimeSeries.yearfrac

Financial Metrics
~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenTimeSeries.all_properties
.. autoattribute:: OpenTimeSeries.arithmetic_ret
.. autoattribute:: OpenTimeSeries.geo_ret
.. autoattribute:: OpenTimeSeries.value_ret
.. autoattribute:: OpenTimeSeries.vol
.. autoattribute:: OpenTimeSeries.downside_deviation
.. autoattribute:: OpenTimeSeries.ret_vol_ratio
.. autoattribute:: OpenTimeSeries.sortino_ratio
.. autoattribute:: OpenTimeSeries.kappa3_ratio
.. autoattribute:: OpenTimeSeries.omega_ratio
.. autoattribute:: OpenTimeSeries.var_down
.. autoattribute:: OpenTimeSeries.cvar_down
.. autoattribute:: OpenTimeSeries.worst
.. autoattribute:: OpenTimeSeries.worst_month
.. autoattribute:: OpenTimeSeries.max_drawdown
.. autoattribute:: OpenTimeSeries.max_drawdown_cal_year
.. autoattribute:: OpenTimeSeries.positive_share
.. autoattribute:: OpenTimeSeries.vol_from_var
.. autoattribute:: OpenTimeSeries.skew
.. autoattribute:: OpenTimeSeries.kurtosis
.. autoattribute:: OpenTimeSeries.z_score

Methods
-------

Data Manipulation
~~~~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.pandas_df
.. automethod:: OpenTimeSeries.set_new_label
.. automethod:: OpenTimeSeries.running_adjustment
.. automethod:: OpenTimeSeries.from_1d_rate_to_cumret
.. automethod:: OpenTimeSeries.align_index_to_local_cdays
.. automethod:: OpenTimeSeries.resample
.. automethod:: OpenTimeSeries.resample_to_business_period_ends
.. automethod:: OpenTimeSeries.value_nan_handle
.. automethod:: OpenTimeSeries.return_nan_handle

Transformations
~~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.to_cumret
.. automethod:: OpenTimeSeries.value_to_ret
.. automethod:: OpenTimeSeries.value_to_diff
.. automethod:: OpenTimeSeries.value_to_log
.. automethod:: OpenTimeSeries.to_drawdown_series

Analysis Methods
~~~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.ewma_vol_func
.. automethod:: OpenTimeSeries.value_ret_calendar_period
.. automethod:: OpenTimeSeries.rolling_return
.. automethod:: OpenTimeSeries.rolling_vol
.. automethod:: OpenTimeSeries.rolling_var_down
.. automethod:: OpenTimeSeries.rolling_cvar_down
.. automethod:: OpenTimeSeries.calc_range

Financial Metrics Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.arithmetic_ret_func
.. automethod:: OpenTimeSeries.geo_ret_func
.. automethod:: OpenTimeSeries.value_ret_func
.. automethod:: OpenTimeSeries.vol_func
.. automethod:: OpenTimeSeries.lower_partial_moment_func
.. automethod:: OpenTimeSeries.ret_vol_ratio_func
.. automethod:: OpenTimeSeries.sortino_ratio_func
.. automethod:: OpenTimeSeries.omega_ratio_func
.. automethod:: OpenTimeSeries.var_down_func
.. automethod:: OpenTimeSeries.cvar_down_func
.. automethod:: OpenTimeSeries.worst_func
.. automethod:: OpenTimeSeries.max_drawdown_func
.. automethod:: OpenTimeSeries.positive_share_func
.. automethod:: OpenTimeSeries.vol_from_var_func
.. automethod:: OpenTimeSeries.skew_func
.. automethod:: OpenTimeSeries.kurtosis_func
.. automethod:: OpenTimeSeries.z_score_func
.. automethod:: OpenTimeSeries.target_weight_from_var

Visualization
~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.plot_series
.. automethod:: OpenTimeSeries.plot_bars
.. automethod:: OpenTimeSeries.plot_histogram

Export Methods
~~~~~~~~~~~~~~

.. automethod:: OpenTimeSeries.to_json
.. automethod:: OpenTimeSeries.to_xlsx

Utility Functions
-----------------

.. autofunction:: timeseries_chain
