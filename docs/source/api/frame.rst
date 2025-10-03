OpenFrame
=========

.. currentmodule:: openseries

.. autoclass:: OpenFrame
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

The OpenFrame class manages collections of OpenTimeSeries objects and provides functionality for:

- Multi-asset analysis and comparison
- Portfolio construction and optimization
- Correlation and regression analysis
- Risk attribution and factor analysis
- Batch processing of multiple time series

Class Methods for Construction
------------------------------

.. automethod:: OpenFrame.from_deepcopy

Properties
----------

Frame-specific Properties
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenFrame.constituents
.. autoattribute:: OpenFrame.columns_lvl_zero
.. autoattribute:: OpenFrame.columns_lvl_one
.. autoattribute:: OpenFrame.item_count
.. autoattribute:: OpenFrame.weights
.. autoattribute:: OpenFrame.first_indices
.. autoattribute:: OpenFrame.last_indices
.. autoattribute:: OpenFrame.lengths_of_items
.. autoattribute:: OpenFrame.span_of_days_all

Common Properties
~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenFrame.first_idx
.. autoattribute:: OpenFrame.last_idx
.. autoattribute:: OpenFrame.length
.. autoattribute:: OpenFrame.span_of_days
.. autoattribute:: OpenFrame.tsdf
.. autoattribute:: OpenFrame.max_drawdown_date
.. autoattribute:: OpenFrame.periods_in_a_year
.. autoattribute:: OpenFrame.yearfrac

Financial Metrics
~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenFrame.all_properties
.. autoattribute:: OpenFrame.arithmetic_ret
.. autoattribute:: OpenFrame.geo_ret
.. autoattribute:: OpenFrame.value_ret
.. autoattribute:: OpenFrame.vol
.. autoattribute:: OpenFrame.downside_deviation
.. autoattribute:: OpenFrame.ret_vol_ratio
.. autoattribute:: OpenFrame.sortino_ratio
.. autoattribute:: OpenFrame.kappa3_ratio
.. autoattribute:: OpenFrame.omega_ratio
.. autoattribute:: OpenFrame.var_down
.. autoattribute:: OpenFrame.cvar_down
.. autoattribute:: OpenFrame.worst
.. autoattribute:: OpenFrame.worst_month
.. autoattribute:: OpenFrame.max_drawdown
.. autoattribute:: OpenFrame.max_drawdown_cal_year
.. autoattribute:: OpenFrame.positive_share
.. autoattribute:: OpenFrame.vol_from_var
.. autoattribute:: OpenFrame.skew
.. autoattribute:: OpenFrame.kurtosis
.. autoattribute:: OpenFrame.z_score

Methods
-------

Frame Management
~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.merge_series
.. automethod:: OpenFrame.trunc_frame
.. automethod:: OpenFrame.add_timeseries
.. automethod:: OpenFrame.delete_timeseries

Portfolio Analysis
~~~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.relative
.. automethod:: OpenFrame.make_portfolio

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.ord_least_squares_fit
.. automethod:: OpenFrame.beta
.. automethod:: OpenFrame.jensen_alpha
.. automethod:: OpenFrame.tracking_error_func
.. automethod:: OpenFrame.info_ratio_func
.. automethod:: OpenFrame.capture_ratio_func
.. automethod:: OpenFrame.multi_factor_linear_regression

Rolling Analysis
~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.rolling_info_ratio
.. automethod:: OpenFrame.rolling_beta
.. automethod:: OpenFrame.rolling_corr
.. automethod:: OpenFrame.rolling_return
.. automethod:: OpenFrame.rolling_vol
.. automethod:: OpenFrame.rolling_var_down
.. automethod:: OpenFrame.rolling_cvar_down

Correlation and Risk
~~~~~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.correl_matrix
.. automethod:: OpenFrame.ewma_risk

Data Manipulation
~~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.align_index_to_local_cdays
.. automethod:: OpenFrame.resample
.. automethod:: OpenFrame.resample_to_business_period_ends
.. automethod:: OpenFrame.value_nan_handle
.. automethod:: OpenFrame.return_nan_handle

Transformations
~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.to_cumret
.. automethod:: OpenFrame.value_to_ret
.. automethod:: OpenFrame.value_to_diff
.. automethod:: OpenFrame.value_to_log
.. automethod:: OpenFrame.to_drawdown_series
.. automethod:: OpenFrame.value_ret_calendar_period

Analysis Methods
~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.calc_range

Financial Metrics Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.arithmetic_ret_func
.. automethod:: OpenFrame.geo_ret_func
.. automethod:: OpenFrame.value_ret_func
.. automethod:: OpenFrame.vol_func
.. automethod:: OpenFrame.lower_partial_moment_func
.. automethod:: OpenFrame.ret_vol_ratio_func
.. automethod:: OpenFrame.sortino_ratio_func
.. automethod:: OpenFrame.omega_ratio_func
.. automethod:: OpenFrame.var_down_func
.. automethod:: OpenFrame.cvar_down_func
.. automethod:: OpenFrame.worst_func
.. automethod:: OpenFrame.max_drawdown_func
.. automethod:: OpenFrame.positive_share_func
.. automethod:: OpenFrame.vol_from_var_func
.. automethod:: OpenFrame.skew_func
.. automethod:: OpenFrame.kurtosis_func
.. automethod:: OpenFrame.z_score_func
.. automethod:: OpenFrame.target_weight_from_var

Visualization
~~~~~~~~~~~~~

.. automethod:: OpenFrame.plot_series
.. automethod:: OpenFrame.plot_bars
.. automethod:: OpenFrame.plot_histogram

Export Methods
~~~~~~~~~~~~~~

.. automethod:: OpenFrame.to_json
.. automethod:: OpenFrame.to_xlsx
