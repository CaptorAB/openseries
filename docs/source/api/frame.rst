OpenFrame
=========

.. currentmodule:: openseries

.. autoclass:: OpenFrame
   :undoc-members:
   :show-inheritance:
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
   :no-index:

Properties
----------

Frame-specific Properties
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenFrame.constituents
   :no-index:
.. autoattribute:: OpenFrame.columns_lvl_zero
   :no-index:
.. autoattribute:: OpenFrame.columns_lvl_one
   :no-index:
.. autoattribute:: OpenFrame.item_count
   :no-index:
.. autoattribute:: OpenFrame.weights
   :no-index:
.. autoattribute:: OpenFrame.first_indices
   :no-index:
.. autoattribute:: OpenFrame.last_indices
   :no-index:
.. autoattribute:: OpenFrame.lengths_of_items
   :no-index:
.. autoattribute:: OpenFrame.span_of_days_all
   :no-index:

Common Properties
~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenFrame.first_idx
   :no-index:
.. autoattribute:: OpenFrame.last_idx
   :no-index:
.. autoattribute:: OpenFrame.length
   :no-index:
.. autoattribute:: OpenFrame.span_of_days
   :no-index:
.. autoattribute:: OpenFrame.tsdf
   :no-index:
.. autoattribute:: OpenFrame.max_drawdown_date
   :no-index:
.. autoattribute:: OpenFrame.periods_in_a_year
   :no-index:
.. autoattribute:: OpenFrame.yearfrac
   :no-index:

Financial Metrics
~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenFrame.all_properties
   :no-index:
.. autoattribute:: OpenFrame.arithmetic_ret
   :no-index:
.. autoattribute:: OpenFrame.geo_ret
   :no-index:
.. autoattribute:: OpenFrame.value_ret
   :no-index:
.. autoattribute:: OpenFrame.vol
   :no-index:
.. autoattribute:: OpenFrame.downside_deviation
   :no-index:
.. autoattribute:: OpenFrame.ret_vol_ratio
   :no-index:
.. autoattribute:: OpenFrame.sortino_ratio
   :no-index:
.. autoattribute:: OpenFrame.kappa3_ratio
   :no-index:
.. autoattribute:: OpenFrame.omega_ratio
   :no-index:
.. autoattribute:: OpenFrame.var_down
   :no-index:
.. autoattribute:: OpenFrame.cvar_down
   :no-index:
.. autoattribute:: OpenFrame.worst
   :no-index:
.. autoattribute:: OpenFrame.worst_month
   :no-index:
.. autoattribute:: OpenFrame.max_drawdown
   :no-index:
.. autoattribute:: OpenFrame.max_drawdown_cal_year
   :no-index:
.. autoattribute:: OpenFrame.positive_share
   :no-index:
.. autoattribute:: OpenFrame.vol_from_var
   :no-index:
.. autoattribute:: OpenFrame.skew
   :no-index:
.. autoattribute:: OpenFrame.kurtosis
   :no-index:
.. autoattribute:: OpenFrame.z_score
   :no-index:

Methods
-------

Frame Management
~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.merge_series
   :no-index:
.. automethod:: OpenFrame.trunc_frame
   :no-index:
.. automethod:: OpenFrame.add_timeseries
   :no-index:
.. automethod:: OpenFrame.delete_timeseries
   :no-index:

Portfolio Analysis
~~~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.relative
   :no-index:
.. automethod:: OpenFrame.make_portfolio
   :no-index:
.. automethod:: OpenFrame.rebalanced_portfolio
   :no-index:

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.ord_least_squares_fit
   :no-index:
.. automethod:: OpenFrame.beta
   :no-index:
.. automethod:: OpenFrame.jensen_alpha
   :no-index:
.. automethod:: OpenFrame.tracking_error_func
   :no-index:
.. automethod:: OpenFrame.info_ratio_func
   :no-index:
.. automethod:: OpenFrame.capture_ratio_func
   :no-index:
.. automethod:: OpenFrame.multi_factor_linear_regression
   :no-index:

Rolling Analysis
~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.rolling_info_ratio
   :no-index:
.. automethod:: OpenFrame.rolling_beta
   :no-index:
.. automethod:: OpenFrame.rolling_corr
   :no-index:
.. automethod:: OpenFrame.rolling_return
   :no-index:
.. automethod:: OpenFrame.rolling_vol
   :no-index:
.. automethod:: OpenFrame.rolling_var_down
   :no-index:
.. automethod:: OpenFrame.rolling_cvar_down
   :no-index:

Correlation and Risk
~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: OpenFrame.correl_matrix
   :no-index:
.. automethod:: OpenFrame.ewma_risk
   :no-index:

Data Manipulation
~~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.align_index_to_local_cdays
   :no-index:
.. automethod:: OpenFrame.resample
   :no-index:
.. automethod:: OpenFrame.resample_to_business_period_ends
   :no-index:
.. automethod:: OpenFrame.value_nan_handle
   :no-index:
.. automethod:: OpenFrame.return_nan_handle
   :no-index:

Transformations
~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.to_cumret
   :no-index:
.. automethod:: OpenFrame.value_to_ret
   :no-index:
.. automethod:: OpenFrame.value_to_diff
   :no-index:
.. automethod:: OpenFrame.value_to_log
   :no-index:
.. automethod:: OpenFrame.to_drawdown_series
   :no-index:
.. automethod:: OpenFrame.value_ret_calendar_period
   :no-index:

Analysis Methods
~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.calc_range
   :no-index:

Financial Metrics Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: OpenFrame.arithmetic_ret_func
   :no-index:
.. automethod:: OpenFrame.geo_ret_func
   :no-index:
.. automethod:: OpenFrame.value_ret_func
   :no-index:
.. automethod:: OpenFrame.vol_func
   :no-index:
.. automethod:: OpenFrame.lower_partial_moment_func
   :no-index:
.. automethod:: OpenFrame.ret_vol_ratio_func
   :no-index:
.. automethod:: OpenFrame.sortino_ratio_func
   :no-index:
.. automethod:: OpenFrame.omega_ratio_func
   :no-index:
.. automethod:: OpenFrame.var_down_func
   :no-index:
.. automethod:: OpenFrame.cvar_down_func
   :no-index:
.. automethod:: OpenFrame.worst_func
   :no-index:
.. automethod:: OpenFrame.max_drawdown_func
   :no-index:
.. automethod:: OpenFrame.positive_share_func
   :no-index:
.. automethod:: OpenFrame.vol_from_var_func
   :no-index:
.. automethod:: OpenFrame.skew_func
   :no-index:
.. automethod:: OpenFrame.kurtosis_func
   :no-index:
.. automethod:: OpenFrame.z_score_func
   :no-index:
.. automethod:: OpenFrame.target_weight_from_var
   :no-index:

Visualization
~~~~~~~~~~~~~

.. automethod:: OpenFrame.plot_series
   :no-index:
.. automethod:: OpenFrame.plot_bars
   :no-index:
.. automethod:: OpenFrame.plot_histogram
   :no-index:

Export Methods
~~~~~~~~~~~~~~

.. automethod:: OpenFrame.to_json
   :no-index:
.. automethod:: OpenFrame.to_xlsx
   :no-index:
