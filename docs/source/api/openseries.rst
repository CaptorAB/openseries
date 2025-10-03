openseries package
==================

.. automodule:: openseries
   :members:
   :undoc-members:
   :show-inheritance:

Main Classes
------------

The openseries package provides two main classes for financial time series analysis:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   openseries.OpenTimeSeries
   openseries.OpenFrame

Utility Functions
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   openseries.timeseries_chain
   openseries.report_html

Portfolio Tools
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   openseries.efficient_frontier
   openseries.simulate_portfolios
   openseries.constrain_optimized_portfolios
   openseries.prepare_plot_data
   openseries.sharpeplot

Date Utilities
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   openseries.date_fix
   openseries.date_offset_foll
   openseries.generate_calendar_date_range
   openseries.get_previous_business_day_before_today
   openseries.holiday_calendar
   openseries.offset_business_days

Simulation
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   openseries.ReturnSimulation

Types and Enums
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   openseries.ValueType
   openseries.Self

Other Utilities
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   openseries.load_plotly_dict
