Types and Enums
===============

.. currentmodule:: openseries.owntypes

.. automodule:: openseries.owntypes
   :members:
   :undoc-members:
   :show-inheritance:

Value Types
-----------

.. autoclass:: ValueType
   :members:
   :undoc-members:
   :show-inheritance:

The ValueType enum identifies the type of values in a time series (prices, returns, etc.).

Type Aliases
------------

.. autodata:: SeriesOrFloat_co
.. autodata:: CountryStringType
.. autodata:: CountrySetType
.. autodata:: CountriesType
.. autodata:: CurrencyStringType
.. autodata:: DateStringType
.. autodata:: DateListType
.. autodata:: ValueListType
.. autodata:: DaysInYearType
.. autodata:: DateType

Literal Types
-------------

.. autodata:: LiteralJsonOutput
.. autodata:: LiteralTrunc
.. autodata:: LiteralLinePlotMode
.. autodata:: LiteralHowMerge
.. autodata:: LiteralQuantileInterp
.. autodata:: LiteralBizDayFreq
.. autodata:: LiteralPandasReindexMethod
.. autodata:: LiteralNanMethod
.. autodata:: LiteralCaptureRatio
.. autodata:: LiteralBarPlotMode
.. autodata:: LiteralPlotlyOutput
.. autodata:: LiteralPlotlyJSlib
.. autodata:: LiteralPlotlyHistogramPlotType
.. autodata:: LiteralPlotlyHistogramBarMode
.. autodata:: LiteralPlotlyHistogramCurveType
.. autodata:: LiteralPlotlyHistogramHistNorm
.. autodata:: LiteralPortfolioWeightings
.. autodata:: LiteralMinimizeMethods
.. autodata:: LiteralSeriesProps
.. autodata:: LiteralFrameProps

Validation Classes
------------------

.. autoclass:: Countries
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Currency
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PropertiesList
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: OpenTimeSeriesPropertiesList
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: OpenFramePropertiesList
   :members:
   :undoc-members:
   :show-inheritance:

Custom Exceptions
-----------------

.. autoexception:: MixedValuetypesError
.. autoexception:: AtLeastOneFrameError
.. autoexception:: DateAlignmentError
.. autoexception:: NumberOfItemsAndLabelsNotSameError
.. autoexception:: InitialValueZeroError
.. autoexception:: CountriesNotStringNorListStrError
.. autoexception:: MarketsNotStringNorListStrError
.. autoexception:: TradingDaysNotAboveZeroError
.. autoexception:: BothStartAndEndError
.. autoexception:: NoWeightsError
.. autoexception:: LabelsNotUniqueError
.. autoexception:: RatioInputError
.. autoexception:: MergingResultedInEmptyError
.. autoexception:: IncorrectArgumentComboError
.. autoexception:: PropertiesInputValidationError
.. autoexception:: ResampleDataLossError
