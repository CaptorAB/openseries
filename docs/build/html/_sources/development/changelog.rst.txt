Changelog
=========

This page documents the changes and improvements made to openseries over time.

Version 1.9.7 (Current)
------------------------

**Release Date**: 2024

**New Features**

- Enhanced portfolio optimization tools
- Improved business day calendar handling
- Extended statistical analysis capabilities
- Better integration with external data sources

**Improvements**

- Performance optimizations for large datasets
- Enhanced type safety with Pydantic v2
- Improved error handling and validation
- Better documentation and examples

**Bug Fixes**

- Fixed edge cases in drawdown calculations
- Resolved issues with date alignment
- Corrected correlation matrix calculations
- Fixed memory leaks in rolling calculations

**Dependencies**

- Updated to Pydantic v2.5.2+
- Pandas 2.1.2+ support
- Python 3.10-3.13 compatibility
- Enhanced exchange-calendars integration

Previous Versions
-----------------

Version 1.9.x Series
~~~~~~~~~~~~~~~~~~~~~

The 1.9.x series focused on:

- Stability improvements
- Enhanced financial calculations
- Better data validation
- Expanded visualization capabilities

Version 1.8.x Series
~~~~~~~~~~~~~~~~~~~~~

Key features introduced:

- Rolling analysis functions
- Enhanced portfolio tools
- Improved risk metrics
- Better correlation analysis

Version 1.7.x Series
~~~~~~~~~~~~~~~~~~~~~

Major additions:

- OpenFrame class enhancements
- Multi-factor regression analysis
- Advanced portfolio optimization
- Enhanced reporting capabilities

Version 1.6.x Series
~~~~~~~~~~~~~~~~~~~~~

Notable improvements:

- Performance optimizations
- Enhanced date handling
- Better error messages
- Expanded test coverage

Version 1.5.x Series
~~~~~~~~~~~~~~~~~~~~~

Foundation improvements:

- Core architecture refinements
- Enhanced data validation
- Better type hints
- Improved documentation

Early Versions (1.0.x - 1.4.x)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The early versions established:

- Core OpenTimeSeries functionality
- Basic financial calculations
- Initial OpenFrame implementation
- Fundamental plotting capabilities

Breaking Changes
----------------

Version 1.9.0
~~~~~~~~~~~~~~

- **Pydantic v2 Migration**: Updated to Pydantic v2, which may affect custom validation code
- **Python Version**: Dropped support for Python < 3.10
- **API Changes**: Some internal APIs were refactored for better performance

Version 1.8.0
~~~~~~~~~~~~~~

- **Method Signatures**: Some method signatures were updated for consistency
- **Return Types**: Enhanced return type annotations may affect type checking

Version 1.7.0
~~~~~~~~~~~~~~

- **Configuration**: Some configuration options were moved or renamed
- **Dependencies**: Updated minimum versions for several dependencies

Migration Guide
---------------

From 1.8.x to 1.9.x
~~~~~~~~~~~~~~~~~~~~

**Pydantic v2 Migration**

If you have custom validation code:

.. code-block:: python

   # Old (Pydantic v1)
   from pydantic import validator

   class CustomSeries(OpenTimeSeries):
       @validator('values')
       def validate_values(cls, v):
           return v

   # New (Pydantic v2)
   from pydantic import field_validator

   class CustomSeries(OpenTimeSeries):
       @field_validator('values')
       def validate_values(cls, v):
           return v

**Python Version**

Ensure you're using Python 3.10 or higher:

.. code-block:: bash

   python --version  # Should be 3.10+

From 1.7.x to 1.8.x
~~~~~~~~~~~~~~~~~~~~

**Method Updates**

Some method signatures were standardized:

.. code-block:: python

   # Old
   series.rolling_vol(window=30, method='std')

   # New
   series.rolling_vol(window=30)  # method parameter removed

From 1.6.x to 1.7.x
~~~~~~~~~~~~~~~~~~~~

**Import Changes**

Some imports were reorganized:

.. code-block:: python

   # Old
   from openseries.utils import some_function

   # New
   from openseries import some_function

Deprecation Notices
-------------------

**Deprecated in 1.9.x**

- ``old_method_name()``: Use ``new_method_name()`` instead
- Legacy configuration options: Will be removed in 2.0.0

**Removed in 1.9.x**

- ``deprecated_function()``: Removed, use ``replacement_function()``
- Old-style configuration: No longer supported

Future Plans
------------

Version 2.0.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~

**Major Changes Planned**

- Enhanced performance with optional Rust backend
- Improved memory efficiency for large datasets
- Streamlined API with breaking changes
- Enhanced visualization with modern plotting libraries

**New Features Planned**

- Advanced risk models
- Machine learning integration
- Real-time data streaming
- Enhanced portfolio optimization algorithms

**Timeline**

Version 2.0.0 is planned for release in 2025, with beta versions available in late 2024.

Version 1.10.x Series
~~~~~~~~~~~~~~~~~~~~~

**Planned Features**

- Enhanced statistical analysis
- Better integration with cloud data sources
- Improved performance monitoring
- Extended visualization options

Contributing to Releases
------------------------

**Feature Requests**

Submit feature requests through GitHub Issues with:

- Clear description of the proposed feature
- Use cases and examples
- Potential implementation approach

**Bug Reports**

Report bugs with:

- Steps to reproduce
- Expected vs. actual behavior
- Environment details
- Minimal code example

**Testing Beta Versions**

Help test pre-release versions:

.. code-block:: bash

   pip install --pre openseries

**Release Notes**

Each release includes detailed notes covering:

- New features and improvements
- Bug fixes and performance enhancements
- Breaking changes and migration guidance
- Updated dependencies and requirements

Stay Updated
------------

**GitHub Releases**

Watch the `openseries repository <https://github.com/CaptorAB/openseries>`_ for release notifications.

**PyPI**

Monitor `openseries on PyPI <https://pypi.org/project/openseries/>`_ for new versions.

**Conda-forge**

Track updates on `conda-forge <https://anaconda.org/conda-forge/openseries>`_.

**Changelog Format**

This changelog follows the `Keep a Changelog <https://keepachangelog.com/>`_ format and openseries adheres to `Semantic Versioning <https://semver.org/>`_.

**Categories**

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Now removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes
