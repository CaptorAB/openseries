Contributing to openseries
==========================

We welcome contributions to openseries! This guide will help you get started with contributing to the project.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/yourusername/openseries.git
   cd openseries

3. Install Poetry (if not already installed):

.. code-block:: bash

   curl -sSL https://install.python-poetry.org | python3 -

4. Install dependencies:

.. code-block:: bash

   poetry install

5. Activate the virtual environment:

.. code-block:: bash

   poetry shell

6. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. Create a new branch for your feature or bug fix:

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. Make your changes
3. Run tests to ensure everything works:

.. code-block:: bash

   make test

4. Run linting and type checking:

.. code-block:: bash

   make lint

5. Commit your changes:

.. code-block:: bash

   git add .
   git commit -m "Add your descriptive commit message"

6. Push to your fork:

.. code-block:: bash

   git push origin feature/your-feature-name

7. Create a pull request on GitHub

Code Standards
--------------

Code Style
~~~~~~~~~~

openseries uses several tools to maintain code quality:

- **Ruff**: For linting and code formatting
- **mypy**: For static type checking
- **pre-commit**: For automated checks before commits

The configuration for these tools is in ``pyproject.toml``.

Type Hints
~~~~~~~~~~

All new code should include proper type hints:

.. code-block:: python

    def calculate_returns(prices: list[float]) -> list[float]:
         """Calculate simple returns from prices."""
         returns = []
         for i in range(1, len(prices)):
              ret = (prices[i] / prices[i-1]) - 1
              returns.append(ret)
         return returns

Docstrings
~~~~~~~~~~

Use Google-style docstrings for all public functions and classes:

.. code-block:: python

    def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
         """Calculate the Sharpe ratio.

         Args:
              returns: List of periodic returns.
              risk_free_rate: Risk-free rate for the same period. Defaults to 0.0.

         Returns:
              The Sharpe ratio.

         Raises:
              ValueError: If returns list is empty.

         Example:
              >>> returns = [0.01, 0.02, -0.01, 0.03]
              >>> sharpe = calculate_sharpe_ratio(returns)
              >>> print(f"Sharpe ratio: {sharpe:.3f}")
         """
         if not returns:
              raise ValueError("Returns list cannot be empty")

         mean_return = sum(returns) / len(returns)
         std_dev = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5

         if std_dev == 0:
              return 0.0

         return (mean_return - risk_free_rate) / std_dev

Testing
-------

Test Structure
~~~~~~~~~~~~~~

Tests are located in the ``tests/`` directory and use pytest:

.. code-block:: bash

   tests/
   ├── __init__.py
   ├── test_series.py
   ├── test_frame.py
   ├── test_portfoliotools.py
   └── ...

Writing Tests
~~~~~~~~~~~~~

Write comprehensive tests for new functionality:

.. code-block:: python

    import pytest
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from openseries import OpenTimeSeries

    class TestOpenTimeSeries:
         """Test cases for OpenTimeSeries class."""

         def test_from_arrays_basic(self):
              """Test basic creation from arrays."""
              dates = ['2023-01-01', '2023-01-02', '2023-01-03']
              values = [100.0, 102.0, 99.0]

              series = OpenTimeSeries.from_arrays(dates=dates, values=values, name="Test")

              if series.label != "Test":
                    msg = f"Expected name 'Test', got '{series.label}'"
                    raise ValueError(msg)
              if series.length != 3:
                    msg = f"Expected length 3, got {series.length}"
                    raise ValueError(msg)
              if series.first_idx != pd.Timestamp('2023-01-01').date():
                    msg = f"Expected first_idx 2023-01-01, got {series.first_idx}"
                    raise ValueError(msg)
              if series.last_idx != pd.Timestamp('2023-01-03').date():
                    msg = f"Expected last_idx 2023-01-03, got {series.last_idx}"
                    raise ValueError(msg)

         def test_from_arrays_invalid_dates(self):
              """Test that invalid dates raise appropriate errors."""
              with pytest.raises(ValueError):
                    OpenTimeSeries.from_arrays(
                         dates=['invalid-date'],
                         values=[100.0],
                         name="Test"
                    )

         def test_calculate_returns(self):
              """Test return calculation."""
              dates = ['2023-01-01', '2023-01-02', '2023-01-03']
              values = [100.0, 102.0, 99.0]

              series = OpenTimeSeries.from_arrays(dates=dates, values=values, name="Test")
              series.value_to_ret()  # Modifies original

              expected_returns = [0.02, -0.0294117647]  # Approximate
              actual_returns = series.values

              if len(actual_returns) != 2:
                    msg = f"Expected 2 returns, got {len(actual_returns)}"
                    raise ValueError(msg)
              # Use tolerance-based comparison
              if abs(actual_returns[0] - expected_returns[0]) >= 1e-6:
                    msg = f"First return mismatch: {actual_returns[0]} vs {expected_returns[0]}"
                    raise ValueError(msg)
              if abs(actual_returns[1] - expected_returns[1]) >= 1e-6:
                    msg = f"Second return mismatch: {actual_returns[1]} vs {expected_returns[1]}"
                    raise ValueError(msg)

Running Tests
~~~~~~~~~~~~~

Run all tests:

.. code-block:: bash

   make test

Run specific test files:

.. code-block:: bash

   pytest tests/test_series.py

Run tests with coverage:

.. code-block:: bash

   pytest --cov=openseries tests/

Test Coverage
~~~~~~~~~~~~~

openseries maintains high test coverage (>99%). New code should include comprehensive tests:

- Test normal use cases
- Test edge cases
- Test error conditions
- Test with different data types and sizes

Documentation
-------------

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

- All public APIs must be documented
- Include examples in docstrings where helpful
- Update relevant documentation files when adding features
- Use clear, concise language

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

To build documentation locally:

.. code-block:: bash

   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.

Contributing Guidelines
-----------------------

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. **Fork and Branch**: Create a feature branch from ``master``
2. **Develop**: Make your changes with tests and documentation
3. **Test**: Ensure all tests pass and coverage remains high
4. **Lint**: Run linting and fix any issues
5. **Document**: Update documentation as needed
6. **Commit**: Use clear, descriptive commit messages
7. **Pull Request**: Create a PR with a clear description

Commit Messages
~~~~~~~~~~~~~~~

Use clear, descriptive commit messages:

.. code-block:: text

   Add support for custom business day calendars

   - Implement custom calendar functionality in datefixer module
   - Add tests for various calendar configurations
   - Update documentation with examples
   - Fixes #123

Code Review Process
~~~~~~~~~~~~~~~~~~~

All contributions go through code review:

1. Automated checks must pass (tests, linting, type checking)
2. At least one maintainer review is required
3. Address any feedback or requested changes
4. Once approved, the PR will be merged

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, OS, etc.)
- Minimal code example if possible

Feature Requests
~~~~~~~~~~~~~~~~

For new features:

- Describe the use case and motivation
- Provide examples of how it would be used
- Consider backward compatibility
- Discuss implementation approach if you have ideas

Code Contributions
~~~~~~~~~~~~~~~~~~

Areas where contributions are especially welcome:

- **New financial metrics**: Additional risk measures, performance ratios
- **Data sources**: Integration with new data providers
- **Visualization**: Enhanced plotting capabilities
- **Performance**: Optimization of calculations
- **Documentation**: Examples, tutorials, API documentation

Documentation Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples to existing documentation
- Create new tutorials or guides
- Improve API documentation

Development Environment
-----------------------

IDE Setup
~~~~~~~~~

For VS Code, recommended extensions:

- Python
- Pylance
- Ruff
- mypy

Recommended settings in ``.vscode/settings.json``:

.. code-block:: json

   {
       "python.defaultInterpreterPath": ".venv/bin/python",
       "python.linting.enabled": true,
       "python.linting.ruffEnabled": true,
       "python.formatting.provider": "ruff",
       "python.typeChecking": "strict"
   }

Debugging
~~~~~~~~~

For debugging tests:

.. code-block:: bash

   pytest --pdb tests/test_specific.py::test_function

For debugging with VS Code, create ``.vscode/launch.json``:

.. code-block:: json

   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Python: Current File",
               "type": "python",
               "request": "launch",
               "program": "${file}",
               "console": "integratedTerminal"
           },
           {
               "name": "Python: Pytest",
               "type": "python",
               "request": "launch",
               "module": "pytest",
               "args": ["${workspaceFolder}/tests"],
               "console": "integratedTerminal"
           }
       ]
   }

Release Process
---------------

openseries follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

Releases are managed by maintainers and include:

1. Version bump in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Create GitHub release with release notes
4. Publish to PyPI and conda-forge

Getting Help
------------

If you need help with contributing:

- Check existing issues and discussions on GitHub
- Ask questions in GitHub Discussions
- Reach out to maintainers

Community Guidelines
--------------------

openseries is committed to providing a welcoming and inclusive environment:

- Be respectful and constructive in all interactions
- Focus on what is best for the community
- Show empathy towards other community members
- Welcome newcomers and help them get started

Thank you for contributing to openseries!
