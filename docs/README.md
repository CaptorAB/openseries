# openseries Documentation

This directory contains the documentation for the openseries project, built using [Sphinx](https://www.sphinx-doc.org/).

## Building the Documentation

### Prerequisites

Install the documentation dependencies using Poetry:

```bash
poetry install --with docs
```

### Building HTML Documentation

To build the HTML documentation:

```bash
make builddocs
# or directly with Poetry:
poetry run sphinx-build -b html source build/html
```

The built documentation will be available in `build/html/index.html`.

### Other Build Targets

- `make clean` - Remove build artifacts
- `make linkcheck` - Check for broken links
- `make livehtml` - Build and serve with auto-reload (requires sphinx-autobuild)
- `make strict` - Build with warnings as errors

### Development

For development with auto-reload:

```bash
make servedocs
# or directly with Poetry:
poetry run sphinx-autobuild source build/html --host 127.0.0.1 --port 8000
```

This will start a local server at `http://localhost:8000` that automatically rebuilds when files change.

## Documentation Structure

```
docs/
├── source/
│   ├── index.rst                 # Main documentation index
│   ├── conf.py                   # Sphinx configuration
│   ├── api/                      # API reference documentation
│   │   ├── openseries.rst
│   │   ├── series.rst
│   │   ├── frame.rst
│   │   └── ...
│   ├── user_guide/               # User guide documentation
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   ├── core_concepts.rst
│   │   └── data_handling.rst
│   ├── tutorials/                # Tutorial documentation
│   │   ├── basic_analysis.rst
│   │   ├── portfolio_analysis.rst
│   │   ├── risk_management.rst
│   │   └── advanced_features.rst
│   ├── examples/                 # Example documentation
│   │   ├── single_asset.rst
│   │   ├── multi_asset.rst
│   │   ├── portfolio_optimization.rst
│   │   └── custom_reports.rst
│   ├── development/              # Development documentation
│   │   ├── contributing.rst
│   │   └── changelog.rst
│   ├── _static/                  # Static files (CSS, images)
│   └── _templates/               # Custom templates
├── build/                        # Built documentation (generated)
├── Makefile                      # Build commands (Unix)
└── make.bat                      # Build commands (Windows)
```

## ReadTheDocs Integration

This documentation is configured for [ReadTheDocs](https://readthedocs.org/) hosting:

- Configuration: `.readthedocs.yaml` in the project root
- Dependencies: Managed through Poetry in `pyproject.toml` docs group
- Build process: Automated on ReadTheDocs using Poetry

## Writing Documentation

### reStructuredText (RST)

Most documentation is written in reStructuredText format. Key syntax:

```rst
Title
=====

Subtitle
--------

**Bold text** and *italic text*

- Bullet points
- Another point

1. Numbered lists
2. Another item

.. code-block:: python

   # Python code example
   from openseries import OpenTimeSeries
   series = OpenTimeSeries.from_arrays(dates, values)

.. note::
   This is a note admonition.

.. warning::
   This is a warning admonition.
```

### API Documentation

API documentation is automatically generated from docstrings using Sphinx autodoc:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Example function with Google-style docstring.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 0.

    Returns:
        Description of return value.

    Raises:
        ValueError: If param1 is empty.

    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    return len(param1) > param2
```

### Cross-References

Link to other parts of the documentation:

```rst
:doc:`installation`                    # Link to installation.rst
:ref:`section-label`                   # Link to labeled section
:class:`openseries.OpenTimeSeries`     # Link to class
:meth:`OpenTimeSeries.from_df`         # Link to method
:func:`openseries.timeseries_chain`    # Link to function
```

## Style Guide

- Use clear, concise language
- Include practical examples
- Add code examples for all public APIs
- Use consistent formatting and structure
- Test all code examples

## Contributing

When contributing to documentation:

1. Follow the existing structure and style
2. Test your changes by building locally
3. Check for broken links with `make linkcheck`
4. Ensure examples work with current openseries version
5. Update the changelog if adding new sections

## Troubleshooting

### Common Build Issues

**Import errors during build:**

- Ensure all dependencies are installed
- Check that the openseries package is importable
- Verify Python path configuration in `conf.py`

**Missing modules:**

- Install missing dependencies: `poetry install --with docs`
- For ReadTheDocs builds, dependencies are managed through Poetry in `pyproject.toml`

**Broken links:**

- Run `make linkcheck` to identify broken links
- Update or remove broken external links
- Fix internal cross-references

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Review existing documentation files for examples
- Ask questions in the project's GitHub discussions
