# openseries Documentation

This directory contains the documentation for the openseries project, built using [Sphinx](https://www.sphinx-doc.org/).

## Building the Documentation

### Prerequisites

From the repository root, install the documentation dependencies from the
lockfile (requires the pinned uv version used in `Makefile` / CI):

```bash
uv sync --locked --extra docs
```

Alternatively, `make install` installs the `dev` and `docs` extras together.

### Building HTML Documentation

To build the HTML documentation:

```bash
make builddocs
# or directly:
sphinx-build -b html source build/html
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
# or directly:
sphinx-autobuild source build/html --host 127.0.0.1 --port 8000
```

This will start a local server at `http://localhost:8000` that automatically rebuilds when files change.

## Documentation Structure

```text
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

## Hosting

Canonical documentation is [https://openseries.readthedocs.io/](https://openseries.readthedocs.io/)
(the URL published on PyPI and conda-forge). The same Sphinx site is also
deployed to [GitHub Pages](https://captorab.github.io/openseries/) by
`.github/workflows/docs.yml` (Homepage in package metadata).

- Pull requests that touch documentation sources or library code build docs
  in GitHub Actions (warnings as errors) without deploying Pages
- Pushes to `master` and manual `workflow_dispatch` (including after a PyPI
  release) build and deploy Pages
- Read the Docs builds `latest` from `master` using `.readthedocs.yaml`;
  disable **Build pull requests** in the RTD dashboard so PRs are not also
  built there
- Dependencies come from the `docs` extra in `pyproject.toml` / `uv.lock`.
  Read the Docs installs that extra via `.readthedocs.yaml`
  (`pip install .[docs]`)

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

- Install missing dependencies: `uv sync --locked --extra docs`

**Broken links:**

- Run `make linkcheck` to identify broken links
- Update or remove broken external links
- Fix internal cross-references

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Review existing documentation files for examples
- Ask questions in the project's GitHub discussions
