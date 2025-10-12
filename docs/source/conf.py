# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""Sphinx configuration for openseries documentation."""

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path
import re

# Add the project root to the path
sys.path.insert(0, str(Path("../../").resolve()))


# Read version from pyproject.toml
def get_version_from_pyproject():
    """Extract version from pyproject.toml using regex (Python 3.10+ compatible)."""
    # Get the absolute path to pyproject.toml
    current_dir = Path(__file__).parent
    pyproject_path = current_dir.parent.parent / "pyproject.toml"

    with pyproject_path.open(mode="r", encoding="utf-8") as f:
        content = f.read()

    # Look for version = "x.y.z" pattern
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)

    # Fail explicitly if version cannot be parsed
    raise RuntimeError(
        f"Could not parse version from {pyproject_path}. "
        "Expected pattern: version = 'x.y.z'"
    )


project = "openseries"
copyright = "Captor Fund Management AB"
author = "Martin Karrin"
release = get_version_from_pyproject()
version = get_version_from_pyproject()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_rtd_theme",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases: dict[str, str] | None = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "no-index": True,
}

# Handle properties correctly
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Suppress specific warnings
suppress_warnings = [
    "autodoc.failed_to_get_signature",
    "autodoc.attribute",
    "autodoc.property",
    "autodoc.method",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "custom.css",
]

# The master toctree document.
master_doc = "index"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# Source file suffixes
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Custom substitutions for dynamic content
# Using GitHub API approach instead of Sphinx substitutions
