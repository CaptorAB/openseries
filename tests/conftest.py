"""Configuration of test suite to silence unrelated DeprecationWarning in plotly."""

import warnings

import pytest


@pytest.fixture(autouse=True)  # type: ignore[misc,unused-ignore]
def suppress_plotly_deprecation_warnings() -> None:
    """Silences unrelated DeprecationWarning in plotly."""
    warnings.filterwarnings(
        "ignore",
        message=".*scattermapbox.* is deprecated!",
        category=DeprecationWarning,
    )
