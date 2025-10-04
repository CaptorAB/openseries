"""Test suite for the openseries/owntypes.py module."""

from __future__ import annotations

import pytest

from openseries.owntypes import (
    OpenFramePropertiesList,
    OpenTimeSeriesPropertiesList,
    PropertiesInputValidationError,
)


class TestTypes:
    """class to run tests on the module owntypes.py."""

    def test_opentimeseriesproplist_validate(self: TestTypes) -> None:
        """Test that the OpenTimeSeries property input is correctly checked."""
        subset = ["z_score", "kurtosis", "positive_share"]
        lst = OpenTimeSeriesPropertiesList(*subset)  # type: ignore[arg-type]
        msg = "A OpenTimeSeriesPropertiesList was not produced"
        if not isinstance(lst, OpenTimeSeriesPropertiesList):
            raise TypeError(msg)

        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=r"Invalid string\(s\): \['boo'\]\.",
        ):
            OpenTimeSeriesPropertiesList(*["z_score", "boo", "positive_share"])  # type: ignore[arg-type]

        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=r"Duplicate string\(s\): \['skew'\]\.",
        ):
            OpenTimeSeriesPropertiesList(
                *["z_score", "skew", "skew", "positive_share"],  # type: ignore[arg-type]
            )

        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=(
                r"(?s)(?=.*Invalid string\(s\): \['boo'\])"
                r"(?=.*Duplicate string\(s\): \['skew'\])"
            ),
        ):
            OpenTimeSeriesPropertiesList(*["z_score", "skew", "skew", "boo"])  # type: ignore[arg-type]

    def test_openframeproplist_validate(self: TestTypes) -> None:
        """Test that the OpenFrame property input is correctly checked."""
        subset = ["z_score", "kurtosis", "positive_share"]
        lst = OpenFramePropertiesList(*subset)  # type: ignore[arg-type]
        msg = "A OpenFramePropertiesList was not produced"
        if not isinstance(lst, OpenFramePropertiesList):
            raise TypeError(msg)

        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=r"Invalid string\(s\): \['boo'\]\.",
        ):
            OpenFramePropertiesList(*["z_score", "boo", "positive_share"])  # type: ignore[arg-type]

        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=r"Duplicate string\(s\): \['skew'\]\.",
        ):
            OpenFramePropertiesList(*["z_score", "skew", "skew", "positive_share"])  # type: ignore[arg-type]

        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=(
                r"(?s)(?=.*Invalid string\(s\): \['boo'\])"
                r"(?=.*Duplicate string\(s\): \['skew'\])"
            ),
        ):
            OpenFramePropertiesList(*["z_score", "skew", "skew", "boo"])  # type: ignore[arg-type]
