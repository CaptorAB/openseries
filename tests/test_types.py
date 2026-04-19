"""Test suite for the openseries/owntypes.py module."""

from __future__ import annotations

from typing import Any, cast

import pytest

from openseries.owntypes import (
    LiteralFrameProps,
    LiteralSeriesProps,
    OpenFramePropertiesList,
    OpenTimeSeriesPropertiesList,
    PropertiesInputValidationError,
)


class TestTypes:
    """class to run tests on the module owntypes.py."""

    def test_opentimeseriesproplist_validate(self: TestTypes) -> None:
        """Test that the OpenTimeSeries property input is correctly checked."""
        subset: tuple[LiteralSeriesProps, ...] = (
            "z_score",
            "kurtosis",
            "positive_share",
        )
        lst = OpenTimeSeriesPropertiesList(*subset)
        msg = "A OpenTimeSeriesPropertiesList was not produced"
        if not isinstance(lst, OpenTimeSeriesPropertiesList):
            raise TypeError(msg)

        bad_invalid = cast("tuple[Any, ...]", ("z_score", "boo", "positive_share"))
        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=r"Invalid string\(s\): \['boo'\]\.",
        ):
            OpenTimeSeriesPropertiesList(*bad_invalid)

        bad_duplicates = cast(
            "tuple[Any, ...]",
            ("z_score", "skew", "skew", "positive_share"),
        )
        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=r"Duplicate string\(s\): \['skew'\]\.",
        ):
            OpenTimeSeriesPropertiesList(*bad_duplicates)

        bad_both = cast("tuple[Any, ...]", ("z_score", "skew", "skew", "boo"))
        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=(
                r"(?s)(?=.*Invalid string\(s\): \['boo'\])"
                r"(?=.*Duplicate string\(s\): \['skew'\])"
            ),
        ):
            OpenTimeSeriesPropertiesList(*bad_both)

    def test_openframeproplist_validate(self: TestTypes) -> None:
        """Test that the OpenFrame property input is correctly checked."""
        subset: tuple[LiteralFrameProps, ...] = (
            "z_score",
            "kurtosis",
            "positive_share",
        )
        lst = OpenFramePropertiesList(*subset)
        msg = "A OpenFramePropertiesList was not produced"
        if not isinstance(lst, OpenFramePropertiesList):
            raise TypeError(msg)

        bad_invalid = cast("tuple[Any, ...]", ("z_score", "boo", "positive_share"))
        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=r"Invalid string\(s\): \['boo'\]\.",
        ):
            OpenFramePropertiesList(*bad_invalid)

        bad_duplicates = cast(
            "tuple[Any, ...]",
            ("z_score", "skew", "skew", "positive_share"),
        )
        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=r"Duplicate string\(s\): \['skew'\]\.",
        ):
            OpenFramePropertiesList(*bad_duplicates)

        bad_both = cast("tuple[Any, ...]", ("z_score", "skew", "skew", "boo"))
        with pytest.raises(
            expected_exception=PropertiesInputValidationError,
            match=(
                r"(?s)(?=.*Invalid string\(s\): \['boo'\])"
                r"(?=.*Duplicate string\(s\): \['skew'\])"
            ),
        ):
            OpenFramePropertiesList(*bad_both)
