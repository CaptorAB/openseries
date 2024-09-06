"""Test suite for the openseries/types.py module."""

from __future__ import annotations

from typing import cast

import pytest

from openseries.types import (
    LiteralFrameProps,
    LiteralSeriesProps,
    OpenFramePropertiesList,
    OpenTimeSeriesPropertiesList,
)


class TestTypes:
    """class to run tests on the module types.py."""

    def test_opentimeseriesproplist_validate(self: TestTypes) -> None:
        """Test that the OpenTimeSeries property input is correctly checked."""
        subset = cast(LiteralSeriesProps, ["z_score", "kurtosis", "positive_share"])
        lst = OpenTimeSeriesPropertiesList(*subset)
        msg = "A OpenTimeSeriesPropertiesList was not produced"
        if not isinstance(lst, OpenTimeSeriesPropertiesList):
            raise TypeError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Invalid string",
        ):
            OpenTimeSeriesPropertiesList(
                *cast(LiteralSeriesProps, ["z_score", "boo", "positive_share"]),
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Duplicate string",
        ):
            OpenTimeSeriesPropertiesList(
                *cast(
                    LiteralSeriesProps,
                    ["z_score", "skew", "skew", "positive_share"],
                ),
            )

    def test_openframeproplist_validate(self: TestTypes) -> None:
        """Test that the OpenFrame property input is correctly checked."""
        subset = cast(LiteralFrameProps, ["z_score", "kurtosis", "positive_share"])
        lst = OpenFramePropertiesList(*subset)
        msg = "A OpenFramePropertiesList was not produced"
        if not isinstance(lst, OpenFramePropertiesList):
            raise TypeError(msg)

        with pytest.raises(expected_exception=ValueError, match="Invalid string"):
            OpenFramePropertiesList(
                *cast(LiteralFrameProps, ["z_score", "boo", "positive_share"]),
            )

        with pytest.raises(expected_exception=ValueError, match="Duplicate string"):
            OpenFramePropertiesList(
                *cast(
                    LiteralFrameProps,
                    ["z_score", "skew", "skew", "positive_share"],
                ),
            )
