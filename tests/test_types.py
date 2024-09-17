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

        msg2 = "OpenTimeSeriesPropertiesList._validate not raising expected Exception"
        with pytest.raises(expected_exception=ValueError) as exc:
            OpenTimeSeriesPropertiesList(
                *cast(LiteralSeriesProps, ["z_score", "boo", "positive_share"]),
            )
        if "Invalid string(s): ['boo']." not in str(exc.value):
            raise ValueError(msg2)

        with pytest.raises(expected_exception=ValueError) as excc:
            OpenTimeSeriesPropertiesList(
                *cast(
                    LiteralSeriesProps,
                    ["z_score", "skew", "skew", "positive_share"],
                ),
            )
        if "Duplicate string(s): ['skew']." not in str(excc.value):
            raise ValueError(msg2)

        with pytest.raises(expected_exception=ValueError) as exccc:
            OpenTimeSeriesPropertiesList(
                *cast(
                    LiteralSeriesProps,
                    ["z_score", "skew", "skew", "boo"],
                ),
            )
        if "Duplicate string(s): ['skew']." not in str(
            exccc.value,
        ) and "Invalid string(s): ['boo']." not in str(exccc.value):
            raise ValueError(msg2)

    def test_openframeproplist_validate(self: TestTypes) -> None:
        """Test that the OpenFrame property input is correctly checked."""
        subset = cast(LiteralFrameProps, ["z_score", "kurtosis", "positive_share"])
        lst = OpenFramePropertiesList(*subset)
        msg = "A OpenFramePropertiesList was not produced"
        if not isinstance(lst, OpenFramePropertiesList):
            raise TypeError(msg)

        msg2 = "OpenFramePropertiesList._validate not raising expected Exception"
        with pytest.raises(expected_exception=ValueError) as exc:
            OpenFramePropertiesList(
                *cast(LiteralFrameProps, ["z_score", "boo", "positive_share"]),
            )
        if "Invalid string(s): ['boo']." not in str(exc.value):
            raise ValueError(msg2)

        with pytest.raises(expected_exception=ValueError) as excc:
            OpenFramePropertiesList(
                *cast(
                    LiteralFrameProps,
                    ["z_score", "skew", "skew", "positive_share"],
                ),
            )
        if "Duplicate string(s): ['skew']." not in str(excc.value):
            raise ValueError(msg2)

        with pytest.raises(expected_exception=ValueError) as exccc:
            OpenFramePropertiesList(
                *cast(
                    LiteralFrameProps,
                    ["z_score", "skew", "skew", "boo"],
                ),
            )
        if "Duplicate string(s): ['skew']." not in str(
            exccc.value,
        ) and "Invalid string(s): ['boo']." not in str(exccc.value):
            raise ValueError(msg2)
