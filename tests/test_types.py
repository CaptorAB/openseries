from unittest import TestCase

from openseries.types import (
    OpenTimeSeriesPropertiesList,
    OpenFramePropertiesList,
)


class TestTypes(TestCase):
    def test_types_opentimeseriesproplist_validate(self: "TestTypes") -> None:
        lst = OpenTimeSeriesPropertiesList(
            "z_score",
            "kurtosis",
            "positive_share",
        )
        self.assertIsInstance(lst, OpenTimeSeriesPropertiesList)

        with self.assertRaises(ValueError) as e_boo:
            OpenTimeSeriesPropertiesList(
                "z_score",
                "boo",
                "positive_share",
            )
        self.assertIsInstance(e_boo.exception, ValueError)
        self.assertIn(member="Invalid string: boo", container=str(e_boo.exception))

        with self.assertRaises(ValueError) as e_booo:
            OpenTimeSeriesPropertiesList(
                "z_score",
                "skew",
                "skew",
                "positive_share",
            )
        self.assertIsInstance(e_booo.exception, ValueError)
        self.assertIn(member="Duplicate string: skew", container=str(e_booo.exception))

    def test_types_openframeproplist_validate(self: "TestTypes") -> None:
        lst = OpenFramePropertiesList(
            "z_score",
            "kurtosis",
            "positive_share",
        )
        self.assertIsInstance(lst, OpenFramePropertiesList)

        with self.assertRaises(ValueError) as e_boo:
            OpenFramePropertiesList(
                "z_score",
                "boo",
                "positive_share",
            )
        self.assertIsInstance(e_boo.exception, ValueError)
        self.assertIn(member="Invalid string: boo", container=str(e_boo.exception))

        with self.assertRaises(ValueError) as e_booo:
            OpenFramePropertiesList(
                "z_score",
                "skew",
                "skew",
                "positive_share",
            )
        self.assertIsInstance(e_booo.exception, ValueError)
        self.assertIn(member="Duplicate string: skew", container=str(e_booo.exception))
