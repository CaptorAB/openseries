"""Defining simulated data used in test suite."""

from __future__ import annotations

import datetime as dt
from unittest import TestCase

from openseries.frame import OpenFrame
from openseries.owntypes import ValueType
from openseries.series import OpenTimeSeries
from openseries.simulation import ReturnSimulation


class CommonTestCase(TestCase):
    """base class to run project tests."""

    seed: int
    seriesim: ReturnSimulation
    randomframe: OpenFrame
    randomseries: OpenTimeSeries
    random_properties: dict[str, dt.date | int | float]

    @classmethod
    def setUpClass(cls: type[CommonTestCase]) -> None:
        """SetUpClass for the CommonTestCase class."""
        seed = 71
        end_date = dt.date(2019, 6, 30)

        seriesim = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=5,
            trading_days=2512,
            mean_annual_return=0.05,
            mean_annual_vol=0.1,
            jumps_lamda=0.1,
            jumps_sigma=0.3,
            jumps_mu=-0.2,
            trading_days_in_year=252,
            seed=seed,
        )

        randomseries = OpenTimeSeries.from_df(
            dframe=seriesim.to_dataframe(name="Asset", end=end_date),
            valuetype=ValueType.RTRN,
        ).to_cumret()

        cls.seed = seed
        cls.seriesim = seriesim
        cls.randomseries = randomseries.from_deepcopy()
        cls.randomframe = OpenFrame(
            [
                OpenTimeSeries.from_df(
                    dframe=seriesim.to_dataframe(name="Asset", end=end_date),
                    column_nmbr=serie,
                    valuetype=ValueType.RTRN,
                )
                for serie in range(seriesim.number_of_sims)
            ],
        )
        cls.random_properties = randomseries.all_properties().to_dict()[
            ("Asset_0", ValueType.PRICE)
        ]

    @classmethod
    def tearDownClass(cls: type[CommonTestCase]) -> None:
        """TearDownClass for the CommonTestCase class."""
        del cls.seed
        del cls.seriesim
        del cls.randomseries
        del cls.randomframe
        del cls.random_properties
