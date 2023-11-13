"""Test suite for the openseries/simulation.py module."""
from __future__ import annotations

import datetime as dt
from copy import copy
from typing import Union
from unittest import TestCase

from pandas import DataFrame

from openseries.series import OpenTimeSeries
from openseries.simulation import (
    ReturnSimulation,
    random_generator,
)
from openseries.types import ValueType
from tests.test_common_sim import SEED, SIMS


class TestSimulation(TestCase):

    """class to run unittests on the module simulation.py."""

    seriesim: ReturnSimulation
    framesim: ReturnSimulation

    @classmethod
    def setUpClass(cls: type[TestSimulation]) -> None:
        """SetUpClass for the TestSimulation class."""
        cls.seriesim = SIMS

    def test_init_with_without_randomizer(self: TestSimulation) -> None:
        """Test instantiating ReturnSimulation with & without random generator."""
        sim_without = ReturnSimulation(
            number_of_sims=1,
            trading_days=2512,
            mean_annual_return=0.05,
            mean_annual_vol=0.1,
            trading_days_in_year=252,
            dframe=DataFrame(),
            seed=SEED,
        )
        if not isinstance(sim_without, ReturnSimulation):
            msg = "ReturnSimulation object not instantiated as expected"
            raise TypeError(msg)

        sim_with = ReturnSimulation(
            number_of_sims=1,
            trading_days=2512,
            mean_annual_return=0.05,
            mean_annual_vol=0.1,
            trading_days_in_year=252,
            dframe=DataFrame(),
            randomizer=random_generator(seed=SEED),
        )
        if not isinstance(sim_with, ReturnSimulation):
            msg = "ReturnSimulation object not instantiated as expected"
            raise TypeError(msg)

    def test_processes(self: TestSimulation) -> None:
        """Test ReturnSimulation based on different stochastic processes."""
        args: dict[str, Union[int, float]] = {
            "number_of_sims": 1,
            "trading_days": 2520,
            "mean_annual_return": 0.05,
            "mean_annual_vol": 0.2,
            "seed": SEED,
        }
        methods = [
            "from_normal",
            "from_lognormal",
            "from_gbm",
            "from_merton_jump_gbm",
            "from_merton_jump_gbm",
        ]
        added: list[dict[str, Union[int, float]]] = [
            {},
            {},
            {},
            {"jumps_lamda": 0.0},
            {"jumps_lamda": 0.3, "jumps_sigma": 0.2, "jumps_mu": -0.2},
        ]
        intended_returns = [
            "-0.005640734",
            "0.013058925",
            "-0.025640734",
            "-0.025640734",
            "-0.011505208",
        ]

        intended_volatilities = [
            "0.193403252",
            "0.193487832",
            "0.193403252",
            "0.193403252",
            "0.211446536",
        ]

        returns = []
        volatilities = []
        for method, adding in zip(methods, added):
            arguments = {**args, **adding}
            onesim = getattr(ReturnSimulation, method)(**arguments)
            returns.append(f"{onesim.realized_mean_return:.9f}")
            volatilities.append(f"{onesim.realized_vol:.9f}")

        if intended_returns != returns:
            msg = "Unexpected calculation result"
            raise ValueError(msg)
        if intended_volatilities != volatilities:
            msg = "Unexpected calculation result"
            raise ValueError(msg)

    def test_properties(self: TestSimulation) -> None:
        """Test ReturnSimulation properties output."""
        days = 2512
        psim = copy(self.seriesim)

        if psim.results.shape[0] != days:
            msg = "Unexpected result"
            raise ValueError(msg)

        if f"{psim.realized_mean_return:.9f}" != "0.014773538":
            msg = f"Unexpected result: '{psim.realized_mean_return:.9f}'"
            raise ValueError(msg)

        if f"{psim.realized_vol:.9f}" != "0.096761956":
            msg = f"Unexpected result: '{psim.realized_vol:.9f}'"
            raise ValueError(msg)

    def test_to_dataframe(self: TestSimulation) -> None:
        """Test method to_dataframe."""
        trading_days = 2512
        one = 1
        seriesim = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=one,
            trading_days=trading_days,
            mean_annual_return=0.05,
            mean_annual_vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            trading_days_in_year=252,
            seed=SEED,
        )
        five = 5
        framesim = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=five,
            trading_days=trading_days,
            mean_annual_return=0.05,
            mean_annual_vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            trading_days_in_year=252,
            seed=SEED,
        )

        start = dt.date(2009, 6, 30)

        onedf = seriesim.to_dataframe(name="Asset", start=start)
        fivedf = framesim.to_dataframe(name="Asset", start=start)

        returnseries = OpenTimeSeries.from_df(onedf)
        startseries = returnseries.from_deepcopy()
        startseries.to_cumret()

        if onedf.shape != (trading_days, one):
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)

        if fivedf.shape != (trading_days, five):
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)

        if startseries.valuetype != ValueType.PRICE:
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)

        if startseries.valuetype != ValueType.PRICE:
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)
