"""Test suite for the openseries/simulation.py module."""

from __future__ import annotations

import datetime as dt
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

    @classmethod
    def setUpClass(cls: type[TestSimulation]) -> None:
        """SetUpClass for the TestSimulation class."""
        cls.seriesim = SIMS

    def test_init_with_without_randomizer(self: TestSimulation) -> None:
        """Test instantiating ReturnSimulation with & without random generator."""
        sim_without = ReturnSimulation(
            number_of_sims=1,
            trading_days=SIMS.trading_days,
            mean_annual_return=SIMS.mean_annual_return,
            mean_annual_vol=SIMS.mean_annual_vol,
            trading_days_in_year=SIMS.trading_days_in_year,
            dframe=DataFrame(),
            seed=SEED,
        )
        if not isinstance(sim_without, ReturnSimulation):
            msg = "ReturnSimulation object not instantiated as expected"
            raise TypeError(msg)

        sim_with = ReturnSimulation(
            number_of_sims=1,
            trading_days=SIMS.trading_days,
            mean_annual_return=SIMS.mean_annual_return,
            mean_annual_vol=SIMS.mean_annual_vol,
            trading_days_in_year=SIMS.trading_days_in_year,
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
            "trading_days": SIMS.trading_days,
            "mean_annual_return": SIMS.mean_annual_return,
            "mean_annual_vol": SIMS.mean_annual_vol,
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
            {"jumps_lamda": SIMS.jumps_lamda},
            {
                "jumps_lamda": SIMS.jumps_lamda,
                "jumps_sigma": SIMS.jumps_sigma,
                "jumps_mu": SIMS.jumps_mu,
            },
        ]
        intended_returns = [
            "0.019523539",
            "0.024204850",
            "0.014523539",
            "0.014523539",
            "0.014523539",
        ]

        intended_volatilities = [
            "0.096761956",
            "0.096790015",
            "0.096761956",
            "0.096761956",
            "0.096761956",
        ]

        returns = []
        volatilities = []
        for method, adding in zip(methods, added):
            arguments = {**args, **adding}
            onesim = getattr(ReturnSimulation, method)(**arguments)
            returns.append(f"{onesim.realized_mean_return:.9f}")
            volatilities.append(f"{onesim.realized_vol:.9f}")

        if intended_returns != returns:
            msg = f"Unexpected returns result {returns}"
            raise ValueError(msg)
        if intended_volatilities != volatilities:
            msg = f"Unexpected volatilities result {volatilities}"
            raise ValueError(msg)

    def test_properties(self: TestSimulation) -> None:
        """Test ReturnSimulation properties output."""
        if self.seriesim.results.shape[0] != SIMS.trading_days:
            msg = "Unexpected result"
            raise ValueError(msg)

        if f"{self.seriesim.realized_mean_return:.9f}" != "0.014773538":
            msg = (
                "Unexpected return result: "
                f"'{self.seriesim.realized_mean_return:.9f}'"
            )
            raise ValueError(msg)

        if f"{self.seriesim.realized_vol:.9f}" != "0.096761956":
            msg = f"Unexpected volatility result: '{self.seriesim.realized_vol:.9f}'"
            raise ValueError(msg)

    def test_to_dataframe(self: TestSimulation) -> None:
        """Test method to_dataframe."""
        one = 1
        seriesim = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=one,
            trading_days=SIMS.trading_days,
            mean_annual_return=SIMS.mean_annual_return,
            mean_annual_vol=SIMS.mean_annual_vol,
            jumps_lamda=SIMS.jumps_lamda,
            jumps_sigma=SIMS.jumps_sigma,
            jumps_mu=SIMS.jumps_mu,
            trading_days_in_year=SIMS.trading_days_in_year,
            seed=SEED,
        )
        five = 5
        framesim = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=five,
            trading_days=SIMS.trading_days,
            mean_annual_return=SIMS.mean_annual_return,
            mean_annual_vol=SIMS.mean_annual_vol,
            jumps_lamda=SIMS.jumps_lamda,
            jumps_sigma=SIMS.jumps_sigma,
            jumps_mu=SIMS.jumps_mu,
            trading_days_in_year=SIMS.trading_days_in_year,
            seed=SEED,
        )

        start = dt.date(2009, 6, 30)

        onedf = seriesim.to_dataframe(name="Asset", start=start)
        fivedf = framesim.to_dataframe(name="Asset", start=start)

        returnseries = OpenTimeSeries.from_df(onedf)
        startseries = returnseries.from_deepcopy()
        startseries.to_cumret()

        if onedf.shape != (SIMS.trading_days, one):
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)

        if fivedf.shape != (SIMS.trading_days, five):
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)

        if returnseries.valuetype != ValueType.RTRN:
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)

        if startseries.valuetype != ValueType.PRICE:
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)
