"""Test suite for the openseries/simulation.py module.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from openseries.owntypes import ValueType
from openseries.series import OpenTimeSeries
from openseries.simulation import ReturnSimulation, _random_generator

from .test_common_sim import CommonTestCase

if TYPE_CHECKING:
    from numpy.random import Generator


class SimulationTestError(Exception):
    """Custom exception used for signaling test failures."""


class TestSimulation(CommonTestCase):
    """class to run tests on the module simulation.py."""

    def test_processes(self: TestSimulation) -> None:
        """Test ReturnSimulation based on different stochastic processes."""
        args: dict[str, int | float] = {
            "number_of_sims": 1,
            "trading_days": self.seriesim.trading_days,
            "mean_annual_return": self.seriesim.mean_annual_return,
            "mean_annual_vol": self.seriesim.mean_annual_vol,
            "seed": self.seed,
        }
        methods = [
            "from_normal",
            "from_normal",
            "from_lognormal",
            "from_lognormal",
            "from_gbm",
            "from_gbm",
            "from_merton_jump_gbm",
            "from_merton_jump_gbm",
        ]
        added: list[dict[str, int | float]] = [
            {},
            {
                "mean_annual_return": self.seriesim.mean_annual_return + 0.01,
                "mean_annual_vol": self.seriesim.mean_annual_vol + 0.01,
            },
            {},
            {
                "mean_annual_return": self.seriesim.mean_annual_return + 0.01,
                "mean_annual_vol": self.seriesim.mean_annual_vol + 0.01,
            },
            {},
            {
                "mean_annual_return": self.seriesim.mean_annual_return + 0.01,
                "mean_annual_vol": self.seriesim.mean_annual_vol + 0.01,
            },
            {"jumps_lamda": self.seriesim.jumps_lamda},
            {
                "jumps_lamda": self.seriesim.jumps_lamda + 0.1,
                "jumps_sigma": self.seriesim.jumps_sigma + 0.1,
                "jumps_mu": self.seriesim.jumps_mu + 0.1,
            },
        ]
        intended_returns = [
            "0.019523539",
            "0.026475893",
            "0.024204850",
            "0.032140993",
            "0.014523539",
            "0.020425893",
            "0.014523539",
            "0.051790043",
        ]

        intended_volatilities = [
            "0.096761956",
            "0.106438152",
            "0.096790015",
            "0.106474544",
            "0.096761956",
            "0.106438152",
            "0.096761956",
            "0.181849820",
        ]

        returns = []
        volatilities = []
        for method, adding in zip(methods, added, strict=True):
            arguments = {**args, **adding}
            onesim = getattr(ReturnSimulation, method)(**arguments)
            returns.append(f"{onesim.realized_mean_return:.9f}")
            volatilities.append(f"{onesim.realized_vol:.9f}")

        if intended_returns != returns:
            msg = f"Unexpected returns result {returns}"
            raise SimulationTestError(msg)
        if intended_volatilities != volatilities:
            msg = f"Unexpected volatilities result {volatilities}"
            raise SimulationTestError(msg)

    def test_processes_with_randomizer(self: TestSimulation) -> None:
        """Test ReturnSimulation with a random generator as input."""
        randomizer = _random_generator(self.seed)
        args: dict[str, int | float | Generator] = {
            "number_of_sims": 1,
            "trading_days": self.seriesim.trading_days,
            "mean_annual_return": self.seriesim.mean_annual_return,
            "mean_annual_vol": self.seriesim.mean_annual_vol,
            "randomizer": randomizer,
        }
        methods = [
            "from_normal",
            "from_normal",
            "from_lognormal",
            "from_lognormal",
            "from_gbm",
            "from_gbm",
            "from_merton_jump_gbm",
            "from_merton_jump_gbm",
        ]
        added: list[dict[str, int | float]] = [
            {},
            {
                "mean_annual_return": self.seriesim.mean_annual_return + 0.01,
                "mean_annual_vol": self.seriesim.mean_annual_vol + 0.01,
            },
            {},
            {
                "mean_annual_return": self.seriesim.mean_annual_return + 0.01,
                "mean_annual_vol": self.seriesim.mean_annual_vol + 0.01,
            },
            {},
            {
                "mean_annual_return": self.seriesim.mean_annual_return + 0.01,
                "mean_annual_vol": self.seriesim.mean_annual_vol + 0.01,
            },
            {"jumps_lamda": self.seriesim.jumps_lamda},
            {
                "jumps_lamda": self.seriesim.jumps_lamda + 0.1,
                "jumps_sigma": self.seriesim.jumps_sigma + 0.1,
                "jumps_mu": self.seriesim.jumps_mu + 0.1,
            },
        ]
        intended_returns = [
            "0.019523539",
            "0.050694317",
            "0.087309636",
            "0.089036349",
            "0.009865720",
            "0.054983999",
            "0.053461274",
            "0.013244309",
        ]

        intended_volatilities = [
            "0.096761956",
            "0.106983837",
            "0.101235865",
            "0.110245567",
            "0.101405667",
            "0.112748630",
            "0.099448330",
            "0.102635387",
        ]

        returns = []
        volatilities = []
        for method, adding in zip(methods, added, strict=True):
            arguments = {**args, **adding}
            onesim = getattr(ReturnSimulation, method)(**arguments)
            returns.append(f"{onesim.realized_mean_return:.9f}")
            volatilities.append(f"{onesim.realized_vol:.9f}")

        if intended_returns != returns:
            msg = f"Unexpected returns result {returns}"
            raise SimulationTestError(msg)
        if intended_volatilities != volatilities:
            msg = f"Unexpected volatilities result {volatilities}"
            raise SimulationTestError(msg)

    def test_properties(self: TestSimulation) -> None:
        """Test ReturnSimulation properties output."""
        if self.seriesim.results.shape[0] != self.seriesim.trading_days:
            msg = "Unexpected result"
            raise SimulationTestError(msg)

        if f"{self.seriesim.realized_mean_return:.9f}" != "0.058650906":
            msg = (
                f"Unexpected return result: '{self.seriesim.realized_mean_return:.9f}'"
            )
            raise SimulationTestError(msg)

        if f"{self.seriesim.realized_vol:.9f}" != "0.140742347":
            msg = f"Unexpected volatility result: '{self.seriesim.realized_vol:.9f}'"
            raise SimulationTestError(msg)

    def test_to_dataframe(self: TestSimulation) -> None:
        """Test method to_dataframe."""
        one = 1
        seriesim = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=one,
            trading_days=self.seriesim.trading_days,
            mean_annual_return=self.seriesim.mean_annual_return,
            mean_annual_vol=self.seriesim.mean_annual_vol,
            jumps_lamda=self.seriesim.jumps_lamda,
            jumps_sigma=self.seriesim.jumps_sigma,
            jumps_mu=self.seriesim.jumps_mu,
            trading_days_in_year=self.seriesim.trading_days_in_year,
            seed=self.seed,
        )
        five = 5
        framesim = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=five,
            trading_days=self.seriesim.trading_days,
            mean_annual_return=self.seriesim.mean_annual_return,
            mean_annual_vol=self.seriesim.mean_annual_vol,
            jumps_lamda=self.seriesim.jumps_lamda,
            jumps_sigma=self.seriesim.jumps_sigma,
            jumps_mu=self.seriesim.jumps_mu,
            trading_days_in_year=self.seriesim.trading_days_in_year,
            seed=self.seed,
        )
        framesim_mkt = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=five,
            trading_days=self.seriesim.trading_days,
            mean_annual_return=self.seriesim.mean_annual_return,
            mean_annual_vol=self.seriesim.mean_annual_vol,
            jumps_lamda=self.seriesim.jumps_lamda,
            jumps_sigma=self.seriesim.jumps_sigma,
            jumps_mu=self.seriesim.jumps_mu,
            trading_days_in_year=self.seriesim.trading_days_in_year,
            seed=self.seed,
        )

        start = dt.date(2009, 6, 30)

        onedf = seriesim.to_dataframe(name="Asset", start=start)
        fivedf = framesim.to_dataframe(name="Asset", start=start)
        fivedf_mkt = framesim_mkt.to_dataframe(
            name="Asset",
            start=start,
            markets=["XSTO"],
        )

        returnseries = OpenTimeSeries.from_df(onedf)
        startseries = returnseries.from_deepcopy()
        startseries.to_cumret()

        if onedf.shape != (self.seriesim.trading_days, one):
            msg = "Method to_dataframe() not working as intended"
            raise SimulationTestError(msg)

        if fivedf.shape != (self.seriesim.trading_days, five):
            msg = "Method to_dataframe() not working as intended"
            raise SimulationTestError(msg)

        if fivedf_mkt.shape != (self.seriesim.trading_days, five):
            msg = "Method to_dataframe() not working as intended"
            raise SimulationTestError(msg)

        if returnseries.valuetype != ValueType.RTRN:
            msg = "Method to_dataframe() not working as intended"
            raise SimulationTestError(msg)

        if startseries.valuetype != ValueType.PRICE:
            msg = "Method to_dataframe() not working as intended"
            raise SimulationTestError(msg)
