"""Test suite for the openseries/simulation.py module."""
# mypy: disable-error-code="type-arg"
from __future__ import annotations

from copy import copy
from datetime import date as dtdate
from typing import Union, cast
from unittest import TestCase

from pandas import DataFrame, Series, date_range

from openseries.frame import OpenFrame
from openseries.series import OpenTimeSeries
from openseries.simulation import (
    ModelParameters,
    ReturnSimulation,
    brownian_motion_levels,
    cox_ingersoll_ross_levels,
    geometric_brownian_motion_jump_diffusion_levels,
    geometric_brownian_motion_levels,
    heston_model_levels,
    ornstein_uhlenbeck_levels,
    random_generator,
)
from openseries.types import ValueType
from tests.common_sim import FIVE_SIMS, ONE_SIM, SEED


class TestSimulation(TestCase):

    """class to run unittests on the module simulation.py."""

    seriesim: ReturnSimulation
    framesim: ReturnSimulation

    @classmethod
    def setUpClass(cls: type[TestSimulation]) -> None:
        """SetUpClass for the TestSimulation class."""
        cls.seriesim = ONE_SIM
        cls.framesim = FIVE_SIMS

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
            "from_heston",
            "from_heston_vol",
            "from_merton_jump_gbm",
        ]
        added: list[dict[str, Union[int, float]]] = [
            {},
            {},
            {},
            {"heston_mu": 0.35, "heston_a": 0.25},
            {"heston_mu": 0.35, "heston_a": 0.25},
            {"jumps_lamda": 0.00125, "jumps_sigma": 0.001, "jumps_mu": -0.2},
        ]
        intended_returns = [
            "-0.005640734",
            "0.013058925",
            "-0.025640734",
            "0.027163652",
            "0.124437404",
            "-0.043708800",
        ]

        intended_volatilities = [
            "0.193403252",
            "0.193487832",
            "0.193403252",
            "0.198417705",
            "0.447234514",
            "0.209297219",
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

        if f"{psim.realized_mean_return:.9f}" != "-0.017030572":
            msg = f"Unexpected result: '{psim.realized_mean_return:.9f}'"
            raise ValueError(msg)

        if f"{psim.realized_vol:.9f}" != "0.125885483":
            msg = f"Unexpected result: '{psim.realized_vol:.9f}'"
            raise ValueError(msg)

    def test_assets(self: TestSimulation) -> None:
        """Test stoch processes output."""
        days = 2520
        intended_returns = [
            "-0.054955955",
            "0.061043422",
            "0.009116732",
            "0.126175407",
            "0.397127475",
        ]

        intended_volatilities = [
            "0.232908982",
            "0.232982933",
            "0.252075511",
            "0.203895259",
            "0.621810014",
        ]

        modelparams = ModelParameters(
            all_s0=1.0,
            all_r0=0.05,
            all_time=days,
            all_delta=1.0 / 252,
            all_sigma=0.2,
            gbm_mu=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            heston_a=0.25,
            heston_mu=0.35,
            heston_vol0=0.06125,
        )

        processes = [
            brownian_motion_levels,
            geometric_brownian_motion_levels,
            geometric_brownian_motion_jump_diffusion_levels,
            heston_model_levels,
            heston_model_levels,
        ]
        res_indices = [None, None, None, 0, 1]

        series = []
        for i, process, residx in zip(range(len(processes)), processes, res_indices):
            modelresult = process(
                param=modelparams,
                randomizer=random_generator(seed=SEED),
            )
            if isinstance(modelresult, tuple):
                modelresult = modelresult[cast(int, residx)]
            d_range = [
                d.date()
                for d in date_range(periods=days, end=dtdate(2019, 6, 30), freq="D")
            ]
            sdf = DataFrame(  # type: ignore[call-overload,unused-ignore]
                data=modelresult,
                index=d_range,
                columns=[f"Simulation_{i}"],
            )
            series.append(
                OpenTimeSeries.from_df(sdf, valuetype=ValueType.PRICE).to_cumret(),
            )

        frame = OpenFrame(series)
        means = [f"{r:.9f}" for r in cast(Series, frame.arithmetic_ret)]
        deviations = [f"{v:.9f}" for v in cast(Series, frame.vol)]

        if intended_returns != means:
            msg = "Unexpected calculation result"
            raise ValueError(msg)
        if intended_volatilities != deviations:
            msg = "Unexpected calculation result"
            raise ValueError(msg)

    def test_cir_and_ou(self: TestSimulation) -> None:
        """Test output of cox_ingersoll_ross_levels & ornstein_uhlenbeck_levels."""
        series = []
        days = 2520
        intended_returns = ["0.024059099", "0.019497031"]
        intended_volatilities = ["0.003014102", "0.019346265"]

        modelparams = ModelParameters(
            all_s0=1.0,
            all_r0=0.025,
            all_time=days,
            all_delta=1.0 / 252,
            all_sigma=0.06,
            gbm_mu=0.01,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            cir_a=3.0,
            cir_mu=0.025,
            cir_rho=0.1,
            ou_a=3.0,
            ou_mu=0.025,
            heston_a=0.25,
            heston_mu=0.35,
            heston_vol0=0.06125,
        )

        processes = [cox_ingersoll_ross_levels, ornstein_uhlenbeck_levels]
        for process in processes:
            onesim = process(param=modelparams, randomizer=random_generator(seed=SEED))
            d_range = [
                d.date()
                for d in date_range(periods=days, end=dtdate(2019, 6, 30), freq="D")
            ]
            sdf = DataFrame(
                data=onesim,
                index=d_range,
                columns=[[f"Asset_{str(process)[:-7]}"], [ValueType.PRICE]],
            )
            series.append(OpenTimeSeries.from_df(sdf, valuetype=ValueType.PRICE))

        frame = OpenFrame(series)
        means = [f"{r:.9f}" for r in frame.tsdf.mean()]
        deviations = [f"{v:.9f}" for v in frame.tsdf.std()]

        if intended_returns != means:
            msg = "Unexpected results from method beta()"
            raise ValueError(msg)

        if intended_volatilities != deviations:
            msg = "Unexpected results from cir or ou random processes()"
            raise ValueError(msg)

    def test_to_dataframe(self: TestSimulation) -> None:
        """Test method to_dataframe."""
        seriesim = copy(self.seriesim)

        start = dtdate(2009, 6, 30)

        startseries = OpenTimeSeries.from_df(
            seriesim.to_dataframe(name="Asset", start=start),
        ).to_cumret()
        returnseries = OpenTimeSeries.from_df(
            seriesim.to_dataframe(name="Asset", start=start),
        )

        if startseries.valuetype != ValueType.PRICE:
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)

        if returnseries.valuetype != ValueType.RTRN:
            msg = "Method to_dataframe() not working as intended"
            raise ValueError(msg)
