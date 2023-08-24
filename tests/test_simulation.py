"""Test suite for the openseries/simulation.py module."""
from __future__ import annotations

from copy import copy
from datetime import date as dtdate
from typing import TypeVar, cast
from unittest import TestCase

from pandas import DataFrame, Series, date_range

from openseries.frame import OpenFrame
from openseries.series import OpenTimeSeries
from openseries.simulation import ModelParameters, ReturnSimulation
from openseries.types import ValueType
from tests.common_sim import FIVE_SIMS, ONE_SIM

TypeTestSimulation = TypeVar("TypeTestSimulation", bound="TestSimulation")


class TestSimulation(TestCase):
    """class to run unittests on the module simulation.py."""

    seriesim: ReturnSimulation
    framesim: ReturnSimulation

    @classmethod
    def setUpClass(cls: type[TypeTestSimulation]) -> None:
        """SetUpClass for the TestSimulation class."""
        cls.seriesim = ONE_SIM
        cls.framesim = FIVE_SIMS

    def test_processes(self: TestSimulation) -> None:
        """Test ReturnSimulation based on different stochastic processes."""
        args: dict[str, int | float] = {
            "number_of_sims": 1,
            "trading_days": 2520,
            "mean_annual_return": 0.05,
            "mean_annual_vol": 0.2,
            "seed": 71,
        }
        methods = [
            "from_normal",
            "from_lognormal",
            "from_gbm",
            "from_heston",
            "from_heston_vol",
            "from_merton_jump_gbm",
        ]
        added: list[dict[str, int | float]] = [
            {},
            {},
            {},
            {"heston_mu": 0.35, "heston_a": 0.25},
            {"heston_mu": 0.35, "heston_a": 0.25},
            {"jumps_lamda": 0.00125, "jumps_sigma": 0.001, "jumps_mu": -0.2},
        ]
        target_returns = [
            "0.008917436",
            "0.029000099",
            "-0.011082564",
            "0.067119310",
            "0.101488620",
            "-0.007388824",
        ]
        target_volatilities = [
            "0.200429415",
            "0.200504640",
            "0.200429415",
            "0.263455329",
            "0.440520211",
            "0.210298179",
        ]

        returns = []
        volatilities = []
        for method, adding in zip(methods, added):
            arguments: dict[str, int | float] = {**args, **adding}
            onesim = getattr(ReturnSimulation, method)(**arguments)
            returns.append(f"{onesim.realized_mean_return:.9f}")
            volatilities.append(f"{onesim.realized_vol:.9f}")

        self.assertListEqual(target_returns, returns)
        self.assertListEqual(target_volatilities, volatilities)

    def test_properties(self: TestSimulation) -> None:
        """Test ReturnSimulation properties output."""
        days = 2512
        psim = copy(self.seriesim)

        self.assertIsInstance(psim.results, DataFrame)

        self.assertEqual(psim.results.shape[0], days)

        self.assertEqual(f"{psim.realized_mean_return:.9f}", "0.009553952")

        self.assertEqual(f"{psim.realized_vol:.9f}", "0.117099479")

    def test_assets(self: TestSimulation) -> None:
        """Test stoch processes output."""
        days = 2520
        target_returns = [
            "-0.031826675",
            "0.084180046",
            "0.058456697",
            "0.034909498",
            "0.353642948",
        ]
        target_volatilities = [
            "0.241393324",
            "0.241469969",
            "0.252469189",
            "0.236601983",
            "0.600404476",
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
            "brownian_motion_levels",
            "geometric_brownian_motion_levels",
            "geometric_brownian_motion_jump_diffusion_levels",
            "heston_model_levels",
            "heston_model_levels",
        ]
        res_indices = [None, None, None, 0, 1]

        series = []
        for i, process, residx in zip(range(len(processes)), processes, res_indices):
            modelresult = getattr(ReturnSimulation, process)(modelparams, seed=71)
            if isinstance(modelresult, tuple):
                modelresult = modelresult[cast(int, residx)]
            d_range = [
                d.date()
                for d in date_range(periods=days, end=dtdate(2019, 6, 30), freq="D")
            ]
            sdf = DataFrame(
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

        self.assertListEqual(target_returns, means)
        self.assertListEqual(target_volatilities, deviations)

    def test_cir_and_ou(self: TestSimulation) -> None:
        """Test output of cox_ingersoll_ross_levels & ornstein_uhlenbeck_levels."""
        series = []
        days = 2520
        target_means = ["0.024184423", "0.019893950"]
        target_deviations = ["0.003590473", "0.023333692"]

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

        processes = ["cox_ingersoll_ross_levels", "ornstein_uhlenbeck_levels"]
        for process in processes:
            onesim = getattr(ReturnSimulation, process)(modelparams, seed=71)
            d_range = [
                d.date()
                for d in date_range(periods=days, end=dtdate(2019, 6, 30), freq="D")
            ]
            sdf = DataFrame(
                data=onesim,
                index=d_range,
                columns=[[f"Asset_{process[:-7]}"], [ValueType.PRICE]],
            )
            series.append(OpenTimeSeries.from_df(sdf, valuetype=ValueType.PRICE))

        frame = OpenFrame(series)
        means = [f"{r:.9f}" for r in frame.tsdf.mean()]
        deviations = [f"{v:.9f}" for v in frame.tsdf.std()]

        self.assertListEqual(target_means, means)
        self.assertListEqual(target_deviations, deviations)

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

        self.assertEqual(ValueType.PRICE, startseries.valuetype)
        self.assertEqual(ValueType.RTRN, returnseries.valuetype)
