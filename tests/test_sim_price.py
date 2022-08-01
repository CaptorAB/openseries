# -*- coding: utf-8 -*-
from pandas import DataFrame
from unittest import TestCase

from openseries.sim_price import ReturnSimulation


class TestSimPrice(TestCase):
    def test_return_simulation_processes(self):

        args = {"n": 1, "d": 2520, "mu": 0.05, "vol": 0.2, "seed": 71}
        methods = [
            "from_normal",
            "from_lognormal",
            "from_gbm",
            "from_heston",
            "from_heston_vol",
            "from_merton_jump_gbm",
        ]
        added = [
            {},
            {},
            {},
            {"heston_mu": 0.35, "heston_a": 0.25},
            {"heston_mu": 0.35, "heston_a": 0.25},
            {"jumps_lamda": 0.00125, "jumps_sigma": 0.001, "jumps_mu": -0.2},
        ]
        target_returns = [
            "-0.011157857",
            "0.008917436",
            "-0.031161130",
            "0.032446979",
            "0.004575385",
            "-0.029813702",
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
        for m, a in zip(methods, added):
            arguments = {**args, **a}
            onesim = getattr(ReturnSimulation, m)(**arguments)
            returns.append(f"{onesim.realized_mean_return:.9f}")
            volatilities.append(f"{onesim.realized_vol:.9f}")

        self.assertListEqual(target_returns, returns)
        self.assertListEqual(target_volatilities, volatilities)

    def test_returnsimulation_properties(self):

        days = 1200
        psim = ReturnSimulation.from_normal(n=1, d=days, mu=0.05, vol=0.1, seed=71)

        self.assertIsInstance(psim.results, DataFrame)

        self.assertEqual(psim.results.shape[0], days)

        self.assertEqual(f"{psim.realized_mean_return:.9f}", "0.028832246")

        self.assertEqual(f"{psim.realized_vol:.9f}", "0.096596353")
