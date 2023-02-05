from pandas import DataFrame
from typing import get_type_hints, TypeVar
from unittest import TestCase

from openseries.sim_price import ReturnSimulation, Simulation

TTestSimPrice = TypeVar("TTestSimPrice", bound="TestSimPrice")


class TestSimPrice(TestCase):
    def test_return_simulations_annotations_and_typehints(self: TTestSimPrice):
        returnsimulation_annotations = dict(ReturnSimulation.__annotations__)
        simulation_annotations = dict(Simulation.__annotations__)

        self.assertDictEqual(
            returnsimulation_annotations,
            {
                "number_of_sims": int,
                "trading_days": int,
                "trading_days_in_year": int,
                "mean_annual_return": float,
                "mean_annual_vol": float,
                "df": DataFrame,
            },
        )
        self.assertDictEqual(returnsimulation_annotations, simulation_annotations)

        returnsimulation_typehints = get_type_hints(ReturnSimulation)
        simulation_typehints = get_type_hints(Simulation)
        self.assertDictEqual(returnsimulation_annotations, returnsimulation_typehints)
        self.assertDictEqual(returnsimulation_typehints, simulation_typehints)

    def test_return_simulation_processes(self: TTestSimPrice):
        args = {
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
        added = [
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
        for m, a in zip(methods, added):
            arguments = {**args, **a}
            onesim = getattr(ReturnSimulation, m)(**arguments)
            returns.append(f"{onesim.realized_mean_return:.9f}")
            volatilities.append(f"{onesim.realized_vol:.9f}")

        self.assertListEqual(target_returns, returns)
        self.assertListEqual(target_volatilities, volatilities)

    def test_return_simulation_properties(self: TTestSimPrice):
        days = 1200
        psim = ReturnSimulation.from_normal(
            number_of_sims=1,
            trading_days=days,
            mean_annual_return=0.05,
            mean_annual_vol=0.1,
            seed=71,
        )

        self.assertIsInstance(psim.results, DataFrame)

        self.assertEqual(psim.results.shape[0], days)

        self.assertEqual(f"{psim.realized_mean_return:.9f}", "0.033493161")

        self.assertEqual(f"{psim.realized_vol:.9f}", "0.096596353")
