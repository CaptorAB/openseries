# -*- coding: utf-8 -*-
import datetime as dt
import pandas as pd
from unittest import TestCase

from openseries.frame import OpenFrame
from openseries.series import OpenTimeSeries
from openseries.stoch_processes import (
    ModelParameters,
    cox_ingersoll_ross_levels,
    ornstein_uhlenbeck_levels,
    brownian_motion_levels,
    geometric_brownian_motion_levels,
    geometric_brownian_motion_jump_diffusion_levels,
    heston_model_levels,
)


class TestStochProcesses(TestCase):
    def test_stoch_processes_assets(self):
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

        mp = ModelParameters(
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
            modelresult = process(param=mp, seed=71)
            if isinstance(modelresult, tuple):
                modelresult = modelresult[residx]
            date_range = [
                d.date()
                for d in pd.date_range(periods=days, end=dt.date(2019, 6, 30), freq="D")
            ]
            sdf = pd.DataFrame(
                data=modelresult,
                index=date_range,
                columns=[f"Simulation_{i}"],
            )
            series.append(
                OpenTimeSeries.from_df(sdf, valuetype="Price(Close)").to_cumret()
            )

        frame = OpenFrame(series)
        means = [f"{r:.9f}" for r in frame.arithmetic_ret]
        deviations = [f"{v:.9f}" for v in frame.vol]

        self.assertListEqual(target_returns, means)
        self.assertListEqual(target_volatilities, deviations)

    def test_stoch_processes_cir_and_ou(self):
        series = []
        days = 2520
        target_means = ["0.024184423", "0.019893950"]
        target_deviations = ["0.003590473", "0.023333692"]

        mp = ModelParameters(
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
            onesim = process(mp, seed=71)
            name = process.__name__
            date_range = [
                d.date()
                for d in pd.date_range(periods=days, end=dt.date(2019, 6, 30), freq="D")
            ]
            sdf = pd.DataFrame(
                data=onesim,
                index=date_range,
                columns=pd.MultiIndex.from_product(
                    [[f"Asset_{name[:-7]}"], ["Price(Close)"]]
                ),
            )
            series.append(OpenTimeSeries.from_df(sdf, valuetype="Price(Close)"))

        frame = OpenFrame(series)
        means = [f"{r:.9f}" for r in frame.tsdf.mean()]
        deviations = [f"{v:.9f}" for v in frame.tsdf.std()]

        self.assertListEqual(target_means, means)
        self.assertListEqual(target_deviations, deviations)
