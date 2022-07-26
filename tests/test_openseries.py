# -*- coding: utf-8 -*-
import datetime as dt
import io
import json
from jsonschema.exceptions import ValidationError
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.tseries.offsets import CDay
from stdnum.exceptions import InvalidChecksum
import sys
from testfixtures import LogCapture
import unittest

from openseries.frenkla_open_api_sdk import FrenklaOpenApiService
from openseries.frame import OpenFrame
from openseries.risk import cvar_down, var_down
from openseries.series import OpenTimeSeries, timeseries_chain, TimeSerie
from openseries.sim_price import ReturnSimulation
from openseries.stoch_processes import (
    ModelParameters,
    cox_ingersoll_ross_levels,
    ornstein_uhlenbeck_levels,
    brownian_motion_levels,
    geometric_brownian_motion_levels,
    geometric_brownian_motion_jump_diffusion_levels,
    heston_model_levels,
)

repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if repo_root not in sys.path:
    sys.path.append(repo_root)


def sim_to_opentimeseries(sim: ReturnSimulation, end: dt.date) -> OpenTimeSeries:
    date_range = [
        d.date()
        for d in pd.date_range(
            periods=sim.trading_days,
            end=end,
            freq=CDay(calendar=OpenTimeSeries.sweden),
        )
    ]
    sdf = sim.df.iloc[0].T.to_frame()
    sdf.index = date_range
    sdf.columns = pd.MultiIndex.from_product([["Asset"], ["Return(Total)"]])
    return OpenTimeSeries.from_df(sdf, valuetype="Return(Total)")


def sim_to_openframe(sim: ReturnSimulation, end: dt.date) -> OpenFrame:
    date_range = [
        d.date()
        for d in pd.date_range(
            periods=sim.trading_days,
            end=end,
            freq=CDay(calendar=OpenTimeSeries.sweden),
        )
    ]
    tslist = []
    for item in range(sim.number_of_sims):
        sdf = sim.df.iloc[item].T.to_frame()
        sdf.index = date_range
        sdf.columns = pd.MultiIndex.from_product([[f"Asset_{item}"], ["Return(Total)"]])
        tslist.append(OpenTimeSeries.from_df(sdf, valuetype="Return(Total)"))
    return OpenFrame(tslist)


class TestOpenTimeSeries(unittest.TestCase):
    randomseries: OpenTimeSeries
    sim: ReturnSimulation

    @classmethod
    def setUpClass(cls):

        OpenTimeSeries.setup_class()

        cls.sim = ReturnSimulation.from_merton_jump_gbm(
            n=1,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )

        cls.randomseries = sim_to_opentimeseries(
            cls.sim, end=dt.date(2019, 6, 30)
        ).to_cumret()
        cls.random_properties = cls.randomseries.all_properties().to_dict()[
            ("Asset", "Price(Close)")
        ]

    def test_opentimeseries_repr(self):

        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        repseries = self.randomseries
        r = (
            "OpenTimeSeries(name=Asset, _id=, instrumentId=, valuetype=Price(Close), currency=SEK, "
            "start=2009-06-30, end=2019-06-28, local_ccy=True)\n"
        )
        print(repseries)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        self.assertEqual(r, output)

    def test_openframe_repr(self):

        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        reprframe = OpenFrame(
            [
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="reprseries1",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Price(Close)",
                        dates=["2022-07-01", "2023-07-01"],
                        values=[1.0, 1.1],
                    )
                ),
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="reprseries2",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Price(Close)",
                        dates=["2022-07-01", "2023-07-01"],
                        values=[1.0, 1.2],
                    )
                ),
            ],
            weights=[0.5, 0.5],
        )
        r = (
            "OpenFrame(constituents=[OpenTimeSeries(name=reprseries1, _id=, instrumentId=, "
            "valuetype=Price(Close), currency=SEK, start=2022-07-01, end=2023-07-01, local_ccy=True), "
            "OpenTimeSeries(name=reprseries2, _id=, instrumentId=, valuetype=Price(Close), currency=SEK, "
            "start=2022-07-01, end=2023-07-01, local_ccy=True)], "
            "weights=[0.5, 0.5])"
        )
        r = r.rstrip() + "\n"
        print(reprframe)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        self.assertEqual(r, output)

    def test_opentimeseries_tsdf_not_empty(self):

        json_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "series.json"
        )
        with open(json_file, "r") as ff:
            output = json.load(ff)
        timeseries = OpenTimeSeries(output)

        self.assertFalse(timeseries.tsdf.empty)

    def test_opentimeseries_duplicates_handling(self):
        class NewTimeSeries(OpenTimeSeries):
            def __init__(self, d):

                super().__init__(d)

            @classmethod
            def from_file(cls, remove_duplicates: bool = False):

                json_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "series.json"
                )
                with open(json_file, "r") as ff:
                    output = json.load(ff)

                dates = (
                    output["dates"][:63]
                    + [output["dates"][63]]
                    + output["dates"][63:128]
                    + [output["dates"][128]] * 2
                    + output["dates"][128:]
                )
                values = (
                    output["values"][:63]
                    + [output["values"][63]]
                    + output["values"][63:128]
                    + [output["values"][128]] * 2
                    + output["values"][128:]
                )
                output.update({"dates": dates, "values": values})

                if remove_duplicates:
                    udf = pd.DataFrame(
                        index=output["dates"],
                        columns=[output["name"]],
                        data=output["values"],
                    )
                    udf.sort_index(inplace=True)
                    removed = udf.loc[udf.index.duplicated(keep="first")]
                    if not removed.empty:
                        udf = udf.loc[~udf.index.duplicated(keep="first")]
                        output.update({"dates": udf.index.tolist()})
                        output.update({"values": udf.iloc[:, 0].values.tolist()})

                        self.assertListEqual(
                            removed.index.tolist(),
                            ["2017-08-28", "2017-11-27", "2017-11-27"],
                        )
                        self.assertListEqual(
                            removed.iloc[:, 0].values.tolist(),
                            [99.4062, 100.8974, 100.8974],
                        )

                return cls(d=output)

        with self.assertRaises(Exception) as e_dup:
            NewTimeSeries.from_file(remove_duplicates=False)

        self.assertIsInstance(e_dup.exception, ValidationError)

        ts = NewTimeSeries.from_file(remove_duplicates=True)
        self.assertIsInstance(ts, OpenTimeSeries)

    def test_create_opentimeseries_from_open_api(self):

        timeseries_id = "59977d91f3fa6319ecb41cbd"
        timeseries = OpenTimeSeries.from_open_api(timeseries_id=timeseries_id)

        self.assertTrue(isinstance(timeseries, OpenTimeSeries))

    def test_create_opentimeseries_from_open_nav(self):

        fund = "SE0009807308"
        timeseries = OpenTimeSeries.from_open_nav(isin=fund)

        self.assertTrue(isinstance(timeseries, OpenTimeSeries))

        with self.assertRaises(Exception) as e_unique:
            fnd = ""
            _ = OpenTimeSeries.from_open_nav(isin=fnd)

        self.assertEqual(
            f"Request for NAV series using ISIN {fnd} returned no data.",
            e_unique.exception.args[0],
        )

    def test_create_opentimeseries_from_open_fundinfo(self):

        fund = "SE0009807308"
        timeseries = OpenTimeSeries.from_open_fundinfo(isin=fund)

        self.assertTrue(isinstance(timeseries, OpenTimeSeries))

        with self.assertRaises(Exception) as e_unique:
            fnd = ""
            _ = OpenTimeSeries.from_open_fundinfo(isin=fnd)

        self.assertEqual(int(e_unique.exception.args[0].split(",")[0]), 400)

        with self.assertRaises(Exception) as e_unique:
            fundd = "SE000"
            _ = OpenTimeSeries.from_open_fundinfo(isin=fundd)

        self.assertTrue(f"{fundd} is not a valid ISIN" in e_unique.exception.args[0])

    def test_create_opentimeseries_from_pandas_df(self):

        se = pd.Series(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            name="Asset_0",
        )
        sen = pd.Series(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            name=("Asset_0", "Price(Close)"),
        )
        df = pd.DataFrame(
            data=[
                [1.0, 1.0],
                [1.01, 0.98],
                [0.99, 1.004],
                [1.015, 0.976],
                [1.003, 0.982],
            ],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=["Asset_0", "Asset_1"],
        )

        seseries = OpenTimeSeries.from_df(df=se)
        senseries = OpenTimeSeries.from_df(df=sen)
        dfseries = OpenTimeSeries.from_df(df=df, column_nmbr=1)

        self.assertTrue(isinstance(seseries, OpenTimeSeries))
        self.assertTrue(isinstance(senseries, OpenTimeSeries))
        self.assertTrue(isinstance(dfseries, OpenTimeSeries))
        self.assertEqual(seseries.label, senseries.label)

    def test_create_opentimeseries_from_frame(self):

        sim_f = ReturnSimulation.from_merton_jump_gbm(
            n=2,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame_f = sim_to_openframe(sim_f, end=dt.date(2019, 6, 30))
        frame_f.to_cumret()
        fseries = OpenTimeSeries.from_frame(frame_f, label="Asset_1")

        self.assertTrue(isinstance(fseries, OpenTimeSeries))

    def test_opentimeseries_save_to_json(self):

        seriesfile = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "irisc.json"
        )
        capirisc = "SE0009807308"
        irisc = OpenTimeSeries.from_open_nav(isin=capirisc)
        irisc.to_json(filename=seriesfile)

        self.assertTrue(os.path.exists(seriesfile))

        os.remove(seriesfile)

        self.assertFalse(os.path.exists(seriesfile))

    def test_create_opentimeseries_from_fixed_rate(self):

        fixseries = OpenTimeSeries.from_fixed_rate(
            rate=0.03, days=756, end_dt=dt.date(2019, 6, 30)
        )

        self.assertTrue(isinstance(fixseries, OpenTimeSeries))

    def test_opentimeseries_periods_in_a_year(self):

        calc = len(self.randomseries.dates) / (
            (self.randomseries.last_idx - self.randomseries.first_idx).days / 365.25
        )

        self.assertEqual(calc, self.randomseries.periods_in_a_year)
        self.assertEqual(
            f"{251.3720547945205:.13f}",
            f"{self.randomseries.periods_in_a_year:.13f}",
        )
        all_prop = self.random_properties["periods_in_a_year"]
        self.assertEqual(
            f"{all_prop:.13f}", f"{self.randomseries.periods_in_a_year:.13f}"
        )

    def test_opentimeseries_yearfrac(self):

        self.assertEqual(
            f"{9.9931553730322:.13f}", f"{self.randomseries.yearfrac:.13f}"
        )
        all_prop = self.random_properties["yearfrac"]
        self.assertEqual(f"{all_prop:.13f}", f"{self.randomseries.yearfrac:.13f}")

    def test_opentimeseries_resample(self):

        rs_sim = ReturnSimulation.from_merton_jump_gbm(
            n=1,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        rs_series = sim_to_opentimeseries(rs_sim, end=dt.date(2019, 6, 30)).to_cumret()

        before = rs_series.value_ret

        rs_series.resample(freq="BM")

        self.assertEqual(121, rs_series.length)
        self.assertEqual(before, rs_series.value_ret)

    def test_opentimeseries_calc_range(self):

        cseries = self.randomseries.from_deepcopy()
        st, en = cseries.first_idx.strftime("%Y-%m-%d"), cseries.last_idx.strftime(
            "%Y-%m-%d"
        )

        rst, ren = cseries.calc_range()

        self.assertListEqual(
            [st, en], [rst.strftime("%Y-%m-%d"), ren.strftime("%Y-%m-%d")]
        )

        with self.assertRaises(AssertionError) as too_far:
            _, _ = cseries.calc_range(months_offset=125)
        self.assertIsInstance(too_far.exception, AssertionError)

        with self.assertRaises(AssertionError) as too_early:
            _, _ = cseries.calc_range(from_dt=dt.date(2009, 5, 31))
        self.assertIsInstance(too_early.exception, AssertionError)

        with self.assertRaises(AssertionError) as too_late:
            _, _ = cseries.calc_range(to_dt=dt.date(2019, 7, 31))
        self.assertIsInstance(too_late.exception, AssertionError)

        with self.assertRaises(AssertionError) as outside:
            _, _ = cseries.calc_range(
                from_dt=dt.date(2009, 5, 31), to_dt=dt.date(2019, 7, 31)
            )
        self.assertIsInstance(outside.exception, AssertionError)

        with self.assertRaises(AssertionError) as outside_end:
            _, _ = cseries.calc_range(
                from_dt=dt.date(2009, 7, 31), to_dt=dt.date(2019, 7, 31)
            )
        self.assertIsInstance(outside_end.exception, AssertionError)

        with self.assertRaises(AssertionError) as outside_start:
            _, _ = cseries.calc_range(
                from_dt=dt.date(2009, 5, 31), to_dt=dt.date(2019, 5, 31)
            )
        self.assertIsInstance(outside_start.exception, AssertionError)

        nst, nen = cseries.calc_range(
            from_dt=dt.date(2009, 7, 3), to_dt=dt.date(2019, 6, 25)
        )
        self.assertEqual(nst, dt.date(2009, 7, 3))
        self.assertEqual(nen, dt.date(2019, 6, 25))

        cseries.resample()

        earlier_moved, _ = cseries.calc_range(from_dt=dt.date(2009, 8, 10))
        self.assertEqual(earlier_moved, dt.date(2009, 7, 31))

        _, later_moved = cseries.calc_range(to_dt=dt.date(2009, 8, 20))
        self.assertEqual(later_moved, dt.date(2009, 8, 31))

    def test_openframe_calc_range(self):

        crsims = ReturnSimulation.from_normal(n=5, d=2512, mu=0.05, vol=0.1, seed=71)
        crframe = sim_to_openframe(crsims, dt.date(2019, 6, 30))
        st, en = crframe.first_idx.strftime("%Y-%m-%d"), crframe.last_idx.strftime(
            "%Y-%m-%d"
        )
        rst, ren = crframe.calc_range()

        self.assertListEqual(
            [st, en], [rst.strftime("%Y-%m-%d"), ren.strftime("%Y-%m-%d")]
        )

        with self.assertRaises(AssertionError) as too_far:
            _, _ = crframe.calc_range(months_offset=125)
        self.assertIsInstance(too_far.exception, AssertionError)

        with self.assertRaises(AssertionError) as too_early:
            _, _ = crframe.calc_range(from_dt=dt.date(2009, 5, 31))
        self.assertIsInstance(too_early.exception, AssertionError)

        with self.assertRaises(AssertionError) as too_late:
            _, _ = crframe.calc_range(to_dt=dt.date(2019, 7, 31))
        self.assertIsInstance(too_late.exception, AssertionError)

        with self.assertRaises(AssertionError) as outside:
            _, _ = crframe.calc_range(
                from_dt=dt.date(2009, 5, 31), to_dt=dt.date(2019, 7, 31)
            )
        self.assertIsInstance(outside.exception, AssertionError)

        with self.assertRaises(AssertionError) as outside_end:
            _, _ = crframe.calc_range(
                from_dt=dt.date(2009, 7, 31), to_dt=dt.date(2019, 7, 31)
            )
        self.assertIsInstance(outside_end.exception, AssertionError)

        with self.assertRaises(AssertionError) as outside_start:
            _, _ = crframe.calc_range(
                from_dt=dt.date(2009, 5, 31), to_dt=dt.date(2019, 5, 31)
            )
        self.assertIsInstance(outside_start.exception, AssertionError)

        nst, nen = crframe.calc_range(
            from_dt=dt.date(2009, 7, 3), to_dt=dt.date(2019, 6, 25)
        )
        self.assertEqual(nst, dt.date(2009, 7, 3))
        self.assertEqual(nen, dt.date(2019, 6, 25))

        crframe.resample()

        earlier_moved, _ = crframe.calc_range(from_dt=dt.date(2009, 8, 10))
        self.assertEqual(earlier_moved, dt.date(2009, 7, 31))

        _, later_moved = crframe.calc_range(to_dt=dt.date(2009, 8, 20))
        self.assertEqual(later_moved, dt.date(2009, 8, 31))

    def test_opentimeseries_calc_range_ouput(self):

        csim = ReturnSimulation.from_normal(n=1, d=1200, mu=0.05, vol=0.1, seed=71)
        cseries = sim_to_opentimeseries(csim, end=dt.date(2019, 6, 30)).to_cumret()

        dates = cseries.calc_range(months_offset=48)

        self.assertListEqual(
            ["2015-06-26", "2019-06-28"],
            [dates[0].strftime("%Y-%m-%d"), dates[1].strftime("%Y-%m-%d")],
        )
        dates = self.randomseries.calc_range(from_dt=dt.date(2016, 6, 30))

        self.assertListEqual(
            ["2016-06-30", "2019-06-28"],
            [dates[0].strftime("%Y-%m-%d"), dates[1].strftime("%Y-%m-%d")],
        )

        gr_0 = cseries.vol_func(months_from_last=48)

        cseries.dates = cseries.dates[-1008:]
        cseries.values = cseries.values[-1008:]
        cseries.pandas_df()
        cseries.set_new_label(lvl_one="Return(Total)")
        cseries.to_cumret()

        gr_1 = cseries.vol

        self.assertEqual(f"{gr_0:.13f}", f"{gr_1:.13f}")

    def test_opentimeseries_value_to_diff(self):

        diffsim = ReturnSimulation.from_normal(n=1, d=15, mu=0.05, vol=0.1, seed=71)
        diffseries = sim_to_opentimeseries(
            diffsim, end=dt.date(2019, 6, 30)
        ).to_cumret()
        diffseries.value_to_diff()
        are_bes = [f"{nn[0]:.12f}" for nn in diffseries.tsdf.values]
        should_bes = [
            "0.000000000000",
            "-0.007322627296",
            "-0.002581366067",
            "0.003248920666",
            "-0.002628519782",
            "0.003851856296",
            "0.007573468698",
            "-0.005893167569",
            "0.001567531620",
            "-0.005246297149",
            "-0.001822686581",
            "0.009014775004",
            "-0.004289844249",
            "-0.008344628763",
            "-0.010412377959",
        ]

        self.assertListEqual(are_bes, should_bes)

    def test_opentimeseries_value_to_ret(self):

        retsim = ReturnSimulation.from_normal(n=1, d=15, mu=0.05, vol=0.1, seed=71)
        retseries = sim_to_opentimeseries(retsim, end=dt.date(2019, 6, 30)).to_cumret()

        retseries.value_to_ret()
        are_bes = [f"{nn[0]:.12f}" for nn in retseries.tsdf.values]
        should_bes = [
            "0.000000000000",
            "-0.007322627296",
            "-0.002600407884",
            "0.003281419826",
            "-0.002646129969",
            "0.003887950443",
            "0.007614830448",
            "-0.005880572955",
            "0.001573434257",
            "-0.005257779632",
            "-0.001836330888",
            "0.009098966716",
            "-0.004290865956",
            "-0.008382584742",
            "-0.010548160048",
        ]

        self.assertListEqual(are_bes, should_bes)

        retseries.to_cumret()

    def test_opentimeseries_valute_to_log(self):

        logsim = ReturnSimulation.from_normal(n=1, d=15, mu=0.05, vol=0.1, seed=71)
        logseries = sim_to_opentimeseries(logsim, end=dt.date(2019, 6, 30)).to_cumret()

        logseries.value_to_log()
        are_log = [f"{nn[0]:.12f}" for nn in logseries.tsdf.values]

        should_log = [
            "0.000000000000",
            "-0.007349569336",
            "-0.009953364154",
            "-0.006677316437",
            "-0.009326953597",
            "-0.005446541699",
            "0.002139442275",
            "-0.003758489335",
            "-0.002186291629",
            "-0.007457942025",
            "-0.009295961036",
            "-0.000238140514",
            "-0.004538238654",
            "-0.012956154844",
            "-0.023560341062",
        ]

        self.assertListEqual(are_log, should_log)

    def test_all_calc_properties(self):

        checks = {
            "arithmetic_ret": f"{0.00953014509:.11f}",
            "cvar_down": f"{-0.01402077271:.11f}",
            "downside_deviation": f"{0.09195729357:.11f}",
            "geo_ret": f"{0.00242231676:.11f}",
            "kurtosis": f"{180.63357183510:.11f}",
            "max_drawdown": f"{-0.40011625413:.11f}",
            "max_drawdown_cal_year": f"{-0.23811167802:.11f}",
            "positive_share": f"{0.49940262843:.11f}",
            "ret_vol_ratio": f"{0.08148662314:.11f}",
            "skew": f"{-6.94679906059:.11f}",
            "sortino_ratio": f"{0.10363664173:.11f}",
            "value_ret": f"{0.02447195802:.11f}",
            "var_down": f"{-0.01059129607:.11f}",
            "vol": f"{0.11695349153:.11f}",
            "vol_from_var": f"{0.10208932904:.11f}",
            "worst": f"{-0.19174232326:.11f}",
            "worst_month": f"{-0.19165644070:.11f}",
            "z_score": f"{1.21195350537:.11f}",
        }
        for c in checks:
            self.assertEqual(
                checks[c],
                f"{getattr(self.randomseries, c):.11f}",
                msg=f"Difference in: {c}",
            )
            self.assertEqual(
                f"{self.random_properties[c]:.11f}",
                f"{getattr(self.randomseries, c):.11f}",
                msg=f"Difference in: {c}",
            )

    def test_all_calc_functions(self):

        checks = {
            "arithmetic_ret_func": f"{0.00885255100:.11f}",
            "cvar_down_func": f"{-0.01331889836:.11f}",
            "downside_deviation_func": f"{0.07335125856:.11f}",
            "geo_ret_func": f"{0.00348439444:.11f}",
            "kurtosis_func": f"{-0.16164566028:.11f}",
            "max_drawdown_func": f"{-0.20565775282:.11f}",
            "positive_share_func": f"{0.50645481629:.11f}",
            "ret_vol_ratio_func": f"{0.08538041030:.11f}",
            "skew_func": f"{-0.03615947531:.11f}",
            "sortino_ratio_func": f"{0.12068710437:.11f}",
            "value_ret_func": f"{0.01402990651:.11f}",
            "var_down_func": f"{-0.01095830172:.11f}",
            "vol_func": f"{0.10368363149:.11f}",
            "vol_from_var_func": f"{0.10568642619:.11f}",
            "worst_func": f"{-0.02063487245:.11f}",
            "z_score_func": f"{1.36825335773:.11f}",
        }
        for c in checks:
            self.assertEqual(
                checks[c],
                f"{getattr(self.randomseries, c)(months_from_last=48):.11f}",
                msg=f"Difference in {c}",
            )

        func = "value_ret_calendar_period"
        self.assertEqual(
            f"{0.076502833914:.12f}",
            f"{getattr(self.randomseries, func)(year=2019):.12f}",
        )

    def test_opentimeseries_max_drawdown_date(self):

        self.assertEqual(dt.date(2018, 11, 8), self.randomseries.max_drawdown_date)
        all_prop = self.random_properties["max_drawdown_date"]
        self.assertEqual(all_prop, self.randomseries.max_drawdown_date)

    def test_openframe_max_drawdown_date(self):

        mddsim = ReturnSimulation.from_normal(n=5, d=252, mu=0.05, vol=0.1, seed=71)
        mddframe = sim_to_openframe(mddsim, dt.date(2019, 6, 30)).to_cumret()
        self.assertListEqual(
            [
                dt.date(2018, 8, 15),
                dt.date(2018, 7, 2),
                dt.date(2018, 8, 3),
                dt.date(2018, 10, 3),
                dt.date(2018, 10, 17),
            ],
            mddframe.max_drawdown_date.tolist(),
        )

    def test_openframe_make_portfolio(self):

        assets = 5
        mpsim = ReturnSimulation.from_normal(n=assets, d=252, mu=0.05, vol=0.1, seed=71)
        mpframe = sim_to_openframe(mpsim, dt.date(2019, 6, 30)).to_cumret()
        mpframe.weights = [1.0 / assets] * assets

        name = "portfolio"
        mptail = mpframe.make_portfolio(name=name).tail()
        mptail = mptail.applymap(lambda nn: f"{nn:.6f}")

        correct = ["1.037311", "1.039374", "1.039730", "1.044433", "1.045528"]
        wrong = ["1.037311", "1.039374", "1.039730", "1.044433", "1.045527"]
        true_tail = pd.DataFrame(
            columns=pd.MultiIndex.from_product([[name], ["Price(Close)"]]),
            index=[
                dt.date(2019, 6, 24),
                dt.date(2019, 6, 25),
                dt.date(2019, 6, 26),
                dt.date(2019, 6, 27),
                dt.date(2019, 6, 28),
            ],
            data=correct,
        )
        false_tail = pd.DataFrame(
            columns=pd.MultiIndex.from_product([[name], ["Price(Close)"]]),
            index=[
                dt.date(2019, 6, 24),
                dt.date(2019, 6, 25),
                dt.date(2019, 6, 26),
                dt.date(2019, 6, 27),
                dt.date(2019, 6, 28),
            ],
            data=wrong,
        )

        assert_frame_equal(true_tail, mptail, check_exact=True)

        try:
            assert_frame_equal(false_tail, mptail, check_exact=True)
        except AssertionError as e:
            self.assertTrue(isinstance(e, AssertionError))

        mpframe.weights = None
        with self.assertRaises(Exception) as e_make:
            _ = mpframe.make_portfolio(name=name)

        self.assertEqual(
            e_make.exception.args[0],
            (
                "OpenFrame weights property must be provided to run the "
                "make_portfolio method."
            ),
        )

    def test_opentimeseries_running_adjustment(self):

        simadj = ReturnSimulation.from_merton_jump_gbm(
            n=1,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        adjustedseries = sim_to_opentimeseries(
            simadj, end=dt.date(2019, 6, 30)
        ).to_cumret()
        adjustedseries.running_adjustment(0.05)

        self.assertEqual(
            f"{1.689055852583:.12f}",
            f"{float(adjustedseries.tsdf.iloc[-1]):.12f}",
        )
        adjustedseries_returns = sim_to_opentimeseries(
            simadj, end=dt.date(2019, 6, 30)
        )
        adjustedseries_returns.running_adjustment(0.05)

        self.assertEqual(
            f"{0.009114963334:.12f}",
            f"{float(adjustedseries_returns.tsdf.iloc[-1]):.12f}",
        )

        adjustedseries_returns.to_cumret()
        self.assertEqual(
            f"{float(adjustedseries.tsdf.iloc[-1]):.12f}",
            f"{float(adjustedseries_returns.tsdf.iloc[-1]):.12f}",
        )

    @staticmethod
    def create_list_randomseries(num_series: int) -> list:

        sims = []
        np.random.seed(71)
        for g in range(num_series):
            sim_0 = ReturnSimulation.from_normal(
                n=1, d=100, mu=0.05, vol=0.1, seed=None
            )
            series = sim_to_opentimeseries(sim_0, end=dt.date(2019, 6, 30))
            series.set_new_label(lvl_zero=f"Asset_{g}")
            sims.append(series)
        return sims

    def test_returnsimulation_toframe_vs_toseries(self):

        n = 10
        frame_0 = OpenFrame(self.create_list_randomseries(n)).to_cumret()
        dict_toseries = frame_0.tsdf.to_dict()

        sim_1 = ReturnSimulation.from_normal(n=n, d=100, mu=0.05, vol=0.1, seed=71)
        frame_1 = sim_to_openframe(sim_1, end=dt.date(2019, 6, 30))
        frame_1.to_cumret()
        dict_toframe = frame_1.tsdf.to_dict()

        self.assertDictEqual(dict_toseries, dict_toframe)

    def test_openframe_add_timeseries(self):

        n = 4
        sims = self.create_list_randomseries(n)

        frame_0 = OpenFrame(sims[:-1])
        frame_0.add_timeseries(sims[-1])

        dict_toseries = frame_0.tsdf.to_dict()

        sim_1 = ReturnSimulation.from_normal(n=4, d=100, mu=0.05, vol=0.1, seed=71)
        frame_1 = sim_to_openframe(sim_1, end=dt.date(2019, 6, 30))

        dict_toframe = frame_1.tsdf.to_dict()

        self.assertDictEqual(dict_toseries, dict_toframe)

    def test_openframe_delete_timeseries(self):

        dsim = ReturnSimulation.from_normal(n=4, d=100, mu=0.05, vol=0.1, seed=71)

        frame = sim_to_openframe(dsim, end=dt.date(2019, 6, 30))
        frame.weights = [0.4, 0.1, 0.2, 0.3]

        lbl = "Asset_1"
        frame.delete_timeseries(lbl)
        labels = [ff.label for ff in frame.constituents]

        self.assertListEqual(labels, ["Asset_0", "Asset_2", "Asset_3"])
        self.assertListEqual(frame.weights, [0.4, 0.2, 0.3])

    def test_openframe_tocumret_and_back_toret(self):

        fmt = "{:.12f}"

        sim_0 = ReturnSimulation.from_normal(n=4, d=61, mu=0.05, vol=0.1, seed=71)
        frame_0 = sim_to_openframe(sim_0, end=dt.date(2019, 6, 30))

        frame_0.to_cumret()
        frame_0.value_to_ret()
        frame_0.to_cumret()
        frame_0.value_to_ret()
        frame_0.tsdf = frame_0.tsdf.applymap(lambda x: fmt.format(x))

        dict_toframe_0 = frame_0.tsdf.to_dict()

        sim_1 = ReturnSimulation.from_normal(n=4, d=61, mu=0.05, vol=0.1, seed=71)
        frame_1 = sim_to_openframe(sim_1, end=dt.date(2019, 6, 30))

        # The below adjustment is not ideal but I believe I implemented it
        # to mimic behaviour of Bbg return series.
        frame_1.tsdf.iloc[0] = 0

        frame_1.tsdf = frame_1.tsdf.applymap(lambda x: fmt.format(x))

        dict_toframe_1 = frame_1.tsdf.to_dict()

        self.assertDictEqual(dict_toframe_0, dict_toframe_1)

    def test_risk_functions_same_as_series_and_frame_methods(self):

        riskdata = ReturnSimulation.from_normal(n=1, d=504, mu=0.05, vol=0.175, seed=71)

        riskseries = sim_to_opentimeseries(riskdata, end=dt.date(2019, 6, 30))
        riskseries.set_new_label(lvl_zero="Asset_0")
        riskframe = sim_to_openframe(riskdata, end=dt.date(2019, 6, 30))
        riskseries.to_cumret()
        riskframe.to_cumret()

        self.assertEqual(
            riskseries.cvar_down,
            cvar_down(riskseries.tsdf.iloc[:, 0].tolist()),
            msg="CVaR for OpenTimeSeries not equal",
        )
        self.assertEqual(
            riskseries.var_down,
            var_down(riskseries.tsdf.iloc[:, 0].tolist()),
            msg="VaR for OpenTimeSeries not equal",
        )

        self.assertEqual(
            riskframe.cvar_down.iloc[0],
            cvar_down(riskframe.tsdf.iloc[:, 0]),
            msg="CVaR for OpenFrame not equal",
        )
        self.assertEqual(
            riskframe.var_down.iloc[0],
            var_down(riskframe.tsdf.iloc[:, 0]),
            msg="VaR for OpenFrame not equal",
        )

        self.assertEqual(
            riskframe.cvar_down.iloc[0],
            cvar_down(riskframe.tsdf),
            msg="CVaR for OpenFrame not equal",
        )
        self.assertEqual(
            riskframe.var_down.iloc[0],
            var_down(riskframe.tsdf),
            msg="VaR for OpenFrame not equal",
        )

    def test_openframe_methods_same_as_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=1, d=504, mu=0.05, vol=0.175, seed=71)

        sameseries = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30))
        sameseries.set_new_label(lvl_zero="Asset_0")
        sameframe = sim_to_openframe(same, end=dt.date(2019, 6, 30))
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        methods = [
            "rolling_return",
            "rolling_vol",
            "rolling_var_down",
            "rolling_cvar_down",
        ]
        for method in methods:
            assert_frame_equal(
                getattr(sameseries, method)(),
                getattr(sameframe, method)(column=0),
            )

        cumseries = sameseries.from_deepcopy()
        cumframe = sameframe.from_deepcopy()

        cumseries.value_to_log()
        cumframe.value_to_log()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.value_to_ret()
        sameframe.value_to_ret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.resample()
        sameframe.resample()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.value_to_diff()
        sameframe.value_to_diff()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

    def test_openframe_calc_methods_same_as_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=2, d=504, mu=0.05, vol=0.175, seed=71)

        sames = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30)).to_cumret()
        sames.set_new_label(lvl_zero="Asset_0")
        samef = sim_to_openframe(same, end=dt.date(2019, 6, 30)).to_cumret()

        methods_to_compare = [
            "arithmetic_ret_func",
            "cvar_down_func",
            "downside_deviation_func",
            "geo_ret_func",
            "kurtosis_func",
            "max_drawdown_func",
            "positive_share_func",
            "skew_func",
            "target_weight_from_var",
            "value_ret_func",
            "var_down_func",
            "vol_from_var_func",
            "vol_func",
            "worst_func",
            "z_score_func",
        ]
        for m in methods_to_compare:
            self.assertEqual(
                f"{getattr(sames, m)(months_from_last=12):.11f}",
                f"{float(getattr(samef, m)(months_from_last=12).iloc[0]):.11f}",
            )

    def test_openframe_ratio_methods_same_as_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=2, d=504, mu=0.05, vol=0.175, seed=71)

        sames = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30)).to_cumret()
        sames.set_new_label(lvl_zero="Asset_0")
        samef = sim_to_openframe(same, end=dt.date(2019, 6, 30)).to_cumret()

        self.assertEqual(
            f"{sames.ret_vol_ratio_func(months_from_last=12):.11f}",
            f"{float(samef.ret_vol_ratio_func(riskfree_rate=0.0, months_from_last=12).iloc[0]):.11f}",
        )

        self.assertEqual(
            f"{sames.sortino_ratio_func(months_from_last=12):.11f}",
            f"{float(samef.sortino_ratio_func(riskfree_rate=0.0, months_from_last=12).iloc[0]):.11f}",
        )

    def test_opentimeseries_measures_same_as_openframe_measures(self):

        sims = []
        np.random.seed(71)
        for g in range(10):
            sim_0 = ReturnSimulation.from_normal(
                n=1, d=100, mu=0.05, vol=0.1, seed=None
            )
            series = sim_to_opentimeseries(sim_0, end=dt.date(2019, 6, 30))
            series.set_new_label(lvl_zero=f"Asset_{g}")
            series.to_cumret()
            sims.append(series)
        frame_0 = OpenFrame(sims).to_cumret()
        common_calc_props = [
            "arithmetic_ret",
            "cvar_down",
            "downside_deviation",
            "geo_ret",
            "kurtosis",
            "max_drawdown",
            "max_drawdown_cal_year",
            "positive_share",
            "ret_vol_ratio",
            "skew",
            "sortino_ratio",
            "value_ret",
            "var_down",
            "vol",
            "vol_from_var",
            "worst",
            "worst_month",
            "z_score",
        ]
        series_measures = []
        frame_measures = []
        for p in common_calc_props:
            fr = getattr(frame_0, p).tolist()
            fr = [f"{ff:.10f}" for ff in fr]
            frame_measures.append(fr)
            se = [f"{getattr(s, p):.10f}" for s in sims]
            series_measures.append(se)

        self.assertListEqual(series_measures, frame_measures)

    def test_openframe_properties_same_as_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=1, d=504, mu=0.05, vol=0.175, seed=71)
        sameseries = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30)).to_cumret()
        sameseries.set_new_label(lvl_zero="Asset_0")
        sameframe = sim_to_openframe(same, end=dt.date(2019, 6, 30)).to_cumret()

        common_props_to_compare = ["periods_in_a_year", "yearfrac"]
        for c in common_props_to_compare:
            self.assertEqual(getattr(sameseries, c), getattr(sameframe, c))

    def test_keeping_attributes_aligned_openframe_vs_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=1, d=255, mu=0.05, vol=0.175, seed=71)
        sameseries = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30)).to_cumret()
        sameseries.set_new_label(lvl_zero="Asset_0")
        sameframe = sim_to_openframe(same, end=dt.date(2019, 6, 30)).to_cumret()

        common_calc_props = [
            "arithmetic_ret",
            "cvar_down",
            "downside_deviation",
            "geo_ret",
            "kurtosis",
            "max_drawdown",
            "max_drawdown_cal_year",
            "positive_share",
            "ret_vol_ratio",
            "skew",
            "sortino_ratio",
            "value_ret",
            "var_down",
            "vol",
            "vol_from_var",
            "worst",
            "worst_month",
            "z_score",
        ]

        common_props = ["periods_in_a_year", "yearfrac", "max_drawdown_date"]

        common_attributes = [
            "length",
            "span_of_days",
            "first_idx",
            "last_idx",
            "tsdf",
        ]

        series_attributes = [
            "values",
            "local_ccy",
            "_id",
            "instrumentId",
            "currency",
            "isin",
            "dates",
            "name",
            "valuetype",
            "label",
            "domestic",
            "sweden",
        ]

        frame_attributes = [
            "constituents",
            "columns_lvl_zero",
            "columns_lvl_one",
            "item_count",
            "weights",
            "first_indices",
            "last_indices",
            "lengths_of_items",
            "span_of_days_all",
        ]

        frame_calc_props = ["correl_matrix"]

        series_props = [
            a
            for a in dir(sameseries)
            if not a.startswith("__") and not callable(getattr(sameseries, a))
        ]
        series_compared = set(series_props).symmetric_difference(
            set(
                common_calc_props + common_props + common_attributes + series_attributes
            )
        )
        self.assertTrue(
            len(series_compared) == 0, msg=f"Difference is: {series_compared}"
        )
        frame_props = [
            a
            for a in dir(sameframe)
            if not a.startswith("__") and not callable(getattr(sameframe, a))
        ]
        frame_compared = set(frame_props).symmetric_difference(
            set(
                common_calc_props
                + common_props
                + common_attributes
                + frame_attributes
                + frame_calc_props
            )
        )
        self.assertTrue(
            len(frame_compared) == 0, msg=f"Difference is: {frame_compared}"
        )

    def test_keeping_methods_aligned_openframe_vs_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=1, d=255, mu=0.05, vol=0.175, seed=71)
        sameseries = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30)).to_cumret()
        sameseries.set_new_label(lvl_zero="Asset_0")
        sameframe = sim_to_openframe(same, end=dt.date(2019, 6, 30)).to_cumret()

        common_calc_methods = [
            "arithmetic_ret_func",
            "cvar_down_func",
            "downside_deviation_func",
            "geo_ret_func",
            "kurtosis_func",
            "max_drawdown_func",
            "positive_share_func",
            "ret_vol_ratio_func",
            "skew_func",
            "sortino_ratio_func",
            "target_weight_from_var",
            "value_ret_func",
            "var_down_func",
            "vol_from_var_func",
            "vol_func",
            "worst_func",
            "z_score_func",
        ]

        common_methods = [
            "align_index_to_local_cdays",
            "all_properties",
            "calc_range",
            "drawdown_details",
            "from_deepcopy",
            "plot_series",
            "resample",
            "return_nan_handle",
            "rolling_return",
            "rolling_vol",
            "rolling_cvar_down",
            "rolling_var_down",
            "to_cumret",
            "to_drawdown_series",
            "value_nan_handle",
            "value_ret_calendar_period",
            "value_to_diff",
            "value_to_log",
            "value_to_ret",
        ]

        series_createmethods = [
            "from_open_api",
            "from_open_nav",
            "from_open_fundinfo",
            "from_df",
            "from_frame",
            "from_fixed_rate",
        ]

        series_unique = [
            "pandas_df",
            "running_adjustment",
            "set_new_label",
            "to_json",
            "setup_class",
        ]

        frame_unique = [
            "add_timeseries",
            "beta",
            "delete_timeseries",
            "rolling_info_ratio",
            "info_ratio_func",
            "tracking_error_func",
            "capture_ratio_func",
            "ord_least_squares_fit",
            "make_portfolio",
            "relative",
            "rolling_corr",
            "rolling_beta",
            "trunc_frame",
        ]

        series_methods = [
            a
            for a in dir(sameseries)
            if not a.startswith("__") and callable(getattr(sameseries, a))
        ]
        series_compared = set(series_methods).symmetric_difference(
            set(
                common_calc_methods
                + common_methods
                + series_createmethods
                + series_unique
            )
        )
        self.assertTrue(
            len(series_compared) == 0, msg=f"Difference is: {series_compared}"
        )

        frame_methods = [
            a
            for a in dir(sameframe)
            if not a.startswith("__") and callable(getattr(sameframe, a))
        ]
        frame_compared = set(frame_methods).symmetric_difference(
            set(common_calc_methods + common_methods + frame_unique)
        )
        self.assertTrue(
            len(frame_compared) == 0, msg=f"Difference is: {frame_compared}"
        )

    def test_openframe_value_to_log(self):

        logsim = ReturnSimulation.from_normal(n=5, d=252, mu=0.05, vol=0.1, seed=71)
        logframe = sim_to_openframe(logsim, dt.date(2019, 6, 30)).to_cumret()

        aa = logframe.tsdf.applymap(lambda nn: f"{nn:.12f}")
        bb = aa.to_dict(orient="list")
        b4_log = [bb[k] for k in bb]

        logframe.value_to_log()

        aa = logframe.tsdf.applymap(lambda nn: f"{nn:.12f}")
        bb = aa.to_dict(orient="list")
        middle_log = [bb[k] for k in bb]

        self.assertNotEqual(b4_log, middle_log)

    def test_openframe_correl_matrix(self):

        corrsim = ReturnSimulation.from_normal(n=5, d=252, mu=0.05, vol=0.1, seed=71)
        corrframe = sim_to_openframe(corrsim, dt.date(2019, 6, 30)).to_cumret()
        dict1 = corrframe.correl_matrix.applymap(lambda nn: f"{nn:.12f}").to_dict()
        dict2 = {
            "Asset_0": {
                "Asset_0": "1.000000000000",
                "Asset_1": "0.008448597235",
                "Asset_2": "0.059458117640",
                "Asset_3": "0.071395739932",
                "Asset_4": "0.087545728279",
            },
            "Asset_1": {
                "Asset_0": "0.008448597235",
                "Asset_1": "1.000000000000",
                "Asset_2": "-0.040605114787",
                "Asset_3": "0.030023445985",
                "Asset_4": "0.074249393671",
            },
            "Asset_2": {
                "Asset_0": "0.059458117640",
                "Asset_1": "-0.040605114787",
                "Asset_2": "1.000000000000",
                "Asset_3": "-0.015715823407",
                "Asset_4": "0.064477746560",
            },
            "Asset_3": {
                "Asset_0": "0.071395739932",
                "Asset_1": "0.030023445985",
                "Asset_2": "-0.015715823407",
                "Asset_3": "1.000000000000",
                "Asset_4": "0.038405133612",
            },
            "Asset_4": {
                "Asset_0": "0.087545728279",
                "Asset_1": "0.074249393671",
                "Asset_2": "0.064477746560",
                "Asset_3": "0.038405133612",
                "Asset_4": "1.000000000000",
            },
        }

        self.assertDictEqual(dict1, dict2)

    def test_timeseries_chain(self):

        full_sim = ReturnSimulation.from_normal(n=1, d=252, mu=0.05, vol=0.1, seed=71)
        full_series = sim_to_opentimeseries(
            full_sim, end=dt.date(2019, 6, 30)
        ).to_cumret()
        full_values = [f"{nn:.10f}" for nn in full_series.tsdf.iloc[:, 0].tolist()]

        front_series = OpenTimeSeries.from_df(full_series.tsdf.iloc[:126])

        back_series = OpenTimeSeries.from_df(
            full_series.tsdf.loc[front_series.last_idx :]
        )

        chained_series = timeseries_chain(front_series, back_series)
        chained_values = [f"{nn:.10f}" for nn in chained_series.values]

        self.assertListEqual(full_series.dates, chained_series.dates)
        self.assertListEqual(full_values, chained_values)

    def test_opentimeseries_plot_series(self):

        plotsim = ReturnSimulation.from_normal(n=1, d=252, mu=0.05, vol=0.1, seed=71)
        plotseries = sim_to_opentimeseries(plotsim, end=dt.date(2019, 6, 30))
        fig, _ = plotseries.plot_series(auto_open=False, output_type="div")
        fig_json = json.loads(fig.to_json())
        fig_keys = list(fig_json.keys())
        self.assertListEqual(fig_keys, ["data", "layout"])

        fig_last, _ = plotseries.plot_series(
            auto_open=False, output_type="div", show_last=True
        )
        fig_last_json = json.loads(fig_last.to_json())
        last = fig_last_json["data"][-1]["y"][0]
        self.assertEqual(f"{last:.12f}", "0.001319755187")

        fig_last_fmt, _ = plotseries.plot_series(
            auto_open=False, output_type="div", show_last=True, tick_fmt=".3%"
        )
        fig_last_fmt_json = json.loads(fig_last_fmt.to_json())
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]
        self.assertEqual(last_fmt, "Last 0.132%")

    def test_openframe_plot_series(self):

        plotsims = ReturnSimulation.from_normal(n=5, d=252, mu=0.05, vol=0.1, seed=71)
        plotframe = sim_to_openframe(plotsims, dt.date(2019, 6, 30)).to_cumret()
        fig, _ = plotframe.plot_series(auto_open=False, output_type="div")
        fig_json = json.loads(fig.to_json())
        fig_keys = list(fig_json.keys())
        self.assertListEqual(fig_keys, ["data", "layout"])

        fig_last, _ = plotframe.plot_series(
            auto_open=False, output_type="div", show_last=True
        )
        fig_last_json = json.loads(fig_last.to_json())
        last = fig_last_json["data"][-1]["y"][0]
        self.assertEqual(f"{last:.12f}", "0.994000592625")

        fig_last_fmt, _ = plotframe.plot_series(
            auto_open=False, output_type="div", show_last=True, tick_fmt=".3%"
        )
        fig_last_fmt_json = json.loads(fig_last_fmt.to_json())
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]
        self.assertEqual(last_fmt, "Last 99.400%")

        with self.assertRaises(AssertionError) as e_plot:
            _, _ = plotframe.plot_series(auto_open=False, labels=["a", "b"])

        self.assertIsInstance(e_plot.exception, AssertionError)

    def test_openframe_passed_empty_list(self):

        with LogCapture() as log:
            OpenFrame([])
            ll = log.actual()
            self.assertEqual(
                ll,
                [("root", "WARNING", "OpenFrame() was passed an empty list.")],
            )

    def test_openframe_wrong_number_of_weights_passed(self):

        n = 5
        wghts = [1.0 / n] * (n + 1)
        wrongsims = ReturnSimulation.from_normal(n=n, d=252, mu=0.05, vol=0.1, seed=71)
        date_range = [
            d.date()
            for d in pd.date_range(
                periods=wrongsims.trading_days,
                end=dt.date(2019, 6, 30),
                freq=CDay(calendar=OpenTimeSeries.sweden),
            )
        ]
        tslist = []
        for item in range(wrongsims.number_of_sims):
            sdf = wrongsims.df.iloc[item].T.to_frame()
            sdf.index = date_range
            sdf.columns = pd.MultiIndex.from_product(
                [[f"Asset_{item}"], ["Return(Total)"]]
            )
            tslist.append(OpenTimeSeries.from_df(sdf, valuetype="Return(Total)"))

        with self.assertRaises(Exception) as e_weights:
            OpenFrame(tslist, weights=wghts)

        self.assertEqual(
            "Number of TimeSeries must equal number of weights.",
            e_weights.exception.args[0],
        )

    def test_opentimeseries_drawdown_details(self):

        details = self.randomseries.drawdown_details()
        self.assertEqual(
            "{:7f}".format(details.loc["Max Drawdown", "Drawdown details"]),
            "-0.400116",
        )
        self.assertEqual(
            details.loc["Start of drawdown", "Drawdown details"],
            dt.date(2012, 7, 5),
        )
        self.assertEqual(
            details.loc["Date of bottom", "Drawdown details"],
            dt.date(2018, 11, 8),
        )
        self.assertEqual(
            details.loc["Days from start to bottom", "Drawdown details"], 2317
        )
        self.assertEqual(
            "{:.9f}".format(details.loc["Average fall per day", "Drawdown details"]),
            "-0.000172687",
        )

    def test_openframe_drawdown_details(self):

        ddsims = ReturnSimulation.from_normal(n=5, d=252, mu=0.05, vol=0.1, seed=71)
        ddframe = sim_to_openframe(ddsims, dt.date(2019, 6, 30)).to_cumret()
        dds = ddframe.drawdown_details().loc["Days from start to bottom"].tolist()
        self.assertListEqual([1, 5, 36, 89, 79], dds)

    def test_openframe_trunc_frame(self):

        sim_long = ReturnSimulation.from_normal(n=1, d=756, mu=0.05, vol=0.1, seed=71)
        series_long = sim_to_opentimeseries(sim_long, end=dt.date(2020, 6, 30))
        series_long.set_new_label("Long")
        sim_short = ReturnSimulation.from_normal(n=1, d=252, mu=0.05, vol=0.1, seed=71)
        series_short = sim_to_opentimeseries(sim_short, end=dt.date(2019, 6, 30))
        series_short.set_new_label("Short")
        frame = OpenFrame([series_long, series_short])

        firsts = [
            dt.date(2018, 6, 27),
            dt.date(2018, 6, 27),
        ]
        lasts = [
            dt.date(2019, 6, 28),
            dt.date(2019, 6, 28),
        ]

        self.assertNotEqual(firsts, frame.first_indices.tolist())
        self.assertNotEqual(lasts, frame.last_indices.tolist())

        frame.trunc_frame()

        self.assertListEqual(firsts, frame.first_indices.tolist())
        self.assertListEqual(lasts, frame.last_indices.tolist())

        trunced = [dt.date(2019, 1, 2), dt.date(2019, 3, 29)]

        frame.trunc_frame(start_cut=trunced[0], end_cut=trunced[1])

        self.assertListEqual(trunced, [frame.first_idx, frame.last_idx])

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

    def test_frenklaopenapiservice_repr(self):

        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        service = FrenklaOpenApiService()
        r = "FrenklaOpenApiService(" "base_url=https://api.frenkla.com/public/api/)\n"
        print(service)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        self.assertEqual(r, output)

    def test_frenklaopenapiservice_get_timeseries(self):

        sevice = FrenklaOpenApiService()
        ts_id = "62d06f9d753964781e81f185"
        series = sevice.get_timeseries(timeseries_id=ts_id)

        self.assertEqual(ts_id, series["id"])

        with self.assertRaises(Exception) as e_unique:
            sevice.get_timeseries(timeseries_id="")

        self.assertEqual(int(str(e_unique.exception)[:3]), 404)

    def test_frenklaopenapiservice_get_fundinfo(self):

        sevice = FrenklaOpenApiService()
        isin_code = "SE0009807308"
        fundinfo = sevice.get_fundinfo(isins=[isin_code])

        self.assertEqual(isin_code, fundinfo[0]["classes"][0]["isin"])

        fundinfo_date = sevice.get_fundinfo(
            isins=[isin_code], report_date=dt.date(2022, 6, 30)
        )

        self.assertEqual(isin_code, fundinfo_date[0]["classes"][0]["isin"])

        with self.assertRaises(Exception) as e_unique:
            isin_cde = ""
            _ = sevice.get_fundinfo(isins=[isin_cde])

        self.assertEqual(int(str(e_unique.exception)[:3]), 400)

    def test_frenklaopenapiservice_get_nav(self):

        sevice = FrenklaOpenApiService()
        isin_code = "SE0009807308"
        series = sevice.get_nav(isin=isin_code)

        self.assertEqual(isin_code, series["isin"])

        with self.assertRaises(Exception) as e_unique:
            isin_cde = ""
            _ = sevice.get_nav(isin=isin_cde)

        self.assertEqual(
            f"Request for NAV series using ISIN {isin_cde} returned no data.",
            e_unique.exception.args[0],
        )

    def test_frenklaopenapiservice_get_nav_to_dataframe(self):

        sevice = FrenklaOpenApiService()
        isin_code = "SE0009807308"
        df = sevice.get_nav_to_dataframe(isin=isin_code).head()
        ddf = pd.DataFrame(
            data=[100.0000, 100.0978, 100.2821, 100.1741, 100.4561],
            index=pd.DatetimeIndex(
                [
                    "2017-05-29",
                    "2017-05-30",
                    "2017-05-31",
                    "2017-06-01",
                    "2017-06-02",
                ]
            ),
            columns=["Captor Iris Bond, SE0009807308"],
        )
        assert_frame_equal(df, ddf)

        with self.assertRaises(Exception) as e_unique:
            isin_cde = ""
            _ = sevice.get_nav_to_dataframe(isin=isin_cde).head()

        self.assertEqual(
            f"Request for NAV series using ISIN {isin_cde} returned no data.",
            e_unique.exception.args[0],
        )

    def test_openframe_all_properties(self):

        prop_index = [
            "Total return",
            "Geometric return",
            "Arithmetic return",
            "Volatility",
            "Downside deviation",
            "Return vol ratio",
            "Sortino ratio",
            "Z-score",
            "Skew",
            "Kurtosis",
            "Positive share",
            "VaR 95.0%",
            "CVaR 95.0%",
            "Imp vol from VaR 95%",
            "Worst",
            "Worst month",
            "Max drawdown",
            "Max drawdown dates",
            "Max drawdown in cal yr",
            "first indices",
            "last indices",
            "observations",
            "span of days",
        ]
        apsims = ReturnSimulation.from_normal(n=5, d=252, mu=0.05, vol=0.1, seed=71)
        apframe = sim_to_openframe(apsims, dt.date(2019, 6, 30)).to_cumret()
        result_index = apframe.all_properties().index.tolist()

        self.assertListEqual(prop_index, result_index)

    def test_opentimeseries_align_index_to_local_cdays(self):

        date_range = [
            d.date() for d in pd.date_range(start="2020-06-15", end="2020-06-25")
        ]
        asim = [1.0] * len(date_range)
        adf = pd.DataFrame(
            data=asim,
            index=date_range,
            columns=pd.MultiIndex.from_product([["Asset"], ["Price(Close)"]]),
        )
        aseries = OpenTimeSeries.from_df(adf, valuetype="Price(Close)")

        midsummer = dt.date(2020, 6, 19)
        self.assertTrue(midsummer in date_range)

        aseries.align_index_to_local_cdays()
        self.assertFalse(midsummer in aseries.tsdf.index)

    def test_openframe_align_index_to_local_cdays(self):

        date_range = [
            d.date() for d in pd.date_range(start="2022-06-01", end="2022-06-15")
        ]
        asim = [1.0] * len(date_range)
        adf = pd.DataFrame(
            data=asim,
            index=date_range,
            columns=pd.MultiIndex.from_product([["Asset_a"], ["Price(Close)"]]),
        )
        aseries = OpenTimeSeries.from_df(adf, valuetype="Price(Close)")
        bseries = OpenTimeSeries.from_df(adf, valuetype="Price(Close)")
        bseries.set_new_label("Asset_b")
        aframe = OpenFrame([aseries, bseries])

        midsummer = dt.date(2022, 6, 6)
        self.assertTrue(midsummer in date_range)

        aframe.align_index_to_local_cdays()
        self.assertFalse(midsummer in aframe.tsdf.index)

    def test_openframe_rolling_info_ratio(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=5,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()

        simdata = frame.rolling_info_ratio(long_column=0, short_column=1)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.22029628522",
            "0.16342938866",
            "0.19954924433",
            "0.19579197546",
            "0.20346268143",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

        simdata_fxd_per_yr = frame.rolling_info_ratio(
            long_column=0, short_column=1, periods_in_a_year_fixed=251
        )

        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5, 0]]
        checkdata_fxd_per_yr = [
            "0.22045949626",
            "0.16355046869",
            "0.19969708449",
            "0.19593703197",
            "0.20361342094",
        ]
        self.assertListEqual(values_fxd_per_yr, checkdata_fxd_per_yr)

    def test_openframe_rolling_beta(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=5,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()
        simdata = frame.rolling_beta(asset_column=0, market_column=1)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "-0.05129437067",
            "-0.07071418405",
            "-0.05236923352",
            "-0.04282093703",
            "-0.08012220038",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

    def test_openframe_ret_vol_ratio_func(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=5,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()

        simdataa = frame.ret_vol_ratio_func(riskfree_column=-1)

        self.assertEqual(f"{simdataa[0]:.10f}", "0.1580040085")

        simdatab = frame.ret_vol_ratio_func(
            riskfree_column=-1, periods_in_a_year_fixed=251
        )

        self.assertEqual(f"{simdatab[0]:.10f}", "0.1578870346")

        simdatac = frame.ret_vol_ratio_func(riskfree_column=("Asset_4", "Price(Close)"))

        self.assertEqual(f"{simdatac[0]:.10f}", "0.1580040085")

        self.assertEqual(f"{simdataa[0]:.10f}", f"{simdatac[0]:.10f}")

        with self.assertRaises(Exception) as e_retvolfunc:
            # noinspection PyTypeChecker
            _ = frame.ret_vol_ratio_func(riskfree_column="string")

        self.assertEqual(
            e_retvolfunc.exception.args[0],
            "base_column should be a tuple or an integer.",
        )

    def test_openframe_sortino_ratio_func(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=5,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()

        simdataa = frame.sortino_ratio_func(riskfree_column=-1)

        self.assertEqual(f"{simdataa[0]:.10f}", "0.2009532877")

        simdatab = frame.sortino_ratio_func(
            riskfree_column=-1, periods_in_a_year_fixed=251
        )

        self.assertEqual(f"{simdatab[0]:.10f}", "0.2008045175")

        simdatac = frame.sortino_ratio_func(riskfree_column=("Asset_4", "Price(Close)"))

        self.assertEqual(f"{simdataa[0]:.10f}", f"{simdatac[0]:.10f}")

        self.assertEqual(f"{simdatac[0]:.10f}", "0.2009532877")

        with self.assertRaises(Exception) as e_func:
            # noinspection PyTypeChecker
            _ = frame.sortino_ratio_func(riskfree_column="string")

        self.assertEqual(
            e_func.exception.args[0],
            "base_column should be a tuple or an integer.",
        )

    def test_openframe_tracking_error_func(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=5,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()

        simdataa = frame.tracking_error_func(base_column=-1)

        self.assertEqual(f"{simdataa[0]:.10f}", "0.2462231908")

        simdatab = frame.tracking_error_func(
            base_column=-1, periods_in_a_year_fixed=251
        )

        self.assertEqual(f"{simdatab[0]:.10f}", "0.2460409063")

        simdatac = frame.tracking_error_func(base_column=("Asset_4", "Price(Close)"))

        self.assertEqual(f"{simdatac[0]:.10f}", "0.2462231908")

        self.assertEqual(f"{simdataa[0]:.10f}", f"{simdatac[0]:.10f}")

        with self.assertRaises(Exception) as e_func:
            # noinspection PyTypeChecker
            _ = frame.tracking_error_func(base_column="string")

        self.assertEqual(
            e_func.exception.args[0],
            "base_column should be a tuple or an integer.",
        )

    def test_openframe_info_ratio_func(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=5,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()

        simdataa = frame.info_ratio_func(base_column=-1)

        self.assertEqual(f"{simdataa[0]:.10f}", "0.2063067697")

        simdatab = frame.info_ratio_func(base_column=-1, periods_in_a_year_fixed=251)

        self.assertEqual(f"{simdatab[0]:.10f}", "0.2061540363")

        simdatac = frame.info_ratio_func(base_column=("Asset_4", "Price(Close)"))

        self.assertEqual(f"{simdatac[0]:.10f}", "0.2063067697")

        with self.assertRaises(Exception) as e_func:
            # noinspection PyTypeChecker
            _ = frame.info_ratio_func(base_column="string")

        self.assertEqual(
            e_func.exception.args[0],
            "base_column should be a tuple or an integer.",
        )

    def test_openframe_rolling_corr(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=5,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()
        simdata = frame.rolling_corr(first_column=0, second_column=1)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "-0.06170179015",
            "-0.08612430578",
            "-0.06462318798",
            "-0.05487880293",
            "-0.10634855725",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

    def test_openframe_rolling_vol(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=1,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()

        simdata = frame.rolling_vol(column=0, observations=21)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.08745000502",
            "0.08809050608",
            "0.08832329638",
            "0.08671269840",
            "0.08300985872",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

        simdata_fxd_per_yr = frame.rolling_vol(
            column=0, observations=21, periods_in_a_year_fixed=251
        )

        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5, 0]]
        checkdata_fxd_per_yr = [
            "0.08738526385",
            "0.08802529073",
            "0.08825790869",
            "0.08664850307",
            "0.08294840469",
        ]
        self.assertListEqual(values_fxd_per_yr, checkdata_fxd_per_yr)

    def test_openframe_rolling_return(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=1,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()

        simdata = frame.rolling_return(column=0, observations=21)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "-0.01477558639",
            "-0.01662326401",
            "-0.01735881460",
            "-0.02138743793",
            "-0.03592486809",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

    def test_openframe_rolling_cvar_down(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=1,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()

        simdata = frame.rolling_cvar_down(column=0, observations=21)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[-5:, 0]]
        checkdata = [
            "-0.01337460746",
            "-0.01337460746",
            "-0.01337460746",
            "-0.01270193467",
            "-0.01270193467",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

    def test_openframe_rolling_var_down(self):

        sims = ReturnSimulation.from_merton_jump_gbm(
            n=1,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        frame = sim_to_openframe(sims, dt.date(2019, 6, 30)).to_cumret()

        simdata = frame.rolling_var_down(column=0, observations=21)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[-5:, 0]]
        checkdata = [
            "-0.01342248045",
            "-0.01342248045",
            "-0.01342248045",
            "-0.01342248045",
            "-0.01342248045",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

    def test_opentimeseries_rolling_vol(self):

        simdata = self.randomseries.rolling_vol(observations=21)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.08745000502",
            "0.08809050608",
            "0.08832329638",
            "0.08671269840",
            "0.08300985872",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

        simdata_fxd_per_yr = self.randomseries.rolling_vol(observations=21, periods_in_a_year_fixed=251)

        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5, 0]]
        checkdata_fxd_per_yr = [
            "0.08738526385",
            "0.08802529073",
            "0.08825790869",
            "0.08664850307",
            "0.08294840469",
        ]
        self.assertListEqual(values_fxd_per_yr, checkdata_fxd_per_yr)

    def test_opentimeseries_rolling_return(self):

        simdata = self.randomseries.rolling_return(observations=21)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "-0.01477558639",
            "-0.01662326401",
            "-0.01735881460",
            "-0.02138743793",
            "-0.03592486809",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

    def test_opentimeseries_rolling_cvar_down(self):

        simdata = self.randomseries.rolling_cvar_down(observations=21)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[-5:, 0]]
        checkdata = [
            "-0.01337460746",
            "-0.01337460746",
            "-0.01337460746",
            "-0.01270193467",
            "-0.01270193467",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

    def test_opentimeseries_rolling_var_down(self):

        simdata = self.randomseries.rolling_var_down(observations=21)
        simseries = OpenTimeSeries.from_df(simdata)

        values = [f"{v:.11f}" for v in simdata.iloc[-5:, 0]]
        checkdata = [
            "-0.01342248045",
            "-0.01342248045",
            "-0.01342248045",
            "-0.01342248045",
            "-0.01342248045",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

    def test_openframe_label_uniqueness(self):

        usim = ReturnSimulation.from_normal(n=1, d=1200, mu=0.05, vol=0.1, seed=71)
        aseries = sim_to_opentimeseries(usim, end=dt.date(2019, 6, 30))
        bseries = sim_to_opentimeseries(usim, end=dt.date(2019, 6, 30))

        with self.assertRaises(Exception) as e_unique:
            OpenFrame([aseries, bseries])

        self.assertEqual(
            "TimeSeries names/labels must be unique.", e_unique.exception.args[0]
        )

        bseries.set_new_label("other_name")
        uframe = OpenFrame([aseries, bseries])

        self.assertEqual(len(set(uframe.columns_lvl_zero)), 2)

    def test_openframe_capture_ratio(self):

        """
        Source: 'Capture Ratios: A Popular Method of Measuring Portfolio Performance
        in Practice', Don R. Cox and Delbert C. Goff, Journal of Economics and
        Finance Education (Vol 2 Winter 2013).
        https://www.economics-finance.org/jefe/volume12-2/11ArticleCox.pdf
        """

        asset = OpenTimeSeries(
            {
                "_id": "",
                "currency": "USD",
                "dates": [
                    "2007-12-31",
                    "2008-01-31",
                    "2008-02-29",
                    "2008-03-31",
                    "2008-04-30",
                    "2008-05-31",
                    "2008-06-30",
                    "2008-07-31",
                    "2008-08-31",
                    "2008-09-30",
                    "2008-10-31",
                    "2008-11-30",
                    "2008-12-31",
                    "2009-01-31",
                    "2009-02-28",
                    "2009-03-31",
                    "2009-04-30",
                    "2009-05-31",
                    "2009-06-30",
                    "2009-07-31",
                    "2009-08-31",
                    "2009-09-30",
                    "2009-10-31",
                    "2009-11-30",
                    "2009-12-31",
                ],
                "instrumentId": "",
                "local_ccy": True,
                "name": "asset",
                "values": [
                    0.0,
                    -0.0436,
                    -0.0217,
                    -0.0036,
                    0.0623,
                    0.0255,
                    -0.0555,
                    -0.0124,
                    -0.0088,
                    -0.0643,
                    -0.1897,
                    -0.0781,
                    0.0248,
                    -0.0666,
                    -0.0993,
                    0.0853,
                    0.1028,
                    0.0634,
                    -0.0125,
                    0.0762,
                    0.0398,
                    0.048,
                    -0.0052,
                    0.0592,
                    0.0195,
                ],
                "valuetype": "Return(Total)",
            }
        )
        indxx = OpenTimeSeries(
            {
                "_id": "",
                "currency": "USD",
                "dates": [
                    "2007-12-31",
                    "2008-01-31",
                    "2008-02-29",
                    "2008-03-31",
                    "2008-04-30",
                    "2008-05-31",
                    "2008-06-30",
                    "2008-07-31",
                    "2008-08-31",
                    "2008-09-30",
                    "2008-10-31",
                    "2008-11-30",
                    "2008-12-31",
                    "2009-01-31",
                    "2009-02-28",
                    "2009-03-31",
                    "2009-04-30",
                    "2009-05-31",
                    "2009-06-30",
                    "2009-07-31",
                    "2009-08-31",
                    "2009-09-30",
                    "2009-10-31",
                    "2009-11-30",
                    "2009-12-31",
                ],
                "instrumentId": "",
                "local_ccy": True,
                "name": "indxx",
                "values": [
                    0.0,
                    -0.06,
                    -0.0325,
                    -0.0043,
                    0.0487,
                    0.013,
                    -0.0843,
                    -0.0084,
                    0.0145,
                    -0.0891,
                    -0.168,
                    -0.0718,
                    0.0106,
                    -0.0843,
                    -0.1065,
                    0.0876,
                    0.0957,
                    0.0559,
                    0.002,
                    0.0756,
                    0.0361,
                    0.0373,
                    -0.0186,
                    0.06,
                    0.0193,
                ],
                "valuetype": "Return(Total)",
            }
        )
        cframe = OpenFrame([asset, indxx]).to_cumret()

        up = cframe.capture_ratio_func(ratio="up")
        down = cframe.capture_ratio_func(ratio="down")
        both = cframe.capture_ratio_func(ratio="both")

        self.assertEqual(f"{up.iloc[0]:.12f}", "1.063842457805")
        self.assertEqual(f"{down.iloc[0]:.12f}", "0.922188852957")
        self.assertEqual(f"{both.iloc[0]:.12f}", "1.153605852417")

        upfixed = cframe.capture_ratio_func(ratio="up", periods_in_a_year_fixed=12)

        self.assertEqual(f"{upfixed.iloc[0]:.12f}", "1.063217236138")
        self.assertAlmostEqual(up.iloc[0], upfixed.iloc[0], places=2)

        uptuple = cframe.capture_ratio_func(
            ratio="up", base_column=("indxx", "Price(Close)")
        )

        self.assertEqual(f"{uptuple.iloc[0]:.12f}", "1.063842457805")
        self.assertEqual(f"{up.iloc[0]:.12f}", f"{uptuple.iloc[0]:.12f}")

        with self.assertRaises(Exception) as e_func:
            # noinspection PyTypeChecker
            _ = cframe.capture_ratio_func(ratio="up", base_column="string")

        self.assertEqual(
            e_func.exception.args[0],
            "base_column should be a tuple or an integer.",
        )

    def test_opentimeseries_downside_deviation(self):

        """
        Source:
        https://www.investopedia.com/terms/d/downside-deviation.asp
        """

        dd_asset = OpenTimeSeries(
            {
                "_id": "",
                "currency": "USD",
                "dates": [
                    "2010-12-31",
                    "2011-12-31",
                    "2012-12-31",
                    "2013-12-31",
                    "2014-12-31",
                    "2015-12-31",
                    "2016-12-31",
                    "2017-12-31",
                    "2018-12-31",
                    "2019-12-31",
                ],
                "instrumentId": "",
                "local_ccy": True,
                "name": "asset",
                "values": [
                    0.0,
                    -0.02,
                    0.16,
                    0.31,
                    0.17,
                    -0.11,
                    0.21,
                    0.26,
                    -0.03,
                    0.38,
                ],
                "valuetype": "Return(Total)",
            }
        ).to_cumret()

        mar = 0.01
        downdev = dd_asset.downside_deviation_func(
            min_accepted_return=mar, periods_in_a_year_fixed=1
        )

        self.assertEqual(f"{downdev:.12f}", "0.043333333333")

    def test_opentimeseries_validations(self):

        valid_isin = "SE0009807308"
        invalid_isin_one = "SE0009807307"
        invalid_isin_two = "SE000980730B"

        timeseries_with_valid_isin = OpenTimeSeries(
            {
                "_id": "",
                "currency": "SEK",
                "dates": [
                    "2017-05-29",
                    "2017-05-30",
                ],
                "instrumentId": "",
                "isin": valid_isin,
                "local_ccy": True,
                "name": "asset",
                "values": [
                    100.0,
                    100.0978,
                ],
                "valuetype": "Price(Close)",
            }
        )
        self.assertIsInstance(timeseries_with_valid_isin, OpenTimeSeries)

        new_dict = dict(timeseries_with_valid_isin.__dict__)
        cleaner_list = [
            "local_ccy",
            "tsdf",
        ]  # 'local_ccy' removed to trigger ValidationError
        for item in cleaner_list:
            new_dict.pop(item)

        with self.assertRaises(Exception) as e_ccy:
            OpenTimeSeries(new_dict)

        # noinspection PyUnresolvedReferences
        self.assertIn(member="local_ccy", container=e_ccy.exception.message)

        new_dict.pop("label")
        new_dict.update(
            {"local_ccy": True, "dates": []}
        )  # Set dates to empty array to trigger minItems ValidationError

        with self.assertRaises(Exception) as e_min_items:
            OpenTimeSeries(new_dict)

        # noinspection PyUnresolvedReferences
        self.assertIn(member="is too short", container=e_min_items.exception.message)

        with self.assertRaises(Exception) as e_one:
            OpenTimeSeries(
                {
                    "_id": "",
                    "currency": "SEK",
                    "dates": [
                        "2017-05-29",
                        "2017-05-30",
                    ],
                    "instrumentId": "",
                    "isin": invalid_isin_one,
                    "local_ccy": True,
                    "name": "asset",
                    "values": [
                        100.0,
                        100.0978,
                    ],
                    "valuetype": "Price(Close)",
                }
            )
        self.assertIsInstance(e_one.exception, InvalidChecksum)

        with self.assertRaises(Exception) as e_two:
            OpenTimeSeries(
                {
                    "_id": "",
                    "currency": "SEK",
                    "dates": [
                        "2017-05-29",
                        "2017-05-30",
                    ],
                    "instrumentId": "",
                    "isin": invalid_isin_two,
                    "local_ccy": True,
                    "name": "asset",
                    "values": [
                        100.0,
                        100.0978,
                    ],
                    "valuetype": "Price(Close)",
                }
            )
        # noinspection PyUnresolvedReferences
        self.assertIn(member="does not match", container=e_two.exception.message)

    def test_opentimeseries_geo_ret_value_ret_exceptions(self):

        geoseries = OpenTimeSeries(
            TimeSerie(
                _id="",
                name="geoseries",
                currency="SEK",
                instrumentId="",
                local_ccy=True,
                valuetype="Price(Close)",
                dates=["2022-07-01", "2023-07-01"],
                values=[1.0, 1.1],
            )
        )
        self.assertEqual(f"{geoseries.geo_ret:.7f}", "0.1000718")
        self.assertEqual(f"{geoseries.geo_ret_func():.7f}", "0.1000718")

        zeroseries = OpenTimeSeries(
            TimeSerie(
                _id="",
                name="zeroseries",
                currency="SEK",
                instrumentId="",
                local_ccy=True,
                valuetype="Price(Close)",
                dates=["2022-07-01", "2023-07-01"],
                values=[0.0, 1.1],
            )
        )
        with self.assertRaises(Exception) as e_gr_zero:
            _ = zeroseries.geo_ret

        self.assertEqual(
            e_gr_zero.exception.args[0],
            "Geometric return cannot be calculated due to an initial value being zero or a negative value.",
        )

        with self.assertRaises(Exception) as e_grf_zero:
            _ = zeroseries.geo_ret_func()

        self.assertEqual(
            e_grf_zero.exception.args[0],
            "Geometric return cannot be calculated due to an initial value being zero or a negative value.",
        )
        with self.assertRaises(Exception) as e_vr_zero:
            _ = zeroseries.value_ret

        self.assertEqual(
            e_vr_zero.exception.args[0],
            "Simple Return cannot be calculated due to an initial value being zero.",
        )

        with self.assertRaises(Exception) as e_vrf_zero:
            _ = zeroseries.value_ret_func()

        self.assertEqual(
            e_vrf_zero.exception.args[0],
            "Simple Return cannot be calculated due to an initial value being zero.",
        )

        negseries = OpenTimeSeries(
            TimeSerie(
                _id="",
                name="negseries",
                currency="SEK",
                instrumentId="",
                local_ccy=True,
                valuetype="Price(Close)",
                dates=["2022-07-01", "2023-07-01"],
                values=[1.0, -0.1],
            )
        )

        with self.assertRaises(Exception) as e_gr_neg:
            _ = negseries.geo_ret

        self.assertEqual(
            e_gr_neg.exception.args[0],
            "Geometric return cannot be calculated due to an initial value being zero or a negative value.",
        )

        with self.assertRaises(Exception) as e_grf_neg:
            _ = negseries.geo_ret_func()

        self.assertEqual(
            e_grf_neg.exception.args[0],
            "Geometric return cannot be calculated due to an initial value being zero or a negative value.",
        )

    def test_openframe_georet_exceptions(self):

        geoframe = OpenFrame(
            [
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="geoseries1",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Price(Close)",
                        dates=["2022-07-01", "2023-07-01"],
                        values=[1.0, 1.1],
                    )
                ),
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="geoseries2",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Price(Close)",
                        dates=["2022-07-01", "2023-07-01"],
                        values=[1.0, 1.2],
                    )
                ),
            ]
        )
        self.assertListEqual(
            [f"{gr:.5f}" for gr in geoframe.geo_ret], ["0.10007", "0.20015"]
        )

        self.assertListEqual(
            [f"{gr:.5f}" for gr in geoframe.geo_ret_func()], ["0.10007", "0.20015"]
        )

        geoframe.add_timeseries(
            OpenTimeSeries(
                TimeSerie(
                    _id="",
                    name="geoseries3",
                    currency="SEK",
                    instrumentId="",
                    local_ccy=True,
                    valuetype="Price(Close)",
                    dates=["2022-07-01", "2023-07-01"],
                    values=[0.0, 1.1],
                )
            )
        )
        with self.assertRaises(Exception) as e_gr_zero:
            _ = geoframe.geo_ret

        self.assertEqual(
            e_gr_zero.exception.args[0],
            "Geometric return cannot be calculated due to an initial value being zero or a negative value.",
        )

        with self.assertRaises(Exception) as e_grf_zero:
            _ = geoframe.geo_ret_func()

        self.assertEqual(
            e_grf_zero.exception.args[0],
            "Geometric return cannot be calculated due to an initial value being zero or a negative value.",
        )

        geoframe.delete_timeseries(lvl_zero_item="geoseries3")

        self.assertListEqual(
            [f"{gr:.5f}" for gr in geoframe.geo_ret], ["0.10007", "0.20015"]
        )

        geoframe.add_timeseries(
            OpenTimeSeries(
                TimeSerie(
                    _id="",
                    name="geoseries4",
                    currency="SEK",
                    instrumentId="",
                    local_ccy=True,
                    valuetype="Price(Close)",
                    dates=["2022-07-01", "2023-07-01"],
                    values=[1.0, -1.1],
                )
            )
        )
        with self.assertRaises(Exception) as e_gr_neg:
            _ = geoframe.geo_ret

        self.assertEqual(
            e_gr_neg.exception.args[0],
            "Geometric return cannot be calculated due to an initial value being zero or a negative value.",
        )

        with self.assertRaises(Exception) as e_grf_neg:
            _ = geoframe.geo_ret_func()

        self.assertEqual(
            e_grf_neg.exception.args[0],
            "Geometric return cannot be calculated due to an initial value being zero or a negative value.",
        )

    def test_returnsimulation_properties(self):

        days = 1200
        psim = ReturnSimulation.from_normal(n=1, d=days, mu=0.05, vol=0.1, seed=71)

        self.assertIsInstance(psim.results, pd.DataFrame)

        self.assertEqual(psim.results.shape[0], days)

        self.assertEqual(f"{psim.realized_mean_return:.9f}", "0.028832246")

        self.assertEqual(f"{psim.realized_vol:.9f}", "0.096596353")

    def test_opentimeseries_value_nan_handle(self):

        nanseries = OpenTimeSeries(
            TimeSerie(
                _id="",
                name="nanseries",
                currency="SEK",
                instrumentId="",
                local_ccy=True,
                valuetype="Price(Close)",
                dates=[
                    "2022-07-11",
                    "2022-07-12",
                    "2022-07-13",
                    "2022-07-14",
                    "2022-07-15",
                ],
                values=[1.1, 1.0, 0.8, 1.1, 1.0],
            )
        )
        nanseries.tsdf.iloc[2, 0] = None
        dropseries = nanseries.from_deepcopy()
        dropseries.value_nan_handle(method="drop")
        self.assertListEqual([1.1, 1.0, 1.1, 1.0], dropseries.tsdf.iloc[:, 0].tolist())

        fillseries = nanseries.from_deepcopy()
        fillseries.value_nan_handle(method="fill")
        self.assertListEqual(
            [1.1, 1.0, 1.0, 1.1, 1.0], fillseries.tsdf.iloc[:, 0].tolist()
        )

        with self.assertRaises(AssertionError) as e_method:
            _ = nanseries.value_nan_handle(method="other")

        self.assertEqual(
            e_method.exception.args[0],
            "Method must be either fill or drop passed as string.",
        )

    def test_opentimeseries_return_nan_handle(self):

        nanseries = OpenTimeSeries(
            TimeSerie(
                _id="",
                name="nanseries",
                currency="SEK",
                instrumentId="",
                local_ccy=True,
                valuetype="Return(Total)",
                dates=[
                    "2022-07-11",
                    "2022-07-12",
                    "2022-07-13",
                    "2022-07-14",
                    "2022-07-15",
                ],
                values=[0.1, 0.05, 0.03, 0.01, 0.04],
            )
        )
        nanseries.tsdf.iloc[2, 0] = None
        dropseries = nanseries.from_deepcopy()
        dropseries.return_nan_handle(method="drop")
        self.assertListEqual(
            [0.1, 0.05, 0.01, 0.04], dropseries.tsdf.iloc[:, 0].tolist()
        )

        fillseries = nanseries.from_deepcopy()
        fillseries.return_nan_handle(method="fill")
        self.assertListEqual(
            [0.1, 0.05, 0.0, 0.01, 0.04], fillseries.tsdf.iloc[:, 0].tolist()
        )

        with self.assertRaises(AssertionError) as e_method:
            _ = nanseries.return_nan_handle(method="other")

        self.assertEqual(
            e_method.exception.args[0],
            "Method must be either fill or drop passed as string.",
        )

    def test_openframe_value_nan_handle(self):

        nanframe = OpenFrame(
            [
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="nanseries1",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Price(Close)",
                        dates=[
                            "2022-07-11",
                            "2022-07-12",
                            "2022-07-13",
                            "2022-07-14",
                            "2022-07-15",
                        ],
                        values=[1.1, 1.0, 0.8, 1.1, 1.0],
                    )
                ),
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="nanseries2",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Price(Close)",
                        dates=[
                            "2022-07-11",
                            "2022-07-12",
                            "2022-07-13",
                            "2022-07-14",
                            "2022-07-15",
                        ],
                        values=[2.1, 2.0, 1.8, 2.1, 2.0],
                    )
                ),
            ]
        )
        nanframe.tsdf.iloc[2, 0] = None
        nanframe.tsdf.iloc[3, 1] = None
        dropframe = nanframe.from_deepcopy()
        dropframe.value_nan_handle(method="drop")
        self.assertListEqual([1.1, 1.0, 1.0], dropframe.tsdf.iloc[:, 0].tolist())
        self.assertListEqual([2.1, 2.0, 2.0], dropframe.tsdf.iloc[:, 1].tolist())

        fillframe = nanframe.from_deepcopy()
        fillframe.value_nan_handle(method="fill")
        self.assertListEqual(
            [1.1, 1.0, 1.0, 1.1, 1.0], fillframe.tsdf.iloc[:, 0].tolist()
        )
        self.assertListEqual(
            [2.1, 2.0, 1.8, 1.8, 2.0], fillframe.tsdf.iloc[:, 1].tolist()
        )

        with self.assertRaises(AssertionError) as e_methd:
            _ = nanframe.value_nan_handle(method="other")

        self.assertEqual(
            e_methd.exception.args[0],
            "Method must be either fill or drop passed as string.",
        )

    def test_openframe_return_nan_handle(self):

        nanframe = OpenFrame(
            [
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="nanseries1",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Return(Total)",
                        dates=[
                            "2022-07-11",
                            "2022-07-12",
                            "2022-07-13",
                            "2022-07-14",
                            "2022-07-15",
                        ],
                        values=[0.1, 0.05, 0.03, 0.01, 0.04],
                    )
                ),
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="nanseries2",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Return(Total)",
                        dates=[
                            "2022-07-11",
                            "2022-07-12",
                            "2022-07-13",
                            "2022-07-14",
                            "2022-07-15",
                        ],
                        values=[0.01, 0.04, 0.02, 0.11, 0.06],
                    )
                ),
            ]
        )
        nanframe.tsdf.iloc[2, 0] = None
        nanframe.tsdf.iloc[3, 1] = None
        dropframe = nanframe.from_deepcopy()
        dropframe.return_nan_handle(method="drop")
        self.assertListEqual([0.1, 0.05, 0.04], dropframe.tsdf.iloc[:, 0].tolist())
        self.assertListEqual([0.01, 0.04, 0.06], dropframe.tsdf.iloc[:, 1].tolist())

        fillframe = nanframe.from_deepcopy()
        fillframe.return_nan_handle(method="fill")
        self.assertListEqual(
            [0.1, 0.05, 0.0, 0.01, 0.04], fillframe.tsdf.iloc[:, 0].tolist()
        )
        self.assertListEqual(
            [0.01, 0.04, 0.02, 0.0, 0.06], fillframe.tsdf.iloc[:, 1].tolist()
        )

        with self.assertRaises(AssertionError) as e_methd:
            _ = nanframe.return_nan_handle(method="other")

        self.assertEqual(
            e_methd.exception.args[0],
            "Method must be either fill or drop passed as string.",
        )

    def test_openframe_relative(self):

        rsims = ReturnSimulation.from_normal(n=5, d=504, mu=0.05, vol=0.1, seed=71)
        rframe = sim_to_openframe(rsims, dt.date(2019, 6, 30)).to_cumret()
        sframe = rframe.from_deepcopy()

        rframe.relative()
        self.assertEqual("Asset_0_over_Asset_1", rframe.columns_lvl_zero[-1])
        rframe.tsdf.iloc[:, -1] = rframe.tsdf.iloc[:, -1].add(1.0)

        sframe.relative(base_zero=False)

        rf = [f"{rr:.11f}" for rr in rframe.tsdf.iloc[:, -1]]
        sf = [f"{rr:.11f}" for rr in sframe.tsdf.iloc[:, -1]]

        self.assertListEqual(rf, sf)

    def test_openframe_miscellaneous(self):

        msims = ReturnSimulation.from_normal(n=5, d=504, mu=0.05, vol=0.1, seed=71)
        mframe = sim_to_openframe(msims, dt.date(2019, 6, 30)).to_cumret()

        methods = [
            "arithmetic_ret_func",
            "vol_func",
            "vol_from_var_func",
            "downside_deviation_func",
            "target_weight_from_var",
        ]
        for methd in methods:
            no_fixed = getattr(mframe, methd)()
            fixed = getattr(mframe, methd)(periods_in_a_year_fixed=252)
            for nf, f in zip(no_fixed, fixed):
                self.assertAlmostEqual(nf, f, places=2)
                self.assertNotAlmostEqual(nf, f, places=6)

        ret = [f"{rr:.9f}" for rr in mframe.value_ret_func()]
        self.assertListEqual(
            [
                "0.275697802",
                "-0.033533800",
                "0.001519387",
                "0.061395949",
                "-0.040958695",
            ],
            ret,
        )

        impvol = [f"{iv:.11f}" for iv in mframe.vol_from_var_func(drift_adjust=False)]
        impvoldrifted = [
            f"{iv:.11f}" for iv in mframe.vol_from_var_func(drift_adjust=True)
        ]

        self.assertListEqual(
            impvol,
            [
                "0.09346157986",
                "0.09356336505",
                "0.10329974747",
                "0.10611467842",
                "0.10533221112",
            ],
        )
        self.assertListEqual(
            impvoldrifted,
            [
                "0.09831616286",
                "0.09307200988",
                "0.10352124920",
                "0.10745704093",
                "0.10474133708",
            ],
        )

        mframe.tsdf.iloc[0, 2] = 0.0

        r = (
            "Error in function value_ret due to an initial value being zero. "
            "(                Asset_0      Asset_1      Asset_2      Asset_3      "
            "Asset_4\n"
            "           Price(Close) Price(Close) Price(Close) Price(Close) Price(Close)\n"
            "2017-06-27     1.000000     1.000000     0.000000     1.000000     1.000000\n"
            "2017-06-28     0.992677     1.013048     1.006154     0.998712     0.999713\n"
            "2017-06-29     0.990096     1.005543     1.013704     0.998940     1.004026)"
        )

        with self.assertRaises(Exception) as e_vr:
            _ = mframe.value_ret

        self.assertEqual(e_vr.exception.args[0], r)

        with self.assertRaises(Exception) as e_vrf:
            _ = mframe.value_ret_func()

        self.assertEqual(e_vrf.exception.args[0], r)

    def test_opentimeseries_miscellaneous(self):

        msims = ReturnSimulation.from_normal(n=1, d=504, mu=0.05, vol=0.1, seed=71)
        mseries = sim_to_opentimeseries(msims, dt.date(2019, 6, 30)).to_cumret()

        methods = [
            "arithmetic_ret_func",
            "vol_func",
            "vol_from_var_func",
            "downside_deviation_func",
            "target_weight_from_var",
        ]
        for methd in methods:
            no_fixed = getattr(mseries, methd)()
            fixed = getattr(mseries, methd)(periods_in_a_year_fixed=252)
            self.assertAlmostEqual(no_fixed, fixed, places=2)
            self.assertNotAlmostEqual(no_fixed, fixed, places=6)

        impvol = mseries.vol_from_var_func(drift_adjust=False)
        self.assertEqual(f"{impvol:.12f}", "0.093461579863")
        impvoldrifted = mseries.vol_from_var_func(drift_adjust=True)
        self.assertEqual(f"{impvoldrifted:.12f}", "0.098316162865")

    def test_value_ret_calendar_period(self):

        vrcsims = ReturnSimulation.from_normal(n=5, d=504, mu=0.05, vol=0.1, seed=71)
        vrcseries = sim_to_opentimeseries(vrcsims, dt.date(2019, 6, 30)).to_cumret()
        vrcframe = sim_to_openframe(vrcsims, dt.date(2019, 6, 30)).to_cumret()

        vrfs_y = vrcseries.value_ret_func(
            from_date=dt.date(2017, 12, 29), to_date=dt.date(2018, 12, 28)
        )
        vrff_y = vrcframe.value_ret_func(
            from_date=dt.date(2017, 12, 29), to_date=dt.date(2018, 12, 28)
        )
        vrffl_y = [f"{rr:.11f}" for rr in vrff_y]

        vrvrcs_y = vrcseries.value_ret_calendar_period(year=2018)
        vrvrcf_y = vrcframe.value_ret_calendar_period(year=2018)
        vrvrcfl_y = [f"{rr:.11f}" for rr in vrvrcf_y]

        self.assertEqual(f"{vrfs_y:.11f}", f"{vrvrcs_y:.11f}")
        self.assertListEqual(vrffl_y, vrvrcfl_y)

        vrfs_ym = vrcseries.value_ret_func(
            from_date=dt.date(2018, 4, 30), to_date=dt.date(2018, 5, 31)
        )
        vrff_ym = vrcframe.value_ret_func(
            from_date=dt.date(2018, 4, 30), to_date=dt.date(2018, 5, 31)
        )
        vrffl_ym = [f"{rr:.11f}" for rr in vrff_ym]

        vrvrcs_ym = vrcseries.value_ret_calendar_period(year=2018, month=5)
        vrvrcf_ym = vrcframe.value_ret_calendar_period(year=2018, month=5)
        vrvrcfl_ym = [f"{rr:.11f}" for rr in vrvrcf_ym]

        self.assertEqual(f"{vrfs_ym:.11f}", f"{vrvrcs_ym:.11f}")
        self.assertListEqual(vrffl_ym, vrvrcfl_ym)

    def test_opentimeseries_to_drawdown_series(self):

        msims = ReturnSimulation.from_normal(n=1, d=2520, mu=0.05, vol=0.1, seed=71)
        mseries = sim_to_opentimeseries(msims, dt.date(2019, 6, 30)).to_cumret()
        dd = mseries.max_drawdown
        mseries.to_drawdown_series()
        dds = float(mseries.tsdf.min())
        self.assertEqual(f"{dd:.11f}", f"{dds:.11f}")

    def test_openframe_to_drawdown_series(self):

        msims = ReturnSimulation.from_normal(n=5, d=2520, mu=0.05, vol=0.1, seed=71)
        mframe = sim_to_openframe(msims, dt.date(2019, 6, 30)).to_cumret()
        dd = [f"{dmax:.11f}" for dmax in mframe.max_drawdown]
        mframe.to_drawdown_series()
        dds = [f"{dmax:.11f}" for dmax in mframe.tsdf.min()]
        self.assertListEqual(dd, dds)

    def test_openframe_ord_least_squares_fit(self):

        osims = ReturnSimulation.from_lognormal(n=5, d=2520, mu=0.05, vol=0.1, seed=71)
        oframe = sim_to_openframe(osims, dt.date(2019, 6, 30)).to_cumret()
        oframe.value_to_log()

        fsframe = oframe.from_deepcopy()
        fsframe.ord_least_squares_fit(y_column=0, x_column=1, fitted_series=True)
        self.assertEqual(fsframe.columns_lvl_zero[-1], oframe.columns_lvl_zero[0])
        self.assertEqual(fsframe.columns_lvl_one[-1], oframe.columns_lvl_zero[1])

        results = []
        for i in range(oframe.item_count):
            for j in range(oframe.item_count):
                results.append(
                    f"{oframe.ord_least_squares_fit(y_column=i, x_column=j, fitted_series=False):.11f}"
                )

        results_tuple = []
        for i in oframe.tsdf:
            for j in oframe.tsdf:
                results_tuple.append(
                    f"{oframe.ord_least_squares_fit(y_column=i, x_column=j, fitted_series=False):.11f}"
                )

        self.assertListEqual(results, results_tuple)

        self.assertListEqual(
            results,
            [
                "1.00000000000",
                "0.55040796387",
                "0.60822717148",
                "0.37824573835",
                "0.32062039005",
                "1.37777232698",
                "1.00000000000",
                "1.03552512853",
                "0.68035802592",
                "0.57966252870",
                "1.34414305664",
                "0.91421337148",
                "1.00000000000",
                "0.63878222178",
                "0.53719279113",
                "1.94917298315",
                "1.40062251534",
                "1.48953078947",
                "1.00000000000",
                "0.82328905407",
                "2.30702559932",
                "1.66626378001",
                "1.74908846662",
                "1.14957491185",
                "1.00000000000",
            ],
        )
        with self.assertRaises(Exception) as e_x:
            # noinspection PyTypeChecker
            _ = oframe.ord_least_squares_fit(
                y_column=0, x_column="string", fitted_series=False
            )

        self.assertEqual(
            e_x.exception.args[0],
            "x_column should be a tuple or an integer.",
        )

        with self.assertRaises(Exception) as e_y:
            # noinspection PyTypeChecker
            _ = oframe.ord_least_squares_fit(
                y_column="string", x_column=1, fitted_series=False
            )

        self.assertEqual(
            e_y.exception.args[0],
            "y_column should be a tuple or an integer.",
        )

    def test_openframe_beta(self):

        bsims = ReturnSimulation.from_lognormal(n=5, d=2520, mu=0.05, vol=0.1, seed=71)
        bframe = sim_to_openframe(bsims, dt.date(2019, 6, 30)).to_cumret()
        bframe.resample("7D")
        results = []
        for i in range(bframe.item_count):
            for j in range(bframe.item_count):
                results.append(f"{bframe.beta(asset=i, market=j):.11f}")

        results_tuple = []
        for i in bframe.tsdf:
            for j in bframe.tsdf:
                results_tuple.append(f"{bframe.beta(asset=i, market=j):.11f}")

        self.assertListEqual(results, results_tuple)

        self.assertListEqual(
            results,
            [
                "1.00000000000",
                "0.14789008484",
                "0.22081056673",
                "0.08289437576",
                "0.06643502183",
                "0.83394218499",
                "1.00000000000",
                "1.03301969320",
                "0.62455936849",
                "0.55871763920",
                "0.92277039702",
                "0.76557107601",
                "1.00000000000",
                "0.53556526892",
                "0.44931511667",
                "0.98059414709",
                "1.31021049367",
                "1.51601268628",
                "1.00000000000",
                "0.77504650499",
                "1.04160776510",
                "1.55346911864",
                "1.68571579339",
                "1.02723700914",
                "1.00000000000",
            ],
        )
        with self.assertRaises(Exception) as e_asset:
            # noinspection PyTypeChecker
            _ = bframe.beta(asset="string", market=1)

        self.assertEqual(
            e_asset.exception.args[0],
            "asset should be a tuple or an integer.",
        )

        with self.assertRaises(Exception) as e_market:
            # noinspection PyTypeChecker
            _ = bframe.beta(asset=0, market="string")

        self.assertEqual(
            e_market.exception.args[0],
            "market should be a tuple or an integer.",
        )

    def test_openframe_beta_returns_input(self):

        bsims = ReturnSimulation.from_lognormal(n=5, d=2520, mu=0.05, vol=0.1, seed=71)
        bframe = sim_to_openframe(bsims, dt.date(2019, 6, 30))
        bframe.resample("7D")
        results = []
        for i in range(bframe.item_count):
            for j in range(bframe.item_count):
                results.append(f"{bframe.beta(asset=i, market=j):.11f}")

        results_tuple = []
        for i in bframe.tsdf:
            for j in bframe.tsdf:
                results_tuple.append(f"{bframe.beta(asset=i, market=j):.11f}")

        self.assertListEqual(results, results_tuple)

        self.assertListEqual(
            results,
            [
                "1.00000000000",
                "0.05604932482",
                "-0.10318577564",
                "0.00403166381",
                "0.03641081338",
                "0.05125978278",
                "1.00000000000",
                "0.07019744912",
                "-0.00011060548",
                "0.03145441859",
                "-0.10798723144",
                "0.08032810625",
                "1.00000000000",
                "-0.01124691224",
                "-0.07845911818",
                "0.00329118754",
                "-0.00009872760",
                "-0.00877301863",
                "1.00000000000",
                "0.00136012991",
                "0.03550321225",
                "0.03353609985",
                "-0.07310180776",
                "0.00162461081",
                "1.00000000000",
            ],
        )
        with self.assertRaises(Exception) as e_asset:
            # noinspection PyTypeChecker
            _ = bframe.beta(asset="string", market=1)

        self.assertEqual(
            e_asset.exception.args[0],
            "asset should be a tuple or an integer.",
        )

        with self.assertRaises(Exception) as e_market:
            # noinspection PyTypeChecker
            _ = bframe.beta(asset=0, market="string")

        self.assertEqual(
            e_market.exception.args[0],
            "market should be a tuple or an integer.",
        )

    def test_opentimeseries_set_new_label(self):

        lsims = ReturnSimulation.from_normal(n=1, d=252, mu=0.05, vol=0.1, seed=71)
        lseries = sim_to_opentimeseries(lsims, dt.date(2019, 6, 30))

        self.assertTupleEqual(lseries.tsdf.columns[0], ("Asset", "Return(Total)"))

        lseries.set_new_label(lvl_zero="zero")
        self.assertTupleEqual(lseries.tsdf.columns[0], ("zero", "Return(Total)"))

        lseries.set_new_label(lvl_one="one")
        self.assertTupleEqual(lseries.tsdf.columns[0], ("zero", "one"))

        lseries.set_new_label(lvl_zero="two", lvl_one="three")
        self.assertTupleEqual(lseries.tsdf.columns[0], ("two", "three"))

        lseries.set_new_label(delete_lvl_one=True)
        self.assertEqual(lseries.tsdf.columns[0], "two")
