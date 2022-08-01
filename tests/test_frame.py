# -*- coding: utf-8 -*-
import datetime as dt
from io import StringIO
from json import loads
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.tseries.offsets import CDay
import sys
from testfixtures import LogCapture
from unittest import TestCase

from openseries.frame import OpenFrame
from openseries.risk import cvar_down, var_down
from openseries.series import OpenTimeSeries, TimeSerie
from openseries.sim_price import ReturnSimulation


class TestOpenFrame(TestCase):
    sim: ReturnSimulation
    randomframe: OpenFrame
    randomseries: OpenTimeSeries

    @classmethod
    def setUpClass(cls):

        OpenTimeSeries.setup_class()

        sim = ReturnSimulation.from_merton_jump_gbm(
            n=5,
            d=2512,
            mu=0.05,
            vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        date_range = [
            d.date()
            for d in pd.date_range(
                periods=sim.trading_days,
                end=dt.date(2019, 6, 30),
                freq=CDay(calendar=OpenTimeSeries.sweden),
            )
        ]
        sdf = sim.df.iloc[0].T.to_frame()
        sdf.index = date_range
        sdf.columns = pd.MultiIndex.from_product([["Asset"], ["Return(Total)"]])

        cls.randomseries = OpenTimeSeries.from_df(
            sdf, valuetype="Return(Total)"
        ).to_cumret()

        tslist = []
        for item in range(sim.number_of_sims):
            sdf = sim.df.iloc[item].T.to_frame()
            sdf.index = date_range
            sdf.columns = pd.MultiIndex.from_product(
                [[f"Asset_{item}"], ["Return(Total)"]]
            )
            tslist.append(OpenTimeSeries.from_df(sdf, valuetype="Return(Total)"))

        cls.randomframe = OpenFrame(tslist)

    def test_openframe_repr(self):

        old_stdout = sys.stdout
        new_stdout = StringIO()
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

    def test_create_opentimeseries_from_frame(self):

        frame_f = self.randomframe.from_deepcopy()
        frame_f.to_cumret()
        fseries = OpenTimeSeries.from_frame(frame_f, label="Asset_1")

        self.assertTrue(isinstance(fseries, OpenTimeSeries))

    def test_openframe_calc_range(self):

        crframe = self.randomframe.from_deepcopy()
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

    def test_openframe_max_drawdown_date(self):

        mddframe = self.randomframe.from_deepcopy()
        mddframe.to_cumret()
        print(mddframe.max_drawdown_date.tolist())
        self.assertListEqual(
            [
                dt.date(2009, 7, 1),
                dt.date(2009, 7, 1),
                dt.date(2012, 4, 17),
                dt.date(2013, 7, 29),
                dt.date(2009, 7, 1),
            ],
            mddframe.max_drawdown_date.tolist(),
        )

    def test_openframe_make_portfolio(self):

        mpframe = self.randomframe.from_deepcopy()
        mpframe.to_cumret()
        mpframe.weights = [1.0 / mpframe.item_count] * mpframe.item_count

        name = "portfolio"
        mptail = mpframe.make_portfolio(name=name).tail()
        mptail = mptail.applymap(lambda nn: f"{nn:.6f}")

        correct = ["0.832536", "0.830516", "0.829576", "0.826926", "0.824288"]
        wrong = ["0.832536", "0.830516", "0.829576", "0.826926", "0.824285"]
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

    def test_openframe_add_timeseries(self):

        frameas = self.randomframe.from_deepcopy()
        items = int(frameas.item_count)
        cols = list(frameas.columns_lvl_zero)
        nbr_cols = int(len(frameas.columns_lvl_zero))
        seriesas = self.randomseries.from_deepcopy()
        seriesas.set_new_label("Asset_6")
        frameas.add_timeseries(seriesas)

        self.assertEqual(items + 1, frameas.item_count)
        self.assertEqual(nbr_cols + 1, len(frameas.columns_lvl_zero))
        self.assertListEqual(cols + ["Asset_6"], frameas.columns_lvl_zero)

    def test_openframe_delete_timeseries(self):

        frame = self.randomframe.from_deepcopy()
        frame.weights = [0.4, 0.1, 0.2, 0.1, 0.2]

        lbl = "Asset_1"
        frame.delete_timeseries(lbl)
        labels = [ff.label for ff in frame.constituents]

        self.assertListEqual(labels, ["Asset_0", "Asset_2", "Asset_3", "Asset_4"])
        self.assertListEqual(frame.weights, [0.4, 0.2, 0.1, 0.2])

    def test_risk_functions_same_as_series_and_frame_methods(self):

        riskseries = self.randomseries.from_deepcopy()
        riskseries.set_new_label(lvl_zero="Asset_0")
        riskframe = self.randomframe.from_deepcopy()
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

        sameseries = self.randomseries.from_deepcopy()
        sameseries.set_new_label(lvl_zero="Asset_0")
        sameseries.value_to_ret()
        sameframe = self.randomframe.from_deepcopy()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

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
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.value_to_ret()
        sameframe.value_to_ret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.resample()
        sameframe.resample()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.value_to_diff()
        sameframe.value_to_diff()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

    def test_openframe_calc_methods_same_as_opentimeseries(self):

        sames = self.randomseries.from_deepcopy()
        sames.to_cumret()
        sames.set_new_label(lvl_zero="Asset_0")
        samef = self.randomframe.from_deepcopy()
        samef.to_cumret()

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

        sames = self.randomseries.from_deepcopy()
        sames.to_cumret()
        sames.set_new_label(lvl_zero="Asset_0")
        samef = self.randomframe.from_deepcopy()
        samef.to_cumret()

        self.assertEqual(
            f"{sames.ret_vol_ratio_func(months_from_last=12):.11f}",
            f"{float(samef.ret_vol_ratio_func(riskfree_rate=0.0, months_from_last=12).iloc[0]):.11f}",
        )

        self.assertEqual(
            f"{sames.sortino_ratio_func(months_from_last=12):.11f}",
            f"{float(samef.sortino_ratio_func(riskfree_rate=0.0, months_from_last=12).iloc[0]):.11f}",
        )

    def test_opentimeseries_measures_same_as_openframe_measures(self):

        frame_0 = self.randomframe.from_deepcopy()
        for s in frame_0.constituents:
            s.to_cumret()
        frame_0.to_cumret()

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
            se = [f"{getattr(s, p):.10f}" for s in frame_0.constituents]
            series_measures.append(se)

        self.assertListEqual(series_measures, frame_measures)

    def test_openframe_properties_same_as_opentimeseries(self):

        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

        common_props_to_compare = ["periods_in_a_year", "yearfrac"]
        for c in common_props_to_compare:
            self.assertEqual(getattr(sameseries, c), getattr(sameframe, c))

    def test_keeping_attributes_aligned_openframe_vs_opentimeseries(self):

        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

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

        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

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

        logframe = self.randomframe.from_deepcopy()
        logframe.to_cumret()

        aa = logframe.tsdf.applymap(lambda nn: f"{nn:.12f}")
        bb = aa.to_dict(orient="list")
        b4_log = [bb[k] for k in bb]

        logframe.value_to_log()

        aa = logframe.tsdf.applymap(lambda nn: f"{nn:.12f}")
        bb = aa.to_dict(orient="list")
        middle_log = [bb[k] for k in bb]

        self.assertNotEqual(b4_log, middle_log)

    def test_openframe_correl_matrix(self):

        corrframe = self.randomframe.from_deepcopy()
        corrframe.to_cumret()
        dict1 = corrframe.correl_matrix.applymap(lambda nn: f"{nn:.12f}").to_dict()
        import pprint

        pprint.pprint(dict1)
        dict2 = {
            "Asset_0": {
                "Asset_0": "1.000000000000",
                "Asset_1": "0.017066083668",
                "Asset_2": "-0.007450037376",
                "Asset_3": "-0.004244941678",
                "Asset_4": "0.018417822676",
            },
            "Asset_1": {
                "Asset_0": "0.017066083668",
                "Asset_1": "1.000000000000",
                "Asset_2": "0.020864777227",
                "Asset_3": "-0.020027446255",
                "Asset_4": "-0.001683308306",
            },
            "Asset_2": {
                "Asset_0": "-0.007450037376",
                "Asset_1": "0.020864777227",
                "Asset_2": "1.000000000000",
                "Asset_3": "-0.026987004596",
                "Asset_4": "-0.002409387706",
            },
            "Asset_3": {
                "Asset_0": "-0.004244941678",
                "Asset_1": "-0.020027446255",
                "Asset_2": "-0.026987004596",
                "Asset_3": "1.000000000000",
                "Asset_4": "-0.000475629043",
            },
            "Asset_4": {
                "Asset_0": "0.018417822676",
                "Asset_1": "-0.001683308306",
                "Asset_2": "-0.002409387706",
                "Asset_3": "-0.000475629043",
                "Asset_4": "1.000000000000",
            },
        }

        self.assertDictEqual(dict1, dict2)

    def test_openframe_plot_series(self):

        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()
        fig, _ = plotframe.plot_series(auto_open=False, output_type="div")
        fig_json = loads(fig.to_json())
        fig_keys = list(fig_json.keys())
        self.assertListEqual(fig_keys, ["data", "layout"])

        fig_last, _ = plotframe.plot_series(
            auto_open=False, output_type="div", show_last=True
        )
        fig_last_json = loads(fig_last.to_json())
        last = fig_last_json["data"][-1]["y"][0]
        self.assertEqual(f"{last:.12f}", "0.778129717727")

        fig_last_fmt, _ = plotframe.plot_series(
            auto_open=False, output_type="div", show_last=True, tick_fmt=".3%"
        )
        fig_last_fmt_json = loads(fig_last_fmt.to_json())
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]
        self.assertEqual(last_fmt, "Last 77.813%")

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

        wrongsims = self.randomframe.from_deepcopy()
        tslist = list(wrongsims.constituents)
        wghts = [1.0 / wrongsims.item_count] * (wrongsims.item_count + 1)

        with self.assertRaises(Exception) as e_weights:
            OpenFrame(tslist, weights=wghts)

        self.assertEqual(
            "Number of TimeSeries must equal number of weights.",
            e_weights.exception.args[0],
        )

    def test_openframe_drawdown_details(self):

        ddframe = self.randomframe.from_deepcopy()
        for s in ddframe.constituents:
            s.to_cumret()
        ddframe.to_cumret()
        dds = ddframe.drawdown_details().loc["Days from start to bottom"].tolist()
        self.assertListEqual([2317, 1797, 2439, 1024, 1278], dds)

    def test_openframe_trunc_frame(self):

        series_long = self.randomseries.from_deepcopy()
        series_long.set_new_label("Long")
        tmp_series = self.randomseries.from_deepcopy()
        series_short = OpenTimeSeries.from_df(
            tmp_series.tsdf.loc[
                dt.date(2017, 6, 27) : dt.date(2018, 6, 27), ("Asset", "Price(Close)")
            ]
        )
        series_short.set_new_label("Short")
        frame = OpenFrame([series_long, series_short])

        firsts = [
            dt.date(2017, 6, 27),
            dt.date(2017, 6, 27),
        ]
        lasts = [
            dt.date(2018, 6, 27),
            dt.date(2018, 6, 27),
        ]

        self.assertNotEqual(firsts, frame.first_indices.tolist())
        self.assertNotEqual(lasts, frame.last_indices.tolist())

        frame.trunc_frame()

        self.assertListEqual(firsts, frame.first_indices.tolist())
        self.assertListEqual(lasts, frame.last_indices.tolist())

        trunced = [dt.date(2017, 12, 29), dt.date(2018, 3, 29)]

        frame.trunc_frame(start_cut=trunced[0], end_cut=trunced[1])

        self.assertListEqual(trunced, [frame.first_idx, frame.last_idx])

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
        apframe = self.randomframe.from_deepcopy()
        apframe.to_cumret()
        result_index = apframe.all_properties().index.tolist()

        self.assertListEqual(prop_index, result_index)

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()
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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

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

    def test_openframe_label_uniqueness(self):

        aseries = self.randomseries.from_deepcopy()
        bseries = self.randomseries.from_deepcopy()

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

        rframe = self.randomframe.from_deepcopy()
        rframe.to_cumret()
        sframe = self.randomframe.from_deepcopy()
        sframe.to_cumret()

        rframe.relative()
        self.assertEqual("Asset_0_over_Asset_1", rframe.columns_lvl_zero[-1])
        rframe.tsdf.iloc[:, -1] = rframe.tsdf.iloc[:, -1].add(1.0)

        sframe.relative(base_zero=False)

        rf = [f"{rr:.11f}" for rr in rframe.tsdf.iloc[:, -1]]
        sf = [f"{rr:.11f}" for rr in sframe.tsdf.iloc[:, -1]]

        self.assertListEqual(rf, sf)

    def test_openframe_to_cumret(self):

        rseries = self.randomseries.from_deepcopy()
        rseries.value_to_ret()
        rrseries = rseries.from_deepcopy()
        rrseries.set_new_label(lvl_zero="Rasset")

        cseries = self.randomseries.from_deepcopy()
        cseries.set_new_label(lvl_zero="Basset")
        ccseries = cseries.from_deepcopy()
        ccseries.set_new_label(lvl_zero="Casset")

        mframe = OpenFrame([rseries, cseries])
        cframe = OpenFrame([cseries, ccseries])
        rframe = OpenFrame([rseries, rrseries])

        self.assertListEqual(["Return(Total)", "Price(Close)"], mframe.columns_lvl_one)
        self.assertListEqual(["Price(Close)", "Price(Close)"], cframe.columns_lvl_one)
        cframe_lvl_one = list(cframe.columns_lvl_one)
        self.assertListEqual(["Return(Total)", "Return(Total)"], rframe.columns_lvl_one)

        mframe.to_cumret()
        cframe.to_cumret()
        rframe.to_cumret()

        self.assertListEqual(["Price(Close)", "Price(Close)"], mframe.columns_lvl_one)
        self.assertListEqual(cframe_lvl_one, cframe.columns_lvl_one)
        self.assertListEqual(["Price(Close)", "Price(Close)"], rframe.columns_lvl_one)

        fmt = "{:.12f}"

        frame_0 = self.randomframe.from_deepcopy()
        frame_0.to_cumret()
        frame_0.value_to_ret()
        frame_0.tsdf = frame_0.tsdf.applymap(lambda x: fmt.format(x))
        dict_toframe_0 = frame_0.tsdf.to_dict()

        frame_1 = self.randomframe.from_deepcopy()
        frame_1.tsdf = frame_1.tsdf.applymap(lambda x: fmt.format(x))
        dict_toframe_1 = frame_1.tsdf.to_dict()

        self.assertDictEqual(dict_toframe_0, dict_toframe_1)

    def test_openframe_miscellaneous(self):

        mframe = self.randomframe.from_deepcopy()
        mframe.to_cumret()

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
                "0.024471958",
                "-0.620625714",
                "-0.399460961",
                "0.245899647",
                "-0.221870282",
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
                "0.10208932904",
                "0.09911226523",
                "0.09785296425",
                "0.09587988606",
                "0.09653565636",
            ],
        )
        self.assertListEqual(
            impvoldrifted,
            [
                "0.10245462160",
                "0.09607641481",
                "0.09644421046",
                "0.09705532014",
                "0.09619264015",
            ],
        )

        mframe.tsdf.iloc[0, 2] = 0.0

        r = (
            "Error in function value_ret due to an initial value being zero. "
            "(                Asset_0      Asset_1      Asset_2      Asset_3      "
            "Asset_4\n"
            "           Price(Close) Price(Close) Price(Close) Price(Close) Price(Close)\n"
            "2009-06-30     1.000000     1.000000     0.000000     1.000000     1.000000\n"
            "2009-07-01     0.997755     0.998202     1.005330     1.006926     0.995778\n"
            "2009-07-02     0.995099     0.996817     1.018729     1.017295     0.996617)"
        )

        with self.assertRaises(Exception) as e_vr:
            _ = mframe.value_ret

        self.assertEqual(e_vr.exception.args[0], r)

        with self.assertRaises(Exception) as e_vrf:
            _ = mframe.value_ret_func()

        self.assertEqual(e_vrf.exception.args[0], r)

    def test_openframe_value_ret_calendar_period(self):

        vrcseries = self.randomseries.from_deepcopy()
        vrcseries.to_cumret()
        vrcframe = self.randomframe.from_deepcopy()
        vrcframe.to_cumret()

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

    def test_openframe_to_drawdown_series(self):

        mframe = self.randomframe.from_deepcopy()
        mframe.to_cumret()
        dd = [f"{dmax:.11f}" for dmax in mframe.max_drawdown]
        mframe.to_drawdown_series()
        dds = [f"{dmax:.11f}" for dmax in mframe.tsdf.min()]
        self.assertListEqual(dd, dds)

    def test_openframe_ord_least_squares_fit(self):

        oframe = self.randomframe.from_deepcopy()
        oframe.to_cumret()
        oframe.value_to_log()

        fsframe = self.randomframe.from_deepcopy()
        fsframe.to_cumret()
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
                "-0.09363759343",
                "-0.16636507875",
                "0.70835395893",
                "0.35961222138",
                "-0.60070110740",
                "1.00000000000",
                "1.79533271373",
                "-1.39409950365",
                "-1.80547164323",
                "-0.24049715964",
                "0.40456148704",
                "1.00000000000",
                "-0.61221397083",
                "-0.68807137926",
                "0.78819572733",
                "-0.24180725905",
                "-0.47123680764",
                "1.00000000000",
                "0.42517123248",
                "0.26256460553",
                "-0.20548693834",
                "-0.34752611919",
                "0.27898564658",
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

        bframe = self.randomframe.from_deepcopy()
        bframe.to_cumret()
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
                "0.28033485966",
                "0.54925184505",
                "0.51812697660",
                "-0.31860438079",
                "1.35008859840",
                "1.00000000000",
                "1.32115950219",
                "0.02193497832",
                "-0.40939945575",
                "0.67189587260",
                "0.33558345229",
                "1.00000000000",
                "0.06126647907",
                "0.04825332968",
                "0.45103333029",
                "0.00396482915",
                "0.04359783320",
                "1.00000000000",
                "-0.39479104514",
                "-0.23399128905",
                "-0.06243240197",
                "0.02896975251",
                "-0.33307559196",
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

        bframe = self.randomframe.from_deepcopy()
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
                "0.00225511385",
                "-0.04957426622",
                "0.01312690613",
                "0.02515147445",
                "0.00589243239",
                "1.00000000000",
                "0.10533452453",
                "-0.01266754819",
                "0.03977607247",
                "-0.04676333947",
                "0.03802715426",
                "1.00000000000",
                "-0.00723248920",
                "-0.03474759800",
                "0.03520481606",
                "-0.01300187928",
                "-0.02056261152",
                "1.00000000000",
                "0.00042231402",
                "0.10271570420",
                "0.06216832423",
                "-0.15043502549",
                "0.00064308622",
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