from datetime import date as dtdate
from io import StringIO
from json import loads
from pandas import DataFrame, date_range
from pandas.testing import assert_frame_equal
from pandas.tseries.offsets import CustomBusinessDay
import sys
from typing import get_type_hints, List, TypeVar
from unittest import TestCase

from openseries.datefixer import holiday_calendar
from openseries.frame import OpenFrame
from openseries.risk import cvar_down, var_down
from openseries.series import OpenTimeSeries, TimeSerie
from openseries.sim_price import ReturnSimulation

TTestOpenFrame = TypeVar("TTestOpenFrame", bound="TestOpenFrame")


class TestOpenFrame(TestCase):
    sim: ReturnSimulation
    randomframe: OpenFrame
    randomseries: OpenTimeSeries

    @classmethod
    def setUpClass(cls):
        OpenTimeSeries.setup_class()

        sim = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=5,
            trading_days=2512,
            mean_annual_return=0.05,
            mean_annual_vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        end = dtdate(2019, 6, 30)
        startyear = 2009
        calendar = holiday_calendar(
            startyear=startyear, endyear=end.year, countries=OpenTimeSeries.countries
        )
        d_range = [
            d.date()
            for d in date_range(
                periods=sim.trading_days,
                end=end,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]
        sdf = sim.df.iloc[0].T.to_frame()
        sdf.index = d_range
        sdf.columns = [["Asset"], ["Return(Total)"]]

        cls.randomseries = OpenTimeSeries.from_df(
            sdf, valuetype="Return(Total)"
        ).to_cumret()

        tslist = []
        for item in range(sim.number_of_sims):
            sdf = sim.df.iloc[item].T.to_frame()
            sdf.index = d_range
            sdf.columns = [[f"Asset_{item}"], ["Return(Total)"]]
            tslist.append(OpenTimeSeries.from_df(sdf, valuetype="Return(Total)"))

        cls.randomframe = OpenFrame(tslist)

    def test_openframe_repr(self: TTestOpenFrame):
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

    def test_openframe_annotations_and_typehints(self: TTestOpenFrame):
        annotations = dict(OpenFrame.__annotations__)
        self.assertDictEqual(
            annotations,
            {
                "constituents": List[OpenTimeSeries],
                "tsdf": DataFrame,
                "weights": List[float],
            },
        )
        typehints = get_type_hints(OpenFrame)
        self.assertDictEqual(annotations, typehints)

    def test_create_opentimeseries_from_frame(self: TTestOpenFrame):
        frame_f = self.randomframe.from_deepcopy()
        frame_f.to_cumret()
        fseries = OpenTimeSeries.from_frame(frame_f, label="Asset_1")

        self.assertTrue(isinstance(fseries, OpenTimeSeries))

    def test_openframe_calc_range(self: TTestOpenFrame):
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
            _, _ = crframe.calc_range(from_dt=dtdate(2009, 5, 31))
        self.assertIsInstance(too_early.exception, AssertionError)

        with self.assertRaises(AssertionError) as too_late:
            _, _ = crframe.calc_range(to_dt=dtdate(2019, 7, 31))
        self.assertIsInstance(too_late.exception, AssertionError)

        with self.assertRaises(AssertionError) as outside:
            _, _ = crframe.calc_range(
                from_dt=dtdate(2009, 5, 31), to_dt=dtdate(2019, 7, 31)
            )
        self.assertIsInstance(outside.exception, AssertionError)

        with self.assertRaises(AssertionError) as outside_end:
            _, _ = crframe.calc_range(
                from_dt=dtdate(2009, 7, 31), to_dt=dtdate(2019, 7, 31)
            )
        self.assertIsInstance(outside_end.exception, AssertionError)

        with self.assertRaises(AssertionError) as outside_start:
            _, _ = crframe.calc_range(
                from_dt=dtdate(2009, 5, 31), to_dt=dtdate(2019, 5, 31)
            )
        self.assertIsInstance(outside_start.exception, AssertionError)

        nst, nen = crframe.calc_range(
            from_dt=dtdate(2009, 7, 3), to_dt=dtdate(2019, 6, 25)
        )
        self.assertEqual(nst, dtdate(2009, 7, 3))
        self.assertEqual(nen, dtdate(2019, 6, 25))

        crframe.resample()

        earlier_moved, _ = crframe.calc_range(from_dt=dtdate(2009, 8, 10))
        self.assertEqual(earlier_moved, dtdate(2009, 7, 31))

        _, later_moved = crframe.calc_range(to_dt=dtdate(2009, 8, 20))
        self.assertEqual(later_moved, dtdate(2009, 8, 31))

    def test_openframe_resample(self: TTestOpenFrame):
        rs_frame = self.randomframe.from_deepcopy()
        rs_frame.to_cumret()

        before = rs_frame.value_ret.to_dict()

        rs_frame.resample(freq="BM")

        self.assertEqual(121, rs_frame.length)
        after = rs_frame.value_ret.to_dict()
        self.assertDictEqual(before, after)

    def test_openframe_resample_to_business_period_ends(self: TTestOpenFrame):
        rsb_stubs_frame = OpenFrame(
            [
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01, days=121, end_dt=dtdate(2023, 5, 15)
                ).set_new_label("A"),
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01, days=123, end_dt=dtdate(2023, 5, 16)
                ).set_new_label("B"),
            ]
        )

        rsb_stubs_frame.resample_to_business_period_ends(freq="BM")
        new_stubs_dates = rsb_stubs_frame.tsdf.index.tolist()

        self.assertListEqual(
            new_stubs_dates,
            [
                dtdate(2023, 1, 15),
                dtdate(2023, 1, 31),
                dtdate(2023, 2, 28),
                dtdate(2023, 3, 31),
                dtdate(2023, 4, 28),
                dtdate(2023, 5, 15),
            ],
        )

        rsb_frame = OpenFrame(
            [
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01, days=88, end_dt=dtdate(2023, 4, 28)
                ).set_new_label("A"),
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01, days=8, end_dt=dtdate(2023, 4, 28)
                ).set_new_label("B"),
            ]
        )

        rsb_frame.resample_to_business_period_ends(freq="BM")
        new_dates = rsb_frame.tsdf.index.tolist()

        self.assertListEqual(
            new_dates,
            [
                dtdate(2023, 1, 31),
                dtdate(2023, 2, 28),
                dtdate(2023, 3, 31),
                dtdate(2023, 4, 28),
            ],
        )

    def test_openframe_max_drawdown_date(self: TTestOpenFrame):
        mddframe = self.randomframe.from_deepcopy()
        mddframe.to_cumret()
        self.assertListEqual(
            [
                dtdate(2009, 7, 1),
                dtdate(2009, 7, 1),
                dtdate(2012, 4, 17),
                dtdate(2013, 7, 29),
                dtdate(2009, 7, 1),
            ],
            mddframe.max_drawdown_date.tolist(),
        )

    def test_openframe_make_portfolio(self: TTestOpenFrame):
        mpframe = self.randomframe.from_deepcopy()
        mpframe.to_cumret()
        mpframe.weights = [1.0 / mpframe.item_count] * mpframe.item_count

        name = "portfolio"
        mptail = mpframe.make_portfolio(name=name).tail()
        mptail = mptail.applymap(lambda nn: f"{nn:.6f}")

        correct = ["0.832536", "0.830516", "0.829576", "0.826926", "0.824288"]
        wrong = ["0.832536", "0.830516", "0.829576", "0.826926", "0.824285"]
        true_tail = DataFrame(
            columns=[[name], ["Price(Close)"]],
            index=[
                dtdate(2019, 6, 24),
                dtdate(2019, 6, 25),
                dtdate(2019, 6, 26),
                dtdate(2019, 6, 27),
                dtdate(2019, 6, 28),
            ],
            data=correct,
        )
        false_tail = DataFrame(
            columns=[[name], ["Price(Close)"]],
            index=[
                dtdate(2019, 6, 24),
                dtdate(2019, 6, 25),
                dtdate(2019, 6, 26),
                dtdate(2019, 6, 27),
                dtdate(2019, 6, 28),
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

    def test_openframe_add_timeseries(self: TTestOpenFrame):
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

    def test_openframe_delete_timeseries(self: TTestOpenFrame):
        frame = self.randomframe.from_deepcopy()
        frame.weights = [0.4, 0.1, 0.2, 0.1, 0.2]

        lbl = "Asset_1"
        frame.delete_timeseries(lbl)
        labels = [ff.label for ff in frame.constituents]

        self.assertListEqual(labels, ["Asset_0", "Asset_2", "Asset_3", "Asset_4"])
        self.assertListEqual(frame.weights, [0.4, 0.2, 0.1, 0.2])

    def test_openframe_risk_functions_same_as_opentimeseries(self: TTestOpenFrame):
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

    def test_openframe_methods_same_as_opentimeseries(self: TTestOpenFrame):
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

    def test_openframe_calc_methods_same_as_opentimeseries(self: TTestOpenFrame):
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

    def test_openframe_ratio_methods_same_as_opentimeseries(self: TTestOpenFrame):
        sames = self.randomseries.from_deepcopy()
        sames.to_cumret()
        sames.set_new_label(lvl_zero="Asset_0")
        samef = self.randomframe.from_deepcopy()
        samef.to_cumret()

        self.assertEqual(
            f"{sames.ret_vol_ratio_func(months_from_last=12):.11f}",
            "{:.11f}".format(
                float(
                    samef.ret_vol_ratio_func(
                        riskfree_rate=0.0, months_from_last=12
                    ).iloc[0]
                )
            ),
        )

        self.assertEqual(
            f"{sames.sortino_ratio_func(months_from_last=12):.11f}",
            "{:.11f}".format(
                float(
                    samef.sortino_ratio_func(
                        riskfree_rate=0.0, months_from_last=12
                    ).iloc[0]
                )
            ),
        )

    def test_openframe_measures_same_as_opentimeseries(self: TTestOpenFrame):
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

    def test_openframe_properties_same_as_opentimeseries(self: TTestOpenFrame):
        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

        common_props_to_compare = ["periods_in_a_year", "yearfrac"]
        for c in common_props_to_compare:
            self.assertEqual(getattr(sameseries, c), getattr(sameframe, c))

    def test_openframe_keeping_attributes_aligned_vs_opentimeseries(
        self: TTestOpenFrame,
    ):
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
            "countries",
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

    def test_openframe_keeping_methods_aligned_vs_opentimeseries(self: TTestOpenFrame):
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
            "plot_bars",
            "plot_series",
            "resample",
            "resample_to_business_period_ends",
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
            "from_df",
            "from_frame",
            "from_fixed_rate",
        ]

        series_unique = [
            "ewma_vol_func",
            "from_1d_rate_to_cumret",
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
            "ewma_risk",
            "rolling_info_ratio",
            "info_ratio_func",
            "jensen_alpha",
            "tracking_error_func",
            "capture_ratio_func",
            "ord_least_squares_fit",
            "make_portfolio",
            "merge_series",
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

    def test_openframe_value_to_log(self: TTestOpenFrame):
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

    def test_openframe_correl_matrix(self: TTestOpenFrame):
        corrframe = self.randomframe.from_deepcopy()
        corrframe.to_cumret()
        dict1 = corrframe.correl_matrix.applymap(lambda nn: f"{nn:.12f}").to_dict()
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

    def test_openframe_plot_series(self: TTestOpenFrame):
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        fig, _ = plotframe.plot_series(auto_open=False, output_type="div")
        fig_json = loads(fig.to_json())

        for i in range(plotframe.item_count):
            rawdata = [f"{x:.11f}" for x in plotframe.tsdf.iloc[1:5, i]]
            fig_data = [f"{x:.11f}" for x in fig_json["data"][i]["y"][1:5]]
            self.assertListEqual(rawdata, fig_data)

        fig_last, _ = plotframe.plot_series(
            auto_open=False, output_type="div", show_last=True
        )
        fig_last_json = loads(fig_last.to_json())
        rawlast = plotframe.tsdf.iloc[-1, -1]
        figlast = fig_last_json["data"][-1]["y"][0]
        self.assertEqual(f"{figlast:.12f}", f"{rawlast:.12f}")

        fig_last_fmt, _ = plotframe.plot_series(
            auto_open=False, output_type="div", show_last=True, tick_fmt=".3%"
        )
        fig_last_fmt_json = loads(fig_last_fmt.to_json())
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]
        self.assertEqual(last_fmt, "Last 77.813%")

        with self.assertRaises(AssertionError) as e_plot:
            _, _ = plotframe.plot_series(auto_open=False, labels=["a", "b"])

        self.assertIsInstance(e_plot.exception, AssertionError)

    def test_openframe_plot_bars(self: TTestOpenFrame):
        plotframe = self.randomframe.from_deepcopy()

        fig_keys = ["hovertemplate", "name", "x", "y", "type"]
        fig, _ = plotframe.plot_bars(auto_open=False, output_type="div")
        fig_json = loads(fig.to_json())
        self.assertListEqual(list(fig_json["data"][0].keys()), fig_keys)

        for i in range(plotframe.item_count):
            rawdata = [f"{x:.11f}" for x in plotframe.tsdf.iloc[1:5, i]]
            fig_data = [f"{x:.11f}" for x in fig_json["data"][i]["y"][1:5]]
            self.assertListEqual(rawdata, fig_data)

        with self.assertRaises(AssertionError) as e_plot:
            _, _ = plotframe.plot_bars(auto_open=False, labels=["a", "b"])

        self.assertIsInstance(e_plot.exception, AssertionError)

        overlayfig, _ = plotframe.plot_bars(
            auto_open=False, output_type="div", mode="overlay"
        )
        overlayfig_json = loads(overlayfig.to_json())

        fig_keys.append("opacity")
        self.assertListEqual(
            sorted(list(overlayfig_json["data"][0].keys())), sorted(fig_keys)
        )

    def test_openframe_passed_empty_list(self: TTestOpenFrame):
        with self.assertLogs() as cm:
            OpenFrame([])
        self.assertListEqual(
            cm.output, ["WARNING:root:OpenFrame() was passed an empty list."]
        )

    def test_openframe_wrong_number_of_weights_passed(self: TTestOpenFrame):
        wrongsims = self.randomframe.from_deepcopy()
        tslist = list(wrongsims.constituents)
        wghts = [1.0 / wrongsims.item_count] * (wrongsims.item_count + 1)

        with self.assertRaises(Exception) as e_weights:
            OpenFrame(tslist, weights=wghts)

        self.assertEqual(
            "Number of TimeSeries must equal number of weights.",
            e_weights.exception.args[0],
        )

    def test_openframe_drawdown_details(self: TTestOpenFrame):
        ddframe = self.randomframe.from_deepcopy()
        for s in ddframe.constituents:
            s.to_cumret()
        ddframe.to_cumret()
        dds = ddframe.drawdown_details().loc["Days from start to bottom"].tolist()
        self.assertListEqual([2317, 1797, 2439, 1024, 1278], dds)

    def test_openframe_trunc_frame(self: TTestOpenFrame):
        series_long = self.randomseries.from_deepcopy()
        series_long.set_new_label("Long")
        tmp_series = self.randomseries.from_deepcopy()
        series_short = OpenTimeSeries.from_df(
            tmp_series.tsdf.loc[
                dtdate(2017, 6, 27) : dtdate(2018, 6, 27), ("Asset", "Price(Close)")
            ]
        )
        series_short.set_new_label("Short")
        frame = OpenFrame([series_long, series_short])

        firsts = [
            dtdate(2017, 6, 27),
            dtdate(2017, 6, 27),
        ]
        lasts = [
            dtdate(2018, 6, 27),
            dtdate(2018, 6, 27),
        ]

        self.assertNotEqual(firsts, frame.first_indices.tolist())
        self.assertNotEqual(lasts, frame.last_indices.tolist())

        frame.trunc_frame()

        self.assertListEqual(firsts, frame.first_indices.tolist())
        self.assertListEqual(lasts, frame.last_indices.tolist())

        trunced = [dtdate(2017, 12, 29), dtdate(2018, 3, 29)]

        frame.trunc_frame(start_cut=trunced[0], end_cut=trunced[1])

        self.assertListEqual(trunced, [frame.first_idx, frame.last_idx])

    def test_openframe_trunc_frame_start_fail(self: TTestOpenFrame):
        frame = OpenFrame(
            [
                OpenTimeSeries.from_df(
                    df=DataFrame(
                        columns=["a"],
                        data=[1, 2, 3, 4, 5],
                        index=[
                            "2022-10-01",
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                        ],
                    )
                ),
                OpenTimeSeries.from_df(
                    df=DataFrame(
                        columns=["b"],
                        data=[6, 7, 8, 9, 10],
                        index=[
                            "2022-10-01",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-06",
                        ],
                    )
                ),
                OpenTimeSeries.from_df(
                    df=DataFrame(
                        columns=["c"],
                        data=[11, 12, 13, 14, 15],
                        index=[
                            "2022-10-02",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-06",
                            "2022-10-07",
                        ],
                    )
                ),
                OpenTimeSeries.from_df(
                    df=DataFrame(
                        columns=["d"],
                        data=[16, 17, 18, 19, 20],
                        index=[
                            "2022-10-01",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-06",
                            "2022-10-07",
                        ],
                    )
                ),
            ]
        )
        with self.assertLogs("root", level="WARNING") as logs:
            frame.trunc_frame()
        self.assertIn(
            (
                "WARNING:root:One or more constituents"
                " still not truncated to same start dates."
            ),
            logs.output[0],
        )

    def test_openframe_trunc_frame_end_fail(self: TTestOpenFrame):
        frame = OpenFrame(
            [
                OpenTimeSeries.from_df(
                    df=DataFrame(
                        columns=["a"],
                        data=[1, 2, 3, 4, 5],
                        index=[
                            "2022-10-01",
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-07",
                        ],
                    )
                ),
                OpenTimeSeries.from_df(
                    df=DataFrame(
                        columns=["b"],
                        data=[6, 7, 8, 9],
                        index=[
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-07",
                        ],
                    )
                ),
                OpenTimeSeries.from_df(
                    df=DataFrame(
                        columns=["c"],
                        data=[10, 11, 12, 13, 14],
                        index=[
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-07",
                        ],
                    )
                ),
                OpenTimeSeries.from_df(
                    df=DataFrame(
                        columns=["d"],
                        data=[15, 16, 17, 18, 19],
                        index=[
                            "2022-10-01",
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                        ],
                    )
                ),
            ]
        )
        with self.assertLogs("root", level="WARNING") as logs:
            frame.trunc_frame()
        self.assertIn(
            (
                "WARNING:root:One or more constituents"
                " still not truncated to same end dates."
            ),
            logs.output[0],
        )

    def test_openframe_merge_series(self: TTestOpenFrame):
        aframe = OpenFrame(
            [
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="Asset_one",
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
                        name="Asset_two",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Price(Close)",
                        dates=[
                            "2022-08-11",
                            "2022-08-12",
                            "2022-08-13",
                            "2022-08-14",
                            "2022-08-15",
                        ],
                        values=[2.1, 2.0, 1.8, 2.1, 2.0],
                    )
                ),
            ]
        )
        bframe = OpenFrame(
            [
                OpenTimeSeries(
                    TimeSerie(
                        _id="",
                        name="Asset_one",
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
                        name="Asset_two",
                        currency="SEK",
                        instrumentId="",
                        local_ccy=True,
                        valuetype="Price(Close)",
                        dates=[
                            "2022-07-11",
                            "2022-07-12",
                            "2022-07-13",
                            "2022-07-14",
                            "2022-07-16",
                        ],
                        values=[2.1, 2.0, 1.8, 2.1, 1.9],
                    )
                ),
            ]
        )

        b4df = aframe.tsdf.copy()
        aframe.merge_series(how="outer")
        assert_frame_equal(b4df, aframe.tsdf, check_exact=True)

        bframe.merge_series(how="inner")
        blist = [d.strftime("%Y-%m-%d") for d in bframe.tsdf.index]
        self.assertListEqual(
            blist,
            [
                "2022-07-11",
                "2022-07-12",
                "2022-07-13",
                "2022-07-14",
            ],
        )

        with self.assertRaises(Exception) as e_merged:
            aframe.merge_series(how="inner")

        self.assertEqual(
            (
                "Merging OpenTimeSeries DataFrames with argument how=inner produced "
                "an empty DataFrame."
            ),
            e_merged.exception.args[0],
        )

    def test_openframe_all_properties(self: TTestOpenFrame):
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

    def test_openframe_align_index_to_local_cdays(self: TTestOpenFrame):
        d_range = [d.date() for d in date_range(start="2022-06-01", end="2022-06-15")]
        asim = [1.0] * len(d_range)
        adf = DataFrame(
            data=asim,
            index=d_range,
            columns=[["Asset_a"], ["Price(Close)"]],
        )
        aseries = OpenTimeSeries.from_df(adf, valuetype="Price(Close)")
        bseries = OpenTimeSeries.from_df(adf, valuetype="Price(Close)")
        bseries.set_new_label("Asset_b")
        aframe = OpenFrame([aseries, bseries])

        midsummer = dtdate(2022, 6, 6)
        self.assertTrue(midsummer in d_range)

        aframe.align_index_to_local_cdays()
        self.assertFalse(midsummer in aframe.tsdf.index)

    def test_openframe_rolling_info_ratio(self: TTestOpenFrame):
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

    def test_openframe_rolling_beta(self: TTestOpenFrame):
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

    def test_openframe_ret_vol_ratio_func(self: TTestOpenFrame):
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

    def test_openframe_sortino_ratio_func(self: TTestOpenFrame):
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

    def test_openframe_info_ratio_func(self: TTestOpenFrame):
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

    def test_openframe_rolling_corr(self: TTestOpenFrame):
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

    def test_openframe_rolling_vol(self: TTestOpenFrame):
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

    def test_openframe_rolling_return(self: TTestOpenFrame):
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

    def test_openframe_rolling_cvar_down(self: TTestOpenFrame):
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

    def test_openframe_rolling_var_down(self: TTestOpenFrame):
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

    def test_openframe_label_uniqueness(self: TTestOpenFrame):
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

    def test_openframe_capture_ratio(self: TTestOpenFrame):
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

    def test_openframe_georet_exceptions(self: TTestOpenFrame):
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
            (
                "Geometric return cannot be calculated due to an initial value being "
                "zero or a negative value."
            ),
        )

        with self.assertRaises(Exception) as e_grf_zero:
            _ = geoframe.geo_ret_func()

        self.assertEqual(
            e_grf_zero.exception.args[0],
            (
                "Geometric return cannot be calculated due to an initial value being "
                "zero or a negative value."
            ),
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
            (
                "Geometric return cannot be calculated due to an initial value being "
                "zero or a negative value."
            ),
        )

        with self.assertRaises(Exception) as e_grf_neg:
            _ = geoframe.geo_ret_func()

        self.assertEqual(
            e_grf_neg.exception.args[0],
            (
                "Geometric return cannot be calculated due to an initial value being "
                "zero or a negative value."
            ),
        )

    def test_openframe_value_nan_handle(self: TTestOpenFrame):
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
            # noinspection PyTypeChecker
            _ = nanframe.value_nan_handle(method="other")

        self.assertEqual(
            e_methd.exception.args[0],
            "Method must be either fill or drop passed as string.",
        )

    def test_openframe_return_nan_handle(self: TTestOpenFrame):
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
            # noinspection PyTypeChecker
            _ = nanframe.return_nan_handle(method="other")

        self.assertEqual(
            e_methd.exception.args[0],
            "Method must be either fill or drop passed as string.",
        )

    def test_openframe_relative(self: TTestOpenFrame):
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

    def test_openframe_to_cumret(self: TTestOpenFrame):
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

    def test_openframe_miscellaneous(self: TTestOpenFrame):
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
        for methd in methods:
            dated = getattr(mframe, methd)(
                from_date=mframe.first_idx, to_date=mframe.last_idx
            )
            undated = getattr(mframe, methd)(
                from_date=mframe.first_idx,
                to_date=mframe.last_idx,
            )
            for dd, ud in zip(dated, undated):
                self.assertEqual(f"{dd:.10f}", f"{ud:.10f}")

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

    def test_openframe_value_ret_calendar_period(self: TTestOpenFrame):
        vrcseries = self.randomseries.from_deepcopy()
        vrcseries.to_cumret()
        vrcframe = self.randomframe.from_deepcopy()
        vrcframe.to_cumret()

        vrfs_y = vrcseries.value_ret_func(
            from_date=dtdate(2017, 12, 29), to_date=dtdate(2018, 12, 28)
        )
        vrff_y = vrcframe.value_ret_func(
            from_date=dtdate(2017, 12, 29), to_date=dtdate(2018, 12, 28)
        )
        vrffl_y = [f"{rr:.11f}" for rr in vrff_y]

        vrvrcs_y = vrcseries.value_ret_calendar_period(year=2018)
        vrvrcf_y = vrcframe.value_ret_calendar_period(year=2018)
        vrvrcfl_y = [f"{rr:.11f}" for rr in vrvrcf_y]

        self.assertEqual(f"{vrfs_y:.11f}", f"{vrvrcs_y:.11f}")
        self.assertListEqual(vrffl_y, vrvrcfl_y)

        vrfs_ym = vrcseries.value_ret_func(
            from_date=dtdate(2018, 4, 30), to_date=dtdate(2018, 5, 31)
        )
        vrff_ym = vrcframe.value_ret_func(
            from_date=dtdate(2018, 4, 30), to_date=dtdate(2018, 5, 31)
        )
        vrffl_ym = [f"{rr:.11f}" for rr in vrff_ym]

        vrvrcs_ym = vrcseries.value_ret_calendar_period(year=2018, month=5)
        vrvrcf_ym = vrcframe.value_ret_calendar_period(year=2018, month=5)
        vrvrcfl_ym = [f"{rr:.11f}" for rr in vrvrcf_ym]

        self.assertEqual(f"{vrfs_ym:.11f}", f"{vrvrcs_ym:.11f}")
        self.assertListEqual(vrffl_ym, vrvrcfl_ym)

    def test_openframe_to_drawdown_series(self: TTestOpenFrame):
        mframe = self.randomframe.from_deepcopy()
        mframe.to_cumret()
        dd = [f"{dmax:.11f}" for dmax in mframe.max_drawdown]
        mframe.to_drawdown_series()
        dds = [f"{dmax:.11f}" for dmax in mframe.tsdf.min()]
        self.assertListEqual(dd, dds)

    def test_openframe_ord_least_squares_fit(self: TTestOpenFrame):
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
                tmp = oframe.ord_least_squares_fit(
                    y_column=i, x_column=j, fitted_series=False
                )
                results.append(f"{float(tmp.params):.11f}")

        results_tuple = []
        for i in oframe.tsdf:
            for j in oframe.tsdf:
                tmp = oframe.ord_least_squares_fit(
                    y_column=i, x_column=j, fitted_series=False
                )
                results_tuple.append(f"{float(tmp.params):.11f}")

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

    def test_openframe_beta(self: TTestOpenFrame):
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

    def test_openframe_beta_returns_input(self: TTestOpenFrame):
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

    def test_openframe_jensen_alpha(self: TTestOpenFrame):
        jframe = self.randomframe.from_deepcopy()
        jframe.to_cumret()
        jframe.resample("7D")
        results = []
        for i in range(jframe.item_count):
            for j in range(jframe.item_count):
                results.append(f"{jframe.jensen_alpha(asset=i, market=j):.9f}")

        results_tuple = []
        for i in jframe.tsdf:
            for j in jframe.tsdf:
                results_tuple.append(f"{jframe.jensen_alpha(asset=i, market=j):.9f}")

        self.assertListEqual(results, results_tuple)
        self.assertListEqual(
            results,
            [
                "0.000000000",
                "0.027643279",
                "0.029626538",
                "-0.008368599",
                "-0.005750392",
                "-0.094690628",
                "0.000000000",
                "-0.025292179",
                "-0.092734191",
                "-0.101964705",
                "-0.051910010",
                "-0.019746745",
                "0.000000000",
                "-0.051921297",
                "-0.049584293",
                "0.018764935",
                "0.019927858",
                "0.021773289",
                "0.000000000",
                "0.010246959",
                "-0.023181122",
                "-0.029357400",
                "-0.022125144",
                "-0.017078985",
                "0.000000000",
            ],
        )
        with self.assertRaises(Exception) as e_asset:
            # noinspection PyTypeChecker
            _ = jframe.jensen_alpha(asset="string", market=1)

        self.assertEqual(
            e_asset.exception.args[0],
            "asset should be a tuple or an integer.",
        )

        with self.assertRaises(Exception) as e_market:
            # noinspection PyTypeChecker
            _ = jframe.jensen_alpha(asset=0, market="string")

        self.assertEqual(
            e_market.exception.args[0],
            "market should be a tuple or an integer.",
        )
        from openseries.datefixer import date_offset_foll

        ninemth = date_offset_foll(jframe.last_idx, months_offset=-9, adjust=True)
        shortframe = jframe.trunc_frame(start_cut=ninemth)
        shortframe.to_cumret()
        sresults = []
        for i in range(shortframe.item_count):
            for j in range(shortframe.item_count):
                sresults.append(f"{shortframe.jensen_alpha(asset=i, market=j):.9f}")

        sresults_tuple = []
        for i in shortframe.tsdf:
            for j in shortframe.tsdf:
                sresults_tuple.append(
                    f"{shortframe.jensen_alpha(asset=i, market=j):.9f}"
                )

        self.assertListEqual(sresults, sresults_tuple)
        self.assertListEqual(
            sresults,
            [
                "0.000000000",
                "-0.043721481",
                "0.091725480",
                "0.108780390",
                "0.012244249",
                "-0.078588038",
                "0.000000000",
                "-0.126003759",
                "-0.137148953",
                "-0.069468578",
                "0.006395139",
                "-0.050191011",
                "0.000000000",
                "0.018726894",
                "-0.000440884",
                "-0.018697252",
                "-0.011027547",
                "-0.013438783",
                "0.000000000",
                "-0.005692602",
                "-0.119326162",
                "0.056709768",
                "-0.293744566",
                "-0.274853038",
                "0.000000000",
            ],
        )

    def test_openframe_jensen_alpha_returns_input(self: TTestOpenFrame):
        jframe = self.randomframe.from_deepcopy()
        jframe.resample("7D")
        results = []
        for i in range(jframe.item_count):
            for j in range(jframe.item_count):
                results.append(f"{jframe.jensen_alpha(asset=i, market=j):.9f}")

        results_tuple = []
        for i in jframe.tsdf:
            for j in jframe.tsdf:
                results_tuple.append(f"{jframe.jensen_alpha(asset=i, market=j):.9f}")

        self.assertListEqual(results, results_tuple)

        self.assertListEqual(
            results,
            [
                "0.000000000",
                "-0.000198851",
                "-0.000187574",
                "-0.000199182",
                "-0.000186796",
                "-0.000198987",
                "0.000000000",
                "-0.000225082",
                "-0.000200278",
                "-0.000180382",
                "0.000227262",
                "0.000244193",
                "0.000000000",
                "0.000236515",
                "0.000219303",
                "-0.000002178",
                "-0.000011797",
                "-0.000004329",
                "0.000000000",
                "-0.000008984",
                "-0.000476795",
                "-0.000484823",
                "-0.000461676",
                "-0.000497260",
                "0.000000000",
            ],
        )
        with self.assertRaises(Exception) as e_asset:
            # noinspection PyTypeChecker
            _ = jframe.jensen_alpha(asset="string", market=1)

        self.assertEqual(
            e_asset.exception.args[0],
            "asset should be a tuple or an integer.",
        )

        with self.assertRaises(Exception) as e_market:
            # noinspection PyTypeChecker
            _ = jframe.jensen_alpha(asset=0, market="string")

        self.assertEqual(
            e_market.exception.args[0],
            "market should be a tuple or an integer.",
        )

    def test_openframe_ewma_risk(self: TTestOpenFrame):
        eframe = self.randomframe.from_deepcopy()
        eframe.to_cumret()
        edf = eframe.ewma_risk()

        list_one = [f"{e:.11f}" for e in edf.head(10).iloc[:, 0]]
        list_two = [f"{e:.11f}" for e in edf.head(10).iloc[:, 1]]
        corr_one = [f"{e:.11f}" for e in edf.head(10).iloc[:, 2]]
        self.assertListEqual(
            list_one,
            [
                "0.07995872621",
                "0.07801248670",
                "0.07634125583",
                "0.07552465738",
                "0.07894138379",
                "0.07989322216",
                "0.07769398173",
                "0.07806577815",
                "0.07603008639",
                "0.08171281006",
            ],
        )
        self.assertListEqual(
            list_two,
            [
                "0.10153833268",
                "0.09869274051",
                "0.09583812971",
                "0.09483161937",
                "0.09470601474",
                "0.09210588859",
                "0.11261673980",
                "0.11113938828",
                "0.11043515326",
                "0.10817921616",
            ],
        )
        self.assertListEqual(
            corr_one,
            [
                "-0.00015294210",
                "0.00380837753",
                "0.00758444757",
                "-0.01259265721",
                "0.03346034482",
                "0.02068245047",
                "-0.00730691767",
                "0.01757764619",
                "0.02745689252",
                "-0.00629004298",
            ],
        )

        fdf = eframe.ewma_risk(
            first_column=3, second_column=4, periods_in_a_year_fixed=251
        )
        list_three = [f"{f:.11f}" for f in fdf.head(10).iloc[:, 0]]
        list_four = [f"{f:.11f}" for f in fdf.head(10).iloc[:, 1]]
        corr_two = [f"{f:.11f}" for f in fdf.head(10).iloc[:, 2]]
        self.assertListEqual(
            list_three,
            [
                "0.07712206989",
                "0.07942595349",
                "0.08666330524",
                "0.09336934376",
                "0.09064864248",
                "0.08834725868",
                "0.08578870069",
                "0.08372351448",
                "0.08828894057",
                "0.08718509958",
            ],
        )
        self.assertListEqual(
            list_four,
            [
                "0.07787841405",
                "0.07727035322",
                "0.07498769117",
                "0.07273500879",
                "0.07786226476",
                "0.07880499823",
                "0.08075244706",
                "0.07832868687",
                "0.07594379202",
                "0.08107054465",
            ],
        )
        self.assertListEqual(
            corr_two,
            [
                "0.00068511835",
                "-0.03519976419",
                "-0.02124735579",
                "-0.02555360096",
                "-0.01204201129",
                "0.00315017923",
                "0.01198035018",
                "0.01363505146",
                "0.01369207054",
                "0.05193595929",
            ],
        )
