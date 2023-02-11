import datetime as dt
from io import StringIO
from json import load, loads
from jsonschema.exceptions import ValidationError
from os import path, remove
from pandas import DataFrame, date_range, DatetimeIndex, Series
from pandas.tseries.offsets import CustomBusinessDay
from stdnum.exceptions import InvalidChecksum
import sys
from typing import get_type_hints, List, TypeVar
from unittest import TestCase

from openseries.datefixer import holiday_calendar
from openseries.series import OpenTimeSeries, timeseries_chain, TimeSerie
from openseries.sim_price import ReturnSimulation
from openseries.exceptions import FromFixedRateDatesInputError

TTestOpenTimeSeries = TypeVar("TTestOpenTimeSeries", bound="TestOpenTimeSeries")


class TestOpenTimeSeries(TestCase):
    sim: ReturnSimulation
    randomseries: OpenTimeSeries
    random_properties: dict

    @classmethod
    def setUpClass(cls):
        OpenTimeSeries.setup_class()

        sim = ReturnSimulation.from_merton_jump_gbm(
            number_of_sims=1,
            trading_days=2512,
            mean_annual_return=0.05,
            mean_annual_vol=0.1,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            seed=71,
        )
        end = dt.date(2019, 6, 30)
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
        cls.random_properties = cls.randomseries.all_properties().to_dict()[
            ("Asset", "Price(Close)")
        ]

    def test_opentimeseries_annotations_and_typehints(self: TTestOpenTimeSeries):
        opentimeseries_annotations = dict(OpenTimeSeries.__annotations__)
        timeserie_annotations = dict(TimeSerie.__annotations__)

        self.assertDictEqual(
            opentimeseries_annotations,
            {
                "_id": str,
                "instrumentId": str,
                "currency": str,
                "dates": List[str],
                "domestic": str,
                "name": str,
                "isin": str,
                "label": str,
                "countries": list | str,
                "valuetype": str,
                "values": List[float],
                "local_ccy": bool,
                "tsdf": DataFrame,
            },
        )
        self.assertDictEqual(opentimeseries_annotations, timeserie_annotations)

        opentimeseries_typehints = get_type_hints(OpenTimeSeries)
        timeserie_typehints = get_type_hints(TimeSerie)
        self.assertDictEqual(opentimeseries_annotations, opentimeseries_typehints)
        self.assertDictEqual(opentimeseries_typehints, timeserie_typehints)

    def test_opentimeseries_repr(self: TTestOpenTimeSeries):
        old_stdout = sys.stdout
        new_stdout = StringIO()
        sys.stdout = new_stdout
        repseries = self.randomseries
        r = (
            "OpenTimeSeries(name=Asset, _id=, instrumentId=, "
            "valuetype=Price(Close), currency=SEK, "
            "start=2009-06-30, end=2019-06-28, local_ccy=True)\n"
        )
        print(repseries)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        self.assertEqual(r, output)

    def test_opentimeseries_duplicates_handling(self: TTestOpenTimeSeries):
        json_file = path.join(path.dirname(path.abspath(__file__)), "series.json")
        with open(json_file, "r") as ff:
            output = load(ff)

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

        with self.assertRaises(Exception) as e_dup:
            OpenTimeSeries(d=output, remove_duplicates=False)

        self.assertIsInstance(e_dup.exception, ValidationError)

        with self.assertLogs("root", level="WARNING") as warn_logs:
            ts = OpenTimeSeries(d=output, remove_duplicates=True)
        self.assertIn(
            (
                "[{'2017-08-28': 99.4062}, "
                "{'2017-11-27': 100.8974}, "
                "{'2017-11-27': 100.8974}]"
            ),
            warn_logs.output[0],
        )
        self.assertIsInstance(ts, OpenTimeSeries)

        with self.assertRaises(AssertionError) as e_clean:
            with self.assertLogs("root", level="WARNING"):
                _ = self.randomseries.from_deepcopy()
        self.assertIsInstance(e_clean.exception, AssertionError)

    def test_opentimeseries_create_from_pandas_df(self: TTestOpenTimeSeries):
        se = Series(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            name="Asset_0",
            dtype="float64",
        )
        sen = Series(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            name=("Asset_0", "Price(Close)"),
            dtype="float64",
        )
        df = DataFrame(
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
            dtype="float64",
        )

        seseries = OpenTimeSeries.from_df(df=se)
        senseries = OpenTimeSeries.from_df(df=sen)
        dfseries = OpenTimeSeries.from_df(df=df, column_nmbr=1)

        self.assertTrue(isinstance(seseries, OpenTimeSeries))
        self.assertTrue(isinstance(senseries, OpenTimeSeries))
        self.assertTrue(isinstance(dfseries, OpenTimeSeries))
        self.assertEqual(seseries.label, senseries.label)

    def test_opentimeseries_save_to_json(self: TTestOpenTimeSeries):
        seriesfile = path.join(path.dirname(path.abspath(__file__)), "seriessaved.json")

        jseries = self.randomseries.from_deepcopy()
        jseries.to_json(filename=seriesfile)

        self.assertTrue(path.exists(seriesfile))

        remove(seriesfile)

        self.assertFalse(path.exists(seriesfile))

    def test_opentimeseries_create_from_fixed_rate(self: TTestOpenTimeSeries):
        fixseries_one = OpenTimeSeries.from_fixed_rate(
            rate=0.03, days=756, end_dt=dt.date(2019, 6, 30)
        )
        self.assertTrue(isinstance(fixseries_one, OpenTimeSeries))

        rnd_series = self.randomseries.from_deepcopy()
        fixseries_two = OpenTimeSeries.from_fixed_rate(
            rate=0.03, d_range=DatetimeIndex(rnd_series.tsdf.index)
        )
        self.assertTrue(isinstance(fixseries_two, OpenTimeSeries))

        with self.assertRaises(FromFixedRateDatesInputError) as only_rate_arg:
            _ = OpenTimeSeries.from_fixed_rate(rate=0.03)
        self.assertIsInstance(only_rate_arg.exception, FromFixedRateDatesInputError)

        self.assertEqual(
            str(only_rate_arg.exception),
            "If d_range is not provided both days and end_dt must be.",
        )

        with self.assertRaises(FromFixedRateDatesInputError) as only_days_noend:
            _ = OpenTimeSeries.from_fixed_rate(rate=0.03, days=30)
        self.assertIsInstance(only_days_noend.exception, FromFixedRateDatesInputError)

    def test_opentimeseries_periods_in_a_year(self: TTestOpenTimeSeries):
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

    def test_opentimeseries_yearfrac(self: TTestOpenTimeSeries):
        self.assertEqual(
            f"{9.9931553730322:.13f}", f"{self.randomseries.yearfrac:.13f}"
        )
        all_prop = self.random_properties["yearfrac"]
        self.assertEqual(f"{all_prop:.13f}", f"{self.randomseries.yearfrac:.13f}")

    def test_opentimeseries_resample(self: TTestOpenTimeSeries):
        rs_series = self.randomseries.from_deepcopy()

        before = rs_series.value_ret

        rs_series.resample(freq="BM")

        self.assertEqual(121, rs_series.length)
        self.assertEqual(before, rs_series.value_ret)

    def test_opentimeseries_resample_to_business_period_ends(self: TTestOpenTimeSeries):
        rsb_stubs_series = OpenTimeSeries.from_fixed_rate(
            rate=0.01, days=121, end_dt=dt.date(2023, 5, 15)
        )

        rsb_stubs_series.resample_to_business_period_ends(freq="BM")
        new_stubs_dates = rsb_stubs_series.tsdf.index.tolist()

        self.assertListEqual(
            new_stubs_dates,
            [
                dt.date(2023, 1, 15),
                dt.date(2023, 1, 31),
                dt.date(2023, 2, 28),
                dt.date(2023, 3, 31),
                dt.date(2023, 4, 28),
                dt.date(2023, 5, 15),
            ],
        )

        rsb_series = OpenTimeSeries.from_fixed_rate(
            rate=0.01, days=88, end_dt=dt.date(2023, 4, 28)
        )

        rsb_series.resample_to_business_period_ends(freq="BM")
        new_dates = rsb_series.tsdf.index.tolist()

        self.assertListEqual(
            new_dates,
            [
                dt.date(2023, 1, 31),
                dt.date(2023, 2, 28),
                dt.date(2023, 3, 31),
                dt.date(2023, 4, 28),
            ],
        )

    def test_opentimeseries_calc_range(self: TTestOpenTimeSeries):
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

    def test_opentimeseries_calc_range_ouput(self: TTestOpenTimeSeries):
        cseries = self.randomseries.from_deepcopy()

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

    def test_opentimeseries_value_to_diff(self: TTestOpenTimeSeries):
        diffseries = self.randomseries.from_deepcopy()
        diffseries.value_to_diff()
        are_bes = [f"{nn[0]:.12f}" for nn in diffseries.tsdf.values[:15]]
        should_bes = [
            "0.000000000000",
            "-0.002244525566",
            "-0.002656444823",
            "0.003856605762",
            "0.007615942129",
            "-0.005921701827",
            "0.001555810865",
            "-0.005275328842",
            "-0.001848758036",
            "0.009075607620",
            "-0.004319311398",
            "-0.008365867931",
            "-0.010422707104",
            "0.003626411898",
            "-0.000274024491",
        ]

        self.assertListEqual(are_bes, should_bes)

    def test_opentimeseries_value_to_ret(self: TTestOpenTimeSeries):
        retseries = self.randomseries.from_deepcopy()
        retseries.value_to_ret()
        are_bes = [f"{nn[0]:.12f}" for nn in retseries.tsdf.values[:15]]
        should_bes = [
            "0.000000000000",
            "-0.002244525566",
            "-0.002662420694",
            "0.003875599963",
            "0.007623904265",
            "-0.005883040967",
            "0.001554800438",
            "-0.005263718728",
            "-0.001854450536",
            "0.009120465722",
            "-0.004301429464",
            "-0.008367224292",
            "-0.010512356183",
            "0.003696462370",
            "-0.000278289067",
        ]

        self.assertListEqual(are_bes, should_bes)

        retseries.to_cumret()

    def test_opentimeseries_valute_to_log(self: TTestOpenTimeSeries):
        logseries = self.randomseries.from_deepcopy()
        logseries.value_to_log()
        are_log = [f"{nn[0]:.12f}" for nn in logseries.tsdf.values[:15]]
        should_log = [
            "0.000000000000",
            "-0.002247048289",
            "-0.004913019528",
            "-0.001044910355",
            "0.006550078823",
            "0.000649664599",
            "0.002203257585",
            "-0.003074363317",
            "-0.004930535474",
            "0.004148589972",
            "-0.000162117254",
            "-0.008564543266",
            "-0.019132544583",
            "-0.015442897340",
            "-0.015721225137",
        ]

        self.assertListEqual(are_log, should_log)

    def test_opentimeseries_all_calc_properties(self: TTestOpenTimeSeries):
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

    def test_opentimeseries_all_calc_functions(self: TTestOpenTimeSeries):
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

    def test_opentimeseries_max_drawdown_date(self: TTestOpenTimeSeries):
        self.assertEqual(dt.date(2018, 11, 8), self.randomseries.max_drawdown_date)
        all_prop = self.random_properties["max_drawdown_date"]
        self.assertEqual(all_prop, self.randomseries.max_drawdown_date)

    def test_opentimeseries_running_adjustment(self: TTestOpenTimeSeries):
        adjustedseries = self.randomseries.from_deepcopy()
        adjustedseries.running_adjustment(0.05)

        self.assertEqual(
            f"{1.689055852583:.12f}",
            f"{float(adjustedseries.tsdf.iloc[-1]):.12f}",
        )
        adjustedseries_returns = self.randomseries.from_deepcopy()
        adjustedseries_returns.value_to_ret()
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

    def test_timeseries_chain(self: TTestOpenTimeSeries):
        full_series = self.randomseries.from_deepcopy()
        full_values = [f"{nn:.10f}" for nn in full_series.tsdf.iloc[:, 0].tolist()]

        front_series = OpenTimeSeries.from_df(full_series.tsdf.iloc[:126])

        back_series = OpenTimeSeries.from_df(
            full_series.tsdf.loc[front_series.last_idx :]
        )

        chained_series = timeseries_chain(front_series, back_series)
        chained_values = [f"{nn:.10f}" for nn in chained_series.values]

        self.assertListEqual(full_series.dates, chained_series.dates)
        self.assertListEqual(full_values, chained_values)

        pushed_date = front_series.last_idx + dt.timedelta(days=10)
        no_overlap_series = OpenTimeSeries.from_df(full_series.tsdf.loc[pushed_date:])
        with self.assertRaises(Exception) as e_chain:
            _ = timeseries_chain(front_series, no_overlap_series)

        self.assertIsInstance(e_chain.exception, AssertionError)

        front_series_two = OpenTimeSeries.from_df(full_series.tsdf.iloc[:136])
        front_series_two.resample(freq="8D")

        self.assertTrue(back_series.first_idx not in front_series_two.tsdf.index)
        new_chained_series = timeseries_chain(front_series_two, back_series)
        self.assertIsInstance(new_chained_series, OpenTimeSeries)

        front_series_three = OpenTimeSeries.from_df(full_series.tsdf.iloc[:136])
        front_series_three.resample(freq="10D")

        self.assertTrue(back_series.first_idx not in front_series_three.tsdf.index)

        with self.assertRaises(Exception) as e_fail:
            _ = timeseries_chain(front_series_three, back_series)

        self.assertEqual(
            e_fail.exception.args[0], "Failed to find a matching date between series"
        )

    def test_opentimeseries_plot_series(self: TTestOpenTimeSeries):
        plotseries = self.randomseries.from_deepcopy()
        rawdata = [f"{x:.11f}" for x in plotseries.tsdf.iloc[1:5, 0]]

        fig, _ = plotseries.plot_series(auto_open=False, output_type="div")
        fig_json = loads(fig.to_json())
        fig_data = [f"{x:.11f}" for x in fig_json["data"][0]["y"][1:5]]

        self.assertListEqual(rawdata, fig_data)

        fig_last, _ = plotseries.plot_series(
            auto_open=False, output_type="div", show_last=True
        )
        fig_last_json = loads(fig_last.to_json())
        last = fig_last_json["data"][-1]["y"][0]
        self.assertEqual(f"{last:.12f}", "1.024471958022")

        fig_last_fmt, _ = plotseries.plot_series(
            auto_open=False, output_type="div", show_last=True, tick_fmt=".3%"
        )
        fig_last_fmt_json = loads(fig_last_fmt.to_json())
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]
        self.assertEqual(last_fmt, "Last 102.447%")

    def test_opentimeseries_plot_bars(self: TTestOpenTimeSeries):
        barseries = self.randomseries.from_deepcopy()
        barseries.resample(freq="BM").value_to_ret()
        rawdata = [f"{x:.11f}" for x in barseries.tsdf.iloc[1:5, 0]]

        fig, _ = barseries.plot_bars(auto_open=False, output_type="div")
        fig_json = loads(fig.to_json())
        fig_data = [f"{x:.11f}" for x in fig_json["data"][0]["y"][1:5]]

        self.assertListEqual(rawdata, fig_data)

    def test_opentimeseries_drawdown_details(self: TTestOpenTimeSeries):
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

    def test_opentimeseries_align_index_to_local_cdays(self: TTestOpenTimeSeries):
        d_range = [d.date() for d in date_range(start="2020-06-15", end="2020-06-25")]
        asim = [1.0] * len(d_range)
        adf = DataFrame(
            data=asim,
            index=d_range,
            columns=[["Asset"], ["Price(Close)"]],
        )
        aseries = OpenTimeSeries.from_df(adf, valuetype="Price(Close)")

        midsummer = dt.date(2020, 6, 19)
        self.assertTrue(midsummer in d_range)

        aseries.align_index_to_local_cdays()
        self.assertFalse(midsummer in aseries.tsdf.index)

    def test_opentimeseries_ewma_vol_func(self: TTestOpenTimeSeries):
        simdata = self.randomseries.ewma_vol_func()
        simseries = OpenTimeSeries.from_df(simdata, valuetype="EWMA")
        values = [f"{v:.11f}" for v in simdata.iloc[:5]]
        checkdata = [
            "0.07995872621",
            "0.07801248670",
            "0.07634125583",
            "0.07552465738",
            "0.07894138379",
        ]

        self.assertListEqual(values, checkdata)
        self.assertIsInstance(simseries, OpenTimeSeries)

        simdata_fxd_per_yr = self.randomseries.ewma_vol_func(
            periods_in_a_year_fixed=251
        )

        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5]]
        checkdata_fxd_per_yr = [
            "0.07989953100",
            "0.07795473234",
            "0.07628473871",
            "0.07546874481",
            "0.07888294174",
        ]
        self.assertListEqual(values_fxd_per_yr, checkdata_fxd_per_yr)

    def test_opentimeseries_rolling_vol(self: TTestOpenTimeSeries):
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

        simdata_fxd_per_yr = self.randomseries.rolling_vol(
            observations=21, periods_in_a_year_fixed=251
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

    def test_opentimeseries_rolling_return(self: TTestOpenTimeSeries):
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

    def test_opentimeseries_rolling_cvar_down(self: TTestOpenTimeSeries):
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

    def test_opentimeseries_rolling_var_down(self: TTestOpenTimeSeries):
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

    def test_opentimeseries_downside_deviation(self: TTestOpenTimeSeries):
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

    def test_opentimeseries_validations(self: TTestOpenTimeSeries):
        valid_isin = "SE0009807308"
        invalid_isin_one = "SE0009807307"
        invalid_isin_two = "SE000980730B"
        valid_instrument_id = "58135911b239b413482758c9"
        invalid_instrument_id_one = "58135911b239b413482758c"
        invalid_instrument_id_two = "5_135911b239b413482758c9"
        valid_timeseries_id = "5813595971051506189ba416"
        invalid_timeseries_id_one = "5813595971051506189ba41"
        invalid_timeseries_id_two = "5_13595971051506189ba416"

        timeseries_with_valid_isin = OpenTimeSeries(
            TimeSerie(
                _id=valid_timeseries_id,
                currency="SEK",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                instrumentId=valid_instrument_id,
                isin=valid_isin,
                local_ccy=True,
                name="asset",
                values=[
                    100.0,
                    100.0978,
                ],
                valuetype="Price(Close)",
            )
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

        self.assertIn(member="local_ccy", container=getattr(e_ccy.exception, "message"))

        new_dict.pop("label")
        new_dict.update(
            {"local_ccy": True, "dates": []}
        )  # Set dates to empty array to trigger minItems ValidationError

        with self.assertRaises(Exception) as e_min_items:
            OpenTimeSeries(new_dict)

        self.assertIn(
            member="is too short", container=getattr(e_min_items.exception, "message")
        )

        with self.assertRaises(Exception) as e_one:
            OpenTimeSeries(
                TimeSerie(
                    _id=valid_timeseries_id,
                    currency="SEK",
                    dates=[
                        "2017-05-29",
                        "2017-05-30",
                    ],
                    instrumentId=valid_instrument_id,
                    isin=invalid_isin_one,
                    local_ccy=True,
                    name="asset",
                    values=[
                        100.0,
                        100.0978,
                    ],
                    valuetype="Price(Close)",
                )
            )
        self.assertIsInstance(e_one.exception, InvalidChecksum)

        with self.assertRaises(Exception) as e_two:
            OpenTimeSeries(
                TimeSerie(
                    _id=valid_timeseries_id,
                    currency="SEK",
                    dates=[
                        "2017-05-29",
                        "2017-05-30",
                    ],
                    instrumentId=valid_instrument_id,
                    isin=invalid_isin_two,
                    local_ccy=True,
                    name="asset",
                    values=[
                        100.0,
                        100.0978,
                    ],
                    valuetype="Price(Close)",
                )
            )
        self.assertIn(
            member="does not match '^$|([A-Z]{2})([\\\\dA-Z]{9})(\\\\d)$'",
            container=getattr(e_two.exception, "message"),
        )

        with self.assertRaises(Exception) as e_three:
            OpenTimeSeries(
                TimeSerie(
                    _id=invalid_timeseries_id_one,
                    currency="SEK",
                    dates=[
                        "2017-05-29",
                        "2017-05-30",
                    ],
                    instrumentId=valid_instrument_id,
                    isin=valid_isin,
                    local_ccy=True,
                    name="asset",
                    values=[
                        100.0,
                        100.0978,
                    ],
                    valuetype="Price(Close)",
                )
            )
        self.assertIn(
            member="does not match '^$|([a-f\\\\d]{24})'",
            container=getattr(e_three.exception, "message"),
        )

        with self.assertRaises(Exception) as e_four:
            OpenTimeSeries(
                TimeSerie(
                    _id=invalid_timeseries_id_two,
                    currency="SEK",
                    dates=[
                        "2017-05-29",
                        "2017-05-30",
                    ],
                    instrumentId=valid_instrument_id,
                    isin=valid_isin,
                    local_ccy=True,
                    name="asset",
                    values=[
                        100.0,
                        100.0978,
                    ],
                    valuetype="Price(Close)",
                )
            )
        self.assertIn(
            member="does not match '^$|([a-f\\\\d]{24})'",
            container=getattr(e_four.exception, "message"),
        )

        with self.assertRaises(Exception) as e_five:
            OpenTimeSeries(
                TimeSerie(
                    _id=valid_timeseries_id,
                    currency="SEK",
                    dates=[
                        "2017-05-29",
                        "2017-05-30",
                    ],
                    instrumentId=invalid_instrument_id_one,
                    isin=valid_isin,
                    local_ccy=True,
                    name="asset",
                    values=[
                        100.0,
                        100.0978,
                    ],
                    valuetype="Price(Close)",
                )
            )
        self.assertIn(
            member="does not match '^$|([a-f\\\\d]{24})'",
            container=getattr(e_five.exception, "message"),
        )

        with self.assertRaises(Exception) as e_six:
            OpenTimeSeries(
                TimeSerie(
                    _id=valid_timeseries_id,
                    currency="SEK",
                    dates=[
                        "2017-05-29",
                        "2017-05-30",
                    ],
                    instrumentId=invalid_instrument_id_two,
                    isin=valid_isin,
                    local_ccy=True,
                    name="asset",
                    values=[
                        100.0,
                        100.0978,
                    ],
                    valuetype="Price(Close)",
                )
            )

        self.assertIn(
            member="does not match '^$|([a-f\\\\d]{24})'",
            container=getattr(e_six.exception, "message"),
        )

        with self.assertRaises(Exception) as e_seven:
            OpenTimeSeries(
                TimeSerie(
                    _id=valid_timeseries_id,
                    currency="SEK",
                    dates=[
                        "2017-05-29",
                        "2017-05-29",
                    ],
                    instrumentId=valid_instrument_id,
                    isin=valid_isin,
                    local_ccy=True,
                    name="asset",
                    values=[
                        100.0,
                        100.0978,
                    ],
                    valuetype="Price(Close)",
                )
            )

        self.assertIn(
            member="has non-unique elements",
            container=getattr(e_seven.exception, "message"),
        )

    def test_opentimeseries_from_1d_rate_to_cumret(self: TTestOpenTimeSeries):
        tms = OpenTimeSeries(
            TimeSerie(
                _id="",
                instrumentId="",
                currency="SEK",
                dates=[
                    "2022-12-05",
                    "2022-12-06",
                    "2022-12-07",
                    "2022-12-08",
                    "2022-12-09",
                    "2022-12-12",
                    "2022-12-13",
                    "2022-12-14",
                    "2022-12-15",
                    "2022-12-16",
                    "2022-12-19",
                ],
                local_ccy=True,
                name="asset",
                values=[
                    0.02434,
                    0.02434,
                    0.02434,
                    0.02434,
                    0.02434,
                    0.02434,
                    0.02434,
                    0.02434,
                    0.02434,
                    0.02434,
                    0.02434,
                ],
                valuetype="Price(Close)",
            )
        )
        ave_rate = f"{tms.tsdf.mean().iloc[0]:.5f}"
        self.assertEqual(ave_rate, "0.02434")

        tms.from_1d_rate_to_cumret()

        val_ret = f"{tms.value_ret:.5f}"
        self.assertEqual(val_ret, "0.00093")

    def test_opentimeseries_geo_ret_value_ret_exceptions(self: TTestOpenTimeSeries):
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
            (
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        )

        with self.assertRaises(Exception) as e_grf_zero:
            _ = zeroseries.geo_ret_func()

        self.assertEqual(
            e_grf_zero.exception.args[0],
            (
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        )
        with self.assertRaises(Exception) as e_vr_zero:
            _ = zeroseries.value_ret

        self.assertEqual(
            e_vr_zero.exception.args[0],
            (
                "Simple Return cannot be calculated due to an "
                "initial value being zero."
            ),
        )

        with self.assertRaises(Exception) as e_vrf_zero:
            _ = zeroseries.value_ret_func()

        self.assertEqual(
            e_vrf_zero.exception.args[0],
            (
                "Simple Return cannot be calculated due to an "
                "initial value being zero."
            ),
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
            (
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        )

        with self.assertRaises(Exception) as e_grf_neg:
            _ = negseries.geo_ret_func()

        self.assertEqual(
            e_grf_neg.exception.args[0],
            (
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        )

    def test_opentimeseries_value_nan_handle(self: TTestOpenTimeSeries):
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
            # noinspection PyTypeChecker
            _ = nanseries.value_nan_handle(method="other")

        self.assertEqual(
            e_method.exception.args[0],
            "Method must be either fill or drop passed as string.",
        )

    def test_opentimeseries_return_nan_handle(self: TTestOpenTimeSeries):
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
            # noinspection PyTypeChecker
            _ = nanseries.return_nan_handle(method="other")

        self.assertEqual(
            e_method.exception.args[0],
            "Method must be either fill or drop passed as string.",
        )

    def test_opentimeseries_miscellaneous(self: TTestOpenTimeSeries):
        mseries = self.randomseries.from_deepcopy()

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
        self.assertEqual(f"{impvol:.12f}", "0.102089329036")
        impvoldrifted = mseries.vol_from_var_func(drift_adjust=True)
        self.assertEqual(f"{impvoldrifted:.12f}", "0.102454621604")

    def test_opentimeseries_value_ret_calendar_period(self: TTestOpenTimeSeries):
        vrcseries = self.randomseries.from_deepcopy()

        vrfs_y = vrcseries.value_ret_func(
            from_date=dt.date(2017, 12, 29), to_date=dt.date(2018, 12, 28)
        )
        vrvrcs_y = vrcseries.value_ret_calendar_period(year=2018)
        self.assertEqual(f"{vrfs_y:.11f}", f"{vrvrcs_y:.11f}")

        vrfs_ym = vrcseries.value_ret_func(
            from_date=dt.date(2018, 4, 30), to_date=dt.date(2018, 5, 31)
        )
        vrvrcs_ym = vrcseries.value_ret_calendar_period(year=2018, month=5)
        self.assertEqual(f"{vrfs_ym:.11f}", f"{vrvrcs_ym:.11f}")

    def test_opentimeseries_to_drawdown_series(self: TTestOpenTimeSeries):
        mseries = self.randomseries.from_deepcopy()
        dd = mseries.max_drawdown
        mseries.to_drawdown_series()
        dds = float(mseries.tsdf.min())
        self.assertEqual(f"{dd:.11f}", f"{dds:.11f}")

    def test_opentimeseries_set_new_label(self: TTestOpenTimeSeries):
        lseries = self.randomseries.from_deepcopy()

        self.assertTupleEqual(lseries.tsdf.columns[0], ("Asset", "Price(Close)"))

        lseries.set_new_label(lvl_zero="zero")
        self.assertTupleEqual(lseries.tsdf.columns[0], ("zero", "Price(Close)"))

        lseries.set_new_label(lvl_one="one")
        self.assertTupleEqual(lseries.tsdf.columns[0], ("zero", "one"))

        lseries.set_new_label(lvl_zero="two", lvl_one="three")
        self.assertTupleEqual(lseries.tsdf.columns[0], ("two", "three"))

        lseries.set_new_label(delete_lvl_one=True)
        self.assertEqual(lseries.tsdf.columns[0], "two")
