"""Test suite for the openseries/frame.py module."""
# mypy: disable-error-code="operator,type-arg,unused-ignore"
from __future__ import annotations

from datetime import date as dtdate
from decimal import ROUND_HALF_UP, Decimal, localcontext
from itertools import product as iter_product
from json import loads
from pathlib import Path
from typing import Hashable, Optional, Union, cast
from unittest import TestCase
from unittest.mock import patch

import pytest
from pandas import DataFrame, Series, date_range, read_excel
from pandas.testing import assert_frame_equal
from requests.exceptions import ConnectionError

from openseries.datefixer import date_offset_foll
from openseries.frame import OpenFrame
from openseries.load_plotly import load_plotly_dict
from openseries.risk import cvar_down_calc, var_down_calc
from openseries.series import OpenTimeSeries
from openseries.types import (
    LiteralFrameProps,
    LiteralPortfolioWeightings,
    ValueType,
)
from tests.common_sim import FIVE_SIMS


class TestOpenFrame(TestCase):

    """class to run unittests on the module frame.py."""

    randomframe: OpenFrame
    randomseries: OpenTimeSeries

    @classmethod
    def setUpClass(cls: type[TestOpenFrame]) -> None:
        """SetUpClass for the TestOpenFrame class."""
        cls.randomseries = OpenTimeSeries.from_df(
            FIVE_SIMS.to_dataframe(name="Asset", end=dtdate(2019, 6, 30)),
        ).to_cumret()
        cls.randomframe = OpenFrame(
            [
                OpenTimeSeries.from_df(
                    FIVE_SIMS.to_dataframe(name="Asset", end=dtdate(2019, 6, 30)),
                    column_nmbr=serie,
                )
                for serie in range(FIVE_SIMS.number_of_sims)
            ],
        )

    def test_to_json(self: TestOpenFrame) -> None:
        """Test to_json method."""
        jframe = self.randomframe.from_deepcopy()

        filename = "framesaved.json"
        if Path.home().joinpath("Documents").exists():
            framefile = Path.home().joinpath("Documents").joinpath(filename)
        else:
            framefile = Path(__file__).resolve().parent.joinpath(filename)

        if Path(framefile).exists():
            msg = "test_save_to_json test case setup failed."
            raise FileExistsError(msg)

        kwargs = [
            {"filename": str(framefile)},
            {"filename": "framesaved.json", "directory": str(framefile.parent)},
        ]
        for kwarg in kwargs:
            data = jframe.to_json(**kwarg)  # type: ignore[arg-type]
            if [item.get("name") for item in data] != [
                "Asset_0",
                "Asset_1",
                "Asset_2",
                "Asset_3",
                "Asset_4",
            ]:
                msg = "Unexpected data from json"
                raise ValueError(msg)

            if not Path(framefile).exists():
                msg = "json file not created"
                raise FileNotFoundError(msg)

            framefile.unlink()

            if Path(framefile).exists():
                msg = "json file not deleted as intended"
                raise FileExistsError(msg)

        localfile = Path(__file__).resolve().parent.joinpath(filename)

        with patch("pathlib.Path.exists") as mock_doesnotexist:
            mock_doesnotexist.return_value = False
            data = jframe.to_json(filename=filename)

        if [item.get("name") for item in data] != [
            "Asset_0",
            "Asset_1",
            "Asset_2",
            "Asset_3",
            "Asset_4",
        ]:
            msg = "Unexpected data from json"
            raise ValueError(msg)

        localfile.unlink()

    def test_to_xlsx(self: TestOpenFrame) -> None:
        """Test to_xlsx method."""
        xseries = self.randomframe.from_deepcopy()

        filename = "trial.xlsx"
        if Path.home().joinpath("Documents").exists():
            basefile = Path.home().joinpath("Documents").joinpath(filename)
        else:
            basefile = Path(__file__).resolve().parent.joinpath(filename)

        if Path(basefile).exists():
            msg = "test_save_to_xlsx test case setup failed."
            raise FileExistsError(msg)

        seriesfile = Path(
            xseries.to_xlsx(filename=filename, sheet_title="boo"),
        ).resolve()

        if basefile != seriesfile:
            msg = "test_save_to_xlsx test case setup failed."
            raise ValueError(msg)

        if not Path(seriesfile).exists():
            msg = "xlsx file not created"
            raise FileNotFoundError(msg)

        seriesfile.unlink()

        directory = Path(__file__).resolve().parent
        seriesfile = Path(
            xseries.to_xlsx(filename="trial.xlsx", directory=directory),
        ).resolve()

        if not Path(seriesfile).exists():
            msg = "xlsx file not created"
            raise FileNotFoundError(msg)

        seriesfile.unlink()

        if Path(seriesfile).exists():
            msg = "xlsx file not deleted as intended"
            raise FileExistsError(msg)

        with pytest.raises(
            expected_exception=NameError,
            match="Filename must end with .xlsx",
        ):
            _ = xseries.to_xlsx(filename="trial.pdf")

        with Path.open(basefile, "w") as fakefile:
            fakefile.write("Hello world")

        with pytest.raises(
            expected_exception=FileExistsError,
            match=f"{filename} already exists.",
        ):
            _ = xseries.to_xlsx(filename=filename, overwrite=False)

        basefile.unlink()

        localfile = Path(__file__).resolve().parent.joinpath(filename)
        with patch("pathlib.Path.exists") as mock_doesnotexist:
            mock_doesnotexist.return_value = False
            seriesfile = Path(xseries.to_xlsx(filename=filename)).resolve()

        if localfile != seriesfile:
            msg = "test_save_to_xlsx test case setup failed."
            raise ValueError(msg)

        dframe = read_excel(
            io=seriesfile,
            header=0,
            index_col=0,
            usecols="A:F",
            skiprows=[1, 2],
            engine="openpyxl",
        )

        df_index = [dejt.date().strftime("%Y-%m-%d") for dejt in dframe.head().index]
        if df_index != [
            "2009-06-30",
            "2009-07-01",
            "2009-07-02",
            "2009-07-03",
            "2009-07-06",
        ]:
            msg = "save_to_xlsx not working as intended."
            raise ValueError(msg)

        seriesfile.unlink()

    def test_calc_range(self: TestOpenFrame) -> None:
        """Test calc_range method."""
        crframe = self.randomframe.from_deepcopy()
        start, end = crframe.first_idx.strftime("%Y-%m-%d"), crframe.last_idx.strftime(
            "%Y-%m-%d",
        )
        rst, ren = crframe.calc_range()

        if [start, end] != [rst.strftime("%Y-%m-%d"), ren.strftime("%Y-%m-%d")]:
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Function calc_range returned earlier date < series start",
        ):
            _, _ = crframe.calc_range(months_offset=125)

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt date < series start",
        ):
            _, _ = crframe.calc_range(from_dt=dtdate(2009, 5, 31))

        with pytest.raises(
            expected_exception=ValueError,
            match="Given to_dt date > series end",
        ):
            _, _ = crframe.calc_range(to_dt=dtdate(2019, 7, 31))

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt or to_dt dates outside series range",
        ):
            _, _ = crframe.calc_range(
                from_dt=dtdate(2009, 5, 31),
                to_dt=dtdate(2019, 7, 31),
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt or to_dt dates outside series range",
        ):
            _, _ = crframe.calc_range(
                from_dt=dtdate(2009, 7, 31),
                to_dt=dtdate(2019, 7, 31),
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt or to_dt dates outside series range",
        ):
            _, _ = crframe.calc_range(
                from_dt=dtdate(2009, 5, 31),
                to_dt=dtdate(2019, 5, 31),
            )

        nst, nen = crframe.calc_range(
            from_dt=dtdate(2009, 7, 3),
            to_dt=dtdate(2019, 6, 25),
        )
        if nst != dtdate(2009, 7, 3):
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)
        if nen != dtdate(2019, 6, 25):
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)

        crframe.resample()

        earlier_moved, _ = crframe.calc_range(from_dt=dtdate(2009, 8, 10))
        if earlier_moved != dtdate(2009, 7, 31):
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)

        _, later_moved = crframe.calc_range(to_dt=dtdate(2009, 8, 20))
        if later_moved != dtdate(2009, 8, 31):
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)

    def test_resample(self: TestOpenFrame) -> None:
        """Test resample method."""
        expected: int = 121
        rs_frame = self.randomframe.from_deepcopy()
        rs_frame.to_cumret()

        before = cast(Series, rs_frame.value_ret).to_dict()

        rs_frame.resample(freq="BM")

        if rs_frame.length != expected:
            msg = "resample() method generated unexpected result"
            raise ValueError(msg)

        after = cast(Series, rs_frame.value_ret).to_dict()
        if before != after:
            msg = "resample() method generated unexpected result"
            raise ValueError(msg)

    def test_resample_to_business_period_ends(self: TestOpenFrame) -> None:
        """Test resample_to_business_period_ends method."""
        rsb_stubs_frame = OpenFrame(
            [
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=121,
                    end_dt=dtdate(2023, 5, 15),
                ).set_new_label("A"),
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=123,
                    end_dt=dtdate(2023, 5, 16),
                ).set_new_label("B"),
            ],
        )

        rsb_stubs_frame.resample_to_business_period_ends(freq="BM")
        new_stubs_dates = rsb_stubs_frame.tsdf.index.tolist()

        if new_stubs_dates != [
            dtdate(2023, 1, 15),
            dtdate(2023, 1, 31),
            dtdate(2023, 2, 28),
            dtdate(2023, 3, 31),
            dtdate(2023, 4, 28),
            dtdate(2023, 5, 15),
        ]:
            msg = (
                "resample_to_business_period_ends() method "
                "generated unexpected result"
            )
            raise ValueError(msg)

        rsb_frame = OpenFrame(
            [
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=88,
                    end_dt=dtdate(2023, 4, 28),
                ).set_new_label("A"),
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=8,
                    end_dt=dtdate(2023, 4, 28),
                ).set_new_label("B"),
            ],
        )

        rsb_frame.resample_to_business_period_ends(freq="BM")
        new_dates = rsb_frame.tsdf.index.tolist()

        if new_dates != [
            dtdate(2023, 1, 31),
            dtdate(2023, 2, 28),
            dtdate(2023, 3, 31),
            dtdate(2023, 4, 28),
        ]:
            msg = (
                "resample_to_business_period_ends() method "
                "generated unexpected result"
            )
            raise ValueError(msg)

    def test_max_drawdown_date(self: TestOpenFrame) -> None:
        """Test max_drawdown_date method."""
        mddframe = self.randomframe.from_deepcopy()
        mddframe.to_cumret()
        if [
            dtdate(2016, 9, 27),
            dtdate(2016, 9, 22),
            dtdate(2015, 7, 8),
            dtdate(2017, 2, 6),
            dtdate(2018, 10, 19),
        ] != cast(Series, mddframe.max_drawdown_date).tolist():
            msg = "max_drawdown_date property generated unexpected result"
            raise ValueError(msg)

    def test_make_portfolio(self: TestOpenFrame) -> None:
        """Test make_portfolio method."""
        mpframe = self.randomframe.from_deepcopy()
        mpframe.to_cumret()
        mpframe.weights = [1.0 / mpframe.item_count] * mpframe.item_count

        name = "portfolio"
        mptail = mpframe.make_portfolio(name=name).tail()
        mptail = mptail.map(lambda nn: f"{nn:.6f}")

        correct = ["0.885890", "0.888745", "0.892225", "0.890144", "0.888390"]
        wrong = ["0.885890", "0.888745", "0.892225", "0.890144", "0.888380"]
        true_tail = DataFrame(
            columns=[[name], [ValueType.PRICE]],
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
            columns=[[name], [ValueType.PRICE]],
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

        with pytest.raises(expected_exception=AssertionError, match="are different"):
            assert_frame_equal(false_tail, mptail, check_exact=True)

        mpframe.weights = None
        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "OpenFrame weights property must be provided "
                "to run the make_portfolio method."
            ),
        ):
            _ = mpframe.make_portfolio(name=name)

    def test_make_portfolio_weight_strat(self: TestOpenFrame) -> None:
        """Test make_portfolio method with weight_strat."""
        mpframe = self.randomframe.from_deepcopy()
        mpframe.to_cumret()
        name = "portfolio"

        _ = mpframe.make_portfolio(name=name, weight_strat="eq_weights")
        weights: Optional[list[float]] = [0.2, 0.2, 0.2, 0.2, 0.2]
        if weights != mpframe.weights:
            msg = "make_portfolio() equal weight strategy not working as intended."
            ValueError(msg)

        with localcontext() as decimal_context:
            decimal_context.rounding = ROUND_HALF_UP

            _ = mpframe.make_portfolio(name=name, weight_strat="eq_risk")
            eq_risk_weights = [
                round(Decimal(wgt), 6) for wgt in cast(list[float], mpframe.weights)
            ]
            if eq_risk_weights != [
                Decimal("0.206999"),
                Decimal("0.193416"),
                Decimal("0.198024"),
                Decimal("0.206106"),
                Decimal("0.195454"),
            ]:
                msg = "make_portfolio() equal risk strategy not working as intended."
                ValueError(msg)

            _ = mpframe.make_portfolio(name=name, weight_strat="inv_vol")
            inv_vol_weights = [
                round(Decimal(wgt), 6) for wgt in cast(list[float], mpframe.weights)
            ]
            if inv_vol_weights != [
                Decimal("0.252280"),
                Decimal("0.163721"),
                Decimal("0.181780"),
                Decimal("0.230792"),
                Decimal("0.171427"),
            ]:
                msg = "make_portfolio() inverse vol strategy not working as intended."
                ValueError(msg)

            _ = mpframe.make_portfolio(name=name, weight_strat="mean_var")
            mean_var_weights = [
                round(Decimal(wgt), 6) for wgt in cast(list[float], mpframe.weights)
            ]
            if mean_var_weights != [
                Decimal("0.244100"),
                Decimal("0.000000"),
                Decimal("0.000000"),
                Decimal("0.755900"),
                Decimal("0.000000"),
            ]:
                msg = (
                    "make_portfolio() mean variance strategy not working as intended."
                )
                ValueError(msg)

        with pytest.raises(
            expected_exception=NotImplementedError,
            match="Weight strategy not implemented",
        ):
            _ = mpframe.make_portfolio(
                name=name,
                weight_strat=cast(LiteralPortfolioWeightings, "bogus"),
            )

    def test_add_timeseries(self: TestOpenFrame) -> None:
        """Test add_timeseries method."""
        frameas = self.randomframe.from_deepcopy()
        items = int(frameas.item_count)
        frameas.weights = [1 / items] * items
        cols = list(frameas.columns_lvl_zero)
        nbr_cols = int(len(frameas.columns_lvl_zero))
        seriesas = self.randomseries.from_deepcopy()
        seriesas.set_new_label("Asset_6")
        frameas.add_timeseries(seriesas)

        if items + 1 != frameas.item_count:
            msg = "add_timeseries() method did not work as intended."
            raise ValueError(msg)
        if nbr_cols + 1 != len(frameas.columns_lvl_zero):
            msg = "add_timeseries() method did not work as intended."
            raise ValueError(msg)
        if [*cols, "Asset_6"] != frameas.columns_lvl_zero:
            msg = "add_timeseries() method did not work as intended."
            raise ValueError(msg)

    def test_delete_timeseries(self: TestOpenFrame) -> None:
        """Test delete_timeseries method."""
        frame = self.randomframe.from_deepcopy()
        frame.weights = [0.4, 0.1, 0.2, 0.1, 0.2]

        lbl = "Asset_1"
        frame.delete_timeseries(lbl)
        labels = [ff.label for ff in frame.constituents]

        if labels != ["Asset_0", "Asset_2", "Asset_3", "Asset_4"]:
            msg = "delete_timeseries() method did not work as intended."
            raise ValueError(msg)
        if frame.weights != [0.4, 0.2, 0.1, 0.2]:
            msg = "delete_timeseries() method did not work as intended."
            raise ValueError(msg)

    def test_risk_functions_same_as_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that risk measures align between OpenFrame and OpenTimeSeries."""
        riskseries = self.randomseries.from_deepcopy()
        riskseries.set_new_label(lvl_zero="Asset_0")
        riskframe = self.randomframe.from_deepcopy()
        riskseries.to_cumret()
        riskframe.to_cumret()

        if riskseries.cvar_down != cvar_down_calc(riskseries.tsdf.iloc[:, 0].tolist()):
            msg = "CVaR for OpenTimeSeries not equal"
            raise ValueError(msg)
        if riskseries.var_down != var_down_calc(riskseries.tsdf.iloc[:, 0].tolist()):
            msg = "VaR for OpenTimeSeries not equal"
            raise ValueError(msg)

        if cast(Series, riskframe.cvar_down).iloc[0] != cvar_down_calc(
            riskframe.tsdf.iloc[:, 0],
        ):
            msg = "CVaR for OpenFrame not equal"
            raise ValueError(msg)
        if cast(Series, riskframe.var_down).iloc[0] != var_down_calc(
            riskframe.tsdf.iloc[:, 0],
        ):
            msg = "VaR for OpenFrame not equal"
            raise ValueError(msg)

        if cast(Series, riskframe.cvar_down).iloc[0] != cvar_down_calc(riskframe.tsdf):
            msg = "CVaR for OpenFrame not equal"
            raise ValueError(msg)
        if cast(Series, riskframe.var_down).iloc[0] != var_down_calc(
            riskframe.tsdf,
        ):
            msg = "VaR for OpenFrame not equal"
            raise ValueError(msg)

    def test_methods_same_as_opentimeseries(self: TestOpenFrame) -> None:
        """Test that method results align between OpenFrame and OpenTimeSeries."""
        sameseries = self.randomseries.from_deepcopy()
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

    def test_calc_methods_same_as_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that calc results align between OpenFrame and OpenTimeSeries."""
        sames = self.randomseries.from_deepcopy()
        sames.to_cumret()
        sames.set_new_label(lvl_zero="Asset_0")
        samef = self.randomframe.from_deepcopy()
        for serie in samef.constituents:
            serie.to_cumret()
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
            "vol_from_var_func",
            "target_weight_from_var",
            "value_ret_func",
            "var_down_func",
            "vol_func",
            "worst_func",
            "z_score_func",
        ]
        for method in methods_to_compare:
            if (
                f"{getattr(sames, method)(months_from_last=12):.11f}"
                != f"{float(getattr(samef, method)(months_from_last=12).iloc[0]):.11f}"
            ):
                msg = (
                    f"Calc method {method} not aligned between "
                    "OpenTimeSeries and OpenFrame"
                )
                raise ValueError(msg)

    def test_ratio_methods_same_as_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that ratio methods align between OpenFrame and OpenTimeSeries."""
        sames = self.randomseries.from_deepcopy()
        sames.to_cumret()
        sames.set_new_label(lvl_zero="Asset_0")
        samef = self.randomframe.from_deepcopy()
        samef.to_cumret()

        smf_vrf = cast(
            Series,
            samef.ret_vol_ratio_func(riskfree_rate=0.0, months_from_last=12),
        ).iloc[0]
        if (
            f"{sames.ret_vol_ratio_func(riskfree_rate=0.0, months_from_last=12):.11f}"
            != f"{smf_vrf:.11f}"
        ):
            msg = (
                "ret_vol_ratio_func() not aligned between "
                "OpenTimeSeries and OpenFrame"
            )
            raise ValueError(msg)

        smf_srf = cast(
            Series,
            samef.sortino_ratio_func(riskfree_rate=0.0, months_from_last=12),
        ).iloc[0]
        if (
            f"{sames.sortino_ratio_func(riskfree_rate=0.0, months_from_last=12):.11f}"
            != f"{smf_srf:.11f}"
        ):
            msg = (
                "sortino_ratio_func() not aligned between "
                "OpenTimeSeries and OpenFrame"
            )
            raise ValueError(msg)

    def test_measures_same_as_opentimeseries(self: TestOpenFrame) -> None:
        """Test that measure results align between OpenFrame and OpenTimeSeries."""
        frame_0 = self.randomframe.from_deepcopy()
        for zerie in frame_0.constituents:
            zerie.to_cumret()
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

        for prop in common_calc_props:
            result = getattr(frame_0, prop).tolist()
            rounded = [f"{item:.10f}" for item in result]
            roundmeasure = [
                f"{getattr(serie, prop):.10f}" for serie in frame_0.constituents
            ]
            if rounded != roundmeasure:
                msg = (
                    f"Property {prop} not aligned between "
                    "OpenTimeSeries and OpenFrame"
                )
                raise ValueError(msg)

    def test_properties_same_as_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that property results align between OpenFrame and OpenTimeSeries."""
        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

        common_props_to_compare = ["periods_in_a_year", "yearfrac"]
        for comnprop in common_props_to_compare:
            if getattr(sameseries, comnprop) != getattr(sameframe, comnprop):
                msg = (
                    f"Property {comnprop} not aligned between "
                    "OpenTimeSeries and OpenFrame"
                )
                raise ValueError(msg)

    def test_keeping_attributes_aligned_vs_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that attributes are aligned between OpenFrame and OpenTimeSeries."""
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
            "timeseries_id",
            "instrument_id",
            "currency",
            "isin",
            "dates",
            "name",
            "valuetype",
            "label",
            "domestic",
            "countries",
        ]

        pydantic_basemodel_attributes = [
            "model_extra",
            "model_fields",
            "model_config",
            "model_computed_fields",
            "model_fields_set",
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
            if not a.startswith("_") and not callable(getattr(sameseries, a))
        ]
        series_compared = set(series_props).symmetric_difference(
            set(
                common_calc_props
                + common_props
                + common_attributes
                + pydantic_basemodel_attributes
                + series_attributes,
            ),
        )
        if len(series_compared) != 0:
            msg = f"Difference is: {series_compared}"
            raise ValueError(msg)

        frame_props = [
            a
            for a in dir(sameframe)
            if not a.startswith("_") and not callable(getattr(sameframe, a))
        ]
        frame_compared = set(frame_props).symmetric_difference(
            set(
                common_calc_props
                + common_props
                + common_attributes
                + pydantic_basemodel_attributes
                + frame_attributes
                + frame_calc_props,
            ),
        )
        if len(frame_compared) != 0:
            msg = f"Difference is: {frame_compared}"
            raise ValueError(msg)

    def test_keeping_methods_aligned_vs_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that methods are aligned between OpenFrame and OpenTimeSeries."""
        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

        pydantic_basemodel_methods = [
            "dict",
            "copy",
            "model_post_init",
            "model_rebuild",
            "model_dump_json",
            "model_validate_json",
            "model_validate_strings",
            "model_copy",
            "model_dump",
            "model_validate",
            "model_construct",
            "model_parametrized_name",
            "model_json_schema",
            "parse_obj",
            "update_forward_refs",
            "parse_file",
            "schema",
            "construct",
            "schema_json",
            "parse_raw",
            "from_orm",
            "json",
            "validate",
        ]

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
            "to_json",
            "to_xlsx",
            "value_nan_handle",
            "value_ret_calendar_period",
            "value_to_diff",
            "value_to_log",
            "value_to_ret",
        ]

        series_createmethods = [
            "from_arrays",
            "from_df",
            "from_fixed_rate",
        ]

        series_unique = [
            "ewma_vol_func",
            "from_1d_rate_to_cumret",
            "pandas_df",
            "running_adjustment",
            "set_new_label",
            "setup_class",
            "dates_and_values_validate",
        ]

        frame_unique = [
            "add_timeseries",
            "beta",
            "check_labels_unique",
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
            "set_tsdf",
            "trunc_frame",
        ]

        series_methods = [
            a
            for a in dir(sameseries)
            if not a.startswith("_") and callable(getattr(sameseries, a))
        ]
        series_compared = set(series_methods).symmetric_difference(
            set(
                pydantic_basemodel_methods
                + common_calc_methods
                + common_methods
                + series_createmethods
                + series_unique,
            ),
        )
        if len(series_compared) != 0:
            msg = f"Difference is: {series_compared}"
            raise ValueError(msg)

        frame_methods = [
            a
            for a in dir(sameframe)
            if not a.startswith("_") and callable(getattr(sameframe, a))
        ]
        frame_compared = set(frame_methods).symmetric_difference(
            set(
                pydantic_basemodel_methods
                + common_calc_methods
                + common_methods
                + frame_unique,
            ),
        )
        if len(frame_compared) != 0:
            msg = f"Difference is: {frame_compared}"
            raise ValueError(msg)

    def test_value_to_log(self: TestOpenFrame) -> None:
        """Test value_to_log method."""
        logframe = self.randomframe.from_deepcopy()
        logframe.to_cumret()

        aaframe = logframe.tsdf.map(lambda item: f"{item:.12f}")
        bbdict = aaframe.to_dict(orient="list")
        b4_log = [bbdict[key] for key in bbdict]

        logframe.value_to_log()

        ccframe = logframe.tsdf.map(lambda item: f"{item:.12f}")
        eedict = ccframe.to_dict(orient="list")
        middle_log = [eedict[key] for key in eedict]

        if b4_log == middle_log:
            msg = "Method value_to_log() did not work as intended."
            raise ValueError(msg)

    def test_correl_matrix(self: TestOpenFrame) -> None:
        """Test correl_matrix method."""
        corrframe = self.randomframe.from_deepcopy()
        corrframe.to_cumret()
        dict1 = corrframe.correl_matrix.map(lambda nn: f"{nn:.12f}").to_dict()
        dict2 = {
            "Asset_0": {
                "Asset_0": "1.000000000000",
                "Asset_1": "0.029908410596",
                "Asset_2": "0.000018749152",
                "Asset_3": "0.013318562627",
                "Asset_4": "0.006281645813",
            },
            "Asset_1": {
                "Asset_0": "0.029908410596",
                "Asset_1": "1.000000000000",
                "Asset_2": "-0.002797510515",
                "Asset_3": "0.000233708343",
                "Asset_4": "-0.010952080556",
            },
            "Asset_2": {
                "Asset_0": "0.000018749152",
                "Asset_1": "-0.002797510515",
                "Asset_2": "1.000000000000",
                "Asset_3": "0.001543294531",
                "Asset_4": "0.019235178850",
            },
            "Asset_3": {
                "Asset_0": "0.013318562627",
                "Asset_1": "0.000233708343",
                "Asset_2": "0.001543294531",
                "Asset_3": "1.000000000000",
                "Asset_4": "0.005297751873",
            },
            "Asset_4": {
                "Asset_0": "0.006281645813",
                "Asset_1": "-0.010952080556",
                "Asset_2": "0.019235178850",
                "Asset_3": "0.005297751873",
                "Asset_4": "1.000000000000",
            },
        }

        if dict1 != dict2:
            msg = "Unexpected result(s) from method correl_matrix()"
            raise ValueError(msg)

    def test_plot_series(self: TestOpenFrame) -> None:
        """Test plot_series method."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        directory = Path(__file__).resolve().parent
        _, figfile = plotframe.plot_series(auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        fig, _ = plotframe.plot_series(auto_open=False, output_type="div")
        fig_json = loads(fig.to_json())

        for i in range(plotframe.item_count):
            rawdata = [f"{x:.11f}" for x in plotframe.tsdf.iloc[1:5, i]]
            fig_data = [f"{x:.11f}" for x in fig_json["data"][i]["y"][1:5]]
            if rawdata != fig_data:
                msg = "Unaligned data between original and data in Figure."
                raise ValueError(msg)

        fig_last, _ = plotframe.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
        )
        fig_last_json = loads(fig_last.to_json())
        rawlast = plotframe.tsdf.iloc[-1, -1]
        figlast = fig_last_json["data"][-1]["y"][0]
        if f"{figlast:.12f}" != f"{rawlast:.12f}":
            msg = "Unaligned data between original and data in Figure."
            raise ValueError(msg)

        fig_last_fmt, _ = plotframe.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
            tick_fmt=".3%",
        )
        fig_last_fmt_json = loads(fig_last_fmt.to_json())
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]

        if last_fmt != "Last 19.964%":
            msg = "Unaligned data between original and data in Figure."
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Must provide same number of labels as items in frame.",
        ):
            _, _ = plotframe.plot_series(auto_open=False, labels=["a", "b"])

        _, logo = load_plotly_dict()

        fig_logo, _ = plotframe.plot_series(
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        fig_logo_json = loads(fig_logo.to_json())
        if fig_logo_json["layout"]["images"][0]["source"] != logo["source"]:
            msg = "plot_series add_logo argument not setup correctly"
            raise ValueError(msg)

        fig_nologo, _ = plotframe.plot_series(
            auto_open=False,
            add_logo=False,
            output_type="div",
        )
        fig_nologo_json = loads(fig_nologo.to_json())
        if fig_nologo_json["layout"].get("images", None):
            msg = "plot_series add_logo argument not setup correctly"
            raise ValueError(msg)

    def test_plot_bars(self: TestOpenFrame) -> None:
        """Test plot_bars method."""
        plotframe = self.randomframe.from_deepcopy()

        directory = Path(__file__).resolve().parent
        _, figfile = plotframe.plot_bars(auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        fig_keys = ["hovertemplate", "name", "type", "x", "y"]
        fig, _ = plotframe.plot_bars(auto_open=False, output_type="div")
        fig_json = loads(fig.to_json())
        made_fig_keys = list(fig_json["data"][0].keys())
        made_fig_keys.sort()
        if made_fig_keys != fig_keys:
            msg = "Data in Figure not as intended."
            raise ValueError(msg)

        for i in range(plotframe.item_count):
            rawdata = [f"{x:.11f}" for x in plotframe.tsdf.iloc[1:5, i]]
            fig_data = [f"{x:.11f}" for x in fig_json["data"][i]["y"][1:5]]
            if rawdata != fig_data:
                msg = "Unaligned data between original and data in Figure."
                raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Must provide same number of labels as items in frame.",
        ):
            _, _ = plotframe.plot_bars(auto_open=False, labels=["a", "b"])

        overlayfig, _ = plotframe.plot_bars(
            auto_open=False,
            output_type="div",
            mode="overlay",
        )
        overlayfig_json = loads(overlayfig.to_json())

        fig_keys.append("opacity")
        if sorted(overlayfig_json["data"][0].keys()) != sorted(fig_keys):
            msg = "Data in Figure not as intended."
            raise ValueError(msg)

        _, logo = load_plotly_dict()

        fig_logo, _ = plotframe.plot_bars(
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        fig_logo_json = loads(fig_logo.to_json())
        if fig_logo_json["layout"]["images"][0]["source"] != logo["source"]:
            msg = "plot_bars add_logo argument not setup correctly"
            raise ValueError(msg)

        fig_nologo, _ = plotframe.plot_bars(
            auto_open=False,
            add_logo=False,
            output_type="div",
        )
        fig_nologo_json = loads(fig_nologo.to_json())
        if fig_nologo_json["layout"].get("images", None):
            msg = "plot_bars add_logo argument not setup correctly"
            raise ValueError(msg)

    def test_plot_methods_mock_logo_url_fail(self: TestOpenFrame) -> None:
        """Test plot_series and plot_bars methods with mock logo file URL fail."""
        plotframe = self.randomframe.from_deepcopy()

        with patch("requests.head") as mock_conn_error:
            mock_conn_error.side_effect = ConnectionError()

            seriesfig, _ = plotframe.plot_series(
                auto_open=False,
                add_logo=True,
                output_type="div",
            )
            seriesfig_json = loads(seriesfig.to_json())
            if seriesfig_json["layout"]["images"][0].get("source", None):
                msg = "plot_series add_logo argument not setup correctly"
                raise ValueError(msg)

            barfig, _ = plotframe.plot_bars(
                auto_open=False,
                add_logo=True,
                output_type="div",
            )
            barfig_json = loads(barfig.to_json())
            if barfig_json["layout"]["images"][0].get("source", None):
                msg = "plot_bars add_logo argument not setup correctly"
                raise ValueError(msg)

            with self.assertLogs() as plotseries_context:
                _, _ = plotframe.plot_series(auto_open=False, output_type="div")
            if (
                "WARNING:root:Failed to add logo image from URL"
                not in plotseries_context.output[0]
            ):
                msg = (
                    "plot_series() method did not warn as "
                    "expected when logo URL not working"
                )
                raise ValueError(msg)

            with self.assertLogs() as plotbars_context:
                _, _ = plotframe.plot_bars(auto_open=False, output_type="div")
            if (
                "WARNING:root:Failed to add logo image from URL"
                not in plotbars_context.output[0]
            ):
                msg = (
                    "plot_bars() method did not warn as "
                    "expected when logo URL not working"
                )
                raise ValueError(msg)

        with patch("requests.head") as mock_statuscode:
            mock_statuscode.return_value.status_code = 400

            with self.assertLogs() as plotseries_context:
                _, _ = plotframe.plot_series(auto_open=False, output_type="div")
            if (
                "WARNING:root:Failed to add logo image from URL"
                not in plotseries_context.output[0]
            ):
                msg = (
                    "plot_series() method did not warn as "
                    "expected when logo URL not working"
                )
                raise ValueError(msg)

            with self.assertLogs() as plotbars_context:
                _, _ = plotframe.plot_bars(auto_open=False, output_type="div")
            if (
                "WARNING:root:Failed to add logo image from URL"
                not in plotbars_context.output[0]
            ):
                msg = (
                    "plot_bars() method did not warn as "
                    "expected when logo URL not working"
                )
                raise ValueError(msg)

    def test_passed_empty_list(self: TestOpenFrame) -> None:
        """Test warning on object construct with empty list."""
        with self.assertLogs() as contextmgr:
            OpenFrame([])
        if contextmgr.output != ["WARNING:root:OpenFrame() was passed an empty list."]:
            msg = "OpenFrame failed to log warning about empty input list."
            raise ValueError(msg)

    def test_drawdown_details(self: TestOpenFrame) -> None:
        """Test drawdown_details method."""
        ddframe = self.randomframe.from_deepcopy()
        for serie in ddframe.constituents:
            serie.to_cumret()
        ddframe.to_cumret()
        dds = ddframe.drawdown_details().loc["Days from start to bottom"].tolist()

        if [1747, 1424, 128, 664, 3397] != dds:
            msg = "Method drawdown_details() did not produce intended result."
            raise ValueError(msg)

    def test_trunc_frame(self: TestOpenFrame) -> None:
        """Test trunc_frame method."""
        series_long = self.randomseries.from_deepcopy()
        series_long.set_new_label("Long")
        tmp_series = self.randomseries.from_deepcopy()
        series_short = OpenTimeSeries.from_df(
            tmp_series.tsdf.loc[
                cast(int, dtdate(2017, 6, 27)) : cast(  # type: ignore[index]
                    int,
                    dtdate(2018, 6, 27),
                ),
                ("Asset_0", ValueType.PRICE),
            ],
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

        if firsts == frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)
        if lasts == frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

        frame.trunc_frame()

        if firsts != frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)
        if lasts != frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

        trunced = [dtdate(2017, 12, 29), dtdate(2018, 3, 29)]

        frame.trunc_frame(start_cut=trunced[0], end_cut=trunced[1])

        if trunced != [frame.first_idx, frame.last_idx]:
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

    def test_trunc_frame_start_fail(self: TestOpenFrame) -> None:
        """Test trunc_frame method start fail scenario."""
        frame = OpenFrame(
            [
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["a"],
                        data=[1, 2, 3, 4, 5],
                        index=[
                            "2022-10-01",
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["b"],
                        data=[6, 7, 8, 9, 10],
                        index=[
                            "2022-10-01",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-06",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["c"],
                        data=[11, 12, 13, 14, 15],
                        index=[
                            "2022-10-02",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-06",
                            "2022-10-07",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["d"],
                        data=[16, 17, 18, 19, 20],
                        index=[
                            "2022-10-01",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-06",
                            "2022-10-07",
                        ],
                    ),
                ),
            ],
        )
        with self.assertLogs("root", level="WARNING") as logs:
            frame.trunc_frame()
        if (
            "WARNING:root:One or more constituents "
            "still not truncated to same start dates."
        ) not in logs.output[0]:
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

    def test_trunc_frame_end_fail(self: TestOpenFrame) -> None:
        """Test trunc_frame method end fail scenario."""
        frame = OpenFrame(
            [
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["a"],
                        data=[1, 2, 3, 4, 5],
                        index=[
                            "2022-10-01",
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-07",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["b"],
                        data=[6, 7, 8, 9],
                        index=[
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-07",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["c"],
                        data=[10, 11, 12, 13, 14],
                        index=[
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-07",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["d"],
                        data=[15, 16, 17, 18, 19],
                        index=[
                            "2022-10-01",
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                        ],
                    ),
                ),
            ],
        )
        with self.assertLogs("root", level="WARNING") as logs:
            frame.trunc_frame()
        if (
            "WARNING:root:One or more constituents "
            "still not truncated to same end dates."
        ) not in logs.output[0]:
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

    def test_merge_series(self: TestOpenFrame) -> None:
        """Test merge_series method."""
        aframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="Asset_one",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
                OpenTimeSeries.from_arrays(
                    name="Asset_two",
                    dates=[
                        "2022-08-11",
                        "2022-08-12",
                        "2022-08-13",
                        "2022-08-14",
                        "2022-08-15",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
            ],
        )
        bframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="Asset_one",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
                OpenTimeSeries.from_arrays(
                    name="Asset_two",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-16",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
            ],
        )

        b4df = aframe.tsdf.copy()
        aframe.merge_series(how="outer")
        assert_frame_equal(b4df, aframe.tsdf, check_exact=True)

        bframe.merge_series(how="inner")
        blist = [d.strftime("%Y-%m-%d") for d in bframe.tsdf.index]
        if blist != [
            "2022-07-11",
            "2022-07-12",
            "2022-07-13",
            "2022-07-14",
        ]:
            msg = "Method merge_series() did not work as intended."
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=Exception,
            match=(
                "Merging OpenTimeSeries DataFrames with argument "
                "how=inner produced an empty DataFrame."
            ),
        ):
            aframe.merge_series(how="inner")

    def test_all_properties(self: TestOpenFrame) -> None:
        """Test all_properties method."""
        prop_index = [
            "Simple return",
            "Geometric return",
            "Arithmetic return",
            "Volatility",
            "Downside deviation",
            "Return vol ratio",
            "Sortino ratio",
            "Z-score",
            "Skew",
            "Kurtosis",
            "Positive Share",
            "VaR 95.0%",
            "CVaR 95.0%",
            "Imp vol from VaR 95%",
            "Worst",
            "Worst month",
            "Max Drawdown",
            "Max drawdown date",
            "Max Drawdown in cal yr",
            "first indices",
            "last indices",
            "observations",
            "span of days",
        ]
        apframe = self.randomframe.from_deepcopy()
        apframe.to_cumret()
        result = apframe.all_properties()
        result_index = result.index.tolist()

        if not isinstance(result, DataFrame):
            msg = "Method all_properties() not working as intended."
            raise TypeError(msg)

        result_arg = apframe.all_properties(
            properties=cast(list[LiteralFrameProps], ["geo_ret"]),
        )
        if not isinstance(result_arg, DataFrame):
            msg = "Method all_properties() not working as intended."
            raise TypeError(msg)

        if set(prop_index) != set(result_index):
            msg = "Method all_properties() output not as intended."
            raise ValueError(msg)

        result_values = {}
        for value in result.index:
            if isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], float):
                result_values[
                    value
                ] = f"{result.loc[value, ('Asset_0', ValueType.PRICE)]:.10f}"
            elif isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], int):
                result_values[value] = cast(
                    str,
                    result.loc[value, ("Asset_0", ValueType.PRICE)],
                )
            elif isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], dtdate):
                result_values[value] = result.loc[  # type: ignore[union-attr]
                    value,
                    ("Asset_0", ValueType.PRICE),
                ].strftime("%Y-%m-%d")
            else:
                msg = f"all_properties returned unexpected type {type(value)}"
                raise TypeError(
                    msg,
                )
        expected_values = {
            "Simple return": "-0.2235962176",
            "Volatility": "0.1257285416",
            "Max drawdown date": "2016-09-27",
            "Imp vol from VaR 95%": "0.0937379442",
            "Geometric return": "-0.0250075875",
            "Kurtosis": "208.0369645588",
            "observations": 2512,
            "CVaR 95.0%": "-0.0150413764",
            "last indices": "2019-06-28",
            "span of days": 3650,
            "Sortino ratio": "-0.1625299336",
            "Return vol ratio": "-0.1351175671",
            "Worst": "-0.1833801800",
            "Skew": "-9.1925124207",
            "Positive Share": "0.5017921147",
            "Z-score": "-0.3646357403",
            "Worst month": "-0.1961065251",
            "Downside deviation": "0.1045231133",
            "first indices": "2009-06-30",
            "VaR 95.0%": "-0.0097248785",
            "Arithmetic return": "-0.0169881347",
            "Max Drawdown": "-0.4577528079",
            "Max Drawdown in cal yr": "-0.2834374358",
        }
        if result_values != expected_values:
            msg = "Method all_properties() results not as expected."
            raise ValueError(msg)

        with pytest.raises(expected_exception=ValueError, match="Invalid string: boo"):
            _ = apframe.all_properties(
                properties=cast(list[LiteralFrameProps], ["geo_ret", "boo"]),
            )

    def test_align_index_to_local_cdays(self: TestOpenFrame) -> None:
        """Test align_index_to_local_cdays method."""
        d_range = [d.date() for d in date_range(start="2022-06-01", end="2022-06-15")]
        asim = [1.0] * len(d_range)
        adf = DataFrame(
            data=asim,
            index=d_range,
            columns=[["Asset_a"], [ValueType.PRICE]],
        )
        aseries = OpenTimeSeries.from_df(adf, valuetype=ValueType.PRICE)
        bseries = OpenTimeSeries.from_df(adf, valuetype=ValueType.PRICE)
        bseries.set_new_label("Asset_b")
        aframe = OpenFrame([aseries, bseries])

        midsummer = dtdate(2022, 6, 6)
        if midsummer not in d_range:
            msg = "Midsummer not in date range"
            raise ValueError(msg)

        aframe.align_index_to_local_cdays()
        if midsummer in aframe.tsdf.index:
            msg = "Midsummer in date range"
            raise ValueError(msg)

    def test_rolling_info_ratio(self: TestOpenFrame) -> None:
        """Test rolling_info_ratio method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_info_ratio(long_column=0, short_column=1)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.06464266057",
            "0.10895202629",
            "0.05884671119",
            "0.04300405194",
            "-0.08603374575",
        ]

        if values != checkdata:
            msg = "Result from method rolling_info_ratio() not as intended."
            raise ValueError(msg)

        simdata_fxd_per_yr = frame.rolling_info_ratio(
            long_column=0,
            short_column=1,
            periods_in_a_year_fixed=251,
        )

        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5, 0]]
        checkdata_fxd_per_yr = [
            "0.06469055241",
            "0.10903274564",
            "0.05889030898",
            "0.04303591238",
            "-0.08609748562",
        ]

        if values_fxd_per_yr != checkdata_fxd_per_yr:
            msg = "Result from method rolling_info_ratio() not as intended."
            raise ValueError(msg)

    def test_rolling_beta(self: TestOpenFrame) -> None:
        """Test rolling_beta method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()
        simdata = frame.rolling_beta(asset_column=0, market_column=1)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.06064704174",
            "0.05591653346",
            "0.11041741881",
            "0.08436822615",
            "0.04771407242",
        ]
        if values != checkdata:
            msg = "Result from method rolling_info_ratio() not as intended."
            raise ValueError(msg)

    def test_tracking_error_func(self: TestOpenFrame) -> None:
        """Test tracking_error_func method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdataa = frame.tracking_error_func(base_column=-1)

        if f"{simdataa.iloc[0]:.10f}" != "0.1047183258":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdataa.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        simdatab = frame.tracking_error_func(
            base_column=-1,
            periods_in_a_year_fixed=251,
        )

        if f"{simdatab.iloc[0]:.10f}" != "0.1046408005":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdatab.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        simdatac = frame.tracking_error_func(base_column=("Asset_4", ValueType.PRICE))

        if f"{simdatac.iloc[0]:.10f}" != "0.1047183258":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdatac.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        if f"{simdataa.iloc[0]:.10f}" != f"{simdatac.iloc[0]:.10f}":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdataa.iloc[0]:.10f}' "
                f"versus '{simdatac.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="base_column should be a tuple",
        ):
            _ = frame.tracking_error_func(
                base_column=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_info_ratio_func(self: TestOpenFrame) -> None:
        """Test info_ratio_func method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdataa = frame.info_ratio_func(base_column=-1)

        if f"{simdataa.iloc[0]:.10f}" != "0.4866461738":
            msg = (
                f"Result from info_ratio_func() not "
                f"as expected: '{simdataa.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        simdatab = frame.info_ratio_func(base_column=-1, periods_in_a_year_fixed=251)

        if f"{simdatab.iloc[0]:.10f}" != "0.4862858989":
            msg = (
                f"Result from info_ratio_func() not "
                f"as expected: '{simdatab.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        simdatac = frame.info_ratio_func(base_column=("Asset_4", ValueType.PRICE))

        if f"{simdatac.iloc[0]:.10f}" != "0.4866461738":
            msg = (
                f"Result from info_ratio_func() not "
                f"as expected: '{simdatac.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="base_column should be a tuple",
        ):
            _ = frame.info_ratio_func(
                base_column=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_rolling_corr(self: TestOpenFrame) -> None:
        """Test rolling_corr method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_corr(first_column=0, second_column=1)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.08034452787",
            "0.07003512008",
            "0.13680724381",
            "0.10613002176",
            "0.06652224599",
        ]
        if values != checkdata:
            msg = "Result from method rolling_corr() not as intended."
            raise ValueError(msg)

    def test_rolling_vol(self: TestOpenFrame) -> None:
        """Test rolling_vol method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_vol(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.07189785661",
            "0.07601706214",
            "0.07690774674",
            "0.07640509098",
            "0.07534254420",
        ]

        if values != checkdata:
            msg = "Result from method rolling_vol() not as intended."
            raise ValueError(msg)

        simdata_fxd_per_yr = frame.rolling_vol(
            column=0,
            observations=21,
            periods_in_a_year_fixed=251,
        )

        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5, 0]]
        checkdata_fxd_per_yr = [
            "0.07184462904",
            "0.07596078503",
            "0.07685081023",
            "0.07634852661",
            "0.07528676645",
        ]

        if values_fxd_per_yr != checkdata_fxd_per_yr:
            msg = "Result from method rolling_vol() not as intended."
            raise ValueError(msg)

    def test_rolling_return(self: TestOpenFrame) -> None:
        """Test rolling_return method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_return(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.00312001031",
            "0.00761489128",
            "0.00155338654",
            "0.00190827758",
            "0.00562845778",
        ]

        if values != checkdata:
            msg = "Result from method rolling_return() not as intended."
            raise ValueError(msg)

    def test_rolling_cvar_down(self: TestOpenFrame) -> None:
        """Test rolling_cvar_down method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_cvar_down(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[-5:, 0]]
        checkdata = [
            "-0.01099453714",
            "-0.01099453714",
            "-0.01099453714",
            "-0.01099453714",
            "-0.01099453714",
        ]
        if values != checkdata:
            msg = "Result from method rolling_cvar_down() not as intended."
            raise ValueError(msg)

    def test_rolling_var_down(self: TestOpenFrame) -> None:
        """Test rolling_var_down method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_var_down(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[-5:, 0]]
        checkdata = [
            "-0.01239751713",
            "-0.01239751713",
            "-0.01239751713",
            "-0.01239751713",
            "-0.01239751713",
        ]

        if values != checkdata:
            msg = "Result from method rolling_var_down() not as intended."
            raise ValueError(msg)

    def test_label_uniqueness(self: TestOpenFrame) -> None:
        """Test label uniqueness."""
        aseries = self.randomseries.from_deepcopy()
        bseries = self.randomseries.from_deepcopy()

        with pytest.raises(
            expected_exception=ValueError,
            match="TimeSeries names/labels must be unique",
        ):
            OpenFrame([aseries, bseries])

        bseries.set_new_label("other_name")
        uframe = OpenFrame([aseries, bseries])

        if uframe.columns_lvl_zero != ["Asset_0", "other_name"]:
            msg = "Fix of non-unique labels unsuccessful."
            raise ValueError(msg)

    def test_capture_ratio(self: TestOpenFrame) -> None:
        """
        Test the capture_ratio_func method.

        Source: 'Capture Ratios: A Popular Method of Measuring Portfolio Performance
        in Practice', Don R. Cox and Delbert C. Goff, Journal of Economics and
        Finance Education (Vol 2 Winter 2013).
        https://www.economics-finance.org/jefe/volume12-2/11ArticleCox.pdf.
        """
        asset = OpenTimeSeries.from_arrays(
            name="asset",
            baseccy="USD",
            valuetype=ValueType.RTRN,
            dates=[
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
            values=[
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
        )
        indxx = OpenTimeSeries.from_arrays(
            name="indxx",
            valuetype=ValueType.RTRN,
            baseccy="USD",
            dates=[
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
            values=[
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
        )
        cframe = OpenFrame([asset, indxx]).to_cumret()

        upp = cframe.capture_ratio_func(ratio="up")
        down = cframe.capture_ratio_func(ratio="down")
        both = cframe.capture_ratio_func(ratio="both")

        if f"{upp.iloc[0]:.12f}" != "1.063842457805":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)
        if f"{down.iloc[0]:.12f}" != "0.922188852957":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)
        if f"{both.iloc[0]:.12f}" != "1.153605852417":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        upfixed = cframe.capture_ratio_func(ratio="up", periods_in_a_year_fixed=12)

        if f"{upfixed.iloc[0]:.12f}" != "1.063217236138":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        if f"{upp.iloc[0]:.2f}" != f"{upfixed.iloc[0]:.2f}":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        uptuple = cframe.capture_ratio_func(
            ratio="up",
            base_column=("indxx", ValueType.PRICE),
        )

        if f"{uptuple.iloc[0]:.12f}" != "1.063842457805":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        if f"{upp.iloc[0]:.12f}" != f"{uptuple.iloc[0]:.12f}":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="base_column should be a tuple",
        ):
            _ = cframe.capture_ratio_func(
                ratio="up",
                base_column=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_georet_exceptions(self: TestOpenFrame) -> None:
        """Test georet property raising exceptions on bad input data."""
        geoframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="geoseries1",
                    dates=["2022-07-01", "2023-07-01"],
                    values=[1.0, 1.1],
                ),
                OpenTimeSeries.from_arrays(
                    name="geoseries2",
                    dates=["2022-07-01", "2023-07-01"],
                    values=[1.0, 1.2],
                ),
            ],
        )
        if [f"{gr:.5f}" for gr in cast(Series, geoframe.geo_ret)] != [
            "0.10007",
            "0.20015",
        ]:
            msg = "Unexpected result from property geo_ret"
            raise ValueError(msg)

        if [f"{gr:.5f}" for gr in cast(Series, geoframe.geo_ret_func())] != [
            "0.10007",
            "0.20015",
        ]:
            msg = "Unexpected result from property geo_ret"
            raise ValueError(msg)

        geoframe.add_timeseries(
            OpenTimeSeries.from_arrays(
                name="geoseries3",
                dates=["2022-07-01", "2023-07-01"],
                values=[0.0, 1.1],
            ),
        )

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret_func()

        geoframe.delete_timeseries(lvl_zero_item="geoseries3")

        if [f"{gr:.5f}" for gr in cast(Series, geoframe.geo_ret)] != [
            "0.10007",
            "0.20015",
        ]:
            msg = "Unexpected result from property geo_ret"
            raise ValueError(msg)

        geoframe.add_timeseries(
            OpenTimeSeries.from_arrays(
                name="geoseries4",
                dates=["2022-07-01", "2023-07-01"],
                values=[1.0, -1.1],
            ),
        )
        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret_func()

    def test_value_nan_handle(self: TestOpenFrame) -> None:
        """Test value_nan_handle method."""
        nanframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="nanseries1",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
                OpenTimeSeries.from_arrays(
                    name="nanseries2",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    values=[2.1, 2.0, 1.8, 2.1, 2.0],
                ),
            ],
        )
        # noinspection PyTypeChecker
        nanframe.tsdf.iloc[2, 0] = None
        # noinspection PyTypeChecker
        nanframe.tsdf.iloc[3, 1] = None
        dropframe = nanframe.from_deepcopy()
        dropframe.value_nan_handle(method="drop")

        if [1.1, 1.0, 1.0] != dropframe.tsdf.iloc[:, 0].tolist():
            msg = "Method value_nan_handle() not working as intended"
            raise ValueError(msg)
        if [2.1, 2.0, 2.0] != dropframe.tsdf.iloc[:, 1].tolist():
            msg = "Method value_nan_handle() not working as intended"
            raise ValueError(msg)

        fillframe = nanframe.from_deepcopy()
        fillframe.value_nan_handle(method="fill")

        if [1.1, 1.0, 1.0, 1.1, 1.0] != fillframe.tsdf.iloc[:, 0].tolist():
            msg = "Method value_nan_handle() not working as intended"
            raise ValueError(msg)
        if [2.1, 2.0, 1.8, 1.8, 2.0] != fillframe.tsdf.iloc[:, 1].tolist():
            msg = "Method value_nan_handle() not working as intended"
            raise ValueError(msg)

    def test_return_nan_handle(self: TestOpenFrame) -> None:
        """Test return_nan_handle method."""
        nanframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="nanseries1",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    valuetype=ValueType.RTRN,
                    values=[0.1, 0.05, 0.03, 0.01, 0.04],
                ),
                OpenTimeSeries.from_arrays(
                    name="nanseries2",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    valuetype=ValueType.RTRN,
                    values=[0.01, 0.04, 0.02, 0.11, 0.06],
                ),
            ],
        )
        # noinspection PyTypeChecker
        nanframe.tsdf.iloc[2, 0] = None
        # noinspection PyTypeChecker
        nanframe.tsdf.iloc[3, 1] = None
        dropframe = nanframe.from_deepcopy()
        dropframe.return_nan_handle(method="drop")

        if [0.1, 0.05, 0.04] != dropframe.tsdf.iloc[:, 0].tolist():
            msg = "Method return_nan_handle() not working as intended"
            raise ValueError(msg)
        if [0.01, 0.04, 0.06] != dropframe.tsdf.iloc[:, 1].tolist():
            msg = "Method return_nan_handle() not working as intended"
            raise ValueError(msg)

        fillframe = nanframe.from_deepcopy()
        fillframe.return_nan_handle(method="fill")

        if [0.1, 0.05, 0.0, 0.01, 0.04] != fillframe.tsdf.iloc[:, 0].tolist():
            msg = "Method return_nan_handle() not working as intended"
            raise ValueError(msg)
        if [0.01, 0.04, 0.02, 0.0, 0.06] != fillframe.tsdf.iloc[:, 1].tolist():
            msg = "Method return_nan_handle() not working as intended"
            raise ValueError(msg)

    def test_relative(self: TestOpenFrame) -> None:
        """Test relative method."""
        rframe = self.randomframe.from_deepcopy()
        rframe.to_cumret()
        sframe = self.randomframe.from_deepcopy()
        sframe.to_cumret()
        series_before = 5
        series_after = 6

        if rframe.item_count != series_before:
            msg = "Metod relative() base case not as intended"
            raise ValueError(msg)

        rframe.relative()

        if rframe.item_count != series_after:
            msg = "Metod relative() not working as intended"
            raise ValueError(msg)

        if rframe.tsdf.shape[1] != series_after:
            msg = "Metod relative() not working as intended"
            raise ValueError(msg)

        if rframe.constituents[-1].label != "Asset_0_over_Asset_1":
            msg = "Metod relative() not working as intended"
            ValueError(msg)

        if rframe.columns_lvl_zero[-1] != "Asset_0_over_Asset_1":
            msg = "Metod relative() not working as intended"
            ValueError(msg)

        rframe.tsdf.iloc[:, -1] = rframe.tsdf.iloc[:, -1].add(1.0)

        sframe.relative(base_zero=False)

        rflist = [f"{rret:.11f}" for rret in rframe.tsdf.iloc[:, -1]]
        sflist = [f"{rret:.11f}" for rret in sframe.tsdf.iloc[:, -1]]

        if rflist != sflist:
            msg = "Metod relative() not working as intended"
            ValueError(msg)

    def test_to_cumret(self: TestOpenFrame) -> None:
        """Test to_cumret method."""
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

        if [ValueType.RTRN, ValueType.PRICE] != mframe.columns_lvl_one:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        if [ValueType.PRICE, ValueType.PRICE] != cframe.columns_lvl_one:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        cframe_lvl_one = list(cframe.columns_lvl_one)

        if [ValueType.RTRN, ValueType.RTRN] != rframe.columns_lvl_one:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        mframe.to_cumret()
        cframe.to_cumret()
        rframe.to_cumret()

        if [ValueType.PRICE, ValueType.PRICE] != mframe.columns_lvl_one:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        if cframe_lvl_one != cframe.columns_lvl_one:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        if [ValueType.PRICE, ValueType.PRICE] != rframe.columns_lvl_one:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        fmt = "{:.12f}"

        frame_0 = self.randomframe.from_deepcopy()
        frame_0.to_cumret()
        frame_0.value_to_ret()
        frame_0.tsdf = frame_0.tsdf.map(fmt.format)
        dict_toframe_0 = frame_0.tsdf.to_dict()

        frame_1 = self.randomframe.from_deepcopy()
        frame_1.tsdf = frame_1.tsdf.map(fmt.format)
        dict_toframe_1 = frame_1.tsdf.to_dict()

        if dict_toframe_0 != dict_toframe_1:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

    def test_miscellaneous(self: TestOpenFrame) -> None:
        """Test miscellaneous methods."""
        zero_str: str = "0"
        zero_float: float = 0.0
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
            for nofix, fix in zip(no_fixed, fixed):
                if f"{100*abs(nofix-fix):.0f}" != zero_str:
                    msg = (
                        "Difference with or without "
                        "fixed periods in year is too great"
                    )
                    raise ValueError(msg)
        for methd in methods:
            dated = getattr(mframe, methd)(
                from_date=mframe.first_idx,
                to_date=mframe.last_idx,
            )
            undated = getattr(mframe, methd)()
            for ddat, undat in zip(dated, undated):
                if f"{ddat:.10f}" != f"{undat:.10f}":
                    msg = (
                        f"Method {methd} with and without date "
                        "arguments returned inconsistent results"
                    )
                    raise ValueError(msg)

        ret = [f"{rr:.9f}" for rr in cast(Series, mframe.value_ret_func())]
        if ret != [
            "-0.223596218",
            "-0.244776630",
            "1.229358191",
            "0.370036823",
            "-0.800357531",
        ]:
            msg = "Results from value_ret_func() not as expected"
            raise ValueError(msg)

        impvol = [
            f"{iv:.11f}"
            for iv in cast(Series, mframe.vol_from_var_func(drift_adjust=False))
        ]
        impvoldrifted = [
            f"{iv:.11f}"
            for iv in cast(Series, mframe.vol_from_var_func(drift_adjust=True))
        ]
        if impvol != [
            "0.09373794422",
            "0.09452356706",
            "0.09758852171",
            "0.09902646305",
            "0.10399391218",
        ]:
            msg = "Results from vol_from_var_func() not as expected"
            raise ValueError(msg)

        if impvoldrifted != [
            "0.09308678526",
            "0.09383528130",
            "0.10086118783",
            "0.10057354052",
            "0.09864366954",
        ]:
            msg = "Results from vol_from_var_func() not as expected"
            raise ValueError(msg)

        # noinspection PyTypeChecker
        mframe.tsdf.iloc[0, 2] = zero_float

        with pytest.raises(
            expected_exception=ValueError,
            match="Simple return cannot be calculated due to an",
        ):
            _ = mframe.value_ret

        with pytest.raises(
            expected_exception=ValueError,
            match="Simple return cannot be calculated due to an",
        ):
            _ = mframe.value_ret_func()

    def test_value_ret_calendar_period(self: TestOpenFrame) -> None:
        """Test value_ret_calendar_period method."""
        vrcseries = self.randomseries.from_deepcopy()
        vrcseries.to_cumret()
        vrcframe = self.randomframe.from_deepcopy()
        vrcframe.to_cumret()

        vrfs_y = vrcseries.value_ret_func(
            from_date=dtdate(2017, 12, 29),
            to_date=dtdate(2018, 12, 28),
        )
        vrff_y = vrcframe.value_ret_func(
            from_date=dtdate(2017, 12, 29),
            to_date=dtdate(2018, 12, 28),
        )
        vrffl_y = [f"{rr:.11f}" for rr in cast(Series, vrff_y)]

        vrvrcs_y = vrcseries.value_ret_calendar_period(year=2018)
        vrvrcf_y = vrcframe.value_ret_calendar_period(year=2018)
        vrvrcfl_y = [f"{rr:.11f}" for rr in cast(Series, vrvrcf_y)]

        if f"{vrfs_y:.11f}" != f"{vrvrcs_y:.11f}":
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise ValueError(msg)

        if vrffl_y != vrvrcfl_y:
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise ValueError(msg)

        vrfs_ym = vrcseries.value_ret_func(
            from_date=dtdate(2018, 4, 30),
            to_date=dtdate(2018, 5, 31),
        )
        vrff_ym = vrcframe.value_ret_func(
            from_date=dtdate(2018, 4, 30),
            to_date=dtdate(2018, 5, 31),
        )
        vrffl_ym = [f"{rr:.11f}" for rr in cast(Series, vrff_ym)]

        vrvrcs_ym = vrcseries.value_ret_calendar_period(year=2018, month=5)
        vrvrcf_ym = vrcframe.value_ret_calendar_period(year=2018, month=5)
        vrvrcfl_ym = [f"{rr:.11f}" for rr in cast(Series, vrvrcf_ym)]

        if f"{vrfs_ym:.11f}" != f"{vrvrcs_ym:.11f}":
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise ValueError(msg)

        if vrffl_ym != vrvrcfl_ym:
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise ValueError(msg)

    def test_to_drawdown_series(self: TestOpenFrame) -> None:
        """Test to_drawdown_series method."""
        mframe = self.randomframe.from_deepcopy()
        mframe.to_cumret()
        ddown = [f"{dmax:.11f}" for dmax in cast(Series, mframe.max_drawdown)]
        mframe.to_drawdown_series()
        ddownserie = [f"{dmax:.11f}" for dmax in mframe.tsdf.min()]

        if ddown != ddownserie:
            msg = "Method to_drawdown_series() not working as intended"
            raise ValueError(msg)

    def test_ord_least_squares_fit(self: TestOpenFrame) -> None:
        """Test ord_least_squares_fit method."""
        oframe = self.randomframe.from_deepcopy()
        oframe.to_cumret()
        oframe.value_to_log()

        fsframe = self.randomframe.from_deepcopy()
        fsframe.to_cumret()
        fsframe.ord_least_squares_fit(y_column=0, x_column=1, fitted_series=True)

        if fsframe.columns_lvl_zero[-1] != oframe.columns_lvl_zero[0]:
            msg = "Method ord_least_squares_fit() not working as intended"
            raise ValueError(msg)

        if fsframe.columns_lvl_one[-1] != oframe.columns_lvl_zero[1]:
            msg = "Method ord_least_squares_fit() not working as intended"
            raise ValueError(msg)

        results = []
        for i in range(oframe.item_count):
            for j in range(oframe.item_count):
                tmp = oframe.ord_least_squares_fit(
                    y_column=i,
                    x_column=j,
                    fitted_series=False,
                )
                results.append(f"{float(tmp.params.iloc[0]):.11f}")

        results_tuple = []
        k_tuple: Hashable
        l_tuple: Hashable
        for k_tuple in oframe.tsdf:
            for l_tuple in oframe.tsdf:
                tmp = oframe.ord_least_squares_fit(
                    y_column=cast(tuple[str, ValueType], k_tuple),
                    x_column=cast(tuple[str, ValueType], l_tuple),
                    fitted_series=False,
                )
                results_tuple.append(f"{float(tmp.params.iloc[0]):.11f}")

        if results != results_tuple:
            msg = "Method ord_least_squares_fit() not working as intended"
            raise ValueError(msg)

        if results != [
            "1.00000000000",
            "1.04260382009",
            "-0.45683811180",
            "-0.78552285510",
            "0.16527540312",
            "0.84475254880",
            "1.00000000000",
            "-0.43186041740",
            "-0.67734009898",
            "0.15614783403",
            "-1.72148183489",
            "-2.00850685241",
            "1.00000000000",
            "1.55985343891",
            "-0.37248169175",
            "-0.71431469722",
            "-0.76019909515",
            "0.37642151527",
            "1.00000000000",
            "-0.14436264610",
            "4.26473892687",
            "4.97290302806",
            "-2.55063925744",
            "-4.09645603877",
            "1.00000000000",
        ]:
            msg = "Method ord_least_squares_fit() not working as intended"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="x_column should be a tuple",
        ):
            _ = oframe.ord_least_squares_fit(
                y_column=0,
                x_column=cast(Union[tuple[str, ValueType], int], "string"),
                fitted_series=False,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="y_column should be a tuple",
        ):
            _ = oframe.ord_least_squares_fit(
                y_column=cast(Union[tuple[str, ValueType], int], "string"),
                x_column=1,
                fitted_series=False,
            )

    def test_beta(self: TestOpenFrame) -> None:
        """Test beta method."""
        bframe = self.randomframe.from_deepcopy()
        bframe.to_cumret()
        bframe.resample("7D")
        results = [
            f"{bframe.beta(asset=comb[0], market=comb[1]):.11f}"
            for comb in iter_product(
                range(bframe.item_count),
                range(bframe.item_count),
            )
        ]
        results_tuple = []
        for comb in iter_product(bframe.tsdf, bframe.tsdf):
            beta = bframe.beta(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            results_tuple.append(f"{beta:.11f}")

        if results != results_tuple:
            msg = "Unexpected results from method beta()"
            raise ValueError(msg)

        if results != [
            "1.00000000000",
            "1.00515556512",
            "-0.60052744878",
            "-0.53044422166",
            "0.24457642880",
            "0.76348713022",
            "1.00000000000",
            "-0.59023265886",
            "-0.39702178376",
            "0.24413076386",
            "-1.04559694975",
            "-1.35296400287",
            "1.00000000000",
            "0.65962991597",
            "-0.41514609304",
            "-0.51858411086",
            "-0.51100527145",
            "0.37038072911",
            "1.00000000000",
            "-0.17838050385",
            "1.96800044877",
            "2.58621522791",
            "-1.91858084466",
            "-1.46817727237",
            "1.00000000000",
        ]:
            msg = "Unexpected results from method beta()"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = bframe.beta(
                asset=cast(Union[tuple[str, ValueType], int], "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = bframe.beta(
                asset=0,
                market=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_beta_returns_input(self: TestOpenFrame) -> None:
        """Test beta method with returns input."""
        bframe = self.randomframe.from_deepcopy()
        bframe.resample("7D")
        results = [
            f"{bframe.beta(asset=comb[0], market=comb[1]):.11f}"
            for comb in iter_product(
                range(bframe.item_count),
                range(bframe.item_count),
            )
        ]

        results_tuple = []
        for comb in iter_product(bframe.tsdf, bframe.tsdf):
            beta = bframe.beta(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            results_tuple.append(f"{beta:.11f}")

        if results != results_tuple:
            msg = "Unexpected results from method beta()"
            raise ValueError(msg)

        if results != [
            "1.00000000000",
            "0.07226556298",
            "-0.00701645043",
            "0.00852383980",
            "-0.01714210697",
            "0.07155867325",
            "1.00000000000",
            "-0.04446704939",
            "-0.03504494393",
            "-0.02778937911",
            "-0.00796692758",
            "-0.05098950916",
            "1.00000000000",
            "0.01640487004",
            "0.01409673443",
            "0.02490042468",
            "-0.10338700517",
            "0.04220567607",
            "1.00000000000",
            "0.07558966796",
            "-0.01996827214",
            "-0.03269070151",
            "0.01446176820",
            "0.03014166568",
            "1.00000000000",
        ]:
            msg = "Unexpected results from method beta()"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = bframe.beta(
                asset=cast(Union[tuple[str, ValueType], int], "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = bframe.beta(
                asset=0,
                market=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_jensen_alpha(self: TestOpenFrame) -> None:
        """Test jensen_alpha method."""
        jframe = self.randomframe.from_deepcopy()
        jframe.to_cumret()
        jframe.resample("7D")
        results = [
            f"{jframe.jensen_alpha(asset=comb[0], market=comb[1]):.9f}"
            for comb in iter_product(
                range(jframe.item_count),
                range(jframe.item_count),
            )
        ]

        results_tuple = []
        for comb in iter_product(jframe.tsdf, jframe.tsdf):
            alpha = jframe.jensen_alpha(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            results_tuple.append(f"{alpha:.9f}")

        if results != results_tuple:
            msg = "Unexpected results from method jensen_alpha()"
            raise ValueError(msg)

        if results != [
            "0.000000000",
            "0.002463470",
            "0.024326786",
            "-0.007061093",
            "0.012082089",
            "-0.008096442",
            "0.000000000",
            "0.021165791",
            "-0.013747860",
            "0.009689925",
            "0.055626999",
            "0.044980879",
            "0.000000000",
            "0.059578031",
            "0.019214998",
            "0.020033149",
            "0.019029629",
            "0.002633982",
            "0.000000000",
            "0.006078728",
            "-0.101143732",
            "-0.080044979",
            "0.006509846",
            "-0.101172406",
            "0.000000000",
        ]:
            msg = "Unexpected results from method jensen_alpha()"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=cast(Union[tuple[str, ValueType], int], "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=0,
                market=cast(Union[tuple[str, ValueType], int], "string"),
            )

        ninemth = date_offset_foll(jframe.last_idx, months_offset=-9, adjust=True)
        shortframe = jframe.trunc_frame(start_cut=ninemth)
        shortframe.to_cumret()
        sresults = [
            f"{shortframe.jensen_alpha(asset=comb[0], market=comb[1]):.9f}"
            for comb in iter_product(
                range(shortframe.item_count),
                range(shortframe.item_count),
            )
        ]

        sresults_tuple: list[str] = []
        for comb in iter_product(shortframe.tsdf, shortframe.tsdf):
            alpha = shortframe.jensen_alpha(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            sresults_tuple.append(f"{alpha:.9f}")

        if sresults != sresults_tuple:
            msg = "Unexpected results from method jensen_alpha()"
            raise ValueError(msg)

        if sresults != [
            "0.000000000",
            "-0.009054439",
            "-0.002815869",
            "-0.050392432",
            "-0.040949344",
            "0.040957863",
            "0.000000000",
            "0.034761441",
            "0.005922775",
            "-0.008764593",
            "0.022091567",
            "0.008028122",
            "0.000000000",
            "-0.039649995",
            "-0.058499472",
            "0.111965510",
            "0.109741848",
            "0.109228693",
            "0.000000000",
            "0.049484243",
            "0.222302921",
            "0.204520974",
            "0.203935785",
            "0.014946357",
            "0.000000000",
        ]:
            msg = "Unexpected results from method jensen_alpha()"
            raise ValueError(msg)

    def test_jensen_alpha_returns_input(self: TestOpenFrame) -> None:
        """Test jensen_alpha method with returns input."""
        jframe = self.randomframe.from_deepcopy()
        jframe.resample("7D")
        results = [
            f"{jframe.jensen_alpha(asset=comb[0], market=comb[1]):.9f}"
            for comb in iter_product(
                range(jframe.item_count),
                range(jframe.item_count),
            )
        ]

        results_tuple = []
        for comb in iter_product(jframe.tsdf, jframe.tsdf):
            alpha = jframe.jensen_alpha(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            results_tuple.append(f"{alpha:.9f}")

        if results != results_tuple:
            msg = "Unexpected results from method jensen_alpha()"
            raise ValueError(msg)

        if results != [
            "0.000000000",
            "-0.000260875",
            "-0.000228696",
            "-0.000238197",
            "-0.000231801",
            "0.000412105",
            "0.000000000",
            "0.000418295",
            "0.000419747",
            "0.000396284",
            "0.000511183",
            "0.000533199",
            "0.000000000",
            "0.000501675",
            "0.000512627",
            "0.000698173",
            "0.000733277",
            "0.000670736",
            "0.000000000",
            "0.000690209",
            "0.000024205",
            "0.000041772",
            "0.000021424",
            "0.000007973",
            "0.000000000",
        ]:
            msg = "Unexpected results from method jensen_alpha()"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=cast(Union[tuple[str, ValueType], int], "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=0,
                market=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_ewma_risk(self: TestOpenFrame) -> None:
        """Test ewma_risk method."""
        eframe = self.randomframe.from_deepcopy()
        eframe.to_cumret()
        edf = eframe.ewma_risk()

        list_one = [f"{e:.11f}" for e in edf.head(10).iloc[:, 0]]
        list_two = [f"{e:.11f}" for e in edf.head(10).iloc[:, 1]]
        corr_one = [f"{e:.11f}" for e in edf.head(10).iloc[:, 2]]

        if list_one != [
            "0.06122660096",
            "0.06080791286",
            "0.05965906881",
            "0.06712801227",
            "0.06641721467",
            "0.06475262488",
            "0.06553052810",
            "0.06355985410",
            "0.06226899798",
            "0.06118340805",
        ]:
            msg = "Unexpected results from method ewma_risk()"
            raise ValueError(msg)

        if list_two != [
            "0.11871785884",
            "0.11512347678",
            "0.11459743663",
            "0.11128657073",
            "0.11021420293",
            "0.12443177493",
            "0.12158958488",
            "0.11989625887",
            "0.12306422721",
            "0.12098150772",
        ]:
            msg = "Unexpected results from method ewma_risk()"
            raise ValueError(msg)

        if corr_one != [
            "0.00001529928",
            "0.00214586533",
            "-0.01528463682",
            "-0.02758421693",
            "-0.00611901784",
            "-0.03215602356",
            "-0.01269689530",
            "-0.01506926809",
            "0.00948568831",
            "-0.00419257181",
        ]:
            msg = "Unexpected results from method ewma_risk()"
            raise ValueError(msg)

    def test_ewma_risk_set_columns(self: TestOpenFrame) -> None:
        """Test ewma_risk method on specified columns."""
        eframe = self.randomframe.from_deepcopy()
        eframe.to_cumret()
        fdf = eframe.ewma_risk(
            first_column=3,
            second_column=4,
            periods_in_a_year_fixed=251,
        )
        list_three = [f"{f:.11f}" for f in fdf.head(10).iloc[:, 0]]
        list_four = [f"{f:.11f}" for f in fdf.head(10).iloc[:, 1]]
        corr_two = [f"{f:.11f}" for f in fdf.head(10).iloc[:, 2]]

        if list_three != [
            "0.06556281884",
            "0.06369857955",
            "0.07278808537",
            "0.07137009681",
            "0.06960008032",
            "0.06826221730",
            "0.06718852899",
            "0.06516551669",
            "0.06556998317",
            "0.06662259931",
        ]:
            msg = "Unexpected results from method ewma_risk()"
            raise ValueError(msg)

        if list_four != [
            "0.09773545998",
            "0.09732667562",
            "0.09648816042",
            "0.09394638674",
            "0.09201496291",
            "0.09021610952",
            "0.09007334957",
            "0.09291101890",
            "0.09063142446",
            "0.09228768353",
        ]:
            msg = "Unexpected results from method ewma_risk()"
            raise ValueError(msg)

        if corr_two != [
            "0.00059325910",
            "-0.00679578441",
            "0.04961115533",
            "0.04198883193",
            "0.04895600628",
            "0.03662399341",
            "0.01445052963",
            "0.00896260153",
            "0.02330792994",
            "0.06689198575",
        ]:
            msg = "Unexpected results from method ewma_risk()"
            raise ValueError(msg)
