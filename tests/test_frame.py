"""Test suite for the openseries/frame.py module.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal, localcontext
from inspect import getmembers, isfunction
from itertools import product as iter_product
from json import load, loads
from pathlib import Path
from pprint import pformat
from re import escape
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

from numpy import array
from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Hashable, Mapping
    from typing import Literal

    from pandas import Timestamp

import pytest
from pandas import DataFrame, Series, date_range, read_excel
from pandas.testing import assert_frame_equal
from requests.exceptions import ConnectionError as RequestsConnectionError

from openseries._risk import _cvar_down_calc, _var_down_calc
from openseries.frame import OpenFrame
from openseries.load_plotly import load_plotly_dict
from openseries.owntypes import (
    DateAlignmentError,
    InitialValueZeroError,
    LabelsNotUniqueError,
    LiteralPortfolioWeightings,
    MixedValuetypesError,
    NoWeightsError,
    NumberOfItemsAndLabelsNotSameError,
    RatioInputError,
    ResampleDataLossError,
    ValueType,
)
from openseries.series import OpenTimeSeries

from .test_common_sim import CommonTestCase


class OpenFrameTestError(Exception):
    """Custom exception used for signaling test failures."""


class TestOpenFrame(CommonTestCase):
    """class to run tests on the module frame.py."""

    def make_mixed_type_openframe(self: TestOpenFrame) -> OpenFrame:
        """Makes an OpenFrame with both PRICE and RTRN type series."""
        series = self.randomseries.from_deepcopy()
        returns = self.randomseries.from_deepcopy()
        returns.set_new_label(lvl_zero="returns")
        returns.value_to_ret()
        return OpenFrame(constituents=[series, returns])

    def test_single_serie_openframe(self: TestOpenFrame) -> None:
        """Test if else on single series in _set_tsdf."""
        series = self.randomseries.from_deepcopy()
        frame = OpenFrame(constituents=[series])
        if series.tsdf.shape != frame.tsdf.shape:
            msg = "_set_tsdf not working as intended."
            raise OpenFrameTestError(msg)

    def test_to_json(self: TestOpenFrame) -> None:
        """Test to_json method."""
        filename = "framesaved.json"
        if Path.home().joinpath("Documents").exists():
            directory = Path.home().joinpath("Documents")
            framefile = directory.joinpath(filename)
        else:
            directory = Path(__file__).parent
            framefile = directory.joinpath(filename)

        if Path(framefile).exists():
            msg = "test_to_json test case setup failed."
            raise FileExistsError(msg)

        kwargs: list[Mapping[str, Any]] = [
            {"what_output": "values", "filename": str(framefile)},
            {"what_output": "values", "filename": filename},
            {
                "what_output": "values",
                "filename": filename,
                "directory": directory,
            },
        ]

        for kwarg in kwargs:
            data = self.randomframe.to_json(**kwarg)
            if [item.get("name") for item in data] != [
                "Asset_0",
                "Asset_1",
                "Asset_2",
                "Asset_3",
                "Asset_4",
            ]:
                msg = "Unexpected data from json"
                raise OpenFrameTestError(msg)

            if not Path(framefile).exists():
                msg = "json file not created"
                raise FileNotFoundError(msg)

            framefile.unlink()

            if Path(framefile).exists():
                msg = "json file not deleted as intended"
                raise FileExistsError(msg)

        localfile = Path(__file__).parent.joinpath(filename)

        with patch("pathlib.Path.exists") as mock_doesnotexist:
            mock_doesnotexist.return_value = False
            data = self.randomframe.to_json(what_output="values", filename=filename)

        if [item.get("name") for item in data] != [
            "Asset_0",
            "Asset_1",
            "Asset_2",
            "Asset_3",
            "Asset_4",
        ]:
            msg = "Unexpected data from json"
            raise OpenFrameTestError(msg)

        localfile.unlink()

        with (
            patch("pathlib.Path.exists") as mock_doesnotexist,
            patch(
                "pathlib.Path.open",
            ) as mock_donotopen,
        ):
            mock_doesnotexist.return_value = True
            mock_donotopen.side_effect = MagicMock()
            data2 = self.randomframe.to_json(what_output="values", filename=filename)

        if [item.get("name") for item in data2] != [
            "Asset_0",
            "Asset_1",
            "Asset_2",
            "Asset_3",
            "Asset_4",
        ]:
            msg = "Unexpected data from json"
            raise OpenFrameTestError(msg)

    def test_to_json_and_back(self: TestOpenFrame) -> None:
        """Test to_json method and creating an OpenFrame from file data."""
        filename = "frame.json"
        dirpath = Path(__file__).parent
        framefile = dirpath.joinpath(filename)

        if Path(framefile).exists():
            msg = "test_to_json_and_back test case setup failed."
            raise FileExistsError(msg)

        intended = ["1.640116", "1.354976", "2.287658", "2.045919", "1.169641"]

        data = self.randomframe.to_json(
            what_output="values",
            filename=filename,
            directory=dirpath,
        )

        frame_one = OpenFrame(
            constituents=[
                OpenTimeSeries.from_arrays(
                    name=cast("str", item["name"]),
                    dates=cast("list[str]", item["dates"]),
                    values=cast("list[float]", item["values"]),
                    valuetype=cast("ValueType", item["valuetype"]),
                    baseccy=cast("str", item["currency"]),
                    local_ccy=cast("bool", item["local_ccy"]),
                )
                for item in data
            ],
        ).to_cumret()

        check_one = [f"{endvalue:.6f}" for endvalue in frame_one.tsdf.iloc[-1]]

        if check_one != intended:
            msg = f"test_to_json_and_back did not output as intended: {check_one}"
            raise OpenFrameTestError(msg)

        with framefile.open(mode="r", encoding="utf-8") as jsonfile:
            output = load(jsonfile)

        frame_two = OpenFrame(
            constituents=[
                OpenTimeSeries.from_arrays(
                    name=cast("str", item["name"]),
                    dates=cast("list[str]", item["dates"]),
                    values=cast("list[float]", item["values"]),
                    valuetype=cast("ValueType", item["valuetype"]),
                    baseccy=cast("str", item["currency"]),
                    local_ccy=cast("bool", item["local_ccy"]),
                )
                for item in output
            ],
        ).to_cumret()

        check_two = [f"{endvalue:.6f}" for endvalue in frame_two.tsdf.iloc[-1]]

        if check_two != intended:
            msg = f"test_to_json_and_back did not output as intended: {check_two}"
            raise OpenFrameTestError(msg)

        framefile.unlink()

        if Path(framefile).exists():
            msg = "json file not deleted as intended"
            raise FileExistsError(msg)

    def test_to_json_and_back_tsdf(self: TestOpenFrame) -> None:
        """Test to_json method and creating an OpenFrame from file data."""
        filename = "frame_tsdf.json"
        dirpath = Path(__file__).parent
        framefile = dirpath.joinpath(filename)

        if Path(framefile).exists():
            msg = "test_to_json_and_back_tsdf test case setup failed."
            raise FileExistsError(msg)

        intended = ["1.640116", "1.354976", "2.287658", "2.045919", "1.169641"]

        data = self.randomframe.to_json(
            what_output="tsdf",
            filename=filename,
            directory=dirpath,
        )

        frame_one = OpenFrame(
            constituents=[
                OpenTimeSeries.from_arrays(
                    name=cast("str", item["name"]),
                    dates=cast("list[str]", item["dates"]),
                    values=cast("list[float]", item["values"]),
                    valuetype=cast("ValueType", item["valuetype"]),
                    baseccy=cast("str", item["currency"]),
                    local_ccy=cast("bool", item["local_ccy"]),
                )
                for item in data
            ],
        ).to_cumret()

        check_one = [f"{endvalue:.6f}" for endvalue in frame_one.tsdf.iloc[-1]]

        if check_one != intended:
            msg = f"test_to_json_and_back_tsdf did not output as intended: {check_one}"
            raise OpenFrameTestError(msg)

        with framefile.open(mode="r", encoding="utf-8") as jsonfile:
            output = load(jsonfile)

        frame_two = OpenFrame(
            constituents=[
                OpenTimeSeries.from_arrays(
                    name=cast("str", item["name"]),
                    dates=cast("list[str]", item["dates"]),
                    values=cast("list[float]", item["values"]),
                    valuetype=cast("ValueType", item["valuetype"]),
                    baseccy=cast("str", item["currency"]),
                    local_ccy=cast("bool", item["local_ccy"]),
                )
                for item in output
            ],
        ).to_cumret()

        check_two = [f"{endvalue:.6f}" for endvalue in frame_two.tsdf.iloc[-1]]

        if check_two != intended:
            msg = f"test_to_json_and_back_tsdf did not output as intended: {check_two}"
            raise OpenFrameTestError(msg)

        framefile.unlink()

        if Path(framefile).exists():
            msg = "json file not deleted as intended"
            raise FileExistsError(msg)

    def test_to_xlsx(self: TestOpenFrame) -> None:
        """Test to_xlsx method."""
        filename = "trial.xlsx"
        if Path.home().joinpath("Documents").exists():
            basefile = Path.home().joinpath("Documents").joinpath(filename)
        else:
            basefile = Path(__file__).parent.joinpath(filename)

        if Path(basefile).exists():
            msg = "test_save_to_xlsx test case setup failed."
            raise FileExistsError(msg)

        seriesfile = Path(
            self.randomframe.to_xlsx(filename=filename, sheet_title="boo"),
        ).resolve()

        if basefile != seriesfile:
            msg = "test_save_to_xlsx test case setup failed."
            raise OpenFrameTestError(msg)

        if not Path(seriesfile).exists():
            msg = "xlsx file not created"
            raise FileNotFoundError(msg)

        seriesfile.unlink()

        directory = Path(__file__).parent
        seriesfile = Path(
            self.randomframe.to_xlsx(filename="trial.xlsx", directory=directory),
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
            match=r"Filename must end with .xlsx",
        ):
            _ = self.randomframe.to_xlsx(filename="trial.pdf")

        with basefile.open(mode="w", encoding="utf-8") as fakefile:
            fakefile.write("Hello world")

        with pytest.raises(
            expected_exception=FileExistsError,
            match=f"{filename} already exists.",
        ):
            _ = self.randomframe.to_xlsx(filename=filename, overwrite=False)

        basefile.unlink()

        localfile = Path(__file__).parent.joinpath(filename)
        with patch("pathlib.Path.exists") as mock_doesnotexist:
            mock_doesnotexist.return_value = False
            seriesfile = Path(self.randomframe.to_xlsx(filename=filename)).resolve()

        if localfile != seriesfile:
            msg = "test_save_to_xlsx test case setup failed."
            raise OpenFrameTestError(msg)

        dframe = read_excel(  # type: ignore[call-overload]
            io=seriesfile,
            header=0,
            index_col=0,
            usecols=cast("int", "A:F"),
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
            raise OpenFrameTestError(msg)

        seriesfile.unlink()

        with (
            patch("pathlib.Path.exists") as mock_doesnotexist,
            patch(
                "openpyxl.workbook.workbook.Workbook.save",
            ) as mock_donotopen,
        ):
            mock_doesnotexist.return_value = True
            mock_donotopen.side_effect = MagicMock()
            seriesfile2 = Path(self.randomframe.to_xlsx(filename=filename)).resolve()

        if seriesfile2.parts[-2:] != ("Documents", "trial.xlsx"):
            msg = "save_to_xlsx not working as intended."
            raise OpenFrameTestError(msg)

    def test_calc_range(self: TestOpenFrame) -> None:
        """Test calc_range method."""
        crframe = self.randomframe.from_deepcopy()
        start, end = crframe.first_idx, crframe.last_idx

        rst, ren = crframe.calc_range()

        if [start, end] != [rst, ren]:
            msg = "Unintended output from calc_range()"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=DateAlignmentError,
            match=(
                r"Argument months_offset implies startdate "
                r"before first date in series."
            ),
        ):
            _, _ = crframe.calc_range(months_offset=125)

        with pytest.raises(
            expected_exception=DateAlignmentError,
            match="Given from_dt date < series start",
        ):
            _, _ = crframe.calc_range(from_dt=dt.date(2009, 5, 31))

        with pytest.raises(
            expected_exception=DateAlignmentError,
            match="Given to_dt date > series end",
        ):
            _, _ = crframe.calc_range(to_dt=dt.date(2019, 7, 31))

        offsetst, offseten = crframe.calc_range(
            months_offset=12,
        )
        if [offsetst, offseten] != [dt.date(2018, 6, 28), end]:
            msg = f"Unintended output from calc_range():{[offsetst, offseten]}"
            raise OpenFrameTestError(msg)

        fromtost, fromtoen = crframe.calc_range(
            from_dt=dt.date(2009, 7, 3),
            to_dt=dt.date(2019, 6, 25),
        )
        if [fromtost, fromtoen] != [dt.date(2009, 7, 3), dt.date(2019, 6, 25)]:
            msg = f"Unintended output from calc_range():{[fromtost, fromtoen]}"
            raise OpenFrameTestError(msg)

        bothst, bothen = crframe.calc_range(
            months_offset=12,
            from_dt=dt.date(2009, 7, 3),
            to_dt=dt.date(2019, 6, 25),
        )
        if [bothst, bothen] != [offsetst, end]:
            msg = f"Unintended output from calc_range():{[bothst, bothen]}"
            raise OpenFrameTestError(msg)

        crframe.resample()

        earlier_moved, _ = crframe.calc_range(from_dt=dt.date(2009, 8, 10))
        if earlier_moved != dt.date(2009, 7, 31):
            msg = "Unintended output from calc_range()"
            raise OpenFrameTestError(msg)

        _, later_moved = crframe.calc_range(to_dt=dt.date(2009, 8, 20))
        if later_moved != dt.date(2009, 8, 31):
            msg = "Unintended output from calc_range()"
            raise OpenFrameTestError(msg)

    def test_resample(self: TestOpenFrame) -> None:
        """Test resample method."""
        rs_frame = self.randomframe.from_deepcopy()
        rs_frame.to_cumret()

        dates = [
            dt.date(2019, 2, 28),
            dt.date(2019, 3, 29),
            dt.date(2019, 4, 30),
            dt.date(2019, 5, 31),
            dt.date(2019, 6, 28),
        ]
        before = list(rs_frame.tsdf.loc[dates].iloc[:, 0])
        before_str = [f"{item:.6f}" for item in before]

        ret = array(before)[1:] / array(before)[:-1] - 1
        before_ret = [f"{item:.6f}" for item in ret]

        rs_frame.resample(freq="BME")

        after = [f"{item:.6f}" for item in rs_frame.tsdf.iloc[-5:, 0]]

        msg = "resample() method generated unexpected result"
        if before_str != after:
            raise OpenFrameTestError(msg)

        rs_frame.value_to_ret()
        after_ret = [f"{item:.6f}" for item in rs_frame.tsdf.iloc[-4:, 0]]

        if before_ret != after_ret:
            raise OpenFrameTestError(msg)

        mixframe = self.make_mixed_type_openframe()
        with pytest.raises(
            expected_exception=MixedValuetypesError,
            match="Mix of series types will give inconsistent results",
        ):
            mixframe.resample()

    def test_resample_to_business_period_ends(self: TestOpenFrame) -> None:
        """Test resample_to_business_period_ends method."""
        rsb_stubs_frame = OpenFrame(
            constituents=[
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=121,
                    end_dt=dt.date(2023, 5, 15),
                ).set_new_label("A"),
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=123,
                    end_dt=dt.date(2023, 5, 16),
                ).set_new_label("B"),
            ],
        )

        rsb_stubs_frame.resample_to_business_period_ends(freq="BME")
        new_stubs_dates = rsb_stubs_frame.tsdf.index.tolist()

        if new_stubs_dates != [
            dt.date(2023, 1, 14),
            dt.date(2023, 1, 15),
            dt.date(2023, 1, 31),
            dt.date(2023, 2, 28),
            dt.date(2023, 3, 31),
            dt.date(2023, 4, 28),
            dt.date(2023, 5, 15),
            dt.date(2023, 5, 16),
        ]:
            msg = (
                "resample_to_business_period_ends() method "
                f"generated unexpected result:\n{new_stubs_dates}"
            )
            raise OpenFrameTestError(msg)

        rsb_frame = OpenFrame(
            constituents=[
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=88,
                    end_dt=dt.date(2023, 4, 28),
                ).set_new_label("A"),
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=8,
                    end_dt=dt.date(2023, 4, 28),
                ).set_new_label("B"),
            ],
        )

        rsb_frame.resample_to_business_period_ends(freq="BME")
        new_dates = rsb_frame.tsdf.index.tolist()

        if new_dates != [
            dt.date(2023, 1, 31),
            dt.date(2023, 2, 28),
            dt.date(2023, 3, 31),
            dt.date(2023, 4, 21),
            dt.date(2023, 4, 28),
        ]:
            msg = (
                "resample_to_business_period_ends() method "
                f"generated unexpected result:\n{new_dates}"
            )
            raise OpenFrameTestError(msg)

        frame = self.randomframe.from_deepcopy()
        with pytest.raises(
            expected_exception=ResampleDataLossError,
            match=r"Do not run resample_to_business_period_ends on return series.",
        ):
            frame.resample_to_business_period_ends()

    def test_resample_to_business_period_ends_renaming(self: TestOpenFrame) -> None:
        """Test resample_to_business_period_ends method and its handling of labels."""
        rename = {
            "Asset_0": "Asset_A",
            "Asset_1": "Asset_B",
            "Asset_2": "Asset_C",
            "Asset_3": "Asset_D",
            "Asset_4": "Asset_E",
        }

        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()
        frame.tsdf = frame.tsdf.rename(columns=rename, level=0)
        frame.resample_to_business_period_ends(freq="BYE")

        if frame.columns_lvl_zero != list(rename.values()):
            msg = (
                "Method .resample_to_business_period_ends() "
                "not considering new columns in .tsdf"
            )
            raise OpenFrameTestError(msg)

    def test_max_drawdown_date(self: TestOpenFrame) -> None:
        """Test max_drawdown_date method."""
        mddframe = self.randomframe.from_deepcopy()
        mddframe.to_cumret()

        mdates = cast("Series", mddframe.max_drawdown_date).tolist()

        checkdates = [
            dt.date(2012, 11, 21),
            dt.date(2019, 6, 11),
            dt.date(2015, 7, 24),
            dt.date(2010, 11, 19),
            dt.date(2011, 6, 28),
        ]

        msg = f"max_drawdown_date property generated unexpected result\n{mdates}"
        if mdates != checkdates:
            raise OpenFrameTestError(msg)

    def test_make_portfolio(self: TestOpenFrame) -> None:
        """Test make_portfolio method."""
        mpframe = self.randomframe.from_deepcopy()
        mrframe = self.randomframe.from_deepcopy()
        mpframe.to_cumret()
        mpframe.weights = [1.0 / mpframe.item_count] * mpframe.item_count
        mrframe.weights = [1.0 / mrframe.item_count] * mrframe.item_count

        name = "portfolio"
        mptail = mpframe.make_portfolio(name=name).tail()
        mptail = mptail.map(lambda nn: f"{nn:.6f}")
        mrtail = mrframe.make_portfolio(name=name).tail()
        mrtail = mrtail.map(lambda nn: f"{nn:.6f}")

        correct = ["1.731448", "1.729862", "1.730238", "1.726204", "1.727963"]
        wrong = ["1.731448", "1.729862", "1.730238", "1.726204", "1.727933"]
        true_tail = DataFrame(
            columns=[[name], [ValueType.PRICE]],
            index=[
                dt.date(2019, 6, 24),
                dt.date(2019, 6, 25),
                dt.date(2019, 6, 26),
                dt.date(2019, 6, 27),
                dt.date(2019, 6, 28),
            ],
            data=correct,
            dtype="object",
        )
        false_tail = DataFrame(
            columns=[[name], [ValueType.PRICE]],
            index=[
                dt.date(2019, 6, 24),
                dt.date(2019, 6, 25),
                dt.date(2019, 6, 26),
                dt.date(2019, 6, 27),
                dt.date(2019, 6, 28),
            ],
            data=wrong,
            dtype="float64",
        )

        assert_frame_equal(left=true_tail, right=mptail, check_exact=True)
        assert_frame_equal(left=true_tail, right=mrtail, check_exact=True)

        with pytest.raises(expected_exception=AssertionError, match="are different"):
            assert_frame_equal(left=false_tail, right=mptail, check_exact=True)

        mpframe.weights = None
        with pytest.raises(
            expected_exception=NoWeightsError,
            match=(
                r"OpenFrame weights property must be provided "
                r"to run the make_portfolio method."
            ),
        ):
            _ = mpframe.make_portfolio(name=name)

    def test_make_portfolio_weight_strat(self: TestOpenFrame) -> None:
        """Test make_portfolio method with weight_strat."""
        mpframe = self.randomframe.from_deepcopy()
        mpframe.to_cumret()
        name = "portfolio"

        _ = mpframe.make_portfolio(name=name, weight_strat="eq_weights")
        weights: list[float] | None = [0.2, 0.2, 0.2, 0.2, 0.2]
        if weights != mpframe.weights:
            msg = "make_portfolio() equal weight strategy not working as intended."
            raise OpenFrameTestError(msg)

        with localcontext() as decimal_context:
            decimal_context.rounding = ROUND_HALF_UP

            _ = mpframe.make_portfolio(name=name, weight_strat="inv_vol")
            inv_vol_weights = [
                round(Decimal(wgt), 6) for wgt in cast("list[float]", mpframe.weights)
            ]
            if inv_vol_weights != [
                Decimal("0.152977"),
                Decimal("0.206984"),
                Decimal("0.212791"),
                Decimal("0.214929"),
                Decimal("0.212319"),
            ]:
                msg = (
                    "make_portfolio() inverse vol strategy not working as intended."
                    f"output is \n{inv_vol_weights}"
                )
                raise OpenFrameTestError(msg)

        series = self.randomseries.from_deepcopy()
        returns = self.randomseries.from_deepcopy()
        returns.set_new_label(lvl_zero="returns")
        returns.value_to_ret()
        mixframe = OpenFrame(constituents=[series, returns])
        with pytest.raises(
            expected_exception=MixedValuetypesError,
            match="Mix of series types will give inconsistent results",
        ):
            _ = mixframe.make_portfolio(name=name, weight_strat="eq_weights")

        with pytest.raises(
            expected_exception=NotImplementedError,
            match="Weight strategy not implemented",
        ):
            _ = mpframe.make_portfolio(
                name=name,
                weight_strat=cast("LiteralPortfolioWeightings", "bogus"),
            )

    def test_make_portfolio_new_weight_strategies(self: TestOpenFrame) -> None:
        """Test make_portfolio method with new weight strategies."""
        mpframe = self.randomframe.from_deepcopy()
        mpframe.to_cumret()
        name = "portfolio"
        tolerance = 1e-10

        # Test max_div strategy
        _ = mpframe.make_portfolio(name=name, weight_strat="max_div")

        # Weights should sum to 1
        weight_sum = sum(mpframe.weights)  # type: ignore[arg-type]
        if abs(weight_sum - 1.0) > tolerance:
            msg = f"make_portfolio() max_div weights do not sum to 1.0: {weight_sum}"
            raise OpenFrameTestError(msg)

        # Test target_risk strategy
        _ = mpframe.make_portfolio(name=name, weight_strat="target_risk")

        # Weights should sum to 1
        weight_sum = sum(mpframe.weights)  # type: ignore[arg-type]
        if abs(weight_sum - 1.0) > tolerance:
            msg = (
                f"make_portfolio() target_risk weights do not sum to 1.0: {weight_sum}"
            )
            raise OpenFrameTestError(msg)

    def test_make_portfolio_max_div_singular_matrix(self: TestOpenFrame) -> None:
        """Test make_portfolio max_div strategy with singular correlation matrix."""
        # Create perfectly correlated assets to make correlation matrix singular
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        base_values = [100.0, 101.0, 99.0, 102.0, 101.0]

        # Create three assets with identical returns (perfect correlation)
        ts1 = OpenTimeSeries.from_arrays(
            name="Asset1", dates=dates, values=base_values
        )
        ts2 = OpenTimeSeries.from_arrays(
            name="Asset2", dates=dates, values=base_values
        )
        ts3 = OpenTimeSeries.from_arrays(
            name="Asset3", dates=dates, values=base_values
        )

        # Create frame with perfectly correlated assets
        singular_frame = OpenFrame(constituents=[ts1, ts2, ts3])
        singular_frame.to_cumret()

        # Test max_div strategy with singular correlation matrix
        _ = singular_frame.make_portfolio(name="Singular Test", weight_strat="max_div")

        # Should fall back to equal weights when matrix is singular
        expected_weight = 1.0 / 3.0
        tolerance = 1e-10

        for weight in singular_frame.weights:  # type: ignore[union-attr]
            if abs(weight - expected_weight) > tolerance:
                msg = (
                    f"max_div with singular matrix should use equal weights: "
                    f"{singular_frame.weights}"
                )
                raise OpenFrameTestError(msg)

        # Verify weights sum to 1
        weight_sum = sum(singular_frame.weights)  # type: ignore[arg-type]
        if abs(weight_sum - 1.0) > tolerance:
            msg = f"max_div singular matrix weights do not sum to 1.0: {weight_sum}"
            raise OpenFrameTestError(msg)

    def test_add_timeseries(self: TestOpenFrame) -> None:
        """Test add_timeseries method."""
        frameas = self.randomframe.from_deepcopy()
        items = int(frameas.item_count)
        frameas.weights = [1 / items] * items
        cols = list(frameas.columns_lvl_zero)
        nbr_cols = len(frameas.columns_lvl_zero)
        seriesas = self.randomseries.from_deepcopy()
        seriesas.set_new_label("Asset_6")
        frameas.add_timeseries(seriesas)

        msg = "add_timeseries() method did not work as intended."
        if items + 1 != frameas.item_count:
            raise OpenFrameTestError(msg)
        if nbr_cols + 1 != len(frameas.columns_lvl_zero):
            raise OpenFrameTestError(msg)
        if [*cols, "Asset_6"] != frameas.columns_lvl_zero:
            raise OpenFrameTestError(msg)

    def test_delete_timeseries(self: TestOpenFrame) -> None:
        """Test delete_timeseries method."""
        frame = self.randomframe.from_deepcopy()
        frame.weights = [0.4, 0.1, 0.2, 0.1, 0.2]

        lbl = "Asset_1"
        frame.delete_timeseries(lbl)
        labels = [ff.label for ff in frame.constituents]

        msg = "delete_timeseries() method did not work as intended."
        if labels != ["Asset_0", "Asset_2", "Asset_3", "Asset_4"]:
            raise OpenFrameTestError(msg)

        if frame.weights != [0.4, 0.2, 0.1, 0.2]:
            raise OpenFrameTestError(msg)

    def test_risk_functions_same_as_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that risk measures align between OpenFrame and OpenTimeSeries."""
        riskseries = self.randomseries.from_deepcopy()
        riskseries.set_new_label(lvl_zero="Asset_0")
        riskframe = self.randomframe.from_deepcopy()
        riskseries.to_cumret()
        riskframe.to_cumret()

        if riskseries.cvar_down != _cvar_down_calc(data=riskseries.tsdf.iloc[:, 0]):
            msg = "CVaR for OpenTimeSeries not equal"
            raise OpenFrameTestError(msg)
        if riskseries.var_down != _var_down_calc(data=riskseries.tsdf.iloc[:, 0]):
            msg = "VaR for OpenTimeSeries not equal"
            raise OpenFrameTestError(msg)

        if cast("Series", riskframe.cvar_down).iloc[0] != _cvar_down_calc(
            data=riskframe.tsdf.iloc[:, 0],
        ):
            msg = "CVaR for OpenFrame not equal"
            raise OpenFrameTestError(msg)
        if cast("Series", riskframe.var_down).iloc[0] != _var_down_calc(
            riskframe.tsdf.iloc[:, 0],
        ):
            msg = "VaR for OpenFrame not equal"
            raise OpenFrameTestError(msg)

        if cast("Series", riskframe.cvar_down).iloc[0] != _cvar_down_calc(
            riskframe.tsdf,
        ):
            msg = "CVaR for OpenFrame not equal"
            raise OpenFrameTestError(msg)
        if cast("Series", riskframe.var_down).iloc[0] != _var_down_calc(
            riskframe.tsdf,
        ):
            msg = "VaR for OpenFrame not equal"
            raise OpenFrameTestError(msg)

    def test_methods_same_as_opentimeseries(self: TestOpenFrame) -> None:
        """Test that method results align between OpenFrame and OpenTimeSeries."""
        sameseries = self.randomseries.from_deepcopy()
        sameseries.value_to_ret()
        sameframe = self.randomframe.from_deepcopy()
        assert_frame_equal(
            left=sameseries.tsdf,
            right=Series(sameframe.tsdf.iloc[:, 0]).to_frame(),
        )

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(
            left=sameseries.tsdf,
            right=Series(sameframe.tsdf.iloc[:, 0]).to_frame(),
        )

        smethods = [
            sameseries.rolling_return,
            sameseries.rolling_vol,
            sameseries.rolling_var_down,
            sameseries.rolling_cvar_down,
        ]
        fmethods = [
            sameframe.rolling_return,
            sameframe.rolling_vol,
            sameframe.rolling_var_down,
            sameframe.rolling_cvar_down,
        ]
        for smethod, fmethod in zip(smethods, fmethods, strict=False):
            assert_frame_equal(
                left=smethod(),  # type: ignore[operator]
                right=fmethod(column=0),  # type: ignore[operator]
            )

        cumseries = sameseries.from_deepcopy()
        cumframe = sameframe.from_deepcopy()

        cumseries.value_to_log()
        cumframe.value_to_log()
        assert_frame_equal(
            left=sameseries.tsdf,
            right=Series(sameframe.tsdf.iloc[:, 0]).to_frame(),
        )

        sameseries.value_to_ret()
        sameframe.value_to_ret()
        assert_frame_equal(
            left=sameseries.tsdf,
            right=Series(sameframe.tsdf.iloc[:, 0]).to_frame(),
        )

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(
            left=sameseries.tsdf,
            right=Series(sameframe.tsdf.iloc[:, 0]).to_frame(),
        )

        sameseries.resample()
        sameframe.resample()
        assert_frame_equal(
            left=sameseries.tsdf,
            right=Series(sameframe.tsdf.iloc[:, 0]).to_frame(),
        )

        sameseries.value_to_diff()
        sameframe.value_to_diff()
        assert_frame_equal(
            left=sameseries.tsdf,
            right=Series(sameframe.tsdf.iloc[:, 0]).to_frame(),
        )

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
            "lower_partial_moment_func",
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
        smethods_to_compare = [
            sames.arithmetic_ret_func,
            sames.cvar_down_func,
            sames.lower_partial_moment_func,
            sames.geo_ret_func,
            sames.kurtosis_func,
            sames.max_drawdown_func,
            sames.positive_share_func,
            sames.skew_func,
            sames.vol_from_var_func,
            sames.target_weight_from_var,
            sames.value_ret_func,
            sames.var_down_func,
            sames.vol_func,
            sames.worst_func,
            sames.z_score_func,
        ]
        fmethods_to_compare = [
            samef.arithmetic_ret_func,
            samef.cvar_down_func,
            samef.lower_partial_moment_func,
            samef.geo_ret_func,
            samef.kurtosis_func,
            samef.max_drawdown_func,
            samef.positive_share_func,
            samef.skew_func,
            samef.vol_from_var_func,
            samef.target_weight_from_var,
            samef.value_ret_func,
            samef.var_down_func,
            samef.vol_func,
            samef.worst_func,
            samef.z_score_func,
        ]
        for method, smethod, fmethod in zip(
            methods_to_compare,
            smethods_to_compare,
            fmethods_to_compare,
            strict=False,
        ):
            if (
                f"{smethod(months_from_last=12):.9f}"  # type: ignore[operator]
                != f"{float(fmethod(months_from_last=12).iloc[0]):.9f}"  # type: ignore[operator]
            ):
                msg = (
                    f"Calc method {method} not aligned between "
                    "OpenTimeSeries and OpenFrame"
                )
                raise OpenFrameTestError(msg)

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
            "Series",
            samef.ret_vol_ratio_func(riskfree_rate=0.0, months_from_last=12),
        ).iloc[0]
        if (
            f"{sames.ret_vol_ratio_func(riskfree_rate=0.0, months_from_last=12):.11f}"
            != f"{smf_vrf:.11f}"
        ):
            msg = (
                "ret_vol_ratio_func() not aligned between OpenTimeSeries and OpenFrame"
            )
            raise OpenFrameTestError(msg)

        smf_srf = cast(
            "Series",
            samef.sortino_ratio_func(riskfree_rate=0.0, months_from_last=12),
        ).iloc[0]
        if (
            f"{sames.sortino_ratio_func(riskfree_rate=0.0, months_from_last=12):.11f}"
            != f"{smf_srf:.11f}"
        ):
            msg = (
                "sortino_ratio_func() not aligned between OpenTimeSeries and OpenFrame"
            )
            raise OpenFrameTestError(msg)

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
            "kappa3_ratio",
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
                    f"Property {prop} not aligned between OpenTimeSeries and OpenFrame"
                )
                raise OpenFrameTestError(msg)

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
                raise OpenFrameTestError(msg)

    def test_keeping_attributes_aligned_vs_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that attributes are aligned between OpenFrame and OpenTimeSeries."""
        common_calc_props = [
            "arithmetic_ret",
            "cvar_down",
            "downside_deviation",
            "geo_ret",
            "kurtosis",
            "max_drawdown",
            "max_drawdown_cal_year",
            "omega_ratio",
            "positive_share",
            "ret_vol_ratio",
            "skew",
            "sortino_ratio",
            "kappa3_ratio",
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
        ]

        pydantic_basemodel_attributes = [
            name for name in set(dir(BaseModel)) if not name.startswith("_")
        ]

        frame_calc_props = [
            "last_indices",
            "lengths_of_items",
            "first_indices",
            "columns_lvl_one",
            "span_of_days_all",
            "item_count",
            "columns_lvl_zero",
            "correl_matrix",
        ]

        series_props = [
            name
            for name, obj in getmembers(OpenTimeSeries)
            if name not in pydantic_basemodel_attributes
            and not name.startswith("_")
            and isinstance(obj, property)
        ]

        series_compared = set(series_props).symmetric_difference(
            set(common_calc_props + common_props + common_attributes),
        )
        if len(series_compared) != 0:
            msg = f"Difference is: {series_compared}"
            raise OpenFrameTestError(msg)

        frame_props = [
            name
            for name, obj in getmembers(OpenFrame)
            if name not in pydantic_basemodel_attributes
            and not name.startswith("_")
            and isinstance(obj, property)
        ]

        frame_compared = set(frame_props).symmetric_difference(
            set(
                common_calc_props
                + common_props
                + common_attributes
                + frame_calc_props,
            ),
        )
        if len(frame_compared) != 0:
            msg = f"Difference is: {frame_compared}"
            raise OpenFrameTestError(msg)

    def test_keeping_methods_aligned_vs_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that methods are aligned between OpenFrame and OpenTimeSeries."""
        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

        pydantic_basemodel_methods = [
            name for name in set(dir(BaseModel)) if not name.startswith("_")
        ]

        common_calc_methods = [
            "arithmetic_ret_func",
            "cvar_down_func",
            "lower_partial_moment_func",
            "geo_ret_func",
            "kurtosis_func",
            "max_drawdown_func",
            "omega_ratio_func",
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
            "from_deepcopy",
            "plot_bars",
            "plot_series",
            "plot_histogram",
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

        series_unique = [
            "ewma_vol_func",
            "from_1d_rate_to_cumret",
            "pandas_df",
            "running_adjustment",
            "set_new_label",
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
            "multi_factor_linear_regression",
            "make_portfolio",
            "merge_series",
            "relative",
            "rolling_corr",
            "rolling_beta",
            "trunc_frame",
        ]
        series_methods = [
            name
            for name, _ in getmembers(OpenTimeSeries, predicate=isfunction)
            if name not in pydantic_basemodel_methods and not name.startswith("_")
        ]

        series_compared = set(series_methods).symmetric_difference(
            set(
                common_calc_methods + common_methods + series_unique,
            ),
        )
        if len(series_compared) != 0:
            msg = f"Difference is: {series_compared}"
            raise OpenFrameTestError(msg)

        frame_methods = [
            name
            for name, _ in getmembers(OpenFrame, predicate=isfunction)
            if name not in pydantic_basemodel_methods and not name.startswith("_")
        ]

        frame_compared = set(frame_methods).symmetric_difference(
            set(
                common_calc_methods + common_methods + frame_unique,
            ),
        )
        if len(frame_compared) != 0:
            msg = f"Difference is: {frame_compared}"
            raise OpenFrameTestError(msg)

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
            raise OpenFrameTestError(msg)

    def test_correl_matrix(self: TestOpenFrame) -> None:
        """Test correl_matrix method."""
        corrframe = self.randomframe.from_deepcopy()
        corrframe.to_cumret()
        dict1 = corrframe.correl_matrix.map(lambda nn: f"{nn:.10f}").to_dict()

        dict2 = {
            "Asset_0": {
                "Asset_0": "1.0000000000",
                "Asset_1": "0.0239047199",
                "Asset_2": "-0.0077164998",
                "Asset_3": "0.0212300816",
                "Asset_4": "-0.0432032137",
            },
            "Asset_1": {
                "Asset_0": "0.0239047199",
                "Asset_1": "1.0000000000",
                "Asset_2": "-0.0027966077",
                "Asset_3": "-0.0275509319",
                "Asset_4": "-0.0078836695",
            },
            "Asset_2": {
                "Asset_0": "-0.0077164998",
                "Asset_1": "-0.0027966077",
                "Asset_2": "1.0000000000",
                "Asset_3": "-0.0138919271",
                "Asset_4": "0.0103654979",
            },
            "Asset_3": {
                "Asset_0": "0.0212300816",
                "Asset_1": "-0.0275509319",
                "Asset_2": "-0.0138919271",
                "Asset_3": "1.0000000000",
                "Asset_4": "0.0264748547",
            },
            "Asset_4": {
                "Asset_0": "-0.0432032137",
                "Asset_1": "-0.0078836695",
                "Asset_2": "0.0103654979",
                "Asset_3": "0.0264748547",
                "Asset_4": "1.0000000000",
            },
        }

        if dict1 != dict2:
            msg = f"Unexpected result(s) from method correl_matrix()\n{pformat(dict1)}"
            raise OpenFrameTestError(msg)

    def test_plot_series(self: TestOpenFrame) -> None:
        """Test plot_series method."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        fig, _ = plotframe.plot_series(auto_open=False, output_type="div")
        fig_json = loads(cast("str", fig.to_json()))

        rawdata = [x.strftime("%Y-%m-%d") for x in plotframe.tsdf.index[1:5]]
        if rawdata != fig_json["data"][0]["x"][1:5]:
            msg = "Unaligned data between original and data in Figure."
            raise OpenFrameTestError(msg)

        fig_last, _ = plotframe.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
        )
        fig_last_json = loads(cast("str", fig_last.to_json()))
        rawlast = plotframe.tsdf.iloc[-1, -1]
        figlast = fig_last_json["data"][-1]["y"][0]
        if f"{figlast:.12f}" != f"{rawlast:.12f}":
            msg = "Unaligned data between original and data in Figure."
            raise OpenFrameTestError(msg)

        fig_last_fmt, _ = plotframe.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
            tick_fmt=".3%",
        )
        fig_last_fmt_json = loads(cast("str", fig_last_fmt.to_json()))
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]

        if last_fmt != "Last 116.964%":
            msg = f"Unaligned data in Figure: '{last_fmt}'"
            raise OpenFrameTestError(msg)

        intended_labels = ["a", "b", "c", "d", "e"]
        fig_labels, _ = plotframe.plot_series(
            auto_open=False,
            output_type="div",
            labels=intended_labels,
        )
        fig_labels_json = loads(cast("str", fig_labels.to_json()))

        labels = [item["name"] for item in fig_labels_json["data"]]
        if labels != intended_labels:
            msg = f"Manual setting of labels not working: {labels}"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=NumberOfItemsAndLabelsNotSameError,
            match=r"Must provide same number of labels as items in frame.",
        ):
            _, _ = plotframe.plot_series(auto_open=False, labels=["a", "b"])

        _, logo = load_plotly_dict()

        fig_logo, _ = plotframe.plot_series(
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        fig_logo_json = loads(cast("str", fig_logo.to_json()))

        if logo == {}:
            if fig_logo_json["layout"]["images"][0] != logo:
                msg = "plot_series add_logo argument not setup correctly"
                raise OpenFrameTestError(msg)
        elif fig_logo_json["layout"]["images"][0]["source"] != logo["source"]:
            msg = "plot_series add_logo argument not setup correctly"
            raise OpenFrameTestError(msg)

        fig_nologo, _ = plotframe.plot_series(
            auto_open=False,
            add_logo=False,
            output_type="div",
        )
        fig_nologo_json = loads(cast("str", fig_nologo.to_json()))
        if fig_nologo_json["layout"].get("images", None):
            msg = "plot_series add_logo argument not setup correctly"
            raise OpenFrameTestError(msg)

        title = "My Plot"
        fig_title, _ = plotframe.plot_series(
            auto_open=False,
            title=title,
            output_type="div",
        )
        fig_title_json = loads(cast("str", fig_title.to_json()))

        if title not in fig_title_json["layout"]["title"]["text"]:
            msg = "plot_series title argument not setup correctly"
            raise OpenFrameTestError(msg)

        fig_no_title, _ = plotframe.plot_series(
            auto_open=False,
            title=None,
            output_type="div",
        )
        fig_no_title_json = loads(cast("str", fig_no_title.to_json()))

        if fig_no_title_json["layout"]["title"].get("text", None):
            msg = "plot_series title argument not setup correctly"
            raise OpenFrameTestError(msg)

    def test_plot_series_filefolders(self: TestOpenFrame) -> None:
        """Test plot_series method with different file folder options."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        directory = Path(__file__).parent
        _, figfile = plotframe.plot_series(auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        if figfile[:5] == "<div>":
            msg = "plot_series method not working as intended"
            raise OpenFrameTestError(msg)

        _, divstring = plotframe.plot_series(auto_open=False, output_type="div")
        if divstring[:5] != "<div>" or divstring[-6:] != "</div>":
            msg = "Html div section not created"
            raise OpenFrameTestError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = True
            mockhomefig, _ = plotframe.plot_series(
                auto_open=False,
                output_type="div",
            )
            mockhomefig_json = loads(cast("str", mockhomefig.to_json()))

        if mockhomefig_json["data"][0]["name"] != "Asset_0":
            msg = "plot_series method not working as intended"
            raise OpenFrameTestError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = False
            _, mockfile = plotframe.plot_series(
                filename="seriesfile.html",
                auto_open=False,
            )
            mockfilepath = Path(mockfile).resolve()

        if mockfilepath.parts[-2:] != ("tests", "seriesfile.html"):
            msg = "plot_series method not working as intended"
            raise OpenFrameTestError(msg)

        mockfilepath.unlink()

    def test_plot_bars(self: TestOpenFrame) -> None:
        """Test plot_bars method."""
        plotframe = self.randomframe.from_deepcopy()

        fig_keys = ["hovertemplate", "name", "type", "x", "y"]
        fig, _ = plotframe.plot_bars(auto_open=False, output_type="div")
        fig_json = loads(cast("str", fig.to_json()))
        made_fig_keys = list(fig_json["data"][0].keys())
        made_fig_keys.sort()
        if made_fig_keys != fig_keys:
            msg = "Data in Figure not as intended."
            raise OpenFrameTestError(msg)

        rawdata = [x.strftime("%Y-%m-%d") for x in plotframe.tsdf.index[1:5]]
        if rawdata != fig_json["data"][0]["x"][1:5]:
            msg = "Unaligned data between original and data in Figure."
            raise OpenFrameTestError(msg)

        intended_labels = ["a", "b", "c", "d", "e"]
        fig_labels, _ = plotframe.plot_bars(
            auto_open=False,
            output_type="div",
            labels=intended_labels,
        )
        fig_labels_json = loads(cast("str", fig_labels.to_json()))

        labels = [item["name"] for item in fig_labels_json["data"]]
        if labels != intended_labels:
            msg = f"Manual setting of labels not working: {labels}"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=NumberOfItemsAndLabelsNotSameError,
            match=r"Must provide same number of labels as items in frame.",
        ):
            _, _ = plotframe.plot_bars(auto_open=False, labels=["a", "b"])

        overlayfig, _ = plotframe.plot_bars(
            auto_open=False,
            output_type="div",
            mode="overlay",
        )
        overlayfig_json = loads(cast("str", overlayfig.to_json()))

        fig_keys.append("opacity")
        if sorted(overlayfig_json["data"][0].keys()) != sorted(fig_keys):
            msg = "Data in Figure not as intended."
            raise OpenFrameTestError(msg)

        _, logo = load_plotly_dict()

        fig_logo, _ = plotframe.plot_bars(
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        fig_logo_json = loads(cast("str", fig_logo.to_json()))

        if logo == {}:
            if fig_logo_json["layout"]["images"][0] != logo:
                msg = "plot_bars add_logo argument not setup correctly"
                raise OpenFrameTestError(msg)
        elif fig_logo_json["layout"]["images"][0]["source"] != logo["source"]:
            msg = "plot_bars add_logo argument not setup correctly"
            raise OpenFrameTestError(msg)

        fig_nologo, _ = plotframe.plot_bars(
            auto_open=False,
            add_logo=False,
            output_type="div",
        )
        fig_nologo_json = loads(cast("str", fig_nologo.to_json()))
        if fig_nologo_json["layout"].get("images", None):
            msg = "plot_bars add_logo argument not setup correctly"
            raise OpenFrameTestError(msg)

        title = "My Plot"
        fig_title, _ = plotframe.plot_bars(
            auto_open=False,
            title=title,
            output_type="div",
        )
        fig_title_json = loads(cast("str", fig_title.to_json()))

        if title not in fig_title_json["layout"]["title"]["text"]:
            msg = "plot_bars title argument not setup correctly"
            raise OpenFrameTestError(msg)

        fig_no_title, _ = plotframe.plot_bars(
            auto_open=False,
            title=None,
            output_type="div",
        )
        fig_no_title_json = loads(cast("str", fig_no_title.to_json()))

        if fig_no_title_json["layout"]["title"].get("text", None):
            msg = "plot_bars title argument not setup correctly"
            raise OpenFrameTestError(msg)

    def test_plot_bars_filefolders(self: TestOpenFrame) -> None:
        """Test plot_bars method with different file folder options."""
        plotframe = self.randomframe.from_deepcopy()

        directory = Path(__file__).parent
        _, figfile = plotframe.plot_bars(auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        if figfile[:5] == "<div>":
            msg = "plot_bars method not working as intended"
            raise OpenFrameTestError(msg)

        _, divstring = plotframe.plot_bars(auto_open=False, output_type="div")
        if divstring[:5] != "<div>" or divstring[-6:] != "</div>":
            msg = "Html div section not created"
            raise OpenFrameTestError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = True
            mockhomefig, _ = plotframe.plot_bars(
                auto_open=False,
                output_type="div",
            )
            mockhomefig_json = loads(cast("str", mockhomefig.to_json()))

        if mockhomefig_json["data"][0]["name"] != "Asset_0":
            msg = "plot_bars method not working as intended"
            raise OpenFrameTestError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = False
            _, mockfile = plotframe.plot_bars(
                filename="barfile.html",
                auto_open=False,
            )
            mockfilepath = Path(mockfile).resolve()

        if mockfilepath.parts[-2:] != ("tests", "barfile.html"):
            msg = "plot_bars method not working as intended"
            raise OpenFrameTestError(msg)

        mockfilepath.unlink()

    def test_plot_histogram_bars(self: TestOpenFrame) -> None:
        """Test plot_histogram method with plot_type bars."""
        plotframe = self.randomframe.from_deepcopy()

        fig_keys = [
            "cumulative",
            "histfunc",
            "histnorm",
            "hovertemplate",
            "name",
            "opacity",
            "type",
            "x",
        ]
        fig, _ = plotframe.plot_histogram(auto_open=False, output_type="div")
        fig_json = loads(cast("str", fig.to_json()))
        made_fig_keys = list(fig_json["data"][0].keys())
        made_fig_keys.sort()
        if made_fig_keys != fig_keys:
            msg = f"Data in Figure not as intended:\n{made_fig_keys}"
            raise OpenFrameTestError(msg)

        fig_fmt, _ = plotframe.plot_histogram(
            auto_open=False,
            output_type="div",
            x_fmt=".2%",
        )
        fig_fmt_json = loads(cast("str", fig_fmt.to_json()))
        x_tickfmt = fig_fmt_json["layout"]["xaxis"].get("tickformat")
        if x_tickfmt != ".2%":
            msg = f"X axis tick format not working: '{x_tickfmt}'"
            raise OpenFrameTestError(msg)

        intended_labels = ["a", "b", "c", "d", "e"]
        fig_labels, _ = plotframe.plot_histogram(
            auto_open=False,
            output_type="div",
            labels=intended_labels,
        )
        fig_labels_json = loads(cast("str", fig_labels.to_json()))
        labels = [trace["name"] for trace in fig_labels_json["data"]]
        if labels != intended_labels:
            msg = f"Manual setting of labels not working: {labels}"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=NumberOfItemsAndLabelsNotSameError,
            match=r"Must provide same number of labels as items in frame.",
        ):
            _, _ = plotframe.plot_histogram(auto_open=False, labels=["a", "b"])

        _, logo = load_plotly_dict()
        fig_logo, _ = plotframe.plot_histogram(
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        fig_logo_json = loads(cast("str", fig_logo.to_json()))
        if logo == {}:
            if fig_logo_json["layout"]["images"][0] != logo:
                msg = "plot_histogram add_logo argument not setup correctly"
                raise OpenFrameTestError(msg)
        elif fig_logo_json["layout"]["images"][0]["source"] != logo["source"]:
            msg = "plot_histogram add_logo argument not setup correctly"
            raise OpenFrameTestError(msg)

        fig_nologo, _ = plotframe.plot_histogram(
            auto_open=False,
            add_logo=False,
            output_type="div",
        )
        fig_nologo_json = loads(cast("str", fig_nologo.to_json()))
        if fig_nologo_json["layout"].get("images", None):
            msg = "plot_histogram add_logo argument not setup correctly"
            raise OpenFrameTestError(msg)

        title = "My Plot"
        fig_title, _ = plotframe.plot_histogram(
            auto_open=False,
            title=title,
            output_type="div",
        )
        fig_title_json = loads(cast("str", fig_title.to_json()))

        if title not in fig_title_json["layout"]["title"]["text"]:
            msg = "plot_histogram title argument not setup correctly"
            raise OpenFrameTestError(msg)

        fig_no_title, _ = plotframe.plot_histogram(
            auto_open=False,
            title=None,
            output_type="div",
        )
        fig_no_title_json = loads(cast("str", fig_no_title.to_json()))

        if fig_no_title_json["layout"]["title"].get("text", None):
            msg = "plot_histogram title argument not setup correctly"
            raise OpenFrameTestError(msg)

    def test_plot_histogram_lines(self: TestOpenFrame) -> None:
        """Test plot_histogram method with plot_type lines."""
        plotframe = self.randomframe.from_deepcopy()

        fig_keys = [
            "legendgroup",
            "marker",
            "mode",
            "name",
            "showlegend",
            "type",
            "x",
            "xaxis",
            "y",
            "yaxis",
        ]
        fig, _ = plotframe.plot_histogram(
            plot_type="lines",
            auto_open=False,
            output_type="div",
        )
        fig_json = loads(cast("str", fig.to_json()))
        made_fig_keys = list(fig_json["data"][0].keys())
        made_fig_keys.sort()
        if made_fig_keys != fig_keys:
            msg = f"Data in Figure not as intended:\n{made_fig_keys}"
            raise OpenFrameTestError(msg)

        intended_labels = ["a", "b", "c", "d", "e"]
        fig_labels, _ = plotframe.plot_histogram(
            plot_type="lines",
            auto_open=False,
            output_type="div",
            labels=intended_labels,
        )
        fig_labels_json = loads(cast("str", fig_labels.to_json()))
        labels = [trace["name"] for trace in fig_labels_json["data"]]
        if labels != intended_labels:
            msg = f"Manual setting of labels not working: {labels}"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match=r"plot_type must be 'bars' or 'lines'.",
        ):
            _, _ = plotframe.plot_histogram(
                plot_type=cast("Literal['bars', 'lines']", "triangles"),
                auto_open=False,
            )

    def test_plot_histogram_filefolders(self: TestOpenFrame) -> None:
        """Test plot_histogram method with different file folder options."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        directory = Path(__file__).parent
        _, figfile = plotframe.plot_histogram(auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)
        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        if figfile.startswith("<div>"):
            msg = "plot_histogram method not working as intended"
            raise OpenFrameTestError(msg)

        _, divstring = plotframe.plot_histogram(auto_open=False, output_type="div")
        if not (divstring.startswith("<div>") and divstring.endswith("</div>")):
            msg = "Html div section not created"
            raise OpenFrameTestError(msg)

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            homefig, _ = plotframe.plot_histogram(auto_open=False, output_type="div")
            homefig_json = loads(cast("str", homefig.to_json()))
        if homefig_json["data"][0]["name"] != "Asset_0":
            msg = "plot_histogram method not working as intended"
            raise OpenFrameTestError(msg)

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            _, mockfile = plotframe.plot_histogram(
                filename="histfile.html",
                auto_open=False,
            )
            mockpath = Path(mockfile).resolve()
        if mockpath.parts[-2:] != ("tests", "histfile.html"):
            msg = "plot_histogram method not working as intended"
            raise OpenFrameTestError(msg)
        mockpath.unlink()

    def test_plot_methods_mock_logo_url_fail(self: TestOpenFrame) -> None:
        """Test plot_series and plot_bars methods with mock logo file URL fail."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        with patch("requests.head") as mock_conn_error:
            mock_conn_error.side_effect = RequestsConnectionError()

            seriesfig, _ = plotframe.plot_series(
                auto_open=False,
                add_logo=True,
                output_type="div",
            )
            seriesfig_json = loads(cast("str", seriesfig.to_json()))
            if seriesfig_json["layout"]["images"][0].get("source", None):
                msg = "plot_series add_logo argument not setup correctly"
                raise OpenFrameTestError(msg)

            barfig, _ = plotframe.plot_bars(
                auto_open=False,
                add_logo=True,
                output_type="div",
            )
            barfig_json = loads(cast("str", barfig.to_json()))
            if barfig_json["layout"]["images"][0].get("source", None):
                msg = "plot_bars add_logo argument not setup correctly"
                raise OpenFrameTestError(msg)

            with self.assertLogs() as plotseries_context:
                _, _ = plotframe.plot_series(auto_open=False, output_type="div")
            if (
                "WARNING:openseries.load_plotly:Failed to add logo image from URL"
                not in plotseries_context.output[0]
            ):
                msg = (
                    "plot_series() method did not warn as "
                    "expected when logo URL not working: "
                    f"{pformat(plotseries_context.output[0])}"
                )
                raise OpenFrameTestError(msg)

            with self.assertLogs() as plotbars_context:
                _, _ = plotframe.plot_bars(auto_open=False, output_type="div")
            if (
                "WARNING:openseries.load_plotly:Failed to add logo image from URL"
                not in plotbars_context.output[0]
            ):
                msg = (
                    "plot_bars() method did not warn as "
                    "expected when logo URL not working"
                )
                raise OpenFrameTestError(msg)

        with patch("requests.head") as mock_statuscode:
            mock_statuscode.return_value.status_code = 400

            with self.assertLogs() as plotseries_context:
                _, _ = plotframe.plot_series(auto_open=False, output_type="div")
            if (
                "WARNING:openseries.load_plotly:Failed to add logo image from URL"
                not in plotseries_context.output[0]
            ):
                msg = (
                    "plot_series() method did not warn as "
                    "expected when logo URL not working"
                )
                raise OpenFrameTestError(msg)

            with self.assertLogs() as plotbars_context:
                _, _ = plotframe.plot_bars(auto_open=False, output_type="div")
            if (
                "WARNING:openseries.load_plotly:Failed to add logo image from URL"
                not in plotbars_context.output[0]
            ):
                msg = (
                    "plot_bars() method did not warn as "
                    "expected when logo URL not working"
                )
                raise OpenFrameTestError(msg)

        with patch("requests.head") as mock_statuscode:
            mock_statuscode.return_value.status_code = 200

            fig, _ = plotframe.plot_series(auto_open=False, output_type="div")
            fig_json = loads(cast("str", fig.to_json()))

            rawdata = [x.strftime("%Y-%m-%d") for x in plotframe.tsdf.index[1:5]]
            if rawdata != fig_json["data"][0]["x"][1:5]:
                msg = "Unaligned data between original and data in Figure."
                raise OpenFrameTestError(msg)

    def test_passed_empty_list(self: TestOpenFrame) -> None:
        """Test warning on object construct with empty list."""
        with self.assertLogs() as contextmgr:
            OpenFrame(constituents=[])
        if contextmgr.output != [
            "WARNING:openseries.frame:OpenFrame() was passed an empty list.",
        ]:
            msg = (
                "OpenFrame failed to log warning about "
                f"empty input list: {pformat(contextmgr.output)}"
            )
            raise OpenFrameTestError(msg)

    def test_trunc_frame_both(self: TestOpenFrame) -> None:
        """Test trunc_frame method."""
        series_long = self.randomseries.from_deepcopy()
        series_long.set_new_label("Long")
        tmp_series = self.randomseries.from_deepcopy()
        series_short = OpenTimeSeries.from_df(
            dframe=tmp_series.tsdf.loc[
                cast("Timestamp", dt.date(2017, 6, 27)) : cast(
                    "Timestamp",
                    dt.date(2018, 6, 27),
                )
            ][("Asset_0", ValueType.PRICE)],
        )
        series_short.set_new_label("Short")
        frame = OpenFrame(constituents=[series_long, series_short])

        firsts = [
            dt.date(2017, 6, 27),
            dt.date(2017, 6, 27),
        ]
        lasts = [
            dt.date(2018, 6, 27),
            dt.date(2018, 6, 27),
        ]

        if firsts == frame.first_indices.tolist():
            msg = "trunc_frame() test not set up as intended."
            raise OpenFrameTestError(msg)
        if lasts == frame.last_indices.tolist():
            msg = "trunc_frame() test not set up as intended."
            raise OpenFrameTestError(msg)

        frame.trunc_frame()

        if firsts != frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)
        if lasts != frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)

        trunced = [dt.date(2017, 12, 29), dt.date(2018, 3, 29)]

        frame.trunc_frame(start_cut=trunced[0], end_cut=trunced[1])

        if trunced != [frame.first_idx, frame.last_idx]:
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)

    def test_trunc_frame_before_after(self: TestOpenFrame) -> None:
        """Test trunc_frame method."""
        series_long = self.randomseries.from_deepcopy()
        series_long.set_new_label("Long")
        tmp_series = self.randomseries.from_deepcopy()
        series_short = OpenTimeSeries.from_df(
            dframe=tmp_series.tsdf.loc[
                cast("Timestamp", dt.date(2017, 6, 27)) : cast(
                    "Timestamp",
                    dt.date(2018, 6, 27),
                )
            ][("Asset_0", ValueType.PRICE)],
        )
        series_short.set_new_label("Short")
        frame = OpenFrame(constituents=[series_long, series_short])

        firsts = [
            dt.date(2017, 6, 27),
            dt.date(2017, 6, 27),
        ]
        lasts = [
            dt.date(2018, 6, 27),
            dt.date(2018, 6, 27),
        ]

        if firsts == frame.first_indices.tolist():
            msg = "trunc_frame() test not set up as intended."
            raise OpenFrameTestError(msg)
        if lasts == frame.last_indices.tolist():
            msg = "trunc_frame() test not set up as intended."
            raise OpenFrameTestError(msg)

        before_frame = frame.from_deepcopy()
        after_frame = frame.from_deepcopy()

        before_frame.trunc_frame(where="before")

        if firsts != before_frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)
        if lasts == before_frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)

        after_frame.trunc_frame(where="after")

        if firsts == after_frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)
        if lasts != after_frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)

        before_frame.trunc_frame(where="after")
        after_frame.trunc_frame(where="before")

        if firsts != after_frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)
        if lasts != before_frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)

    def test_trunc_frame_start_fail(self: TestOpenFrame) -> None:
        """Test trunc_frame method start fail scenario."""
        frame = OpenFrame(
            constituents=[
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
            "WARNING:openseries.frame:One or more constituents "
            "still not truncated to same start dates."
        ) not in logs.output[0]:
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)

    def test_trunc_frame_end_fail(self: TestOpenFrame) -> None:
        """Test trunc_frame method end fail scenario."""
        frame = OpenFrame(
            constituents=[
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
            "WARNING:openseries.frame:One or more constituents "
            "still not truncated to same end dates."
        ) not in logs.output[0]:
            msg = "Method trunc_frame() did not work as intended."
            raise OpenFrameTestError(msg)

    def test_merge_series(self: TestOpenFrame) -> None:
        """Test merge_series method."""
        aframe = OpenFrame(
            constituents=[
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
            constituents=[
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

        oldlabels = list(aframe.columns_lvl_zero)
        aframe.tsdf = aframe.tsdf.rename(
            columns={
                "Asset_one": "Asset_three",
                "Asset_two": "Asset_four",
            },
            level=0,
        )
        newlabels = list(aframe.columns_lvl_zero)

        b4df = aframe.tsdf.copy()

        if oldlabels == newlabels:
            msg = "Setup to test merge_series() did not work as intended."
            raise OpenFrameTestError(msg)

        aframe.merge_series(how="outer")
        labelspostmerge = list(aframe.columns_lvl_zero)

        assert_frame_equal(left=b4df, right=aframe.tsdf, check_exact=True)

        if newlabels != labelspostmerge:
            msg = "Method merge_series() did not work as intended."
            raise OpenFrameTestError(msg)

        bframe.merge_series(how="inner")
        blist = [d.strftime("%Y-%m-%d") for d in bframe.tsdf.index]
        if blist != [
            "2022-07-11",
            "2022-07-12",
            "2022-07-13",
            "2022-07-14",
        ]:
            msg = "Method merge_series() did not work as intended."
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=Exception,
            match=(
                r"Merging OpenTimeSeries DataFrames with argument "
                r"how=inner produced an empty DataFrame."
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
            "Kappa-3 ratio",
            "Omega ratio",
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
            "Max drawdown date",
            "Max drawdown in cal yr",
            "first indices",
            "last indices",
            "observations",
            "span of days",
        ]
        apframe = self.randomframe.from_deepcopy()
        apframe.to_cumret()
        result = apframe.all_properties()
        result_index = result.index.tolist()

        msg = "Method all_properties() not working as intended."
        if not isinstance(result, DataFrame):
            raise TypeError(msg)

        result_arg = apframe.all_properties(properties=["geo_ret"])

        msg = "Method all_properties() not working as intended."
        if not isinstance(result_arg, DataFrame):
            raise TypeError(msg)

        if set(prop_index) != set(result_index):
            msg = "Method all_properties() output not as intended."
            raise OpenFrameTestError(msg)

        result_values = {}
        for value in result.index:
            if isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], float):
                result_values[value] = (
                    f"{result.loc[value, ('Asset_0', ValueType.PRICE)]:.10f}"
                )
            elif isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], int):
                result_values[value] = cast(
                    "str",
                    result.loc[value, ("Asset_0", ValueType.PRICE)],
                )
            elif isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], dt.date):
                result_values[value] = cast(
                    "dt.date",
                    result.loc[value, ("Asset_0", ValueType.PRICE)],
                ).strftime("%Y-%m-%d")
            else:
                msg = f"all_properties returned unexpected type {type(value)}"
                raise TypeError(msg)

        expected_values = {
            "Arithmetic return": "0.0585047569",
            "CVaR 95.0%": "-0.0123803429",
            "Downside deviation": "0.0667228073",
            "Geometric return": "0.0507567099",
            "Imp vol from VaR 95%": "0.0936737165",
            "Kurtosis": "696.0965168893",
            "Max drawdown": "-0.1314808074",
            "Max drawdown in cal yr": "-0.1292814491",
            "Max drawdown date": "2012-11-21",
            "Omega ratio": "1.0983709757",
            "Positive share": "0.5057745918",
            "Return vol ratio": "0.4162058331",
            "Simple return": "0.6401159258",
            "Skew": "19.1911712502",
            "Sortino ratio": "0.8768329634",
            "Kappa-3 ratio": "0.6671520235",
            "VaR 95.0%": "-0.0097182152",
            "Volatility": "0.1405668835",
            "Worst": "-0.0191572882",
            "Worst month": "-0.0581245494",
            "Z-score": "0.3750685522",
            "first indices": "2009-06-30",
            "last indices": "2019-06-28",
            "observations": 2512,
            "span of days": 3650,
        }

        if result_values != expected_values:
            msg = (
                "Method all_properties() results "
                f"not as expected.\n{pformat(result_values)}"
            )
            raise OpenFrameTestError(msg)

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
        aframe = OpenFrame(constituents=[aseries, bseries])
        anotherframe = OpenFrame(constituents=[aseries, bseries])
        yetoneframe = OpenFrame(constituents=[aseries, bseries])
        noneframe = OpenFrame(constituents=[aseries, bseries])

        swedennationalday = dt.date(2022, 6, 6)

        msg = "align_index_to_local_cdays not working as intended"
        msg_in = "Sweden National Day in date range"
        msg_notin = "Sweden National Day not in date range"

        if swedennationalday not in d_range:
            raise OpenFrameTestError(msg_notin)

        aframe.align_index_to_local_cdays(countries="SE")
        if swedennationalday in aframe.tsdf.index:
            msg = "Sweden National Day in date range"
            raise OpenFrameTestError(msg_in)

        anotherframe.align_index_to_local_cdays(countries="US", markets="XSTO")
        if swedennationalday in anotherframe.tsdf.index:
            msg = "Sweden National Day in date range"
            raise OpenFrameTestError(msg_in)

        ctries = [ctry.countries for ctry in anotherframe.constituents]
        mkts = [mkt.markets for mkt in anotherframe.constituents]

        if ctries != ["US", "US"]:
            raise OpenFrameTestError(msg)

        if mkts != ["XSTO", "XSTO"]:
            raise OpenFrameTestError(msg)

        yetoneframe.align_index_to_local_cdays(countries="US")
        if swedennationalday not in yetoneframe.tsdf.index:
            msg = "Sweden National Day not in date range"
            raise OpenFrameTestError(msg_notin)

        noneframe.align_index_to_local_cdays(countries=None)  # default option
        if swedennationalday in noneframe.tsdf.index:
            msg = "Sweden National Day in date range"
            raise OpenFrameTestError(msg_in)

    def test_rolling_info_ratio(self: TestOpenFrame) -> None:
        """Test rolling_info_ratio method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_info_ratio(long_column=0, short_column=1)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "-0.10944130198",
            "-0.12182987211",
            "-0.02845717747",
            "-0.04862442310",
            "-0.02521145077",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_info_ratio() not as intended\n{values}"
            raise OpenFrameTestError(msg)

        simdata_fxd_per_yr = frame.rolling_info_ratio(
            long_column=0,
            short_column=1,
            periods_in_a_year_fixed=251,
        )

        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5, 0]]
        checkdata_fxd_per_yr = [
            "-0.10952238381",
            "-0.12192013228",
            "-0.02847826056",
            "-0.04866044751",
            "-0.02523012919",
        ]

        if values_fxd_per_yr != checkdata_fxd_per_yr:
            msg = (
                "Result from method rolling_info_ratio()"
                f" not as intended\n{values_fxd_per_yr}"
            )
            raise OpenFrameTestError(msg)

    def test_rolling_beta(self: TestOpenFrame) -> None:
        """Test rolling_beta method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()
        simdata = frame.rolling_beta(asset_column=0, market_column=1)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.11870239224",
            "0.13743313660",
            "0.10203126346",
            "0.11658696118",
            "0.09445337602",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_beta() not as intended\n{values}"
            raise OpenFrameTestError(msg)

    def test_tracking_error_func(self: TestOpenFrame) -> None:
        """Test tracking_error_func method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdataa = frame.tracking_error_func(base_column=-1)

        if f"{simdataa.iloc[0]:.9f}" != "0.176767248":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdataa.iloc[0]:.9f}'"
            )
            raise OpenFrameTestError(msg)

        simdatab = frame.tracking_error_func(
            base_column=-1,
            periods_in_a_year_fixed=251,
        )

        if f"{simdatab.iloc[0]:.9f}" != "0.176636383":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdatab.iloc[0]:.9f}'"
            )
            raise OpenFrameTestError(msg)

        simdatac = frame.tracking_error_func(base_column=("Asset_4", ValueType.PRICE))

        if f"{simdatac.iloc[0]:.9f}" != "0.176767248":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdatac.iloc[0]:.9f}'"
            )
            raise OpenFrameTestError(msg)

        if f"{simdataa.iloc[0]:.9f}" != f"{simdatac.iloc[0]:.9f}":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdataa.iloc[0]:.9f}' "
                f"versus '{simdatac.iloc[0]:.9f}'"
            )
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="base_column should be a tuple",
        ):
            _ = frame.tracking_error_func(
                base_column=cast("tuple[str, ValueType] | int", "string"),
            )

    def test_info_ratio_func(self: TestOpenFrame) -> None:
        """Test info_ratio_func method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdataa = frame.info_ratio_func(base_column=-1)

        if f"{simdataa.iloc[0]:.10f}" != "0.2132240667":
            msg = (
                f"Result from info_ratio_func() not "
                f"as expected: '{simdataa.iloc[0]:.10f}'"
            )
            raise OpenFrameTestError(msg)

        simdatab = frame.info_ratio_func(base_column=-1, periods_in_a_year_fixed=251)

        if f"{simdatab.iloc[0]:.10f}" != "0.2130662123":
            msg = (
                f"Result from info_ratio_func() not "
                f"as expected: '{simdatab.iloc[0]:.10f}'"
            )
            raise OpenFrameTestError(msg)

        simdatac = frame.info_ratio_func(base_column=("Asset_4", ValueType.PRICE))

        if f"{simdatac.iloc[0]:.10f}" != "0.2132240667":
            msg = (
                f"Result from info_ratio_func() not "
                f"as expected: '{simdatac.iloc[0]:.10f}'"
            )
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="base_column should be a tuple",
        ):
            _ = frame.info_ratio_func(
                base_column=cast("tuple[str, ValueType] | int", "string"),
            )

    def test_rolling_corr(self: TestOpenFrame) -> None:
        """Test rolling_corr method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_corr(first_column=0, second_column=1)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.17883816332",
            "0.20010259575",
            "0.14454747431",
            "0.16212311694",
            "0.12819083799",
        ]

        if values != checkdata:
            msg = "Result from method rolling_corr() not as intended."
            raise OpenFrameTestError(msg)

    def test_rolling_vol(self: TestOpenFrame) -> None:
        """Test rolling_vol method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_vol(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.06597558501",
            "0.06856064777",
            "0.07136508824",
            "0.07272307603",
            "0.07199918606",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_vol() not as intended.\n{values}"
            raise OpenFrameTestError(msg)

        simdata_fxd_per_yr = frame.rolling_vol(
            column=0,
            observations=21,
            periods_in_a_year_fixed=251,
        )

        values_fxd = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5, 0]]
        checkdata_fxd = [
            "0.06592674183",
            "0.06850989081",
            "0.07131225509",
            "0.07266923753",
            "0.07194588348",
        ]

        if values_fxd != checkdata_fxd:
            msg = f"Result from method rolling_vol() not as intended.\n{values_fxd}"
            raise OpenFrameTestError(msg)

    def test_rolling_return(self: TestOpenFrame) -> None:
        """Test rolling_return method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_return(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.00700872599",
            "-0.00261771394",
            "0.00348938855",
            "0.00809475448",
            "0.00383062672",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_return() not as intended\n{values}"
            raise OpenFrameTestError(msg)

    def test_rolling_cvar_down(self: TestOpenFrame) -> None:
        """Test rolling_cvar_down method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_cvar_down(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[-14:-9, 0]]
        checkdata = [
            "-0.01044706636",
            "-0.00974280167",
            "-0.00974280167",
            "-0.01139997783",
            "-0.01077418053",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_cvar_down() not as intended\n{values}"
            raise OpenFrameTestError(msg)

    def test_rolling_var_down(self: TestOpenFrame) -> None:
        """Test rolling_var_down method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_var_down(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[-5:, 0]]
        checkdata = [
            "-0.01243135668",
            "-0.01243135668",
            "-0.01243135668",
            "-0.01243135668",
            "-0.01243135668",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_var_down() not as intended\n{values}"
            raise OpenFrameTestError(msg)

    def test_label_uniqueness(self: TestOpenFrame) -> None:
        """Test label uniqueness."""
        aseries = self.randomseries.from_deepcopy()
        bseries = self.randomseries.from_deepcopy()

        with pytest.raises(
            expected_exception=LabelsNotUniqueError,
            match="TimeSeries names/labels must be unique",
        ):
            OpenFrame(constituents=[aseries, bseries])

        bseries.set_new_label("other_name")
        uframe = OpenFrame(constituents=[aseries, bseries])

        if uframe.columns_lvl_zero != ["Asset_0", "other_name"]:
            msg = "Fix of non-unique labels unsuccessful."
            raise OpenFrameTestError(msg)

    def test_capture_ratio(self: TestOpenFrame) -> None:
        """Test the capture_ratio_func method.

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
        cframe = OpenFrame(constituents=[asset, indxx]).to_cumret()

        upp = cframe.capture_ratio_func(ratio="up")
        down = cframe.capture_ratio_func(ratio="down")
        both = cframe.capture_ratio_func(ratio="both")

        if f"{upp.iloc[0]:.12f}" != "1.063842457805":
            msg = "Result from capture_ratio_func() not as expected."
            raise OpenFrameTestError(msg)
        if f"{down.iloc[0]:.12f}" != "0.922188852957":
            msg = "Result from capture_ratio_func() not as expected."
            raise OpenFrameTestError(msg)
        if f"{both.iloc[0]:.12f}" != "1.153605852417":
            msg = "Result from capture_ratio_func() not as expected."
            raise OpenFrameTestError(msg)

        upfixed = cframe.capture_ratio_func(ratio="up", periods_in_a_year_fixed=12)

        if f"{upfixed.iloc[0]:.12f}" != "1.063217236138":
            msg = "Result from capture_ratio_func() not as expected."
            raise OpenFrameTestError(msg)

        if f"{upp.iloc[0]:.2f}" != f"{upfixed.iloc[0]:.2f}":
            msg = "Result from capture_ratio_func() not as expected."
            raise OpenFrameTestError(msg)

        uptuple = cframe.capture_ratio_func(
            ratio="up",
            base_column=("indxx", ValueType.PRICE),
        )

        if f"{uptuple.iloc[0]:.12f}" != "1.063842457805":
            msg = "Result from capture_ratio_func() not as expected."
            raise OpenFrameTestError(msg)

        if f"{upp.iloc[0]:.12f}" != f"{uptuple.iloc[0]:.12f}":
            msg = "Result from capture_ratio_func() not as expected."
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match=r"base_column should be a tuple",
        ):
            _ = cframe.capture_ratio_func(
                ratio="up",
                base_column=cast("tuple[str, ValueType] | int", "string"),
            )

        with pytest.raises(
            expected_exception=RatioInputError,
            match=r"ratio must be one of 'up', 'down' or 'both'.",
        ):
            _ = cframe.capture_ratio_func(
                ratio=cast("Literal['up', 'down', 'both']", "boo"),
            )

    def test_georet_exceptions(self: TestOpenFrame) -> None:
        """Test georet property raising exceptions on bad input data."""
        geoframe = OpenFrame(
            constituents=[
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
        if [f"{gr:.5f}" for gr in cast("Series", geoframe.geo_ret)] != [
            "0.10007",
            "0.20015",
        ]:
            msg = "Unexpected result from property geo_ret"
            raise OpenFrameTestError(msg)

        if [f"{gr:.5f}" for gr in cast("Series", geoframe.geo_ret_func())] != [
            "0.10007",
            "0.20015",
        ]:
            msg = "Unexpected result from property geo_ret"
            raise OpenFrameTestError(msg)

        geoframe.add_timeseries(
            OpenTimeSeries.from_arrays(
                name="geoseries3",
                dates=["2022-07-01", "2023-07-01"],
                values=[0.0, 1.1],
            ),
        )

        with pytest.raises(
            expected_exception=InitialValueZeroError,
            match=(
                r"Geometric return cannot be calculated due to an "
                r"initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret

        with pytest.raises(
            expected_exception=InitialValueZeroError,
            match=(
                r"Geometric return cannot be calculated due to an "
                r"initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret_func()

        geoframe.delete_timeseries(lvl_zero_item="geoseries3")

        if [f"{gr:.5f}" for gr in cast("Series", geoframe.geo_ret)] != [
            "0.10007",
            "0.20015",
        ]:
            msg = "Unexpected result from property geo_ret"
            raise OpenFrameTestError(msg)

        geoframe.add_timeseries(
            OpenTimeSeries.from_arrays(
                name="geoseries4",
                dates=["2022-07-01", "2023-07-01"],
                values=[1.0, -1.1],
            ),
        )
        with pytest.raises(
            expected_exception=InitialValueZeroError,
            match=(
                r"Geometric return cannot be calculated due to an "
                r"initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret

        with pytest.raises(
            expected_exception=InitialValueZeroError,
            match=(
                r"Geometric return cannot be calculated due to an "
                r"initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret_func()

    def test_value_nan_handle(self: TestOpenFrame) -> None:
        """Test value_nan_handle method."""
        nanframe = OpenFrame(
            constituents=[
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

        nanframe.tsdf.iloc[2, 0] = None
        nanframe.tsdf.iloc[3, 1] = None
        dropframe = nanframe.from_deepcopy()
        dropframe.value_nan_handle(method="drop")

        if Series(dropframe.tsdf.iloc[:, 0]).tolist() != [1.1, 1.0, 1.0]:
            msg = "Method value_nan_handle() not working as intended"
            raise OpenFrameTestError(msg)
        if Series(dropframe.tsdf.iloc[:, 1]).tolist() != [2.1, 2.0, 2.0]:
            msg = "Method value_nan_handle() not working as intended"
            raise OpenFrameTestError(msg)

        fillframe = nanframe.from_deepcopy()
        fillframe.value_nan_handle(method="fill")

        if Series(fillframe.tsdf.iloc[:, 0]).tolist() != [1.1, 1.0, 1.0, 1.1, 1.0]:
            msg = "Method value_nan_handle() not working as intended"
            raise OpenFrameTestError(msg)
        if Series(fillframe.tsdf.iloc[:, 1]).tolist() != [2.1, 2.0, 1.8, 1.8, 2.0]:
            msg = "Method value_nan_handle() not working as intended"
            raise OpenFrameTestError(msg)

    def test_return_nan_handle(self: TestOpenFrame) -> None:
        """Test return_nan_handle method."""
        nanframe = OpenFrame(
            constituents=[
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

        nanframe.tsdf.iloc[2, 0] = None
        nanframe.tsdf.iloc[3, 1] = None
        dropframe = nanframe.from_deepcopy()
        dropframe.return_nan_handle(method="drop")

        if Series(dropframe.tsdf.iloc[:, 0]).tolist() != [0.1, 0.05, 0.04]:
            msg = "Method return_nan_handle() not working as intended"
            raise OpenFrameTestError(msg)
        if Series(dropframe.tsdf.iloc[:, 1]).tolist() != [0.01, 0.04, 0.06]:
            msg = "Method return_nan_handle() not working as intended"
            raise OpenFrameTestError(msg)

        fillframe = nanframe.from_deepcopy()
        fillframe.return_nan_handle(method="fill")

        if Series(fillframe.tsdf.iloc[:, 0]).tolist() != [0.1, 0.05, 0.0, 0.01, 0.04]:
            msg = "Method return_nan_handle() not working as intended"
            raise OpenFrameTestError(msg)
        if Series(fillframe.tsdf.iloc[:, 1]).tolist() != [0.01, 0.04, 0.02, 0.0, 0.06]:
            msg = "Method return_nan_handle() not working as intended"
            raise OpenFrameTestError(msg)

    def test_relative(self: TestOpenFrame) -> None:
        """Test relative method."""
        rframe = self.randomframe.from_deepcopy()
        rframe.to_cumret()
        sframe = self.randomframe.from_deepcopy()
        sframe.to_cumret()
        series_before = 5
        series_after = 6

        if rframe.item_count != series_before:
            msg = "Method relative() base case not as intended"
            raise OpenFrameTestError(msg)

        rframe.relative()

        if rframe.item_count != series_after:
            msg = "Method relative() not working as intended"
            raise OpenFrameTestError(msg)

        if rframe.tsdf.shape[1] != series_after:
            msg = "Method relative() not working as intended"
            raise OpenFrameTestError(msg)

        if rframe.constituents[-1].label != "Asset_0_over_Asset_1":
            msg = "Method relative() not working as intended"
            raise OpenFrameTestError(msg)

        if rframe.columns_lvl_zero[-1] != "Asset_0_over_Asset_1":
            msg = "Method relative() not working as intended"
            raise OpenFrameTestError(msg)

        rframe.tsdf.iloc[:, -1] = Series(rframe.tsdf.iloc[:, -1]).add(1.0)

        sframe.relative(base_zero=False)

        rflist = [f"{rret:.11f}" for rret in rframe.tsdf.iloc[:, -1]]
        sflist = [f"{rret:.11f}" for rret in sframe.tsdf.iloc[:, -1]]

        if rflist != sflist:
            msg = "Method relative() not working as intended"
            raise OpenFrameTestError(msg)

    def test_to_cumret(self: TestOpenFrame) -> None:
        """Test to_cumret method."""
        rseries = self.randomseries.from_deepcopy()
        rseries.value_to_ret()
        rrseries = rseries.from_deepcopy()
        rrseries.value_to_ret()
        rrseries.set_new_label(lvl_zero="Rasset")

        cseries = self.randomseries.from_deepcopy()
        cseries.set_new_label(lvl_zero="Basset")
        ccseries = cseries.from_deepcopy()
        ccseries.set_new_label(lvl_zero="Casset")

        mframe = OpenFrame(constituents=[rseries, cseries])
        cframe = OpenFrame(constituents=[cseries, ccseries])
        rframe = OpenFrame(constituents=[rseries, rrseries])

        if mframe.columns_lvl_one != [ValueType.RTRN, ValueType.PRICE]:
            msg = "Method to_cumret() not working as intended"
            raise OpenFrameTestError(msg)

        if cframe.columns_lvl_one != [ValueType.PRICE, ValueType.PRICE]:
            msg = "Method to_cumret() not working as intended"
            raise OpenFrameTestError(msg)

        cframe_lvl_one = list(cframe.columns_lvl_one)

        if rframe.columns_lvl_one != [ValueType.RTRN, ValueType.RTRN]:
            msg = "Method to_cumret() not working as intended"
            raise OpenFrameTestError(msg)

        cframe.to_cumret()
        rframe.to_cumret()

        if mframe.columns_lvl_one != [ValueType.RTRN, ValueType.PRICE]:
            msg = "Method to_cumret() not working as intended"
            raise OpenFrameTestError(msg)

        if cframe_lvl_one != cframe.columns_lvl_one:
            msg = "Method to_cumret() not working as intended"
            raise OpenFrameTestError(msg)

        if rframe.columns_lvl_one != [ValueType.PRICE, ValueType.PRICE]:
            msg = "Method to_cumret() not working as intended"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=MixedValuetypesError,
            match="Mix of series types will give inconsistent results",
        ):
            mframe.to_cumret()

        fmt = "{:.8f}"

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
            raise OpenFrameTestError(msg)

    def test_miscellaneous(self: TestOpenFrame) -> None:
        """Test miscellaneous methods."""
        zero_str: str = "0"
        zero_float: float = 0.0
        mframe = self.randomframe.from_deepcopy()
        mframe.to_cumret()

        methods = [
            mframe.arithmetic_ret_func,
            mframe.vol_func,
            mframe.vol_from_var_func,
            mframe.lower_partial_moment_func,
            mframe.target_weight_from_var,
        ]
        for methd in methods:
            no_fixed = methd()  # type: ignore[operator]
            fixed = methd(periods_in_a_year_fixed=252)  # type: ignore[operator]
            for nofix, fix in zip(no_fixed, fixed, strict=True):
                if f"{100 * abs(nofix - fix):.0f}" != zero_str:
                    msg = (
                        "Difference with or without fixed periods in year is too great"
                    )
                    raise OpenFrameTestError(msg)
        for methd in methods:
            dated = methd(  # type: ignore[operator]
                from_date=mframe.first_idx,
                to_date=mframe.last_idx,
            )
            undated = methd()  # type: ignore[operator]
            for ddat, undat in zip(dated, undated, strict=True):
                if f"{ddat:.10f}" != f"{undat:.10f}":
                    msg = (
                        f"Method {methd} with and without date "
                        "arguments returned inconsistent results"
                    )
                    raise OpenFrameTestError(msg)

        ret = [f"{rr:.9f}" for rr in cast("Series", mframe.value_ret_func())]
        if ret != [
            "0.640115926",
            "0.354975641",
            "1.287658441",
            "1.045918527",
            "0.169641332",
        ]:
            msg = f"Results from value_ret_func() not as expected\n{ret}"
            raise OpenFrameTestError(msg)

        impvol = [
            f"{iv:.11f}"
            for iv in cast("Series", mframe.vol_from_var_func(drift_adjust=False))
        ]
        impvoldrifted = [
            f"{iv:.11f}"
            for iv in cast("Series", mframe.vol_from_var_func(drift_adjust=True))
        ]

        if impvol != [
            "0.09367371648",
            "0.09357837907",
            "0.09766514288",
            "0.09903125918",
            "0.10121521823",
        ]:
            msg = f"Results from vol_from_var_func() not as expected\n{impvol}"
            raise OpenFrameTestError(msg)

        if impvoldrifted != [
            "0.09591621674",
            "0.09495303438",
            "0.10103656927",
            "0.10197016748",
            "0.10201301292",
        ]:
            msg = f"Results from vol_from_var_func() not as expected\n{impvoldrifted}"
            raise OpenFrameTestError(msg)

        mframe.tsdf.iloc[0, 2] = zero_float

        with pytest.raises(
            expected_exception=InitialValueZeroError,
            match="Simple return cannot be calculated due to an",
        ):
            _ = mframe.value_ret

        with pytest.raises(
            expected_exception=InitialValueZeroError,
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
            from_date=dt.date(2017, 12, 29),
            to_date=dt.date(2018, 12, 28),
        )
        vrff_y = vrcframe.value_ret_func(
            from_date=dt.date(2017, 12, 29),
            to_date=dt.date(2018, 12, 28),
        )
        vrffl_y = [f"{rr:.11f}" for rr in cast("Series", vrff_y)]

        vrvrcs_y = vrcseries.value_ret_calendar_period(year=2018)
        vrvrcf_y = vrcframe.value_ret_calendar_period(year=2018)
        vrvrcfl_y = [f"{rr:.11f}" for rr in cast("Series", vrvrcf_y)]

        if f"{vrfs_y:.11f}" != f"{vrvrcs_y:.11f}":
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise OpenFrameTestError(msg)

        if vrffl_y != vrvrcfl_y:
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise OpenFrameTestError(msg)

        vrfs_ym = vrcseries.value_ret_func(
            from_date=dt.date(2018, 4, 30),
            to_date=dt.date(2018, 5, 31),
        )
        vrff_ym = vrcframe.value_ret_func(
            from_date=dt.date(2018, 4, 30),
            to_date=dt.date(2018, 5, 31),
        )
        vrffl_ym = [f"{rr:.11f}" for rr in cast("Series", vrff_ym)]

        vrvrcs_ym = vrcseries.value_ret_calendar_period(year=2018, month=5)
        vrvrcf_ym = vrcframe.value_ret_calendar_period(year=2018, month=5)
        vrvrcfl_ym = [f"{rr:.11f}" for rr in cast("Series", vrvrcf_ym)]

        if f"{vrfs_ym:.11f}" != f"{vrvrcs_ym:.11f}":
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise OpenFrameTestError(msg)

        if vrffl_ym != vrvrcfl_ym:
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise OpenFrameTestError(msg)

    def test_to_drawdown_series(self: TestOpenFrame) -> None:
        """Test to_drawdown_series method."""
        mframe = self.randomframe.from_deepcopy()
        mframe.to_cumret()
        ddown = [f"{dmax:.11f}" for dmax in cast("Series", mframe.max_drawdown)]
        mframe.to_drawdown_series()
        ddownserie = [f"{dmax:.11f}" for dmax in mframe.tsdf.min()]

        if ddown != ddownserie:
            msg = "Method to_drawdown_series() not working as intended"
            raise OpenFrameTestError(msg)

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
            raise OpenFrameTestError(msg)

        if fsframe.columns_lvl_one[-1] != oframe.columns_lvl_zero[1]:
            msg = "Method ord_least_squares_fit() not working as intended"
            raise OpenFrameTestError(msg)

        results = []
        for i in range(oframe.item_count):
            for j in range(oframe.item_count):
                ols = oframe.ord_least_squares_fit(
                    y_column=i,
                    x_column=j,
                    fitted_series=False,
                )
                results.append(f"{ols['coefficient']:.11f}")

        results_tuple = []
        k_tuple: Hashable
        l_tuple: Hashable
        for k_tuple in oframe.tsdf:
            for l_tuple in oframe.tsdf:
                ols = oframe.ord_least_squares_fit(
                    y_column=cast("tuple[str, ValueType]", k_tuple),
                    x_column=cast("tuple[str, ValueType]", l_tuple),
                    fitted_series=False,
                )
                results_tuple.append(f"{ols['coefficient']:.11f}")

        if results != results_tuple:
            msg = "Method ord_least_squares_fit() not working as intended"
            raise OpenFrameTestError(msg)

        intended = [
            "1.00000000000",
            "1.30086569003",
            "0.70760700025",
            "0.67465227245",
            "1.10187251514",
            "0.58311900396",
            "1.00000000000",
            "0.44879185229",
            "0.41664068548",
            "0.66341168704",
            "1.26345414943",
            "1.78767099287",
            "1.00000000000",
            "0.92874072536",
            "1.59673304423",
            "1.29428495525",
            "1.78314580648",
            "0.99787706235",
            "1.00000000000",
            "1.60740416727",
            "0.48167638079",
            "0.64696737805",
            "0.39092095264",
            "0.36626816720",
            "1.00000000000",
        ]

        if results != intended:
            msg = f"Method ord_least_squares_fit() not working as intended\n{results}"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="x_column should be a tuple",
        ):
            _ = oframe.ord_least_squares_fit(
                y_column=0,
                x_column=cast("tuple[str, ValueType] | int", "string"),
                fitted_series=False,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="y_column should be a tuple",
        ):
            _ = oframe.ord_least_squares_fit(
                y_column=cast("tuple[str, ValueType] | int", "string"),
                x_column=1,
                fitted_series=False,
            )

    def test_beta(self: TestOpenFrame) -> None:
        """Test beta method."""
        bframe = self.randomframe.from_deepcopy()
        bframe.to_cumret()
        bframe.resample("7D")
        results = [
            f"{bframe.beta(asset=comb[0], market=comb[1]):.9f}"
            for comb in iter_product(
                range(bframe.item_count),
                range(bframe.item_count),
            )
        ]
        results_tuple = []
        for comb in iter_product(bframe.tsdf, bframe.tsdf):
            beta = bframe.beta(
                asset=cast("tuple[str, ValueType] | int", comb[0]),
                market=cast("tuple[str, ValueType] | int", comb[1]),
            )
            results_tuple.append(f"{beta:.9f}")

        if results != results_tuple:
            msg = "Unexpected results from method beta()"
            raise OpenFrameTestError(msg)

        if results != [
            "1.000000000",
            "0.022866669",
            "-0.014043923",
            "0.046279565",
            "-0.084766517",
            "0.011121651",
            "1.000000000",
            "-0.016786144",
            "-0.052121141",
            "-0.005276238",
            "-0.006444169",
            "-0.015836642",
            "1.000000000",
            "0.024878534",
            "-0.002303547",
            "0.019857392",
            "-0.045981223",
            "0.023263723",
            "1.000000000",
            "0.034819945",
            "-0.040556389",
            "-0.005190306",
            "-0.002401892",
            "0.038826665",
            "1.000000000",
        ]:
            msg = f"Unexpected results from method beta()\n{results}"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = bframe.beta(
                asset=cast("tuple[str, ValueType] | int", "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = bframe.beta(
                asset=0,
                market=cast("tuple[str, ValueType] | int", "string"),
            )

        mixframe = self.make_mixed_type_openframe()
        with pytest.raises(
            expected_exception=MixedValuetypesError,
            match="Mix of series types will give inconsistent results",
        ):
            _ = mixframe.beta(asset=0, market=-1)

    def test_beta_returns_input(self: TestOpenFrame) -> None:
        """Test beta method with returns input."""
        bframe = self.randomframe.from_deepcopy()
        bframe.resample("7D")
        results = [
            f"{bframe.beta(asset=comb[0], market=comb[1]):.9f}"
            for comb in iter_product(
                range(bframe.item_count),
                range(bframe.item_count),
            )
        ]

        results_tuple = []
        for comb in iter_product(bframe.tsdf, bframe.tsdf):
            beta = bframe.beta(
                asset=cast("tuple[str, ValueType] | int", comb[0]),
                market=cast("tuple[str, ValueType] | int", comb[1]),
            )
            results_tuple.append(f"{beta:.9f}")

        if results != results_tuple:
            msg = "Unexpected results from method beta()"
            raise OpenFrameTestError(msg)

        if results != [
            "1.000000000",
            "0.022369167",
            "-0.013894862",
            "0.047138369",
            "-0.085577120",
            "0.010914962",
            "1.000000000",
            "-0.016814808",
            "-0.048363738",
            "-0.005954032",
            "-0.006400844",
            "-0.015874590",
            "1.000000000",
            "0.022169540",
            "-0.000779595",
            "0.020488479",
            "-0.043080690",
            "0.020917456",
            "1.000000000",
            "0.032443857",
            "-0.041313652",
            "-0.005890805",
            "-0.000817000",
            "0.036035721",
            "1.000000000",
        ]:
            msg = f"Unexpected results from method beta()\n{results}"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = bframe.beta(
                asset=cast("tuple[str, ValueType] | int", "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = bframe.beta(
                asset=0,
                market=cast("tuple[str, ValueType] | int", "string"),
            )

    def test_jensen_alpha(self: TestOpenFrame) -> None:
        """Test jensen_alpha method."""
        jframe = self.randomframe.from_deepcopy()
        jframe.to_cumret()
        jframe.resample("7D")
        results = [
            f"{jframe.jensen_alpha(asset=comb[0], market=comb[1]):.9f}".replace(
                "-0.000000000",
                "0.000000000",
            )
            for comb in iter_product(
                range(jframe.item_count),
                range(jframe.item_count),
            )
        ]

        results_tuple = []
        for comb in iter_product(jframe.tsdf, jframe.tsdf):
            alpha = jframe.jensen_alpha(
                asset=cast("tuple[str, ValueType] | int", comb[0]),
                market=cast("tuple[str, ValueType] | int", comb[1]),
            )
            results_tuple.append(f"{alpha:.9f}".replace("-0.000000000", "0.000000000"))

        if results != results_tuple:
            msg = "Unexpected results from method jensen_alpha()"
            raise OpenFrameTestError(msg)

        if results != [
            "0.000000000",
            "0.057775286",
            "0.059811269",
            "0.055158277",
            "0.060404749",
            "0.034039808",
            "0.000000000",
            "0.036176551",
            "0.038531926",
            "0.034805479",
            "0.088864910",
            "0.089036876",
            "0.000000000",
            "0.086654217",
            "0.088537383",
            "0.072525683",
            "0.075283845",
            "0.071630154",
            "0.000000000",
            "0.072934441",
            "0.024037077",
            "0.021841806",
            "0.021874285",
            "0.018800661",
            "0.000000000",
        ]:
            msg = f"Unexpected results from method jensen_alpha()\n{results}"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=cast("tuple[str, ValueType] | int", "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=0,
                market=cast("tuple[str, ValueType] | int", "string"),
            )

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
                asset=cast("tuple[str, ValueType] | int", comb[0]),
                market=cast("tuple[str, ValueType] | int", comb[1]),
            )
            results_tuple.append(f"{alpha:.9f}")

        if results != results_tuple:
            msg = "Unexpected results from method jensen_alpha()"
            raise OpenFrameTestError(msg)

        if results != [
            "0.000000000",
            "0.057726997",
            "0.059752248",
            "0.054913781",
            "0.060311507",
            "0.035239850",
            "0.000000000",
            "0.037358312",
            "0.039588483",
            "0.036002676",
            "0.088369343",
            "0.088564264",
            "0.000000000",
            "0.086294170",
            "0.088010938",
            "0.075506739",
            "0.078251600",
            "0.074865296",
            "0.000000000",
            "0.076030357",
            "0.023240625",
            "0.021033909",
            "0.020894446",
            "0.018058401",
            "0.000000000",
        ]:
            msg = f"Unexpected results from method jensen_alpha()\n{results}"
            raise OpenFrameTestError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=cast("tuple[str, ValueType] | int", "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=0,
                market=cast("tuple[str, ValueType] | int", "string"),
            )

        mixframe = self.make_mixed_type_openframe()
        with pytest.raises(
            expected_exception=MixedValuetypesError,
            match="Mix of series types will give inconsistent results",
        ):
            _ = mixframe.jensen_alpha(asset=0, market=1)

    def test_ewma_risk(self: TestOpenFrame) -> None:
        """Test ewma_risk method."""
        eframe = self.randomframe.from_deepcopy()
        eframe.to_cumret()
        edf = eframe.ewma_risk()

        list_one = [f"{e:.11f}" for e in edf.head(10).iloc[:, 0]]
        list_two = [f"{e:.11f}" for e in edf.head(10).iloc[:, 1]]
        corr_one = [f"{e:.11f}" for e in edf.head(10).iloc[:, 2]]

        if list_one != [
            "0.06250431742",
            "0.06208916909",
            "0.06022552031",
            "0.05840562180",
            "0.05812960782",
            "0.05791392918",
            "0.05691361221",
            "0.06483761105",
            "0.06421248171",
            "0.06260970713",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{list_one}"
            raise OpenFrameTestError(msg)

        if list_two != [
            "0.06783309941",
            "0.06799180977",
            "0.06592070398",
            "0.06491357256",
            "0.06990142770",
            "0.06794101498",
            "0.06878554654",
            "0.06979371620",
            "0.07423575130",
            "0.07469953921",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{list_two}"
            raise OpenFrameTestError(msg)

        if corr_one != [
            "-0.00018950439",
            "0.02743890783",
            "0.02746345872",
            "0.02900303362",
            "0.07459903397",
            "0.08052954645",
            "0.09959709234",
            "0.00357025403",
            "-0.03875791933",
            "-0.02293443963",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{corr_one}"
            raise OpenFrameTestError(msg)

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
            "0.07361260347",
            "0.08380092327",
            "0.08392771222",
            "0.08501304571",
            "0.08362635928",
            "0.08175874942",
            "0.07996606071",
            "0.07806839320",
            "0.07604456321",
            "0.07468607166",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{list_three}"
            raise OpenFrameTestError(msg)

        if list_four != [
            "0.10703091058",
            "0.10476082189",
            "0.10165576633",
            "0.10346063549",
            "0.10280828404",
            "0.09967975602",
            "0.09693631061",
            "0.10848146972",
            "0.11025146341",
            "0.11658815836",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{list_four}"
            raise OpenFrameTestError(msg)

        if corr_two != [
            "-0.00112585517",
            "0.03499961968",
            "0.03901924990",
            "-0.00845828372",
            "-0.02665539891",
            "-0.02696637211",
            "-0.02152694854",
            "-0.04779886556",
            "-0.05984710080",
            "-0.08603979121",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{corr_two}"
            raise OpenFrameTestError(msg)

    def test_multi_factor_linear_regression(self: TestOpenFrame) -> None:
        """Test multi_factor_linear_regression method."""
        frame = self.randomframe.from_deepcopy()
        portfolio = OpenTimeSeries.from_df(
            dframe=frame.make_portfolio(name="Portfolio", weight_strat="eq_weights"),
        ).value_to_ret()
        frame.add_timeseries(portfolio)

        intended = {
            "R-square": "1.00000",
            "Intercept": "0.00000",
            "Asset_0": "0.20000",
            "Asset_1": "0.20000",
            "Asset_2": "0.20000",
            "Asset_3": "0.20000",
            "Asset_4": "0.20000",
        }

        output, _ = frame.multi_factor_linear_regression(
            dependent_column=(cast("str", portfolio.label), ValueType.RTRN),
        )
        result = output.to_dict()[portfolio.label]
        rounded = {}
        for key, value in result.items():
            rounded[key] = f"{value:.5f}"

        msg = (
            "Unexpected results from method "
            f"multi_factor_linear_regression()\n{pformat(rounded)}"
        )
        if intended != rounded:
            raise OpenFrameTestError(msg)

        nonexistantlabel = "nonexistantlabel"
        with pytest.raises(
            expected_exception=KeyError,
            match=escape(
                f"Tuple ({nonexistantlabel}, Return(Total)) not found in data.",
            ),
        ):
            _, _ = frame.multi_factor_linear_regression(
                dependent_column=(nonexistantlabel, ValueType.RTRN),
            )

        gframe = self.randomframe.from_deepcopy()
        gportfolio = OpenTimeSeries.from_df(
            dframe=gframe.make_portfolio(name="Portfolio", weight_strat="eq_weights"),
        )
        gframe.add_timeseries(gportfolio)
        with pytest.raises(
            expected_exception=MixedValuetypesError,
            match=r"All series should be of ValueType.RTRN.",
        ):
            _, _ = gframe.multi_factor_linear_regression(
                dependent_column=(cast("str", gportfolio.label), ValueType.PRICE),
            )

    def test_worst_month(self: TestOpenFrame) -> None:
        """Test worst_month property."""
        mixframe = self.make_mixed_type_openframe()
        with pytest.raises(
            expected_exception=ResampleDataLossError,
            match=r"Do not run worst_month on return series.",
        ):
            _ = mixframe.worst_month
