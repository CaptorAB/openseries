"""Test suite for the openseries/series.py module."""
from __future__ import annotations

import datetime as dt
import sys
from io import StringIO
from json import load, loads
from os import path, remove
from typing import TypeVar, Union, cast
from unittest import TestCase

import pytest
from pandas import DataFrame, DatetimeIndex, Series, date_range
from pydantic import ValidationError as PydanticValidationError

from openseries.series import (
    OpenTimeSeries,
    check_if_none,
    timeseries_chain,
)
from openseries.types import CountriesType, LiteralSeriesProps, ValueType
from tests.common_sim import ONE_SIM

TypeTestOpenTimeSeries = TypeVar("TypeTestOpenTimeSeries", bound="TestOpenTimeSeries")


class NewTimeSeries(OpenTimeSeries):

    """class to test correct pass-through of classes."""

    extra_info: str = "cool"


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "valuetype",
    [ValueType.PRICE, "Price(Close)"],
)
def test_opentimeseries_valid_valuetype(valuetype: ValueType) -> None:
    """Pytest on valid valuetype as input."""
    assert isinstance(
        OpenTimeSeries.from_arrays(
            name="Asset",
            dates=["2023-01-01", "2023-01-02"],
            valuetype=valuetype,
            values=[1.0, 1.1],
        ),
        OpenTimeSeries,
    )


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "valuetype",
    [None, "Price", 12, 1.2],
)
def test_opentimeseries_invalid_valuetype(valuetype: ValueType) -> None:
    """Pytest on invalid valuetype as input."""
    with pytest.raises(PydanticValidationError) as e_invalid_valuetype:
        OpenTimeSeries.from_arrays(
            name="Asset",
            dates=["2023-01-01", "2023-01-02"],
            valuetype=valuetype,
            values=[1.0, 1.1],
        )
    assert "type=enum" in str(
        e_invalid_valuetype.getrepr(style="short"),
    ) or "type=string_type" in str(e_invalid_valuetype.getrepr(style="short"))


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "currency",
    ["SE", True, "12", 1, None],
)
def test_opentimeseries_invalid_currency(currency: str) -> None:
    """Pytest on invalid currency code as input for currency."""
    with pytest.raises(PydanticValidationError) as e_invalid_currency:
        OpenTimeSeries.from_arrays(
            name="Asset",
            baseccy=currency,
            dates=["2023-01-01", "2023-01-02"],
            valuetype=ValueType.PRICE,
            values=[1.0, 1.1],
        )
    assert "type=string_too_short" in str(
        e_invalid_currency.getrepr(style="short"),
    ) or "type=string_type" in str(e_invalid_currency.getrepr(style="short"))


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "domestic",
    ["SE", True, "12", 1, None],
)
def test_opentimeseries_invalid_domestic(domestic: str) -> None:
    """Pytest on invalid currency code as input for domestic."""
    with pytest.raises(PydanticValidationError) as e_dom:
        serie = OpenTimeSeries.from_arrays(
            name="Asset",
            dates=["2023-01-01", "2023-01-02"],
            values=[1.0, 1.1],
        )
        serie.domestic = domestic
    assert "type=string_too_short" in str(
        e_dom.getrepr(style="short"),
    ) or "type=string_type" in str(e_dom.getrepr(style="short"))


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "countries",
    ["SEK", True, "12", 1, None, ["SEK"], [True], ["12"], [1], [None], []],
)
def test_opentimeseries_invalid_countries(countries: CountriesType) -> None:
    """Pytest on invalid country codes as input."""
    with pytest.raises(PydanticValidationError) as e_ctries:
        serie = OpenTimeSeries.from_arrays(
            name="Asset",
            dates=["2023-01-01", "2023-01-02"],
            values=[1.0, 1.1],
        )
        serie.countries = countries
    assert "type=list_type" in str(
        e_ctries.getrepr(style="short"),
    ) or "type=string_type" in str(e_ctries.getrepr(style="short"))


class TestOpenTimeSeries(TestCase):

    """class to run unittests on the module series.py."""

    randomseries: OpenTimeSeries
    random_properties: dict[str, Union[dt.date, int, float]]

    @classmethod
    def setUpClass(cls: type[TypeTestOpenTimeSeries]) -> None:
        """SetUpClass for the TestOpenTimeSeries class."""
        cls.randomseries = OpenTimeSeries.from_df(
            ONE_SIM.to_dataframe(name="Asset", end=dt.date(2019, 6, 30)),
        ).to_cumret()

        cls.random_properties = cls.randomseries.all_properties().to_dict()[
            ("Asset", ValueType.PRICE)
        ]

    def test_setup_class(self: TestOpenTimeSeries) -> None:
        """Test setup_class method."""
        with pytest.raises(
            expected_exception=ValueError,
            match="domestic currency must be a code according to ISO 4217",
        ):
            OpenTimeSeries.setup_class(domestic_ccy="12")

        with pytest.raises(
            expected_exception=ValueError,
            match="domestic currency must be a code according to ISO 4217",
        ):
            OpenTimeSeries.setup_class(domestic_ccy=cast(str, 12))

        with pytest.raises(
            expected_exception=ValueError,
            match="according to ISO 3166-1 alpha-2",
        ):
            OpenTimeSeries.setup_class(countries="12")

        with pytest.raises(
            expected_exception=TypeError,
            match=(
                "countries must be a list of country "
                "codes according to ISO 3166-1 alpha-2"
            ),
        ):
            OpenTimeSeries.setup_class(countries=["SE", cast(str, 12)])

        with pytest.raises(
            expected_exception=TypeError,
            match=(
                "countries must be a list of country "
                "codes according to ISO 3166-1 alpha-2"
            ),
        ):
            OpenTimeSeries.setup_class(countries=["SE", "12"])

        with pytest.raises(
            expected_exception=TypeError,
            match="according to ISO 3166-1 alpha-2",
        ):
            OpenTimeSeries.setup_class(countries=cast(CountriesType, None))

        OpenTimeSeries.setup_class(domestic_ccy="USD", countries="US")
        assert OpenTimeSeries.domestic == "USD"
        assert OpenTimeSeries.countries == "US"

    def test_invalid_dates(self: TestOpenTimeSeries) -> None:
        """Test invalid dates as input."""
        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="Input should be a valid string",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=["2023-01-01", cast(str, None)],
                values=[1.0, 1.1],
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="must be called with a collection of some kind",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=cast(list[str], None),
                values=[1.0, 1.1],
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="must be called with a collection of some kind",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=cast(list[str], "2023-01-01"),
                values=[1.0, 1.1],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Unknown datetime string format",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=["2023-01-bb", "2023-01-02"],
                values=[1.0, 1.1],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Shape of passed values is",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=[],
                values=[1.0, 1.1],
            )

    def test_invalid_values(self: TestOpenTimeSeries) -> None:
        """Test invalid values as input."""
        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="Input should be a valid number",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=["2023-01-01", "2023-01-02"],
                values=[1.0, cast(float, None)],
            )

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="Input should be a valid list",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=["2023-01-01", "2023-01-02"],
                values=cast(list[float], None),
            )

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="Input should be a valid list",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=["2023-01-01", "2023-01-02"],
                values=cast(list[float], 1.0),
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="could not convert string to float",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=["2023-01-01", "2023-01-02"],
                values=[1.0, cast(float, "bb")],
            )

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="There must be at least 1 value",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset",
                dates=["2023-01-01", "2023-01-02"],
                values=[],
            )

    def test_duplicates_handling(self: TestOpenTimeSeries) -> None:
        """Test duplicate handling."""
        json_file = path.join(path.dirname(path.abspath(__file__)), "series.json")
        with open(json_file, encoding="utf-8") as jsonfile:
            output = load(jsonfile)

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

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="Dates are not unique",
        ):
            _ = OpenTimeSeries.from_arrays(
                name="Bond Fund",
                dates=dates,
                values=values,
            )

    def test_valid_tsdf(self: TestOpenTimeSeries) -> None:
        """Test valid pandas.DataFrame property."""
        dframe = DataFrame(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=[["Asset_0"], [ValueType.PRICE]],
            dtype="float64",
        )
        serie = Series(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            name=("Asset_0", ValueType.PRICE),
            dtype="float64",
        )
        data = {
            "timeseries_id": "",
            "currency": "SEK",
            "dates": [
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            "instrument_id": "",
            "isin": "",
            "local_ccy": True,
            "name": "Asset_0",
            "values": [1.0, 1.01, 0.99, 1.015, 1.003],
            "valuetype": ValueType.PRICE,
        }
        df_data = {"tsdf": dframe, **data}
        serie_data = {"tsdf": serie, **data}

        df_obj = OpenTimeSeries(**df_data)
        self.assertListEqual(list(df_obj.tsdf.to_numpy()), df_obj.values)
        assert isinstance(df_obj, OpenTimeSeries)

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="Input should be an instance of DataFrame",
        ):
            OpenTimeSeries(**serie_data)

    def test_create_from_arrays(self: TestOpenTimeSeries) -> None:
        """Test from_arrays construct method."""
        arrseries = OpenTimeSeries.from_arrays(
            name="arrseries",
            dates=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            values=[1.0, 1.01, 0.99, 1.015, 1.003],
        )
        assert isinstance(arrseries, OpenTimeSeries)

    def test_create_from_pandas_df(self: TestOpenTimeSeries) -> None:
        """Test construct from pandas.DataFrame."""
        serie = Series(
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
            name=("Asset_0", ValueType.PRICE),
            dtype="float64",
        )
        df1 = DataFrame(
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
        df2 = DataFrame(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=[["Asset_0"], [ValueType.PRICE]],
            dtype="float64",
        )
        df3 = DataFrame(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=[[""], [ValueType.PRICE]],
            dtype="float64",
        )
        df4 = DataFrame(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=[["Asset_0"], [None]],
            dtype="float64",
        )

        seseries = OpenTimeSeries.from_df(dframe=serie)
        senseries = OpenTimeSeries.from_df(dframe=sen)
        df1series = OpenTimeSeries.from_df(dframe=df1, column_nmbr=1)
        df2series = OpenTimeSeries.from_df(dframe=df2, column_nmbr=0)

        assert isinstance(seseries, OpenTimeSeries)
        assert isinstance(senseries, OpenTimeSeries)
        assert seseries.label == senseries.label

        assert isinstance(df1series, OpenTimeSeries)
        assert isinstance(df2series, OpenTimeSeries)
        label_message = "label missing. Adding 'Series' as label"
        type_message = "valuetype missing. Adding 'Price(Close)' as valuetype"
        old_stdout = sys.stdout
        new_stdout = StringIO()
        sys.stdout = new_stdout

        df3series = OpenTimeSeries.from_df(dframe=df3, column_nmbr=0)
        df3_output = new_stdout.getvalue()
        df4series = OpenTimeSeries.from_df(dframe=df4, column_nmbr=0)
        df4_output = new_stdout.getvalue()

        sys.stdout = old_stdout
        assert label_message in df3_output
        assert type_message in df4_output

        assert isinstance(df3series, OpenTimeSeries)
        assert isinstance(df4series, OpenTimeSeries)

        assert check_if_none(None)
        assert not check_if_none(0.0)

    def test_save_to_json(self: TestOpenTimeSeries) -> None:
        """Test to_json method."""
        seriesfile = path.join(
            path.dirname(path.abspath(__file__)),
            "seriessaved.json",
        )

        jseries = self.randomseries.from_deepcopy()
        data = jseries.to_json(filename=seriesfile)

        self.assertListEqual(
            [item.get("name") for item in data],
            ["Asset"],
        )

        assert path.exists(seriesfile)

        remove(seriesfile)

        assert not path.exists(seriesfile)

    def test_create_from_fixed_rate(self: TestOpenTimeSeries) -> None:
        """Test from_fixed_rate construct method."""
        fixseries_one = OpenTimeSeries.from_fixed_rate(
            rate=0.03,
            days=756,
            end_dt=dt.date(2019, 6, 30),
        )
        assert isinstance(fixseries_one, OpenTimeSeries)

        rnd_series = self.randomseries.from_deepcopy()
        fixseries_two = OpenTimeSeries.from_fixed_rate(
            rate=0.03,
            d_range=DatetimeIndex(rnd_series.tsdf.index),
        )
        assert isinstance(fixseries_two, OpenTimeSeries)

        with pytest.raises(
            expected_exception=ValueError,
            match="If d_range is not provided both days and end_dt must be.",
        ):
            _ = OpenTimeSeries.from_fixed_rate(rate=0.03)

        with pytest.raises(
            expected_exception=ValueError,
            match="If d_range is not provided both days and end_dt must be.",
        ):
            _ = OpenTimeSeries.from_fixed_rate(rate=0.03, days=30)

    def test_periods_in_a_year(self: TestOpenTimeSeries) -> None:
        """Test periods_in_a_year property."""
        calc = len(self.randomseries.dates) / (
            (self.randomseries.last_idx - self.randomseries.first_idx).days / 365.25
        )

        assert calc == self.randomseries.periods_in_a_year
        assert (
            f"{251.3720547945205:.13f}"
            == f"{self.randomseries.periods_in_a_year:.13f}"
        )
        all_prop = self.random_properties["periods_in_a_year"]
        assert f"{all_prop:.13f}" == f"{self.randomseries.periods_in_a_year:.13f}"

    def test_yearfrac(self: TestOpenTimeSeries) -> None:
        """Test yearfrac property."""
        assert f"{9.9931553730322:.13f}" == f"{self.randomseries.yearfrac:.13f}"
        all_prop = self.random_properties["yearfrac"]
        assert f"{all_prop:.13f}" == f"{self.randomseries.yearfrac:.13f}"

    def test_resample(self: TestOpenTimeSeries) -> None:
        """Test resample method."""
        rs_series = self.randomseries.from_deepcopy()
        intended_length: int = 121

        before = rs_series.value_ret

        rs_series.resample(freq="BM")

        assert rs_series.length == intended_length
        assert before == rs_series.value_ret

    def test_resample_to_business_period_ends(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test resample_to_business_period_ends method."""
        rsb_stubs_series = OpenTimeSeries.from_fixed_rate(
            rate=0.01,
            days=121,
            end_dt=dt.date(2023, 5, 15),
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
            rate=0.01,
            days=88,
            end_dt=dt.date(2023, 4, 28),
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

    def test_calc_range(self: TestOpenTimeSeries) -> None:
        """Test calc_range method."""
        cseries = self.randomseries.from_deepcopy()
        start, end = cseries.first_idx.strftime("%Y-%m-%d"), cseries.last_idx.strftime(
            "%Y-%m-%d",
        )

        rst, ren = cseries.calc_range()

        self.assertListEqual(
            [start, end],
            [rst.strftime("%Y-%m-%d"), ren.strftime("%Y-%m-%d")],
        )

        with pytest.raises(
            expected_exception=ValueError,
            match="Function calc_range returned earlier date < series start",
        ):
            _, _ = cseries.calc_range(months_offset=125)

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt date < series start",
        ):
            _, _ = cseries.calc_range(from_dt=dt.date(2009, 5, 31))

        with pytest.raises(
            expected_exception=ValueError,
            match="Given to_dt date > series end",
        ):
            _, _ = cseries.calc_range(to_dt=dt.date(2019, 7, 31))

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt or to_dt dates outside series range",
        ):
            _, _ = cseries.calc_range(
                from_dt=dt.date(2009, 5, 31),
                to_dt=dt.date(2019, 7, 31),
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt or to_dt dates outside series range",
        ):
            _, _ = cseries.calc_range(
                from_dt=dt.date(2009, 7, 31),
                to_dt=dt.date(2019, 7, 31),
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt or to_dt dates outside series range",
        ):
            _, _ = cseries.calc_range(
                from_dt=dt.date(2009, 5, 31),
                to_dt=dt.date(2019, 5, 31),
            )

        nst, nen = cseries.calc_range(
            from_dt=dt.date(2009, 7, 3),
            to_dt=dt.date(2019, 6, 25),
        )
        assert nst == dt.date(2009, 7, 3)
        assert nen == dt.date(2019, 6, 25)

        cseries.resample()

        earlier_moved, _ = cseries.calc_range(from_dt=dt.date(2009, 8, 10))
        assert earlier_moved == dt.date(2009, 7, 31)

        _, later_moved = cseries.calc_range(to_dt=dt.date(2009, 8, 20))
        assert later_moved == dt.date(2009, 8, 31)

    def test_calc_range_output(self: TestOpenTimeSeries) -> None:
        """Test output consistency after calc_range applied."""
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

        cseries.model_config.update({"validate_assignment": False})
        cseries.dates = cseries.dates[-1008:]
        cseries.values = list(cseries.values)[-1008:]
        cseries.model_config.update({"validate_assignment": True})
        cseries.pandas_df()
        cseries.set_new_label(lvl_one=ValueType.RTRN)
        cseries.to_cumret()

        gr_1 = cseries.vol

        assert f"{gr_0:.13f}" == f"{gr_1:.13f}"

    def test_value_to_diff(self: TestOpenTimeSeries) -> None:
        """Test value_to_diff method."""
        diffseries = self.randomseries.from_deepcopy()
        diffseries.value_to_diff()
        are_bes = [f"{nn[0]:.12f}" for nn in diffseries.tsdf.to_numpy()[:15]]
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

    def test_value_to_ret(self: TestOpenTimeSeries) -> None:
        """Test value_to_ret method."""
        retseries = self.randomseries.from_deepcopy()
        retseries.value_to_ret()
        are_bes = [f"{nn[0]:.12f}" for nn in retseries.tsdf.to_numpy()[:15]]
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

    def test_valute_to_log(self: TestOpenTimeSeries) -> None:
        """Test value_to_log method."""
        logseries = self.randomseries.from_deepcopy()
        logseries.value_to_log()
        are_log = [f"{nn[0]:.12f}" for nn in logseries.tsdf.to_numpy()[:15]]
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

    def test_all_properties(self: TestOpenTimeSeries) -> None:
        """Test all_properties method."""
        prop_index = [
            "vol",
            "last_idx",
            "geo_ret",
            "first_idx",
            "max_drawdown",
            "periods_in_a_year",
            "z_score",
            "downside_deviation",
            "worst",
            "value_ret",
            "ret_vol_ratio",
            "worst_month",
            "max_drawdown_date",
            "arithmetic_ret",
            "skew",
            "cvar_down",
            "sortino_ratio",
            "positive_share",
            "kurtosis",
            "vol_from_var",
            "max_drawdown_cal_year",
            "yearfrac",
            "var_down",
            "length",
            "span_of_days",
        ]
        apseries = self.randomseries.from_deepcopy()
        apseries.to_cumret()
        result = apseries.all_properties()
        result_index = result.index.tolist()
        assert set(prop_index) == set(result_index)
        result_values = {}
        for value in result.index:
            if isinstance(result.loc[value, ("Asset", ValueType.PRICE)], float):
                result_values[
                    value
                ] = f"{result.loc[value, ('Asset', ValueType.PRICE)]:.10f}"
            elif isinstance(result.loc[value, ("Asset", ValueType.PRICE)], int):
                result_values[value] = result.loc[value, ("Asset", ValueType.PRICE)]
            elif isinstance(result.loc[value, ("Asset", ValueType.PRICE)], dt.date):
                result_values[value] = result.loc[
                    value,
                    ("Asset", ValueType.PRICE),
                ].strftime("%Y-%m-%d")
            else:
                msg = f"all_properties returned unexpected type {type(value)}"
                raise TypeError(
                    msg,
                )
        expected_values = {
            "positive_share": "0.4994026284",
            "vol": "0.1169534915",
            "worst_month": "-0.1916564407",
            "ret_vol_ratio": "0.0814866231",
            "last_idx": "2019-06-28",
            "max_drawdown": "-0.4001162541",
            "z_score": "1.2119535054",
            "geo_ret": "0.0024223168",
            "span_of_days": 3650,
            "yearfrac": "9.9931553730",
            "length": 2512,
            "first_idx": "2009-06-30",
            "value_ret": "0.0244719580",
            "skew": "-6.9467990606",
            "kurtosis": "180.6335718351",
            "max_drawdown_date": "2018-11-08",
            "var_down": "-0.0105912961",
            "max_drawdown_cal_year": "-0.2381116780",
            "sortino_ratio": "0.1036366417",
            "vol_from_var": "0.1020893290",
            "periods_in_a_year": "251.3720547945",
            "arithmetic_ret": "0.0095301451",
            "worst": "-0.1917423233",
            "downside_deviation": "0.0919572936",
            "cvar_down": "-0.0140207727",
        }
        self.assertDictEqual(result_values, expected_values)

        props = apseries.all_properties(properties=["geo_ret", "vol"])
        assert isinstance(props, DataFrame)

        with pytest.raises(expected_exception=ValueError, match="Invalid string: boo"):
            _ = apseries.all_properties(
                cast(list[LiteralSeriesProps], ["geo_ret", "boo"]),
            )

    def test_all_calc_properties(self: TestOpenTimeSeries) -> None:
        """Test all calculated properties."""
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
        for c_key, c_value in checks.items():
            assert (
                c_value == f"{getattr(self.randomseries, c_key):.11f}"
            ), f"Difference in: {c_key}"
            assert (
                f"{self.random_properties[c_key]:.11f}"
                == f"{getattr(self.randomseries, c_key):.11f}"
            ), f"Difference in: {c_key}"

    def test_all_calc_functions(self: TestOpenTimeSeries) -> None:
        """Test all calculation methods."""
        excel_geo_ret = (1.02447195802235 / 1.0102975774591) ** (
            1 / ((dt.date(2019, 6, 28) - dt.date(2015, 6, 26)).days / 365.25)
        ) - 1
        checks = {
            "arithmetic_ret_func": f"{0.00885255100:.11f}",
            "cvar_down_func": f"{-0.01331889836:.11f}",
            "downside_deviation_func": f"{0.07335125856:.11f}",
            "geo_ret_func": f"{excel_geo_ret:.11f}",
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
        for c_key, c_value in checks.items():
            assert (
                c_value
                == f"{getattr(self.randomseries, c_key)(months_from_last=48):.11f}"
            ), f"Difference in {c_key}"

        func = "value_ret_calendar_period"
        assert (
            f"{0.076502833914:.12f}"
            == f"{getattr(self.randomseries, func)(year=2019):.12f}"
        )

    def test_max_drawdown_date(self: TestOpenTimeSeries) -> None:
        """Test max_drawdown_date property."""
        assert dt.date(2018, 11, 8) == self.randomseries.max_drawdown_date
        all_prop = self.random_properties["max_drawdown_date"]
        assert all_prop == self.randomseries.max_drawdown_date

    def test_running_adjustment(self: TestOpenTimeSeries) -> None:
        """Test running_adjustment method."""
        adjustedseries = self.randomseries.from_deepcopy()
        adjustedseries.running_adjustment(0.05)

        assert (
            f"{1.689055852583:.12f}"
            == f"{float(adjustedseries.tsdf.iloc[-1, 0]):.12f}"
        )
        adjustedseries_returns = self.randomseries.from_deepcopy()
        adjustedseries_returns.value_to_ret()
        adjustedseries_returns.running_adjustment(0.05)

        assert (
            f"{0.009114963334:.12f}"
            == f"{float(adjustedseries_returns.tsdf.iloc[-1, 0]):.12f}"
        )

        adjustedseries_returns.to_cumret()
        assert (
            f"{float(adjustedseries.tsdf.iloc[-1, 0]):.12f}"
            == f"{float(adjustedseries_returns.tsdf.iloc[-1, 0]):.12f}"
        )

    def test_timeseries_chain(self: TestOpenTimeSeries) -> None:
        """Test timeseries_chain function."""
        full_series = self.randomseries.from_deepcopy()
        full_values = [f"{nn:.10f}" for nn in full_series.tsdf.iloc[:, 0].tolist()]

        front_series = OpenTimeSeries.from_df(full_series.tsdf.iloc[:126])

        back_series = OpenTimeSeries.from_df(
            full_series.tsdf.iloc[
                full_series.tsdf.index.get_loc(front_series.last_idx) :
            ],
        )
        full_series.tsdf.index.get_loc(front_series.last_idx)
        chained_series = timeseries_chain(front_series, back_series)
        chained_values = [f"{nn:.10f}" for nn in list(chained_series.values)]

        self.assertListEqual(full_series.dates, chained_series.dates)
        self.assertListEqual(full_values, chained_values)

        pushed_date = front_series.last_idx + dt.timedelta(days=10)
        no_overlap_series = OpenTimeSeries.from_df(
            full_series.tsdf.loc[cast(int, pushed_date) :],
        )
        with pytest.raises(
            expected_exception=ValueError,
            match="Timeseries dates must overlap to allow them to be chained.",
        ):
            _ = timeseries_chain(front_series, no_overlap_series)

        front_series_two = OpenTimeSeries.from_df(full_series.tsdf.iloc[:136])
        front_series_two.resample(freq="8D")

        assert back_series.first_idx not in front_series_two.tsdf.index
        new_chained_series = timeseries_chain(front_series_two, back_series)
        assert isinstance(new_chained_series, OpenTimeSeries)

        front_series_three = OpenTimeSeries.from_df(full_series.tsdf.iloc[:136])
        front_series_three.resample(freq="10D")

        assert back_series.first_idx not in front_series_three.tsdf.index

        with pytest.raises(
            expected_exception=ValueError,
            match="Failed to find a matching date between series",
        ):
            _ = timeseries_chain(front_series_three, back_series)

    def test_timeserieschain_newclass(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test correct pass-through of classes in timeseries_chain."""
        base_series_one = self.randomseries.from_deepcopy()

        sub_series_one = NewTimeSeries.from_arrays(
            name="sub_series_one",
            dates=base_series_one.dates,
            values=list(base_series_one.tsdf.iloc[:, 0].to_numpy()),
        )
        base_series_two = OpenTimeSeries.from_arrays(
            name="base_series_two",
            dates=[
                "2019-06-28",
                "2019-06-29",
                "2019-06-30",
            ],
            values=[
                1.0,
                1.009,
                1.011,
            ],
        )
        sub_series_two = NewTimeSeries.from_arrays(
            name="sub_series_two",
            dates=[
                "2019-06-28",
                "2019-06-29",
                "2019-06-30",
            ],
            values=[
                1.0,
                1.009,
                1.011,
            ],
        )
        assert sub_series_one.extra_info == "cool"
        new_base = timeseries_chain(front=base_series_one, back=base_series_two)
        new_sub = timeseries_chain(front=sub_series_one, back=sub_series_two)

        assert isinstance(new_base, OpenTimeSeries)
        assert isinstance(new_sub, NewTimeSeries)

        with pytest.raises(AssertionError):
            assert isinstance(new_base, NewTimeSeries)

        with pytest.raises(AssertionError):
            assert new_sub.__class__.__subclasscheck__(OpenTimeSeries)

        self.assertListEqual(list1=new_base.dates, list2=new_sub.dates)
        self.assertListEqual(list1=new_base.values, list2=new_sub.values)

    def test_chained_methods_newclass(self: TestOpenTimeSeries) -> None:
        """Test that chained methods on subclass returns subclass and not baseclass."""
        cseries = self.randomseries.from_deepcopy()
        assert isinstance(cseries, OpenTimeSeries)

        copyseries = NewTimeSeries.from_arrays(
            name="moo",
            dates=cseries.dates,
            values=list(cseries.tsdf.iloc[:, 0].to_numpy()),
        )
        assert isinstance(copyseries, NewTimeSeries)

        copyseries.set_new_label("boo").running_adjustment(0.001).resample(
            "BM",
        ).value_to_ret()
        assert isinstance(copyseries, NewTimeSeries)

    def test_plot_series(self: TestOpenTimeSeries) -> None:
        """Test plot_series method."""
        plotseries = self.randomseries.from_deepcopy()
        rawdata = [f"{x:.11f}" for x in plotseries.tsdf.iloc[1:5, 0]]

        fig, _ = plotseries.plot_series(auto_open=False, output_type="div")
        fig_json = loads(fig.to_json())
        fig_data = [f"{x:.11f}" for x in fig_json["data"][0]["y"][1:5]]

        self.assertListEqual(rawdata, fig_data)

        fig_last, _ = plotseries.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
        )
        fig_last_json = loads(fig_last.to_json())
        last = fig_last_json["data"][-1]["y"][0]
        assert f"{last:.12f}" == "1.024471958022"

        fig_last_fmt, _ = plotseries.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
            tick_fmt=".3%",
        )
        fig_last_fmt_json = loads(fig_last_fmt.to_json())
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]
        assert last_fmt == "Last 102.447%"

    def test_plot_bars(self: TestOpenTimeSeries) -> None:
        """Test plot_bars method."""
        barseries = self.randomseries.from_deepcopy()
        barseries.resample(freq="BM").value_to_ret()
        rawdata = [f"{x:.11f}" for x in barseries.tsdf.iloc[1:5, 0]]

        fig, _ = barseries.plot_bars(auto_open=False, output_type="div")
        fig_json = loads(fig.to_json())
        fig_data = [f"{x:.11f}" for x in fig_json["data"][0]["y"][1:5]]

        self.assertListEqual(rawdata, fig_data)

    def test_drawdown_details(self: TestOpenTimeSeries) -> None:
        """Test drawdown_details method."""
        details = self.randomseries.drawdown_details()
        assert f"{details.loc['Max Drawdown', 'Drawdown details']:7f}" == "-0.400116"
        assert details.loc["Start of drawdown", "Drawdown details"] == dt.date(
            2012,
            7,
            5,
        )
        assert details.loc["Date of bottom", "Drawdown details"] == dt.date(
            2018,
            11,
            8,
        )
        assert details.loc["Days from start to bottom", "Drawdown details"] == 2317
        assert (
            f"{details.loc['Average fall per day', 'Drawdown details']:.9f}"
            == "-0.000172687"
        )

    def test_align_index_to_local_cdays(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test align_index_to_local_cdays method."""
        d_range = [d.date() for d in date_range(start="2020-06-15", end="2020-06-25")]
        asim = [1.0] * len(d_range)
        adf = DataFrame(
            data=asim,
            index=d_range,
            columns=[["Asset"], [ValueType.PRICE]],
        )
        aseries = OpenTimeSeries.from_df(adf, valuetype=ValueType.PRICE)

        midsummer = dt.date(2020, 6, 19)
        assert midsummer in d_range

        aseries.align_index_to_local_cdays()
        assert midsummer not in aseries.tsdf.index

    def test_ewma_vol_func(self: TestOpenTimeSeries) -> None:
        """Test ewma_vol_func method."""
        simdata = self.randomseries.ewma_vol_func()
        simseries = OpenTimeSeries.from_df(simdata, valuetype=ValueType.PRICE)
        values = [f"{v:.11f}" for v in simdata.iloc[:5]]
        checkdata = [
            "0.07995872621",
            "0.07801248670",
            "0.07634125583",
            "0.07552465738",
            "0.07894138379",
        ]

        self.assertListEqual(values, checkdata)
        assert isinstance(simseries, OpenTimeSeries)

        simdata_fxd_per_yr = self.randomseries.ewma_vol_func(
            periods_in_a_year_fixed=251,
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

    def test_downside_deviation(self: TestOpenTimeSeries) -> None:
        """
        Test downside_deviation_func method.

        Source: https://www.investopedia.com/terms/d/downside-deviation.asp.
        """
        dd_asset = OpenTimeSeries.from_arrays(
            name="asset",
            valuetype=ValueType.RTRN,
            baseccy="USD",
            dates=[
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
            values=[
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
        ).to_cumret()

        mar = 0.01
        downdev = dd_asset.downside_deviation_func(
            min_accepted_return=mar,
            periods_in_a_year_fixed=1,
        )

        assert f"{downdev:.12f}" == "0.043333333333"

    def test_validations(self: TestOpenTimeSeries) -> None:
        """Test input validations."""
        valid_instrument_id = "58135911b239b413482758c9"
        invalid_instrument_id_one = "58135911b239b413482758c"
        invalid_instrument_id_two = "5_135911b239b413482758c9"
        valid_timeseries_id = "5813595971051506189ba416"
        invalid_timeseries_id_one = "5813595971051506189ba41"
        invalid_timeseries_id_two = "5_13595971051506189ba416"

        basecase = OpenTimeSeries.from_arrays(
            name="asset",
            dates=["2017-05-29"],
            values=[100.0],
        )
        self.assertListEqual(basecase.dates, ["2017-05-29"])
        self.assertListEqual(basecase.values, [100.0])

        with pytest.raises(
            expected_exception=ValueError,
            match="Shape of passed values is",
        ):
            OpenTimeSeries.from_arrays(
                name="asset",
                timeseries_id=valid_timeseries_id,
                instrument_id=valid_instrument_id,
                dates=[],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="There must be at least 1 value",
        ):
            OpenTimeSeries.from_arrays(
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[],
            )

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="String should match pattern",
        ):
            OpenTimeSeries.from_arrays(
                timeseries_id=invalid_timeseries_id_one,
                instrument_id=valid_instrument_id,
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                name="asset",
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="String should match pattern",
        ):
            OpenTimeSeries.from_arrays(
                timeseries_id=invalid_timeseries_id_two,
                instrument_id=valid_instrument_id,
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="String should match pattern",
        ):
            OpenTimeSeries.from_arrays(
                timeseries_id=valid_timeseries_id,
                instrument_id=invalid_instrument_id_one,
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="String should match pattern",
        ):
            OpenTimeSeries.from_arrays(
                timeseries_id=valid_timeseries_id,
                instrument_id=invalid_instrument_id_two,
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=PydanticValidationError,
            match="Dates are not unique",
        ):
            OpenTimeSeries.from_arrays(
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-29",
                ],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Shape of passed values is",
        ):
            OpenTimeSeries.from_arrays(
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[
                    100.0,
                    100.0978,
                    100.2821,
                ],
            )

    def test_from_1d_rate_to_cumret(self: TestOpenTimeSeries) -> None:
        """Test from_1d_rate_to_cumret method."""
        tms = OpenTimeSeries.from_arrays(
            name="asset",
            valuetype=ValueType.RTRN,
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
        )
        ave_rate = f"{tms.tsdf.mean().iloc[0]:.5f}"
        assert ave_rate == "0.02434"

        tms.from_1d_rate_to_cumret()

        val_ret = f"{tms.value_ret:.5f}"
        assert val_ret == "0.00093"

    def test_geo_ret_value_ret_exceptions(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test georet property raising exceptions on bad input data."""
        geoseries = OpenTimeSeries.from_arrays(
            name="asset",
            dates=["2022-07-01", "2023-07-01"],
            values=[
                1.0,
                1.1,
            ],
        )
        assert f"{geoseries.geo_ret:.7f}" == "0.1000718"
        assert f"{geoseries.geo_ret_func():.7f}" == "0.1000718"

        zeroseries = OpenTimeSeries.from_arrays(
            name="asset",
            dates=["2022-07-01", "2023-07-01"],
            values=[
                0.0,
                1.1,
            ],
        )
        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = zeroseries.geo_ret

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = zeroseries.geo_ret_func()

        with pytest.raises(
            expected_exception=ValueError,
            match="Simple return cannot be calculated due to an",
        ):
            _ = zeroseries.value_ret

        with pytest.raises(
            expected_exception=ValueError,
            match="Simple return cannot be calculated due to an",
        ):
            _ = zeroseries.value_ret_func()

        negseries = OpenTimeSeries.from_arrays(
            name="asset",
            dates=["2022-07-01", "2023-07-01"],
            values=[
                1.0,
                -0.1,
            ],
        )

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = negseries.geo_ret

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = negseries.geo_ret_func()

    def test_miscellaneous(self: TestOpenTimeSeries) -> None:
        """Test miscellaneous methods."""
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
        assert f"{impvol:.12f}" == "0.102089329036"
        impvoldrifted = mseries.vol_from_var_func(drift_adjust=True)
        assert f"{impvoldrifted:.12f}" == "0.102454621604"

    def test_value_ret_calendar_period(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test value_ret_calendar_period method."""
        vrcseries = self.randomseries.from_deepcopy()

        vrfs_y = vrcseries.value_ret_func(
            from_date=dt.date(2017, 12, 29),
            to_date=dt.date(2018, 12, 28),
        )
        vrvrcs_y = vrcseries.value_ret_calendar_period(year=2018)
        assert f"{vrfs_y:.11f}" == f"{vrvrcs_y:.11f}"

        vrfs_ym = vrcseries.value_ret_func(
            from_date=dt.date(2018, 4, 30),
            to_date=dt.date(2018, 5, 31),
        )
        vrvrcs_ym = vrcseries.value_ret_calendar_period(year=2018, month=5)
        assert f"{vrfs_ym:.11f}" == f"{vrvrcs_ym:.11f}"

    def test_to_drawdown_series(self: TestOpenTimeSeries) -> None:
        """Test to_drawdown_series method."""
        mseries = self.randomseries.from_deepcopy()
        ddvalue = mseries.max_drawdown
        mseries.to_drawdown_series()
        ddserievalue = float((mseries.tsdf.min()).iloc[0])
        assert f"{ddvalue:.11f}" == f"{ddserievalue:.11f}"

    def test_set_new_label(self: TestOpenTimeSeries) -> None:
        """Test set_new_label method."""
        lseries = self.randomseries.from_deepcopy()

        self.assertTupleEqual(lseries.tsdf.columns[0], ("Asset", ValueType.PRICE))

        lseries.set_new_label(lvl_zero="zero")
        self.assertTupleEqual(lseries.tsdf.columns[0], ("zero", ValueType.PRICE))

        lseries.set_new_label(lvl_one=ValueType.RTRN)
        self.assertTupleEqual(lseries.tsdf.columns[0], ("zero", ValueType.RTRN))

        lseries.set_new_label(lvl_zero="two", lvl_one=ValueType.PRICE)
        self.assertTupleEqual(lseries.tsdf.columns[0], ("two", ValueType.PRICE))

        lseries.set_new_label(delete_lvl_one=True)
        assert lseries.tsdf.columns[0] == "two"
