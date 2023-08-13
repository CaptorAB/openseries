"""
Defining the OpenTimeSeries class
"""
from __future__ import annotations
import datetime as dt
from os import path
from typing import cast, Optional, Tuple, Union
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from pandas import DataFrame, Series

from openseries.datefixer import date_offset_foll


def save_to_xlsx(
    data: DataFrame,
    filename: str,
    sheet_title: Optional[str] = None,
    directory: Optional[str] = None,
) -> str:
    """Saves the data in the .tsdf DataFrame to an Excel spreadsheet file

    Parameters
    ----------
    data: pandas.DataFrame
        The data to be saved to an Excel file
    filename: str
        Filename that should include .xlsx
    sheet_title: str, optional
        Name of the sheet in the Excel file
    directory: str, optional
        The file directory where the Excel file is saved.
    Returns
    -------
    str
        The Excel file path
    """

    if filename[-5:].lower() != ".xlsx":
        raise NameError("Filename must end with .xlsx")
    if directory:
        sheetfile = path.join(directory, filename)
    else:
        script_path = path.abspath(__file__)
        sheetfile = path.join(path.dirname(script_path), filename)

    wrkbook = Workbook()
    wrksheet = wrkbook.active

    if sheet_title:
        wrksheet.title = sheet_title

    for row in dataframe_to_rows(df=data, index=True, header=True):
        wrksheet.append(row)

    wrkbook.save(sheetfile)

    return sheetfile


def get_calc_range(
    data: DataFrame,
    months_offset: Optional[int] = None,
    from_dt: Optional[dt.date] = None,
    to_dt: Optional[dt.date] = None,
) -> Tuple[dt.date, dt.date]:
    """Creates user defined date range

    Parameters
    ----------
    data: pandas.DataFrame
        The data with the date range
    months_offset: int, optional
        Number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_dt: datetime.date, optional
        Specific from date
    to_dt: datetime.date, optional
        Specific from date

    Returns
    -------
    Tuple[datetime.date, datetime.date]
        Start and end date of the chosen date range
    """
    earlier, later = data.index[0], data.index[-1]
    if months_offset is not None or from_dt is not None or to_dt is not None:
        if months_offset is not None:
            earlier = date_offset_foll(
                raw_date=data.index[-1],
                months_offset=-months_offset,
                adjust=False,
                following=True,
            )
            assert (
                earlier >= data.index[0]
            ), "Function calc_range returned earlier date < series start"
            later = data.index[-1]
        else:
            if from_dt is not None and to_dt is None:
                assert (
                    from_dt >= data.index[0]
                ), "Function calc_range returned earlier date < series start"
                earlier, later = from_dt, data.index[-1]
            elif from_dt is None and to_dt is not None:
                assert (
                    to_dt <= data.index[-1]
                ), "Function calc_range returned later date > series end"
                earlier, later = data.index[0], to_dt
            elif from_dt is not None and to_dt is not None:
                assert (
                    to_dt <= data.index[-1] and from_dt >= data.index[0]
                ), "Function calc_range returned dates outside series range"
                earlier, later = from_dt, to_dt
        while earlier not in data.index.tolist():
            earlier -= dt.timedelta(days=1)
        while later not in data.index.tolist():
            later += dt.timedelta(days=1)

    return earlier, later


def calc_geo_ret(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
) -> Union[float, Series]:
    """https://www.investopedia.com/terms/c/cagr.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date

    Returns
    -------
    Union[float, Pandas.Series]
        Compounded Annual Growth Rate (CAGR)
    """
    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    fraction = (later - earlier).days / 365.25

    if (
        0.0 in data.loc[earlier].tolist()
        or data.loc[[earlier, later]].lt(0.0).any().any()
    ):
        raise ValueError(
            "Geometric return cannot be calculated due to an initial "
            "value being zero or a negative value."
        )

    result = (data.iloc[-1] / data.iloc[0]) ** (1 / fraction) - 1

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        name="Geometric return",
        dtype="float64",
    )


def calc_arithmetic_ret(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    periods_in_a_year_fixed: Optional[int] = None,
) -> Union[float, Series]:
    """https://www.investopedia.com/terms/a/arithmeticmean.asp

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date
    periods_in_a_year_fixed : int, optional
        Allows locking the periods-in-a-year to simplify test cases and
        comparisons

    Returns
    -------
    Union[float, Pandas.Series]
        Annualized arithmetic mean of returns
    """

    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    if periods_in_a_year_fixed:
        time_factor = float(periods_in_a_year_fixed)
    else:
        fraction = (later - earlier).days / 365.25
        how_many = data.loc[
            cast(int, earlier) : cast(int, later), data.columns.values[0]
        ].count()
        time_factor = how_many / fraction

    result = (
        data.loc[cast(int, earlier) : cast(int, later)].pct_change().mean()
        * time_factor
    )

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        name="Arithmetic return",
        dtype="float64",
    )


def calc_value_ret(
    data: DataFrame,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
) -> Union[float, Series]:
    """
    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    months_from_last : int, optional
        number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_date : datetime.date, optional
        Specific from date
    to_date : datetime.date, optional
        Specific to date

    Returns
    -------
    Union[float, Pandas.Series]
        Simple return
    """

    earlier, later = get_calc_range(
        data=data, months_offset=months_from_last, from_dt=from_date, to_dt=to_date
    )
    if 0.0 in data.iloc[0].tolist():
        raise ValueError(
            f"Simple return cannot be calculated due to an "
            f"initial value being zero. ({data.head(3)})"
        )

    result = data.loc[later] / data.loc[earlier] - 1

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        name="Simple return",
        dtype="float64",
    )
