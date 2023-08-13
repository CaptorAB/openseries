"""
Defining common tool functions
"""
import datetime as dt
from os import path
from typing import Optional, Tuple, Union
from dateutil.relativedelta import relativedelta
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from pandas import concat, DataFrame, DatetimeIndex, Series

from openseries.datefixer import date_offset_foll
from openseries.types import (
    CountriesType,
    LiteralBizDayFreq,
    LiteralPandasResampleConvention,
)


def do_resample(data: DataFrame, freq: Union[LiteralBizDayFreq, str]) -> DataFrame:
    """Resamples timeseries data to a new frequency

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    freq: Union[LiteralBizDayFreq, str], default "BM"
        The date offset string that sets the resampled frequency
        Examples are "7D", "B", "M", "BM", "Q", "BQ", "A", "BA"

    Returns
    -------
    pandas.DataFrame
        The resampled data
    """
    data.index = DatetimeIndex(data.index)
    data = data.resample(freq).last()
    data.index = [d.date() for d in DatetimeIndex(data.index)]
    return data


def do_resample_to_business_period_ends(
    data: DataFrame,
    head: Series,
    tail: Series,
    freq: LiteralBizDayFreq,
    countries: CountriesType,
    convention: LiteralPandasResampleConvention,
) -> DatetimeIndex:
    """Resamples timeseries frequency to the business calendar
    month end dates of each period while leaving any stubs
    in place. Stubs will be aligned to the shortest stub

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    head: pandas:Series
        Data point at maximum first date of all series
    tail: pandas:Series
        Data point at minimum last date of all series
    freq: LiteralBizDayFreq
        The date offset string that sets the resampled frequency
    countries: CountriesType
        (List of) country code(s) according to ISO 3166-1 alpha-2
        to create a business day calendar used for date adjustments
    convention: LiteralPandasResampleConvention
        Controls whether to use the start or end of `rule`.

    Returns
    -------
    Pandas.DatetimeIndex
        A date range aligned to business period ends
    """

    head = head.to_frame().T
    tail = tail.to_frame().T
    data.index = DatetimeIndex(data.index)
    data = data.resample(rule=freq, convention=convention).last()
    data.drop(index=data.index[-1], inplace=True)
    data.index = [d.date() for d in DatetimeIndex(data.index)]

    if head.index[0] not in data.index:
        data = concat([data, head])

    if tail.index[0] not in data.index:
        data = concat([data, tail])

    data.sort_index(inplace=True)

    dates = DatetimeIndex(
        [data.index[0]]
        + [
            date_offset_foll(
                dt.date(d.year, d.month, 1)
                + relativedelta(months=1)
                - dt.timedelta(days=1),
                countries=countries,
                months_offset=0,
                adjust=True,
                following=False,
            )
            for d in data.index[1:-1]
        ]
        + [data.index[-1]]
    )
    dates = dates.drop_duplicates()
    return dates


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
