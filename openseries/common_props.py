"""
Defining common properties
"""
import datetime as dt
from json import dump
from pathlib import Path
from random import choices
from string import ascii_letters
from os import path
from typing import Any, cast, Dict, List, Optional, Tuple, TypeVar, Union
from numpy import log
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from pandas import DataFrame, Series
from plotly.graph_objs import Figure
from plotly.offline import plot

from openseries.risk import drawdown_series
from openseries.types import (
    LiteralBarPlotMode,
    LiteralLinePlotMode,
    LiteralPlotlyOutput,
    LiteralNanMethod,
    ValueType,
)
from openseries.load_plotly import load_plotly_dict
from openseries.common_tools import get_calc_range

TypeCommonProps = TypeVar("TypeCommonProps", bound="CommonProps")


class CommonProps:
    """Common props declared"""

    tsdf: DataFrame

    @property
    def length(self: TypeCommonProps) -> int:
        """
        Returns
        -------
        int
            Number of observations
        """

        return len(self.tsdf.index)

    @property
    def first_idx(self: TypeCommonProps) -> dt.date:
        """
        Returns
        -------
        datetime.date
            The first date in the timeseries
        """

        return cast(dt.date, self.tsdf.index[0])

    @property
    def last_idx(self: TypeCommonProps) -> dt.date:
        """
        Returns
        -------
        datetime.date
            The last date in the timeseries
        """

        return cast(dt.date, self.tsdf.index[-1])

    @property
    def span_of_days(self: TypeCommonProps) -> int:
        """
        Returns
        -------
        int
            Number of days from the first date to the last
        """

        return (self.last_idx - self.first_idx).days

    @property
    def yearfrac(self: TypeCommonProps) -> float:
        """
        Returns
        -------
        float
            Length of the timeseries expressed in years assuming all years
            have 365.25 days
        """

        return self.span_of_days / 365.25

    @property
    def periods_in_a_year(self: TypeCommonProps) -> float:
        """
        Returns
        -------
        float
            The average number of observations per year
        """

        return self.length / self.yearfrac

    def value_to_log(self: TypeCommonProps) -> TypeCommonProps:
        """Converts a valueseries into logarithmic weighted series \n
        Equivalent to LN(value[t] / value[t=0]) in MS Excel

        Returns
        -------
        self
            An object of the same class
        """

        self.tsdf = DataFrame(
            data=log(self.tsdf / self.tsdf.iloc[0]),
            index=self.tsdf.index,
            columns=self.tsdf.columns,
        )
        return self

    def value_nan_handle(
        self: TypeCommonProps, method: LiteralNanMethod = "fill"
    ) -> TypeCommonProps:
        """Handling of missing values in a valueseries

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with last known or drop

        Returns
        -------
        self
            An object of the same class
        """
        if method == "fill":
            self.tsdf.fillna(method="pad", inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def return_nan_handle(
        self: TypeCommonProps, method: LiteralNanMethod = "fill"
    ) -> TypeCommonProps:
        """Handling of missing values in a returnseries

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with zero or drop

        Returns
        -------
        self
            An object of the same class
        """
        if method == "fill":
            self.tsdf.fillna(value=0.0, inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def to_drawdown_series(self: TypeCommonProps) -> TypeCommonProps:
        """Converts timeseries into a drawdown series

        Returns
        -------
        self
            An object of the same class
        """

        for serie in self.tsdf:
            self.tsdf.loc[:, serie] = drawdown_series(self.tsdf.loc[:, serie])
        return self

    def to_json(
        self: TypeCommonProps, filename: str, directory: Optional[str] = None
    ) -> List[Dict[str, Union[str, bool, ValueType, List[str], List[float]]]]:
        """Dumps timeseries data into a json file

        The label and tsdf parameters are deleted before the json file is saved

        Parameters
        ----------
        filename: str
            Filename including filetype
        directory: str, optional
            File folder location
        Returns
        -------
        List[Dict[str, Union[str, bool, ValueType, List[str], List[float]]]]
            A list of dictionaries with the raw original data of the series
        """
        if not directory:
            directory = path.dirname(path.abspath(__file__))

        cleaner_list = ["label", "tsdf"]
        data = self.__dict__

        output = []
        if "label" in data:
            for item in cleaner_list:
                data.pop(item)
            output.append(dict(data))
        else:
            series = [
                serie.__dict__ for serie in cast(List[Any], data.get("constituents"))
            ]
            for data in series:
                for item in cleaner_list:
                    data.pop(item)
                output.append(data)

        with open(path.join(directory, filename), "w", encoding="utf-8") as jsonfile:
            dump(data, jsonfile, indent=2, sort_keys=False)

        return output

    def to_xlsx(
        self: TypeCommonProps,
        filename: str,
        sheet_title: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> str:
        """Saves the data in the .tsdf DataFrame to an Excel spreadsheet file

        Parameters
        ----------
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

        for row in dataframe_to_rows(df=self.tsdf, index=True, header=True):
            wrksheet.append(row)

        wrkbook.save(sheetfile)

        return sheetfile

    def plot_bars(
        self: TypeCommonProps,
        mode: LiteralBarPlotMode = "group",
        tick_fmt: Optional[str] = None,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        labels: Optional[List[str]] = None,
        auto_open: bool = True,
        add_logo: bool = True,
        output_type: LiteralPlotlyOutput = "file",
    ) -> Tuple[Figure, str]:
        """Creates a Plotly Bar Figure

        Parameters
        ----------
        self.tsdf: pandas.DataFrame
            The timeseries self.tsdf
        mode: LiteralBarPlotMode
            The type of bar to use
        tick_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show
        filename: str, optional
            Name of the Plotly html file
        directory: str, optional
            Directory where Plotly html file is saved
        labels: List[str], optional
            A list of labels to manually override using the names of
            the input self.tsdf
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type

        Returns
        -------
        (plotly.go.Figure, str)
            Plotly Figure and html filename with location
        """
        if labels:
            assert (
                len(labels) == self.tsdf.shape[1]
            ), "Must provide same number of labels as items in frame."
        else:
            labels = list(self.tsdf.columns.get_level_values(0))
        if not directory:
            directory = path.join(str(Path.home()), "Documents")
        if not filename:
            filename = "".join(choices(ascii_letters, k=6)) + ".html"
        plotfile = path.join(path.abspath(directory), filename)

        if mode == "overlay":
            opacity = 0.7
        else:
            opacity = None

        fig, logo = load_plotly_dict()
        figure = Figure(fig)
        for item in range(self.tsdf.shape[1]):
            figure.add_bar(
                x=self.tsdf.index,
                y=self.tsdf.iloc[:, item],
                hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                name=labels[item],
                opacity=opacity,
            )
        figure.update_layout(barmode=mode, yaxis={"tickformat": tick_fmt})

        if add_logo:
            figure.add_layout_image(logo)

        plot(
            figure,
            filename=plotfile,
            auto_open=auto_open,
            link_text="",
            include_plotlyjs="cdn",
            config=fig["config"],
            output_type=output_type,
        )

        return figure, plotfile

    def plot_series(
        self: TypeCommonProps,
        mode: LiteralLinePlotMode = "lines",
        tick_fmt: Optional[str] = None,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        labels: Optional[List[str]] = None,
        auto_open: bool = True,
        add_logo: bool = True,
        show_last: bool = False,
        output_type: LiteralPlotlyOutput = "file",
    ) -> Tuple[Figure, str]:
        """Creates a Plotly Figure

        To scale the bubble size, use the attribute sizeref.
        We recommend using the following formula to calculate a sizeref value:
        sizeref = 2. * max(array of size values) / (desired maximum marker size ** 2)

        Parameters
        ----------
        self.tsdf: pandas.DataFrame
            The timeseries self.tsdf
        mode: LiteralLinePlotMode, default: "lines"
            The type of scatter to use
        tick_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show
        filename: str, optional
            Name of the Plotly html file
        directory: str, optional
            Directory where Plotly html file is saved
        labels: List[str], optional
            A list of labels to manually override using the names of
            the input self.tsdf
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot
        show_last: bool, default: False
            If True the last self.tsdf point is highlighted as red dot with a label
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type

        Returns
        -------
        (plotly.go.Figure, str)
            Plotly Figure and html filename with location
        """

        if labels:
            assert (
                len(labels) == self.tsdf.shape[1]
            ), "Must provide same number of labels as items in frame."
        else:
            labels = list(self.tsdf.columns.get_level_values(0))
        if not directory:
            directory = path.join(str(Path.home()), "Documents")
        if not filename:
            filename = "".join(choices(ascii_letters, k=6)) + ".html"
        plotfile = path.join(path.abspath(directory), filename)

        fig, logo = load_plotly_dict()
        figure = Figure(fig)
        for item in range(self.tsdf.shape[1]):
            figure.add_scatter(
                x=self.tsdf.index,
                y=self.tsdf.iloc[:, item],
                hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                line={"width": 2.5, "dash": "solid"},
                mode=mode,
                name=labels[item],
            )
        figure.update_layout(yaxis={"tickformat": tick_fmt})

        if add_logo:
            figure.add_layout_image(logo)

        if show_last is True:
            if tick_fmt:
                txt = f"Last {{:{tick_fmt}}}"
            else:
                txt = "Last {}"

            for item in range(self.tsdf.shape[1]):
                figure.add_scatter(
                    x=[self.tsdf.iloc[:, item].index[-1]],
                    y=[self.tsdf.iloc[-1, item]],
                    mode="markers + text",
                    marker={"color": "red", "size": 12},
                    hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                    showlegend=False,
                    name=labels[item],
                    text=[txt.format(self.tsdf.iloc[-1, item])],
                    textposition="top center",
                )

        plot(
            figure,
            filename=plotfile,
            auto_open=auto_open,
            link_text="",
            include_plotlyjs="cdn",
            config=fig["config"],
            output_type=output_type,
        )

        return figure, plotfile

    def z_score_func(
        self: TypeCommonProps,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series]:
        """https://www.investopedia.com/terms/z/zscore.asp

        Parameters
        ----------
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
            Z-score as (last return - mean return) / standard deviation of returns
        """

        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        zscframe = self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change()
        result = (zscframe.iloc[-1] - zscframe.mean()) / zscframe.std()

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Z-score",
            dtype="float64",
        )
