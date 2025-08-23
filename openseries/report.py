"""Functions related to HTML reports.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

# mypy: disable-error-code="assignment"
from __future__ import annotations

from inspect import stack
from logging import getLogger
from pathlib import Path
from secrets import choice
from string import ascii_letters
from typing import TYPE_CHECKING, cast
from warnings import catch_warnings, simplefilter

if TYPE_CHECKING:  # pragma: no cover
    from pandas import Series
    from plotly.graph_objs import Figure

    from .frame import OpenFrame
    from .owntypes import LiteralPlotlyJSlib, LiteralPlotlyOutput


from pandas import DataFrame, Series, Timestamp, concat
from plotly.io import to_html
from plotly.offline import plot
from plotly.subplots import make_subplots

from .load_plotly import load_plotly_dict
from .owntypes import (
    LiteralBizDayFreq,
    ValueType,
)

logger = getLogger(__name__)


__all__ = ["report_html"]


def calendar_period_returns(
    data: OpenFrame,
    freq: LiteralBizDayFreq = "BYE",
    *,
    relabel: bool = True,
) -> DataFrame:
    """Generate a table of returns with appropriate table labels.

    Parameters
    ----------
    data: OpenFrame
        The timeseries data
    freq: LiteralBizDayFreq
        The date offset string that sets the resampled frequency
    relabel: bool, default: True
        Whether to set new appropriate labels

    Returns:
    -------
    pandas.DataFrame
        The resulting data

    """
    copied = data.from_deepcopy()
    copied.resample_to_business_period_ends(freq=freq)
    copied.value_to_ret()
    cldr = copied.tsdf.iloc[1:].copy()
    if relabel:
        if freq.upper() == "BYE":
            cldr.index = [d.year for d in cldr.index]
        elif freq.upper() == "BQE":
            cldr.index = [
                Timestamp(d).to_period("Q").strftime("Q%q %Y") for d in cldr.index
            ]
        else:
            cldr.index = [d.strftime("%b %y") for d in cldr.index]

    return cldr  # type: ignore[no-any-return]


def report_html(
    data: OpenFrame,
    bar_freq: LiteralBizDayFreq = "BYE",
    filename: str | None = None,
    title: str | None = None,
    directory: Path | None = None,
    output_type: LiteralPlotlyOutput = "file",
    include_plotlyjs: LiteralPlotlyJSlib = "cdn",
    *,
    auto_open: bool = False,
    add_logo: bool = True,
    vertical_legend: bool = True,
) -> tuple[Figure, str]:
    """Generate a HTML report page with line and bar plots and a table.

    Parameters
    ----------
    data: OpenFrame
        The timeseries data
    bar_freq: LiteralBizDayFreq
        The date offset string that sets the bar plot frequency
    filename: str, optional
        Name of the Plotly html file
    title: str, optional
        The report page title
    directory: DirectoryPath, optional
        Directory where Plotly html file is saved
    output_type: LiteralPlotlyOutput, default: "file"
        Determines output type
    include_plotlyjs: LiteralPlotlyJSlib, default: "cdn"
        Determines how the plotly.js library is included in the output
    auto_open: bool, default: True
        Determines whether to open a browser window with the plot
    add_logo: bool, default: True
        If True a Captor logo is added to the plot
    vertical_legend: bool, default: True
        Determines whether to vertically align the legend's labels

    Returns:
    -------
    tuple[plotly.go.Figure, str]
        Plotly Figure and a div section or a html filename with location

    """
    data.trunc_frame().value_nan_handle().to_cumret()

    if data.yearfrac > 1.0:
        properties = [
            "geo_ret",
            "vol",
            "ret_vol_ratio",
            "sortino_ratio",
            "worst_month",
            "first_indices",
            "last_indices",
        ]
        labels_init = [
            "Return (CAGR)",
            "Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Worst Month",
            "Comparison Start",
            "Comparison End",
            "Jensen's Alpha",
            "Information Ratio",
            "Tracking Error (weekly)",
            "Capture Ratio (monthly)",
            "Index Beta (weekly)",
        ]
        labels_final = [
            "Return (CAGR)",
            "Year-to-Date",
            "Month-to-Date",
            "Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Jensen's Alpha",
            "Information Ratio",
            "Tracking Error (weekly)",
            "Index Beta (weekly)",
            "Capture Ratio (monthly)",
            "Worst Month",
            "Comparison Start",
            "Comparison End",
        ]
    else:
        properties = [
            "value_ret",
            "vol",
            "ret_vol_ratio",
            "sortino_ratio",
            "worst",
            "first_indices",
            "last_indices",
        ]
        labels_init = [
            "Return (simple)",
            "Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Worst Day",
            "Comparison Start",
            "Comparison End",
            "Jensen's Alpha",
            "Information Ratio",
            "Tracking Error (weekly)",
            "Index Beta (weekly)",
        ]
        labels_final = [
            "Return (simple)",
            "Year-to-Date",
            "Month-to-Date",
            "Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Jensen's Alpha",
            "Information Ratio",
            "Tracking Error (weekly)",
            "Index Beta (weekly)",
            "Worst Day",
            "Comparison Start",
            "Comparison End",
        ]

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"rowspan": 2, "type": "table"}],
            [{"type": "xy"}, None],
        ],
    )

    for item, lbl in enumerate(data.columns_lvl_zero):
        figure.add_scatter(
            x=data.tsdf.index,
            y=data.tsdf.iloc[:, item],
            hovertemplate="%{y:.2%}<br>%{x|%Y-%m-%d}",
            line={"width": 2.5, "dash": "solid"},
            mode="lines",
            name=lbl,
            showlegend=True,
            row=1,
            col=1,
        )

    quarter_of_year = 0.25
    if data.yearfrac < quarter_of_year:
        tmp = data.from_deepcopy()
        bdf = tmp.value_to_ret().tsdf.iloc[1:]
    else:
        bdf = calendar_period_returns(data, freq=bar_freq)

    for item in range(data.item_count):
        figure.add_bar(
            x=bdf.index,
            y=bdf.iloc[:, item],
            hovertemplate="%{y:.2%}<br>%{x}",
            name=bdf.iloc[:, item].name[0],  # type: ignore[index]
            showlegend=False,
            row=2,
            col=1,
        )

    formats = [
        "{:.2%}",
        "{:.2%}",
        "{:.2f}",
        "{:.2f}",
        "{:.2%}",
        "{:%Y-%m-%d}",
        "{:%Y-%m-%d}",
        "{:.2%}",
        "{:.2f}",
        "{:.2%}",
        "{:.2f}",
    ]

    # noinspection PyTypeChecker
    rpt_df = data.all_properties(properties=properties)  # type: ignore[arg-type]
    alpha_frame = data.from_deepcopy()
    alpha_frame.to_cumret()
    with catch_warnings():
        simplefilter("ignore")
        alphas: list[str | float] = [
            alpha_frame.jensen_alpha(
                asset=(aname, ValueType.PRICE),
                market=(alpha_frame.columns_lvl_zero[-1], ValueType.PRICE),
                riskfree_rate=0.0,
            )
            for aname in alpha_frame.columns_lvl_zero[:-1]
        ]
    alphas.append("")
    ar = DataFrame(data=alphas, index=data.tsdf.columns, columns=["Jensen's Alpha"]).T
    rpt_df = concat([rpt_df, ar])
    ir = data.info_ratio_func()
    ir.name = "Information Ratio"
    ir.iloc[-1] = None
    ir = ir.to_frame().T
    rpt_df = concat([rpt_df, ir])
    te_frame = data.from_deepcopy()
    te_frame.resample("7D")
    with catch_warnings():
        simplefilter("ignore")
        te = te_frame.tracking_error_func()
    if te.hasnans:
        te = Series(
            data=[""] * te_frame.item_count,
            index=te_frame.tsdf.columns,
            name="Tracking Error (weekly)",
        )
    else:
        te.iloc[-1] = None
        te.name = "Tracking Error (weekly)"
    te = te.to_frame().T
    rpt_df = concat([rpt_df, te])

    if data.yearfrac > 1.0:
        crm = data.from_deepcopy()
        crm.resample("ME")
        cru_save = Series(
            data=[""] * crm.item_count,
            index=crm.tsdf.columns,
            name="Capture Ratio (monthly)",
        )
        with catch_warnings():
            simplefilter("ignore")
            try:
                cru = crm.capture_ratio_func(ratio="both")
            except ZeroDivisionError as exc:  # pragma: no cover
                msg = f"Capture ratio calculation error: {exc!s}"  # pragma: no cover
                logger.warning(msg=msg)  # pragma: no cover
                cru = cru_save  # pragma: no cover
        if cru.hasnans:
            cru = cru_save
        else:
            cru.iloc[-1] = None
            cru.name = "Capture Ratio (monthly)"
        cru = cru.to_frame().T
        rpt_df = concat([rpt_df, cru])
        formats.append("{:.2f}")
    beta_frame = data.from_deepcopy()
    beta_frame.resample("7D").value_nan_handle("drop")
    beta_frame.to_cumret()
    betas: list[str | float] = [
        beta_frame.beta(
            asset=(bname, ValueType.PRICE),
            market=(beta_frame.columns_lvl_zero[-1], ValueType.PRICE),
        )
        for bname in beta_frame.columns_lvl_zero[:-1]
    ]
    # noinspection PyTypeChecker
    betas.append("")
    br = DataFrame(
        data=betas,
        index=data.tsdf.columns,
        columns=["Index Beta (weekly)"],
    ).T
    rpt_df = concat([rpt_df, br])

    for item, f in zip(rpt_df.index, formats, strict=False):
        rpt_df.loc[item] = rpt_df.loc[item].apply(
            lambda x, fmt=f: x if (isinstance(x, str) or x is None) else fmt.format(x),  # type: ignore[return-value]
        )

    rpt_df.index = labels_init

    this_year = data.last_idx.year
    this_month = data.last_idx.month
    ytd = cast("Series[float]", data.value_ret_calendar_period(year=this_year)).map(
        "{:.2%}".format
    )
    ytd.name = "Year-to-Date"
    mtd = cast(
        "Series[float]",
        data.value_ret_calendar_period(year=this_year, month=this_month),
    ).map(
        "{:.2%}".format,
    )
    mtd.name = "Month-to-Date"
    ytd = ytd.to_frame().T
    mtd = mtd.to_frame().T
    rpt_df = concat([rpt_df, ytd])
    rpt_df = concat([rpt_df, mtd])
    rpt_df = rpt_df.reindex(labels_final)

    rpt_df.index = [f"<b>{x}</b>" for x in rpt_df.index]
    rpt_df = rpt_df.reset_index()

    colmns = ["", *data.columns_lvl_zero]
    columns = [f"<b>{x}</b>" for x in colmns]
    aligning = ["left"] + ["center"] * (len(columns) - 1)

    col_even_color = "lightgrey"
    col_odd_color = "white"
    color_lst = ["grey"] + [col_odd_color] * (data.item_count - 1) + [col_even_color]

    tablevalues = rpt_df.transpose().to_numpy().tolist()
    cleanedtablevalues = list(tablevalues)[:-1]
    cleanedcol = [
        valu if valu not in ["nan", "nan%"] else "" for valu in tablevalues[-1]
    ]
    cleanedtablevalues.append(cleanedcol)

    figure.add_table(
        header={
            "values": columns,
            "align": "center",
            "fill_color": "grey",
            "font": {"color": "white"},
        },
        cells={
            "values": cleanedtablevalues,
            "align": aligning,
            "height": 25,
            "fill_color": color_lst,
            "font": {"color": ["white"] + ["black"] * len(columns)},
        },
        row=1,
        col=2,
    )

    if directory:
        dirpath = Path(directory).resolve()
    elif Path.home().joinpath("Documents").exists():
        dirpath = Path.home() / "Documents"
    else:
        dirpath = Path(stack()[1].filename).parent

    if not filename:
        filename = "".join(choice(ascii_letters) for _ in range(6)) + ".html"

    plotfile = dirpath / filename

    fig, logo = load_plotly_dict()

    if add_logo:
        figure.add_layout_image(logo)

    figure.update_layout(fig.get("layout"))
    colorway: list[str] = cast("dict[str, list[str]]", fig["layout"]).get("colorway")

    if vertical_legend:
        legend = {
            "yanchor": "bottom",
            "y": -0.04,
            "xanchor": "right",
            "x": 0.98,
            "orientation": "v",
        }
    else:
        legend = {
            "yanchor": "bottom",
            "y": -0.2,
            "xanchor": "right",
            "x": 0.98,
            "orientation": "h",
        }

    figure.update_layout(
        legend=legend,
        colorway=colorway[: data.item_count],
    )
    figure.update_xaxes(gridcolor="#EEEEEE", automargin=True, tickangle=-45)
    figure.update_yaxes(tickformat=".2%", gridcolor="#EEEEEE", automargin=True)

    if isinstance(title, str):
        figure.update_layout(
            {"title": {"text": f"<b>{title}</b><br>", "font": {"size": 36}}},
        )

    if output_type == "file":
        plot(
            figure_or_data=figure,
            filename=str(plotfile),
            auto_open=auto_open,
            auto_play=False,
            link_text="",
            include_plotlyjs=cast("bool", include_plotlyjs),
            output_type=output_type,
            config=fig["config"],
        )
        string_output = str(plotfile)
    else:
        div_id = filename.split(sep=".")[0]
        string_output = to_html(
            fig=figure,
            div_id=div_id,
            auto_play=False,
            full_html=False,
            include_plotlyjs=cast("bool", include_plotlyjs),
            config=fig["config"],
        )

    return figure, string_output
