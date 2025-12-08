"""Functions related to HTML reports."""

from __future__ import annotations

import webbrowser
from inspect import stack
from logging import getLogger
from pathlib import Path
from secrets import choice
from string import ascii_letters
from typing import TYPE_CHECKING, cast
from warnings import catch_warnings, simplefilter

if TYPE_CHECKING:  # pragma: no cover
    from pandas import Series
    from plotly.graph_objs import Figure  # type: ignore[import-untyped]

    from .frame import OpenFrame
    from .owntypes import LiteralPlotlyJSlib, LiteralPlotlyOutput


from pandas import DataFrame, Index, Series, Timestamp, concat
from plotly.io import to_html  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from .load_plotly import load_plotly_dict
from .owntypes import (
    LiteralBizDayFreq,
    LiteralFrameProps,
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

    Args:
        data: The timeseries data.
        freq: The date offset string that sets the resampled frequency.
        relabel: Whether to set new appropriate labels. Defaults to True.

    Returns:
        The resulting data.

    """
    copied = data.from_deepcopy()
    copied.resample_to_business_period_ends(freq=freq)
    copied.value_to_ret()
    cldr = copied.tsdf.iloc[1:].copy()
    if relabel:
        if freq.upper() == "BYE":
            cldr.index = Index([d.year for d in cldr.index])
        elif freq.upper() == "BQE":
            cldr.index = Index(
                [Timestamp(d).to_period("Q").strftime("Q%q %Y") for d in cldr.index],
            )
        else:
            cldr.index = Index([d.strftime("%b %y") for d in cldr.index])

    return cldr


def _get_report_properties_and_labels(
    yearfrac: float,
) -> tuple[list[str], list[str], list[str]]:
    """Get properties and labels based on year fraction.

    Args:
        yearfrac: Year fraction to determine report type.

    Returns:
        Tuple of (properties, labels_init, labels_final).
    """
    if yearfrac > 1.0:
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

    return properties, labels_init, labels_final


def _add_jensen_alpha_to_report(
    copied: OpenFrame,
    rpt_df: DataFrame,
) -> DataFrame:
    """Add Jensen's Alpha to report DataFrame.

    Args:
        copied: Copied OpenFrame data.
        rpt_df: Report DataFrame to update.

    Returns:
        Updated report DataFrame.
    """
    alpha_frame = copied.from_deepcopy()
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
    ar = DataFrame(
        data=alphas,
        index=copied.tsdf.columns,
        columns=["Jensen's Alpha"],
    ).T
    return concat([rpt_df, ar])


def _add_tracking_error_to_report(
    copied: OpenFrame,
    rpt_df: DataFrame,
) -> DataFrame:
    """Add Tracking Error to report DataFrame.

    Args:
        copied: Copied OpenFrame data.
        rpt_df: Report DataFrame to update.

    Returns:
        Updated report DataFrame.
    """
    te_frame = copied.from_deepcopy()
    te_frame.resample("7D")
    with catch_warnings():
        simplefilter("ignore")
        te: Series[float] | Series[str] = te_frame.tracking_error_func()
    if te.hasnans:
        te = Series(
            data=[""] * te_frame.item_count,
            index=te_frame.tsdf.columns,
            name="Tracking Error (weekly)",
        )
    else:
        te.iloc[-1] = None
        te.name = "Tracking Error (weekly)"
    te_df = te.to_frame().T
    return concat([rpt_df, te_df])


def _add_capture_ratio_to_report(
    copied: OpenFrame,
    rpt_df: DataFrame,
    formats: list[str],
) -> tuple[DataFrame, list[str]]:
    """Add Capture Ratio to report DataFrame.

    Args:
        copied: Copied OpenFrame data.
        rpt_df: Report DataFrame to update.
        formats: Format strings list to update.

    Returns:
        Tuple of (updated report DataFrame, updated formats).
    """
    crm = copied.from_deepcopy()
    crm.resample("ME")
    cru_save = Series(
        data=[""] * crm.item_count,
        index=crm.tsdf.columns,
        name="Capture Ratio (monthly)",
    )
    with catch_warnings():
        simplefilter("ignore")
        try:
            cru: Series[float] | Series[str] = crm.capture_ratio_func(ratio="both")
        except ZeroDivisionError as exc:  # pragma: no cover
            msg = f"Capture ratio calculation error: {exc!s}"  # pragma: no cover
            logger.warning(msg)  # pragma: no cover
            cru = cru_save  # pragma: no cover
    if cru.hasnans:
        cru = cru_save
    else:
        cru.iloc[-1] = None
        cru.name = "Capture Ratio (monthly)"
    cru_df = cru.to_frame().T
    rpt_df = concat([rpt_df, cru_df])
    formats.append("{:.2f}")
    return rpt_df, formats


def _add_beta_to_report(
    copied: OpenFrame,
    rpt_df: DataFrame,
) -> DataFrame:
    """Add Index Beta to report DataFrame.

    Args:
        copied: Copied OpenFrame data.
        rpt_df: Report DataFrame to update.

    Returns:
        Updated report DataFrame.
    """
    beta_frame = copied.from_deepcopy()
    beta_frame.resample("7D").value_nan_handle("drop")
    beta_frame.to_cumret()
    betas: list[str | float] = [
        beta_frame.beta(
            asset=(bname, ValueType.PRICE),
            market=(beta_frame.columns_lvl_zero[-1], ValueType.PRICE),
        )
        for bname in beta_frame.columns_lvl_zero[:-1]
    ]
    betas.append("")
    br = DataFrame(
        data=betas,
        index=copied.tsdf.columns,
        columns=["Index Beta (weekly)"],
    ).T
    return concat([rpt_df, br])


def _build_report_dataframe(
    copied: OpenFrame,
    properties: list[str],
    formats: list[str],
    yearfrac: float,
) -> tuple[DataFrame, list[str]]:
    """Build the complete report DataFrame.

    Args:
        copied: Copied OpenFrame data.
        properties: List of properties to include.
        formats: Format strings for values.
        yearfrac: Year fraction to determine if capture ratio is included.

    Returns:
        Tuple of (report DataFrame, updated formats).
    """
    rpt_df = copied.all_properties(
        properties=cast("list[LiteralFrameProps]", properties),
    )

    rpt_df = _add_jensen_alpha_to_report(copied, rpt_df)

    ir = copied.info_ratio_func()
    ir.name = "Information Ratio"
    ir.iloc[-1] = None
    ir_df = ir.to_frame().T
    rpt_df = concat([rpt_df, ir_df])

    rpt_df = _add_tracking_error_to_report(copied, rpt_df)

    if yearfrac > 1.0:
        rpt_df, formats = _add_capture_ratio_to_report(copied, rpt_df, formats)

    rpt_df = _add_beta_to_report(copied, rpt_df)

    return rpt_df, formats


def _format_report_dataframe(
    rpt_df: DataFrame,
    formats: list[str],
    labels_init: list[str],
    labels_final: list[str],
    copied: OpenFrame,
) -> DataFrame:
    """Format the report DataFrame with labels and calendar period returns.

    Args:
        rpt_df: Report DataFrame to format.
        formats: Format strings for values.
        labels_init: Initial labels for rows.
        labels_final: Final labels for rows.
        copied: Copied OpenFrame data.

    Returns:
        Formatted report DataFrame.
    """
    for item, f in zip(rpt_df.index, formats, strict=False):
        rpt_df.loc[item] = rpt_df.loc[item].apply(
            lambda x, fmt=f: str(x)
            if (isinstance(x, str) or x is None)
            else fmt.format(x),
        )

    rpt_df.index = Index(labels_init)

    this_year = copied.last_idx.year
    this_month = copied.last_idx.month
    ytd = copied.value_ret_calendar_period(year=this_year).map("{:.2%}".format)
    ytd.name = "Year-to-Date"
    mtd = copied.value_ret_calendar_period(year=this_year, month=this_month).map(
        "{:.2%}".format,
    )
    mtd.name = "Month-to-Date"
    ytd_df = ytd.to_frame().T
    mtd_df = mtd.to_frame().T
    rpt_df = concat([rpt_df, ytd_df])
    rpt_df = concat([rpt_df, mtd_df])
    rpt_df = rpt_df.reindex(labels_final)

    rpt_df.index = Index([f"<b>{x}</b>" for x in rpt_df.index])
    return rpt_df.reset_index()


def _prepare_table_data(
    rpt_df: DataFrame,
    copied: OpenFrame,
) -> tuple[list[list[str]], list[str], list[str], list[str]]:
    """Prepare table data for Plotly table.

    Args:
        rpt_df: Formatted report DataFrame.
        copied: Copied OpenFrame data.

    Returns:
        Tuple of (cleaned table values, columns, aligning, color_lst).
    """
    colmns = ["", *copied.columns_lvl_zero]
    columns = [f"<b>{x}</b>" for x in colmns]
    aligning = ["left"] + ["center"] * (len(columns) - 1)

    col_even_color = "lightgrey"
    col_odd_color = "white"
    color_lst = ["grey"] + [col_odd_color] * (copied.item_count - 1) + [col_even_color]

    tablevalues = rpt_df.transpose().to_numpy().tolist()
    cleanedtablevalues = list(tablevalues)[:-1]
    cleanedcol = [
        valu if valu not in ["nan", "nan%"] else "" for valu in tablevalues[-1]
    ]
    cleanedtablevalues.append(cleanedcol)

    return cleanedtablevalues, columns, aligning, color_lst


def _configure_figure_layout(
    figure: Figure,
    copied: OpenFrame,
    *,
    add_logo: bool,
    vertical_legend: bool,
    title: str | None,
    mobile: bool = False,
    total_min_height: int | None = None,
    table_min_height: int | None = None,
) -> None:
    """Configure figure layout with logo, legend, and title.

    Args:
        figure: Plotly figure to configure.
        copied: Copied OpenFrame data.
        add_logo: Whether to add logo.
        vertical_legend: Whether to use vertical legend.
        title: Optional title for the figure.
        mobile: Whether this is a mobile layout. Defaults to False.
        total_min_height: Minimum total height for mobile layout in pixels.
        table_min_height: Minimum height for table subplot in pixels.
    """
    fig, logo = load_plotly_dict()

    if add_logo:
        figure.add_layout_image(logo)

    figure.update_layout(fig.get("layout"))
    colorway: list[str] = cast("dict[str, list[str]]", fig["layout"]).get(
        "colorway",
        [],
    )

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

    if mobile:
        if vertical_legend:
            legend = {
                "yanchor": "top",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
                "orientation": "h",
            }
        else:
            legend = {
                "yanchor": "top",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
                "orientation": "h",
            }

    layout_updates: dict[str, object] = {
        "legend": legend,
        "colorway": colorway[: copied.item_count],
        "autosize": True,
        "margin": {"l": 50, "r": 50, "t": 80, "b": 50, "pad": 10},
    }

    if mobile and total_min_height is not None:
        layout_updates["height"] = total_min_height
        layout_updates["autosize"] = False

    figure.update_layout(**layout_updates)

    if mobile:
        figure.update_xaxes(
            gridcolor="#EEEEEE",
            automargin=True,
            tickangle=-45,
            row=1,
            col=1,
        )
        figure.update_xaxes(
            gridcolor="#EEEEEE",
            automargin=True,
            tickangle=-45,
            row=2,
            col=1,
        )
        figure.update_yaxes(
            tickformat=".2%",
            gridcolor="#EEEEEE",
            automargin=True,
            row=1,
            col=1,
        )
        figure.update_yaxes(
            tickformat=".2%",
            gridcolor="#EEEEEE",
            automargin=True,
            row=2,
            col=1,
        )
        if table_min_height is not None and total_min_height is not None:
            plot_height = 400
            bar_height = 350
            spacing = 0.08
            plot_domain_top = 1.0
            plot_domain_bottom = 1.0 - (plot_height / total_min_height)
            bar_domain_top = plot_domain_bottom - spacing
            bar_domain_bottom = bar_domain_top - (bar_height / total_min_height)

            figure.update_layout(
                yaxis_domain=[plot_domain_bottom, plot_domain_top],
                yaxis2_domain=[bar_domain_bottom, bar_domain_top],
            )
        title_size = 24
    else:
        figure.update_xaxes(gridcolor="#EEEEEE", automargin=True, tickangle=-45)
        figure.update_yaxes(tickformat=".2%", gridcolor="#EEEEEE", automargin=True)
        title_size = 36

    if title:
        figure.update_layout(
            {"title": {"text": f"<b>{title}</b><br>", "font": {"size": title_size}}},
        )


def _get_bar_dataframe(
    copied: OpenFrame,
    bar_freq: LiteralBizDayFreq,
) -> DataFrame:
    """Get bar chart DataFrame based on year fraction.

    Args:
        copied: Copied OpenFrame data.
        bar_freq: The date offset string for bar plot frequency.

    Returns:
        Bar chart DataFrame.
    """
    quarter_of_year = 0.25
    if copied.yearfrac < quarter_of_year:
        tmp = copied.from_deepcopy()
        return tmp.value_to_ret().tsdf.iloc[1:]
    return calendar_period_returns(data=copied, freq=bar_freq)


def _build_desktop_figure(
    copied: OpenFrame,
    bdf: DataFrame,
    cleanedtablevalues: list[list[str]],
    columns: list[str],
    aligning: list[str],
    color_lst: list[str],
) -> Figure:
    """Build the desktop figure with plots and table.

    Args:
        copied: Copied OpenFrame data.
        bdf: Bar chart DataFrame.
        cleanedtablevalues: Table cell values.
        columns: Table column headers.
        aligning: Table cell alignment.
        color_lst: Table cell colors.

    Returns:
        Desktop figure with plots and table.
    """
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"rowspan": 2, "type": "table"}],
            [{"type": "xy"}, None],
        ],
    )

    for item, lbl in enumerate(copied.columns_lvl_zero):
        figure.add_scatter(
            x=copied.tsdf.index,
            y=copied.tsdf.iloc[:, item],
            hovertemplate="%{y:.2%}<br>%{x|%Y-%m-%d}",
            line={"width": 2.5, "dash": "solid"},
            mode="lines",
            name=lbl,
            showlegend=True,
            row=1,
            col=1,
        )

    for item in range(copied.item_count):
        col_name = cast("tuple[str, ValueType]", bdf.iloc[:, item].name)
        figure.add_bar(
            x=bdf.index,
            y=bdf.iloc[:, item],
            hovertemplate="%{y:.2%}<br>%{x}",
            name=col_name[0],
            showlegend=False,
            row=2,
            col=1,
        )

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

    return figure


def _build_mobile_figure(
    copied: OpenFrame,
    bdf: DataFrame,
    *,
    add_logo: bool,
    vertical_legend: bool,
    title: str | None,
) -> Figure:
    """Build the mobile figure with charts only.

    Args:
        copied: Copied OpenFrame data.
        bdf: Bar chart DataFrame.
        add_logo: Whether to add logo.
        vertical_legend: Whether to use vertical legend.
        title: Optional title for the figure.

    Returns:
        Mobile layout figure.
    """
    plot_height = 400
    bar_height = 350
    total_min_height = plot_height + bar_height + 200

    plot_ratio = plot_height / (plot_height + bar_height)
    bar_ratio = bar_height / (plot_height + bar_height)

    figure_mobile = make_subplots(
        rows=2,
        cols=1,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
        vertical_spacing=0.08,
        row_heights=[plot_ratio, bar_ratio],
        subplot_titles=("", ""),
    )

    for item, lbl in enumerate(copied.columns_lvl_zero):
        figure_mobile.add_scatter(
            x=copied.tsdf.index,
            y=copied.tsdf.iloc[:, item],
            hovertemplate="%{y:.2%}<br>%{x|%Y-%m-%d}",
            line={"width": 2.5, "dash": "solid"},
            mode="lines",
            name=lbl,
            showlegend=True,
            row=1,
            col=1,
        )

    for item in range(copied.item_count):
        col_name = cast("tuple[str, ValueType]", bdf.iloc[:, item].name)
        figure_mobile.add_bar(
            x=bdf.index,
            y=bdf.iloc[:, item],
            hovertemplate="%{y:.2%}<br>%{x}",
            name=col_name[0],
            showlegend=False,
            row=2,
            col=1,
        )

    _configure_figure_layout(
        figure_mobile,
        copied,
        add_logo=add_logo,
        vertical_legend=vertical_legend,
        title=title,
        mobile=True,
        total_min_height=total_min_height,
    )

    return figure_mobile


def _generate_html_table_string(
    cleanedtablevalues: list[list[str]],
    columns: list[str],
    aligning: list[str],
    color_lst: list[str],
) -> str:
    """Generate HTML table string for mobile layout.

    Args:
        cleanedtablevalues: Table cell values.
        columns: Table column headers.
        aligning: Table cell alignment.
        color_lst: Table cell colors.

    Returns:
        HTML table string.
    """
    num_rows = len(cleanedtablevalues[0]) if cleanedtablevalues else 0
    num_cols = len(columns)

    table_html = '<table style="width:100%; border-collapse:collapse; '
    table_html += 'margin-top:20px; font-family:Poppins, sans-serif;">\n'
    table_html += "<thead>\n<tr>\n"

    for col_idx, col in enumerate(columns):
        align = aligning[col_idx] if col_idx < len(aligning) else "center"
        table_html += (
            f'<th style="background-color:grey; color:white; '
            f"padding:10px; text-align:{align}; vertical-align:middle; "
            f'font-size:11px; font-weight:bold;">{col}</th>\n'
        )

    table_html += "</tr>\n</thead>\n<tbody>\n"

    col_even_color = color_lst[-1] if len(color_lst) > 1 else "white"
    col_odd_color = color_lst[1] if len(color_lst) > 1 else "white"

    for row_idx in range(num_rows):
        table_html += "<tr>\n"
        for col_idx in range(num_cols):
            align = aligning[col_idx] if col_idx < len(aligning) else "center"
            cell_value = (
                cleanedtablevalues[col_idx][row_idx]
                if col_idx < len(cleanedtablevalues)
                and row_idx < len(cleanedtablevalues[col_idx])
                else ""
            )
            if col_idx == 0:
                bg_color = "grey"
                text_color = "white"
            elif col_idx == num_cols - 1:
                bg_color = col_even_color
                text_color = "black"
            else:
                bg_color = col_odd_color
                text_color = "black"
            table_html += (
                f'<td style="background-color:{bg_color}; color:{text_color}; '
                f"padding:8px; text-align:{align}; vertical-align:middle; "
                f'font-size:10px; height:25px;">{cell_value}</td>\n'
            )
        table_html += "</tr>\n"

    table_html += "</tbody>\n</table>\n"
    return table_html


def _get_output_directory(directory: Path | None) -> Path:
    """Get the output directory path.

    Args:
        directory: Optional directory path.

    Returns:
        Resolved directory path.
    """
    if directory:
        return Path(directory).resolve()
    if Path.home().joinpath("Documents").exists():
        return Path.home() / "Documents"
    return Path(stack()[2].filename).parent


def _generate_responsive_html_string(
    html_desktop: str,
    html_mobile: str,
    div_id_desktop: str,
    div_id_mobile: str,
    table_html: str,
) -> str:
    """Generate responsive HTML wrapper with CSS and JavaScript.

    Args:
        html_desktop: Desktop layout HTML.
        html_mobile: Mobile layout HTML.
        div_id_desktop: Desktop container div ID.
        div_id_mobile: Mobile container div ID.
        table_html: HTML table string for mobile layout.

    Returns:
        Responsive HTML string.
    """
    desktop_style = "width:100%;"
    mobile_style = "width:100%; display:none;"
    match_media = 'window.matchMedia("(max-width: 960px)").matches'
    desktop_get = f'document.getElementById("{div_id_desktop}_container")'
    mobile_get = f'document.getElementById("{div_id_mobile}_container")'

    mobile_content = html_mobile + f"\n{table_html}"

    return (
        f'<div id="{div_id_desktop}_container" '
        f'class="plotly-desktop" style="{desktop_style}">\n'
        f"{html_desktop}\n"
        f"</div>\n"
        f'<div id="{div_id_mobile}_container" '
        f'class="plotly-mobile" style="{mobile_style}">\n'
        f"{mobile_content}\n"
        f"</div>\n"
        "<style>\n"
        "body { overflow-y: auto; }\n"
        "@media (max-width: 960px) {\n"
        "    .plotly-desktop { display: none !important; }\n"
        "    .plotly-mobile { display: block !important; "
        "overflow: visible !important; }\n"
        "    .plotly-mobile .js-plotly-plot { "
        "min-height: auto !important; overflow: visible !important; "
        "height: auto !important; max-height: none !important; }\n"
        "    .plotly-mobile .js-plotly-plot > div { "
        "overflow: visible !important; overflow-y: visible !important; "
        "overflow-x: visible !important; height: auto !important; "
        "max-height: none !important; }\n"
        "    .plotly-mobile .js-plotly-plot svg { "
        "overflow: visible !important; height: auto !important; "
        "max-height: none !important; }\n"
        "    .plotly-mobile .js-plotly-plot g { "
        "overflow: visible !important; }\n"
        "    .plotly-mobile .js-plotly-plot [class*='table'] { "
        "overflow: visible !important; }\n"
        "    .plotly-mobile * { "
        "overflow-y: visible !important; max-height: none !important; }\n"
        "    .plotly-mobile .scrollbar { display: none !important; }\n"
        "    .plotly-mobile [style*='overflow'] { "
        "overflow: visible !important; overflow-y: visible !important; }\n"
        "}\n"
        "@media (min-width: 961px) {\n"
        "    .plotly-desktop { display: block !important; }\n"
        "    .plotly-mobile { display: none !important; }\n"
        "}\n"
        "</style>\n"
        "<script>\n"
        "(function() {\n"
        "    function updateLayout() {\n"
        f"        var isMobile = {match_media};\n"
        f"        var desktopContainer = {desktop_get};\n"
        f"        var mobileContainer = {mobile_get};\n"
        "\n"
        "        if (isMobile) {\n"
        "            if (desktopContainer) "
        'desktopContainer.style.display = "none";\n'
        "            if (mobileContainer) "
        'mobileContainer.style.display = "block";\n'
        "            disableTableScrolling(mobileContainer);\n"
        "            if (mobileContainer && !mobileContainer._scrollObserver) {\n"
        "                mobileContainer._scrollObserver = "
        "setupScrollObserver(mobileContainer);\n"
        "            }\n"
        "        } else {\n"
        "            if (desktopContainer) "
        'desktopContainer.style.display = "block";\n'
        "            if (mobileContainer) "
        'mobileContainer.style.display = "none";\n'
        "            disableTableScrolling(desktopContainer);\n"
        "            setTimeout(function() { "
        "adjustTableSize(desktopContainer); }, 100);\n"
        "        }\n"
        "    }\n"
        "\n"
        "    function adjustTableSize(container) {\n"
        "        if (!container) return;\n"
        "        var plots = container.querySelectorAll('.js-plotly-plot');\n"
        "        plots.forEach(function(plot) {\n"
        "            var svg = plot.querySelector('svg');\n"
        "            if (!svg) return;\n"
        "            var tableGroup = svg.querySelector('g[class*=\"table\"]');\n"
        "            if (!tableGroup) return;\n"
        "            var plotRect = plot.getBoundingClientRect();\n"
        "            if (plotRect.height === 0) return;\n"
        "            setTimeout(function() {\n"
        "                try {\n"
        "                    var bbox = tableGroup.getBBox();\n"
        "                    if (bbox.height === 0) return;\n"
        "                    var availableHeight = plotRect.height - 60;\n"
        "                    var currentHeight = bbox.height;\n"
        "                    if (currentHeight > availableHeight && "
        "availableHeight > 0) {\n"
        "                        var scaleFactor = availableHeight / "
        "currentHeight;\n"
        "                        var minScale = 0.4;\n"
        "                        var maxScale = 1.0;\n"
        "                        scaleFactor = Math.max(minScale, "
        "Math.min(maxScale, scaleFactor));\n"
        "                        var existingTransform = "
        "tableGroup.getAttribute('transform') || '';\n"
        "                        var translateMatch = "
        "existingTransform.match(/translate\\(([^)]+)\\)/);\n"
        "                        var translate = translateMatch ? "
        "translateMatch[0] : 'translate(0,0)';\n"
        "                        var scaleTransform = translate + ' scale(' + "
        "scaleFactor + ')';\n"
        "                        tableGroup.setAttribute('transform', "
        "scaleTransform);\n"
        "                    } else if (currentHeight <= availableHeight) {\n"
        "                        var existingTransform = "
        "tableGroup.getAttribute('transform') || '';\n"
        "                        var translateMatch = "
        "existingTransform.match(/translate\\(([^)]+)\\)/);\n"
        "                        var translate = translateMatch ? "
        "translateMatch[0] : 'translate(0,0)';\n"
        "                        tableGroup.setAttribute('transform', "
        "translate);\n"
        "                    }\n"
        "                } catch (e) {\n"
        "                    console.log('Table adjustment error:', e);\n"
        "                }\n"
        "            }, 300);\n"
        "        });\n"
        "    }\n"
        "\n"
        "    function disableTableScrolling(container) {\n"
        "        if (!container) return;\n"
        "        var plots = container.querySelectorAll('.js-plotly-plot');\n"
        "        plots.forEach(function(plot) {\n"
        "            var allElements = plot.querySelectorAll('*');\n"
        "            allElements.forEach(function(el) {\n"
        "                var style = window.getComputedStyle(el);\n"
        "                var overflow = style.overflow;\n"
        "                var overflowY = style.overflowY;\n"
        "                if (overflow === 'auto' || overflow === 'scroll' || "
        "overflowY === 'auto' || overflowY === 'scroll') {\n"
        "                    el.style.setProperty('overflow', 'visible', "
        "'important');\n"
        "                    el.style.setProperty('overflow-y', 'visible', "
        "'important');\n"
        "                    el.style.setProperty('overflow-x', 'visible', "
        "'important');\n"
        "                }\n"
        "            });\n"
        "            plot.style.setProperty('overflow', 'visible', 'important');\n"
        "            plot.style.setProperty('overflow-y', 'visible', "
        "'important');\n"
        "            var plotDivs = plot.querySelectorAll('div');\n"
        "            plotDivs.forEach(function(div) {\n"
        "                div.style.setProperty('overflow', 'visible', "
        "'important');\n"
        "                div.style.setProperty('overflow-y', 'visible', "
        "'important');\n"
        "            });\n"
        "            var svgs = plot.querySelectorAll('svg');\n"
        "            svgs.forEach(function(svg) {\n"
        "                svg.style.setProperty('overflow', 'visible', "
        "'important');\n"
        "                svg.setAttribute('overflow', 'visible');\n"
        "            });\n"
        "        });\n"
        "    }\n"
        "\n"
        "    function setupScrollObserver(container) {\n"
        "        if (!container) return;\n"
        "        var observer = new MutationObserver(function(mutations) {\n"
        "            disableTableScrolling(container);\n"
        "        });\n"
        "        observer.observe(container, {\n"
        "            childList: true,\n"
        "            subtree: true,\n"
        "            attributes: true,\n"
        "            attributeFilter: ['style', 'class']\n"
        "        });\n"
        "        return observer;\n"
        "    }\n"
        "\n"
        "    window.addEventListener('resize', function() {\n"
        "        updateLayout();\n"
        "        setTimeout(function() {\n"
        f"            var desktopContainer = {desktop_get};\n"
        "            if (desktopContainer && "
        'desktopContainer.style.display !== "none") {\n'
        "                adjustTableSize(desktopContainer);\n"
        "            }\n"
        "        }, 100);\n"
        "    });\n"
        "    updateLayout();\n"
        "    var checkInterval = setInterval(function() {\n"
        f"        var mobileContainer = {mobile_get};\n"
        f"        var desktopContainer = {desktop_get};\n"
        "        if (mobileContainer && "
        'mobileContainer.style.display !== "none") {\n'
        "            disableTableScrolling(mobileContainer);\n"
        "            if (!mobileContainer._scrollObserver) {\n"
        "                mobileContainer._scrollObserver = "
        "setupScrollObserver(mobileContainer);\n"
        "            }\n"
        "        }\n"
        "        if (desktopContainer && "
        'desktopContainer.style.display !== "none") {\n'
        "            disableTableScrolling(desktopContainer);\n"
        "            adjustTableSize(desktopContainer);\n"
        "            if (!desktopContainer._scrollObserver) {\n"
        "                desktopContainer._scrollObserver = "
        "setupScrollObserver(desktopContainer);\n"
        "            }\n"
        "        }\n"
        "    }, 50);\n"
        "    setTimeout(function() { clearInterval(checkInterval); }, 2000);\n"
        "})();\n"
        "</script>"
    )


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
    vertical_legend: bool = False,
) -> tuple[Figure, str]:
    """Generate a HTML report page with line and bar plots and a table.

    Args:
        data: The timeseries data.
        bar_freq: The date offset string that sets the bar plot frequency.
        filename: Name of the Plotly HTML file.
        title: The report page title.
        directory: Directory where Plotly HTML file is saved.
        output_type: Determines output type. Defaults to "file".
        include_plotlyjs: Determines how the plotly.js library is included in
            the output.
            Defaults to "cdn".
        auto_open: Determines whether to open a browser window with the plot.
            Defaults to False.
        add_logo: If True a Captor logo is added to the plot. Defaults to True.
        vertical_legend: Determines whether to vertically align the legend's
            labels. Defaults to False.

    Returns:
        Plotly Figure and a div section or a HTML filename with location.

    """
    copied = data.from_deepcopy()
    copied.trunc_frame().value_nan_handle().to_cumret()

    properties, labels_init, labels_final = _get_report_properties_and_labels(
        copied.yearfrac
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

    rpt_df, formats = _build_report_dataframe(
        copied=copied,
        properties=properties,
        formats=formats,
        yearfrac=copied.yearfrac,
    )

    rpt_df = _format_report_dataframe(
        rpt_df=rpt_df,
        formats=formats,
        labels_init=labels_init,
        labels_final=labels_final,
        copied=copied,
    )

    cleanedtablevalues, columns, aligning, color_lst = _prepare_table_data(
        rpt_df=rpt_df,
        copied=copied,
    )

    bdf = _get_bar_dataframe(copied, bar_freq)
    figure = _build_desktop_figure(
        copied=copied,
        bdf=bdf,
        cleanedtablevalues=cleanedtablevalues,
        columns=columns,
        aligning=aligning,
        color_lst=color_lst,
    )

    dirpath = _get_output_directory(directory)

    if not filename:
        filename = "".join(choice(ascii_letters) for _ in range(6)) + ".html"

    plotfile = dirpath / filename

    _configure_figure_layout(
        figure,
        copied,
        add_logo=add_logo,
        vertical_legend=vertical_legend,
        title=title,
        mobile=False,
    )

    figure_mobile = _build_mobile_figure(
        copied=copied,
        bdf=bdf,
        add_logo=add_logo,
        vertical_legend=vertical_legend,
        title=title,
    )

    table_html = _generate_html_table_string(
        cleanedtablevalues=cleanedtablevalues,
        columns=columns,
        aligning=aligning,
        color_lst=color_lst,
    )

    fig, _ = load_plotly_dict()

    if output_type == "file":
        div_id_desktop = filename.split(sep=".")[0] + "_desktop"
        div_id_mobile = filename.split(sep=".")[0] + "_mobile"

        html_desktop = to_html(
            fig=figure,
            div_id=div_id_desktop,
            auto_play=False,
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            config=fig["config"],
        )

        html_mobile = to_html(
            fig=figure_mobile,
            div_id=div_id_mobile,
            auto_play=False,
            full_html=False,
            include_plotlyjs=False,
            config=fig["config"],
        )

        responsive_html = _generate_responsive_html_string(
            html_desktop=html_desktop,
            html_mobile=html_mobile,
            div_id_desktop=div_id_desktop,
            div_id_mobile=div_id_mobile,
            table_html=table_html,
        )

        with plotfile.open(mode="w", encoding="utf-8") as f:
            f.write(responsive_html)

        if auto_open:
            webbrowser.open(f"file://{plotfile.resolve()}")

        string_output = str(plotfile)
    else:
        div_id = filename.split(sep=".")[0]
        div_id_mobile = div_id + "_mobile"
        html_desktop = to_html(
            fig=figure,
            div_id=div_id,
            auto_play=False,
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            config=fig["config"],
        )
        html_mobile = to_html(
            fig=figure_mobile,
            div_id=div_id_mobile,
            auto_play=False,
            full_html=False,
            include_plotlyjs=False,
            config=fig["config"],
        )

        string_output = _generate_responsive_html_string(
            html_desktop=html_desktop,
            html_mobile=html_mobile,
            div_id_desktop=div_id,
            div_id_mobile=div_id_mobile,
            table_html=table_html,
        )

    return figure, string_output
