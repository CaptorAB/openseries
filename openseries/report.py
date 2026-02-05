"""Functions related to HTML reports."""

from __future__ import annotations

from inspect import stack
from itertools import cycle
from json import dumps as json_dumps
from logging import getLogger
from pathlib import Path
from secrets import choice
from string import ascii_letters
from typing import TYPE_CHECKING, Any, cast
from warnings import catch_warnings, simplefilter
from webbrowser import open as webbrowser_open

from pandas import DataFrame, Index, Series, Timestamp, concat, isna
from plotly.graph_objs import Bar, Figure, Scatter  # type: ignore[import-untyped]
from plotly.utils import PlotlyJSONEncoder  # type: ignore[import-untyped]

from .html_utils import _get_base_css, _get_plotly_script
from .load_plotly import load_plotly_dict
from .owntypes import (
    CaptorLogoType,
    LiteralBizDayFreq,
    LiteralFrameProps,
    LiteralPlotlyJSlib,
    LiteralPlotlyOutput,
    ValueType,
)

if TYPE_CHECKING:  # pragma: no cover
    from .frame import OpenFrame

logger = getLogger(__name__)

__all__ = ["report_html"]


def calendar_period_returns(
    data: OpenFrame,
    freq: LiteralBizDayFreq = "BYE",
    *,
    relabel: bool = True,
) -> DataFrame:
    """Generate a table of returns with appropriate table labels."""
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


def _dumps_plotly(obj: object) -> str:
    return json_dumps(obj, cls=PlotlyJSONEncoder)


def _fmt_dates(idx: Index) -> list[str]:
    return [Timestamp(d).strftime("%Y-%m-%d") for d in idx]


def _metrics_table_html(df: DataFrame) -> str:
    return df.to_html(index=False, escape=False, classes=["metrics"], border=0)


def _get_report_properties_and_labels(
    yearfrac: float,
) -> tuple[list[str], list[str], list[str]]:
    """Get properties and labels based on yearfrac."""
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


def _create_line_traces(data: OpenFrame) -> list[Scatter]:
    """Create line traces for the plot."""
    x_line = _fmt_dates(data.tsdf.index)
    line_traces: list[Scatter] = []
    for item, lbl in enumerate(data.columns_lvl_zero):
        line_traces.append(
            Scatter(
                x=x_line,
                y=data.tsdf.iloc[:, item].tolist(),
                hovertemplate=f"{lbl}<br>%{{y:.2%}}<br>%{{x}}<extra></extra>",
                line={"width": 2.5, "dash": "solid"},
                mode="lines",
                name=lbl,
                showlegend=True,
            ),
        )
    return line_traces


def _create_bar_traces(
    data: OpenFrame,
    bar_freq: LiteralBizDayFreq,
) -> list[Bar]:
    """Create bar traces for the plot."""
    quarter_of_year = 0.25
    if data.yearfrac < quarter_of_year:
        tmp = data.from_deepcopy()
        bdf = tmp.value_to_ret().tsdf.iloc[1:]
    else:
        bdf = calendar_period_returns(data=data, freq=bar_freq)

    x_bar = [str(x) for x in bdf.index]
    bar_traces: list[Bar] = []
    for item in range(data.item_count):
        col_name = cast("tuple[str, ValueType]", bdf.iloc[:, item].name)
        bar_traces.append(
            Bar(
                x=x_bar,
                y=bdf.iloc[:, item].tolist(),
                hovertemplate=f"{col_name[0]}<br>%{{y:.2%}}<br>%{{x}}<extra></extra>",
                name=col_name[0],
                showlegend=False,
            ),
        )
    return bar_traces


def _add_jensen_alpha(
    rpt_df: DataFrame,
    data: OpenFrame,
) -> DataFrame:
    """Add Jensen's Alpha to the report dataframe."""
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
    ar = DataFrame(
        data=alphas,
        index=data.tsdf.columns,
        columns=["Jensen's Alpha"],
    ).T
    return concat([rpt_df, ar])


def _add_information_ratio(
    rpt_df: DataFrame,
    data: OpenFrame,
) -> DataFrame:
    """Add Information Ratio to the report dataframe."""
    ir = data.info_ratio_func()
    ir.name = "Information Ratio"
    ir.iloc[-1] = None
    ir_df = ir.to_frame().T
    return concat([rpt_df, ir_df])


def _add_tracking_error(
    rpt_df: DataFrame,
    data: OpenFrame,
) -> DataFrame:
    """Add Tracking Error to the report dataframe."""
    te_frame = data.from_deepcopy()
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


def _add_capture_ratio(
    rpt_df: DataFrame,
    data: OpenFrame,
    formats: list[str],
) -> tuple[DataFrame, list[str]]:
    """Add Capture Ratio to the report dataframe."""
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
    return concat([rpt_df, cru_df]), formats


def _add_beta(
    rpt_df: DataFrame,
    data: OpenFrame,
) -> DataFrame:
    """Add Index Beta to the report dataframe."""
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
    betas.append("")
    br = DataFrame(
        data=betas,
        index=data.tsdf.columns,
        columns=["Index Beta (weekly)"],
    ).T
    return concat([rpt_df, br])


def _add_ytd_mtd(
    rpt_df: DataFrame,
    data: OpenFrame,
) -> DataFrame:
    """Add Year-to-Date and Month-to-Date to the report dataframe."""
    this_year = data.last_idx.year
    this_month = data.last_idx.month
    ytd = data.value_ret_calendar_period(year=this_year).map("{:.2%}".format)
    ytd.name = "Year-to-Date"
    mtd = data.value_ret_calendar_period(year=this_year, month=this_month).map(
        "{:.2%}".format,
    )
    mtd.name = "Month-to-Date"
    ytd_df = ytd.to_frame().T
    mtd_df = mtd.to_frame().T
    return concat([rpt_df, ytd_df, mtd_df])


def _get_output_directory(directory: Path | None) -> Path:
    """Determine the output directory."""
    if directory:
        return Path(directory).resolve()
    if Path.home().joinpath("Documents").exists():
        return Path.home() / "Documents"
    return Path(stack()[1].filename).parent


def _get_plotly_layouts(
    layout_theme: dict[str, Any],
    colorway: list[str],
    item_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Get line and bar layouts for plotly."""
    line_layout = dict(layout_theme)
    line_layout.update(
        {
            "colorway": colorway[:item_count] if colorway else None,
            "margin": {"l": 50, "r": 20, "t": 20, "b": 40},
            "xaxis": {"gridcolor": "#EEEEEE", "automargin": True, "tickangle": -45},
            "yaxis": {"tickformat": ".2%", "gridcolor": "#EEEEEE", "automargin": True},
            "showlegend": False,
        },
    )

    bar_layout = dict(layout_theme)
    bar_layout.update(
        {
            "barmode": "group",
            "margin": {"l": 50, "r": 20, "t": 10, "b": 80},
            "xaxis": {"gridcolor": "#EEEEEE", "automargin": True, "tickangle": -45},
            "yaxis": {"tickformat": ".2%", "gridcolor": "#EEEEEE", "automargin": True},
            "showlegend": False,
        },
    )

    return line_layout, bar_layout


def _get_logo_html(logo: CaptorLogoType, *, add_logo: bool) -> str:
    """Get logo HTML."""
    if not add_logo:
        return ""
    try:
        src = cast("dict[str, Any]", logo).get("source", "")
    except (KeyError, AttributeError, TypeError):
        src = ""
    if src:
        return f'<img src="{src}" alt="Captor" style="height:68px;" />'
    return "CAPTOR"


def _get_legend_html(line_traces: list[Scatter], colorway: list[str]) -> str:
    """Generate HTML for the legend at the bottom of the page."""
    legend_items = []
    color_cycle = cycle(colorway or ["#66725B"])
    for trace in line_traces:
        name = trace.name or ""
        color = next(color_cycle)
        legend_items.append(
            f'<div class="legend-item">'
            f'<div class="legend-color" style="background-color:{color};"></div>'
            f"<span>{name}</span>"
            f"</div>",
        )
    if legend_items:
        return f'<div class="legend-container">{"".join(legend_items)}</div>'
    return ""


def _get_css() -> str:
    """Get CSS styles for the HTML report."""
    base_css = _get_base_css()
    return (
        base_css
        + """
    .header{display:grid;grid-template-columns:140px 1fr 140px;gap:12px;
    align-items:start;}
    h1{margin:0;text-align:center;font-size:45px;font-weight:800;}
    .layout{display:grid;grid-template-columns:1.2fr .9fr;
    grid-template-areas:"charts table";gap:22px;align-items:start;margin-top:12px;}
    .charts{grid-area:charts;display:grid;grid-template-rows:auto auto;gap:18px;}
    .table{grid-area:table;}
    .plot{width:100%;height:380px;}
    .plot.bar{height:300px;}
    @media (max-width:980px){
      .page{padding:24px;padding-bottom:24px;}
      .header{grid-template-columns:120px 1fr;}
      h1{font-size:36px;}
      .layout{grid-template-columns:1fr;grid-template-areas:"table" "charts";gap:16px;}
      .plot{height:380px;}
      .plot.bar{height:300px;}
      table.metrics{table-layout:fixed;width:auto;}
      table.metrics thead th{min-width:120px;width:120px;white-space:nowrap;}
      table.metrics thead th:first-child{width:180px;}
      table.metrics tbody td{min-width:120px;width:120px;}
      table.metrics tbody td:first-child{width:180px;}
    }
    table.metrics{width:100%;border-collapse:separate;border-spacing:0;font-size:12px;
    border-radius:4px;overflow:hidden;table-layout:fixed;}
    table.metrics thead th{background:var(--header);color:white;padding:8px 10px;
    font-weight:700;text-align:center;word-wrap:break-word;word-break:break-word;}
    table.metrics thead th:first-child{background:var(--header2);text-align:left;
    width:180px;}
    table.metrics tbody td{padding:7px 10px;border-bottom:1px solid white;
    border-right:1px solid white;text-align:center;background:var(--paper);}
    table.metrics tbody td:first-child{text-align:left;font-weight:600;color:white;
    background:var(--header);width:180px;}
    table.metrics tbody td:last-child{background:var(--cell2);}
    .legend-container{margin-top:24px;padding-top:20px;padding-bottom:16px;
    display:flex;justify-content:center;flex-wrap:wrap;gap:24px;flex-shrink:0;}
    .legend-item{display:flex;align-items:center;gap:8px;font-size:14px;}
    .legend-color{width:20px;height:3px;border-radius:2px;}
    @media (min-width:981px){
      html,body{overflow-y:auto;}
    }
    """
    )


def _write_html_file(
    plotfile: Path,
    html: str,
    *,
    auto_open: bool,
) -> str:
    """Write HTML file and optionally open it."""
    plotfile.parent.mkdir(parents=True, exist_ok=True)
    plotfile.write_text(html, encoding="utf-8")
    if auto_open:
        try:
            webbrowser_open(plotfile.as_uri())
        except OSError as exc:
            logger.warning("Failed to open browser: %s", exc)
    return str(plotfile)


def _generate_html(
    title: str | None,
    css: str,
    plotly_script: str,
    logo_html: str,
    table_html: str,
    line_payload: dict[str, Any],
    bar_payload: dict[str, Any],
    legend_html: str,
) -> str:
    """Generate the HTML string."""
    return f"""<!doctype html>
<html lang="sv">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{title or ""}</title>
<style>{css}</style>
{plotly_script}
</head>
<body>
<div class="page">
  <div class="header">
    <div>{logo_html}</div>
    <div><h1>{title or ""}</h1></div>
    <div></div>
  </div>
  <div class="layout">
    <div class="charts">
      <div id="lineplot" class="plot"></div>
      <div id="barplot" class="plot bar"></div>
    </div>
    <div class="table">{table_html}</div>
  </div>
  {legend_html}
</div>
<script>
const line = {_dumps_plotly(line_payload)};
const bar = {_dumps_plotly(bar_payload)};
Plotly.newPlot("lineplot", line.data, line.layout, line.config);
Plotly.newPlot("barplot", bar.data, bar.layout, bar.config);
window.addEventListener("resize", () => {{
  Plotly.Plots.resize("lineplot");
  Plotly.Plots.resize("barplot");
}});
</script>
</body>
</html>
"""


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
    vertical_legend: bool = True,  # noqa: ARG001
) -> tuple[Figure, str]:
    """Generate a responsive HTML report page with line and bar plots and a table."""
    copied = data.from_deepcopy()
    copied.trunc_frame().value_nan_handle().to_cumret()

    properties, labels_init, labels_final = _get_report_properties_and_labels(
        copied.yearfrac,
    )

    line_traces = _create_line_traces(copied)
    bar_traces = _create_bar_traces(copied, bar_freq)

    rpt_df = copied.all_properties(
        properties=cast("list[LiteralFrameProps]", properties),
    )
    rpt_df = _add_jensen_alpha(rpt_df, copied)
    rpt_df = _add_information_ratio(rpt_df, copied)
    rpt_df = _add_tracking_error(rpt_df, copied)

    if copied.yearfrac > 1.0:
        rpt_df, _ = _add_capture_ratio(rpt_df, copied, [])

    rpt_df = _add_beta(rpt_df, copied)
    rpt_df.index = Index(labels_init)
    rpt_df = _add_ytd_mtd(rpt_df, copied)
    rpt_df = rpt_df.reindex(labels_final)

    format_map = {
        "Return (CAGR)": "{:.2%}",
        "Return (simple)": "{:.2%}",
        "Year-to-Date": "{:.2%}",
        "Month-to-Date": "{:.2%}",
        "Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}",
        "Jensen's Alpha": "{:.2%}",
        "Information Ratio": "{:.2f}",
        "Tracking Error (weekly)": "{:.2%}",
        "Index Beta (weekly)": "{:.2f}",
        "Capture Ratio (monthly)": "{:.2f}",
        "Worst Month": "{:.2%}",
        "Worst Day": "{:.2%}",
        "Comparison Start": "{:%Y-%m-%d}",
        "Comparison End": "{:%Y-%m-%d}",
    }
    formats = [format_map.get(label, "{:.2f}") for label in labels_final]

    for item, f in zip(rpt_df.index, formats, strict=False):
        rpt_df.loc[item] = rpt_df.loc[item].apply(
            lambda x, fmt=f: (
                ""
                if (
                    x is None
                    or (not isinstance(x, str) and isna(x))
                    or (isinstance(x, str) and x.lower() in ("nan", "nan%", ""))
                )
                else (
                    str(x)
                    if isinstance(x, str)
                    else (
                        Timestamp(x).strftime("%Y-%m-%d")
                        if "%Y-%m-%d" in fmt and not isinstance(x, str)
                        else fmt.format(x)
                    )
                )
            ),
        )

    rpt_df.index = Index([f"<b>{x}</b>" for x in rpt_df.index])
    rpt_df = rpt_df.reset_index()

    colmns = ["", *copied.columns_lvl_zero]
    rpt_df.columns = colmns
    table_html = _metrics_table_html(rpt_df)

    dirpath = _get_output_directory(directory=directory)

    if not filename:
        filename = "".join(choice(ascii_letters) for _ in range(6)) + ".html"

    plotfile = dirpath / filename

    fig_theme, logo = load_plotly_dict()
    layout_theme = cast("dict[str, Any]", fig_theme.get("layout", {}))
    colorway: list[str] = cast("dict[str, list[str]]", layout_theme).get(
        "colorway", []
    )

    line_layout, bar_layout = _get_plotly_layouts(
        layout_theme=layout_theme,
        colorway=colorway,
        item_count=copied.item_count,
    )

    config = cast("dict[str, Any]", fig_theme.get("config", {})) or {}
    config = {**config, "responsive": True, "displayModeBar": False}

    plotly_script = _get_plotly_script(include_plotlyjs=include_plotlyjs)
    logo_html = _get_logo_html(logo=logo, add_logo=add_logo)
    css = _get_css()

    line_payload = {
        "data": [t.to_plotly_json() for t in line_traces],
        "layout": line_layout,
        "config": config,
    }
    bar_payload = {
        "data": [t.to_plotly_json() for t in bar_traces],
        "layout": bar_layout,
        "config": config,
    }

    legend_html = _get_legend_html(line_traces=line_traces, colorway=colorway)

    html = _generate_html(
        title=title,
        css=css,
        plotly_script=plotly_script,
        logo_html=logo_html,
        table_html=table_html,
        line_payload=line_payload,
        bar_payload=bar_payload,
        legend_html=legend_html,
    )

    if output_type == "file":
        output = _write_html_file(plotfile=plotfile, html=html, auto_open=auto_open)
    else:
        output = html

    fig_return = Figure(data=line_traces)
    return fig_return, output
