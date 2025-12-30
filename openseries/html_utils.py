"""Shared HTML utilities for responsive Plotly outputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from webbrowser import open as webbrowser_open

from plotly.io import to_html  # type: ignore[import-untyped]

if TYPE_CHECKING:  # pragma: no cover
    from plotly.graph_objs import Figure  # type: ignore[import-untyped]

    from .owntypes import LiteralPlotlyJSlib, LiteralPlotlyOutput, PlotlyConfigType

__all__ = [
    "_generate_responsive_plot_html",
    "_get_base_css",
    "_get_plot_css",
    "_get_plotly_script",
    "export_plotly_figure",
]


def _get_base_css() -> str:
    """Get base CSS styles for responsive HTML reports."""
    return """
    :root{--ink:#1f2a44;--muted:#6b778c;--header:#4a4a4a;--header2:#6a6a6a;--cell:#f3f3f3;--cell2:#e6e6e6;--paper:#ffffff;}
    html,body{margin:0;padding:0;background:var(--paper);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
    .page{max-width:calc(100% - 64px);margin:0 auto;padding:32px;
    padding-bottom:48px;}
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
      .table{overflow-x:auto;}
      table.metrics{table-layout:auto;font-size:14px;}
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


def _get_plot_css() -> str:
    """Get CSS styles for full-screen responsive plots.

    Returns:
        CSS string for responsive plots.
    """
    return """
    *{box-sizing:border-box;}
    html,body{margin:0;padding:0;width:100%;height:100%;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    overflow-x:hidden;}
    .page{width:100%;height:100vh;display:flex;flex-direction:column;}
    .title-container{width:100%;flex-shrink:0;padding:15px 20px 10px 20px;
    display:flex;align-items:center;justify-content:center;background:white;z-index:10;
    gap:15px;}
    .title-logo{height:60px;width:auto;flex-shrink:0;}
    .title-container h1{margin:0;padding:0;font-size:36px;font-weight:bold;
    color:#253551;word-wrap:break-word;word-break:break-word;
    white-space:normal;line-height:1.2;flex:1;text-align:center;}
    .plot-container{width:100%;flex:1;position:relative;overflow:hidden;
    min-height:0;}
    .plot{width:100%;height:100%;display:block;position:absolute;top:0;left:0;}
    .plot > div{width:100% !important;height:100% !important;}
    .plot .js-plotly-plot{width:100% !important;height:100% !important;}
    .plot .js-plotly-plot .plotly{width:100% !important;height:100% !important;}
    @media (max-width: 980px) {
      .title-container{padding:12px 15px 8px 15px;gap:12px;}
      .title-container h1{font-size:24px;line-height:1.3;}
      .title-logo{height:32px;}
    }
    @media (max-width: 480px) {
      .title-container{padding:10px 12px 6px 12px;gap:10px;}
      .title-container h1{font-size:18px;line-height:1.2;}
      .title-logo{height:24px;}
    }
    """


def _get_plotly_script(include_plotlyjs: LiteralPlotlyJSlib) -> str:
    """Get plotly script tag."""
    if include_plotlyjs == "cdn":
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    return ""


def _generate_responsive_plot_html(
    title: str | None,
    plot_div: str,
    include_plotlyjs: LiteralPlotlyJSlib,
    plot_id: str,
    logo_url: str | None = None,
) -> str:
    """Generate responsive HTML for a single Plotly plot."""
    css = _get_plot_css()
    plotly_script = _get_plotly_script(include_plotlyjs)

    plot_div = plot_div.replace(
        f'<div id="{plot_id}"', f'<div id="{plot_id}" class="plot"'
    )

    title_html = ""
    if (title is not None and title) or logo_url:
        logo_html = ""
        if logo_url:
            logo_html = f'<img src="{logo_url}" alt="Logo" class="title-logo" />'
        title_text = f"<h1><b>{title}</b></h1>" if title else ""
        title_html = f'<div class="title-container">{logo_html}{title_text}</div>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1,
maximum-scale=5,user-scalable=yes" />
<title>{title or ""}</title>
<style>{css}</style>
{plotly_script}
</head>
<body>
<div class="page">
  {title_html}
  <div class="plot-container">
    {plot_div}
  </div>
</div>
<script>
(function() {{
  var plotDiv = document.getElementById("{plot_id}");
  var container = plotDiv ? plotDiv.closest('.plot-container') : null;

  function getContainerSize() {{
    if (!container) return {{width: window.innerWidth, height: window.innerHeight}};
    var rect = container.getBoundingClientRect();
    return {{width: rect.width, height: rect.height}};
  }}

  function resizePlot() {{
    if (typeof Plotly === 'undefined' || !Plotly.Plots || !plotDiv) return;

    var size = getContainerSize();
    if (size.width > 0 && size.height > 0) {{
      Plotly.Plots.resize(plotDiv);
    }}
  }}

  function debounceResize() {{
    clearTimeout(window._resizeTimeout);
    window._resizeTimeout = setTimeout(resizePlot, 150);
  }}

  window.addEventListener("resize", debounceResize);

  window.addEventListener("orientationchange", function() {{
    setTimeout(function() {{
      resizePlot();
    }}, 300);
  }});

  function initResize() {{
    if (typeof Plotly !== 'undefined' && plotDiv) {{
      setTimeout(resizePlot, 100);
      setTimeout(resizePlot, 500);
      setTimeout(resizePlot, 1000);
    }} else {{
      setTimeout(initResize, 100);
    }}
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', initResize);
  }} else {{
    initResize();
  }}

  window.addEventListener('load', function() {{
    setTimeout(resizePlot, 200);
  }});
  }})();
</script>
</body>
</html>
"""


def export_plotly_figure(
    figure: Figure,
    fig_config: PlotlyConfigType,
    output_type: LiteralPlotlyOutput,
    filename: str,
    *,
    include_plotlyjs: LiteralPlotlyJSlib,
    auto_open: bool = False,
    plotfile: Path | str,
    title: str | None = None,
    logo_url: str | None = None,
) -> str:
    """Export a Plotly figure to a mobile-responsive HTML file or inline div.

    This function converts a Plotly figure to HTML output, with support for
    mobile-responsive standalone HTML files or inline HTML divs. When exporting
    to a file, the output includes proper viewport settings, responsive CSS,
    and JavaScript for handling window resizing and orientation changes.

    Args:
        figure: Plotly figure to render.
        fig_config: Plotly config dictionary.
        output_type: Output type: ``"file"`` or ``"div"``.
        filename: Output filename used for the ``div_id`` when inline, or to
            derive the div_id when exporting to file.
        include_plotlyjs: How plotly.js is included. Can be ``True`` (inline),
            ``False`` (not included), or ``"cdn"`` (CDN link).
        auto_open: Whether to auto-open the file in a browser (only used when
            ``output_type="file"``). Defaults to ``False``.
        plotfile: Full path to the output HTML file. Required when
            ``output_type="file"``, ignored when ``output_type="div"``.
        title: Title for the HTML page (used for file output). Defaults to
            ``None``.
        logo_url: Optional logo URL to display in the title container.
            Defaults to ``None``.

    Returns:
        If ``output_type`` is ``"file"``, the path to the file as a string;
        otherwise an inline HTML string (div).


    Example:
        Export a Plotly figure to a responsive HTML file:

        >>> import plotly.graph_objects as go
        >>> from openseries.html_utils import export_plotly_figure
        >>> from pathlib import Path
        >>>
        >>> fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        >>> output_path = export_plotly_figure(
        ...     figure=fig,
        ...     fig_config={},
        ...     output_type="file",
        ...     filename="my_plot.html",
        ...     include_plotlyjs="cdn",
        ...     plotfile=Path("output/my_plot.html"),
        ...     title="My Plot",
        ...     auto_open=True,
        ... )
    """
    if output_type == "file":
        plotfile_path = Path(plotfile) if isinstance(plotfile, str) else plotfile
        div_id = filename.rsplit(".", 1)[0]
        plot_div = cast(
            "str",
            to_html(
                fig=figure,
                config=fig_config,
                auto_play=False,
                include_plotlyjs=False,
                full_html=False,
                div_id=div_id,
            ),
        )
        html_content = _generate_responsive_plot_html(
            title=title,
            plot_div=plot_div,
            include_plotlyjs=include_plotlyjs,
            plot_id=div_id,
            logo_url=logo_url,
        )
        plotfile_path.write_text(html_content, encoding="utf-8")
        if auto_open:
            webbrowser_open(plotfile_path.as_uri())
        return str(plotfile_path)

    div_id = filename.rsplit(".", 1)[0]
    return cast(
        "str",
        to_html(
            fig=figure,
            config=fig_config,
            auto_play=False,
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            div_id=div_id,
        ),
    )
