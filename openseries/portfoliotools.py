"""Defining the portfolio tools for the OpenFrame class."""

from __future__ import annotations

from inspect import stack
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from numpy import (
    append,
    array,
    einsum,
    float64,
    inf,
    isnan,
    linspace,
    nan,
    sqrt,
)
from numpy import (
    sum as npsum,
)
from pandas import (
    DataFrame,
    Series,
    concat,
)
from plotly.graph_objs import Figure  # type: ignore[import-untyped]
from plotly.io import to_html  # type: ignore[import-untyped]
from plotly.offline import plot  # type: ignore[import-untyped]
from scipy.optimize import minimize

from .load_plotly import load_plotly_dict
from .owntypes import (
    AtLeastOneFrameError,
    LiteralLinePlotMode,
    LiteralMinimizeMethods,
    LiteralPlotlyJSlib,
    LiteralPlotlyOutput,
    MixedValuetypesError,
    ValueType,
)
from .series import OpenTimeSeries
from .simulation import _random_generator

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from numpy.typing import NDArray
    from pydantic import DirectoryPath

    from .frame import OpenFrame

__all__ = [
    "constrain_optimized_portfolios",
    "efficient_frontier",
    "prepare_plot_data",
    "sharpeplot",
    "simulate_portfolios",
]


def simulate_portfolios(
    simframe: OpenFrame,
    num_ports: int,
    seed: int,
) -> DataFrame:
    """Generate random weights for simulated portfolios.

    Args:
        simframe: Return data for portfolio constituents.
        num_ports: Number of possible portfolios to simulate.
        seed: The seed for the random process.

    Returns:
        The resulting data.
    """
    copi = simframe.from_deepcopy()

    vtypes = [x == ValueType.RTRN for x in copi.tsdf.columns.get_level_values(1)]
    if not any(vtypes):
        copi.value_to_ret()
        log_ret = copi.tsdf.copy()[1:]
    elif all(vtypes):
        log_ret = copi.tsdf.copy()
    else:
        msg = "Mix of series types will give inconsistent results"
        raise MixedValuetypesError(msg)

    log_ret.columns = log_ret.columns.droplevel(level=1)

    cov_matrix = log_ret.cov() * simframe.periods_in_a_year
    mean_returns = log_ret.mean() * simframe.periods_in_a_year

    randomizer = _random_generator(seed=seed)
    all_weights = randomizer.random((num_ports, simframe.item_count))
    all_weights = all_weights / all_weights.sum(axis=1, keepdims=True)

    ret_arr = all_weights @ mean_returns
    vol_arr = sqrt(einsum("ij,jk,ik->i", all_weights, cov_matrix, all_weights))
    sharpe_arr = ret_arr / vol_arr

    simdf = concat(
        [
            DataFrame({"stdev": vol_arr, "ret": ret_arr, "sharpe": sharpe_arr}),
            DataFrame(all_weights, columns=simframe.columns_lvl_zero),
        ],
        axis="columns",
    )
    simdf = simdf.replace([inf, -inf], nan)
    return simdf.dropna()


def _prepare_returns_for_frontier(eframe: OpenFrame) -> tuple[DataFrame, OpenFrame]:
    """Prepare returns DataFrame for frontier calculation.

    Args:
        eframe: Portfolio data.

    Returns:
        Tuple of (log_ret DataFrame, copied frame).

    Raises:
        MixedValuetypesError: If series types are mixed.
    """
    if eframe.weights is None:
        eframe.weights = [1.0 / eframe.item_count] * eframe.item_count

    copi = eframe.from_deepcopy()

    vtypes = [x == ValueType.RTRN for x in copi.tsdf.columns.get_level_values(1)]
    if not any(vtypes):
        copi.value_to_ret()
        log_ret = copi.tsdf.copy()[1:]
    elif all(vtypes):
        log_ret = copi.tsdf.copy()
    else:
        msg = "Mix of series types will give inconsistent results"
        raise MixedValuetypesError(msg)

    log_ret.columns = log_ret.columns.droplevel(level=1)
    return log_ret, copi


def _calculate_frontier_bounds(
    simulated: DataFrame,
    log_ret: DataFrame,
    periods_in_a_year: float,
) -> tuple[float, float]:
    """Calculate frontier return bounds.

    Args:
        simulated: Simulated portfolios DataFrame.
        log_ret: Returns DataFrame.
        periods_in_a_year: Periods in a year.

    Returns:
        Tuple of (min_return, max_return).
    """
    frontier_min = float(simulated.loc[simulated["stdev"].idxmin()]["ret"])

    arithmetic_means = array(log_ret.mean() * periods_in_a_year)
    cleaned_arithmetic_means = arithmetic_means[~isnan(arithmetic_means)]

    frontier_max = float(cleaned_arithmetic_means.max())
    return frontier_min, frontier_max


def _build_frontier_line(
    log_ret: DataFrame,
    frontier_min: float,
    frontier_max: float,
    frontier_points: int,
    periods_in_a_year: float,
    init_guess: NDArray[float64],
    bounds: tuple[tuple[float, float], ...],
    minimize_method: LiteralMinimizeMethods,
) -> tuple[list[float], list[NDArray[float64]]]:
    """Build frontier line points.

    Args:
        log_ret: Returns DataFrame.
        frontier_min: Minimum return.
        frontier_max: Maximum return.
        frontier_points: Number of points.
        periods_in_a_year: Periods in a year.
        init_guess: Initial guess for optimization.
        bounds: Optimization bounds.
        minimize_method: Minimization method.

    Returns:
        Tuple of (frontier_x, frontier_weights).
    """

    def _check_sum(weights: NDArray[float64]) -> float:
        return cast("float", npsum(weights) - 1)

    def _get_ret_vol_sr(
        lg_ret: DataFrame,
        weights: NDArray[float64],
        per_in_yr: float,
    ) -> NDArray[float64]:
        ret = npsum(lg_ret.mean() * weights) * per_in_yr
        volatility = sqrt(weights.T @ (lg_ret.cov() * per_in_yr @ weights))
        sr = ret / volatility
        return cast("NDArray[float64]", array([ret, volatility, sr]))

    def _diff_return(
        lg_ret: DataFrame,
        weights: NDArray[float64],
        per_in_yr: float,
        poss_return: float,
    ) -> float64:
        return cast(
            "float64",
            _get_ret_vol_sr(lg_ret=lg_ret, weights=weights, per_in_yr=per_in_yr)[0]
            - poss_return,
        )

    def _minimize_volatility(
        weights: NDArray[float64],
    ) -> float64:
        return cast(
            "float64",
            _get_ret_vol_sr(
                lg_ret=log_ret,
                weights=weights,
                per_in_yr=periods_in_a_year,
            )[1],
        )

    frontier_y = linspace(start=frontier_min, stop=frontier_max, num=frontier_points)
    frontier_x = []
    frontier_weights = []

    for possible_return in frontier_y:
        cons = cast(
            "dict[str, str | Callable[[float, NDArray[float64]], float64]]",
            (
                {"type": "eq", "fun": _check_sum},
                {
                    "type": "eq",
                    "fun": lambda w, poss_return=possible_return: _diff_return(
                        lg_ret=log_ret,
                        weights=w,
                        per_in_yr=periods_in_a_year,
                        poss_return=poss_return,
                    ),
                },
            ),
        )

        result = minimize(  # type: ignore[call-overload]
            fun=_minimize_volatility,
            x0=init_guess,
            method=minimize_method,
            bounds=bounds,
            constraints=cons,
        )

        frontier_x.append(result["fun"])
        frontier_weights.append(result["x"])

    return frontier_x, frontier_weights


def _build_frontier_dataframe(
    frontier_x: list[float],
    frontier_y: NDArray[float64],
    frontier_weights: list[NDArray[float64]],
    columns_lvl_zero: list[str],
) -> DataFrame:
    """Build frontier DataFrame.

    Args:
        frontier_x: Frontier volatility values.
        frontier_y: Frontier return values.
        frontier_weights: Frontier weight arrays.
        columns_lvl_zero: Column names.

    Returns:
        Frontier DataFrame.
    """
    line_df = concat(
        [
            DataFrame(data=frontier_weights, columns=columns_lvl_zero),
            DataFrame({"stdev": frontier_x, "ret": frontier_y}),
        ],
        axis="columns",
    )
    line_df["sharpe"] = line_df.ret / line_df.stdev

    limit_small = 0.0001
    line_df = line_df.mask(line_df.abs() < limit_small, 0.0)

    weight_cols = columns_lvl_zero
    weight_header = "<br><br>Weights:<br>"
    line_df["text"] = line_df[weight_cols].apply(
        lambda row: weight_header
        + "<br>".join([f"{row[col]:.1%}  {col}" for col in weight_cols]),
        axis=1,
    )

    return line_df


def _apply_tweak(line_df: DataFrame) -> DataFrame:
    """Apply tweak to frontier DataFrame.

    Args:
        line_df: Frontier DataFrame.

    Returns:
        Tweaked DataFrame.
    """
    limit_tweak = 0.001
    line_df["stdev_diff"] = line_df.stdev.ffill().pct_change()
    line_df = line_df.loc[line_df.stdev_diff.abs() > limit_tweak]
    return line_df.drop(columns="stdev_diff")


def _create_optimization_functions(
    log_ret: DataFrame,
    periods_in_a_year: float,
) -> tuple[
    Callable[[NDArray[float64]], float],
    Callable[[NDArray[float64]], NDArray[float64]],
    Callable[[NDArray[float64]], float64],
]:
    """Create optimization helper functions.

    Args:
        log_ret: Returns DataFrame.
        periods_in_a_year: Periods in a year.

    Returns:
        Tuple of (_check_sum, _get_ret_vol_sr, _neg_sharpe) functions.
    """

    def _check_sum(weights: NDArray[float64]) -> float:
        return cast("float", npsum(weights) - 1)

    def _get_ret_vol_sr(weights: NDArray[float64]) -> NDArray[float64]:
        ret = npsum(log_ret.mean() * weights) * periods_in_a_year
        volatility = sqrt(weights.T @ (log_ret.cov() * periods_in_a_year @ weights))
        sr = ret / volatility
        return cast("NDArray[float64]", array([ret, volatility, sr]))

    def _neg_sharpe(weights: NDArray[float64]) -> float64:
        return cast("float64", _get_ret_vol_sr(weights)[2] * -1)

    return _check_sum, _get_ret_vol_sr, _neg_sharpe


def _optimize_max_sharpe_portfolio(
    init_guess: NDArray[float64],
    bounds: tuple[tuple[float, float], ...],
    minimize_method: LiteralMinimizeMethods,
    _check_sum: Callable[[NDArray[float64]], float],
    _get_ret_vol_sr: Callable[[NDArray[float64]], NDArray[float64]],
    _neg_sharpe: Callable[[NDArray[float64]], float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Optimize maximum Sharpe ratio portfolio.

    Args:
        init_guess: Initial guess for optimization.
        bounds: Optimization bounds.
        minimize_method: Minimization method.
        _check_sum: Check sum constraint function.
        _get_ret_vol_sr: Get return, volatility, Sharpe ratio function.
        _neg_sharpe: Negative Sharpe ratio function.

    Returns:
        Tuple of (optimal metrics, optimal weights).
    """
    constraints = {"type": "eq", "fun": _check_sum}
    opt_results = minimize(  # type: ignore[call-overload]
        fun=_neg_sharpe,
        x0=init_guess,
        method=minimize_method,
        bounds=bounds,
        constraints=constraints,
    )

    optimal = _get_ret_vol_sr(opt_results.x)

    return optimal, opt_results.x


def efficient_frontier(
    eframe: OpenFrame,
    num_ports: int = 5000,
    seed: int = 71,
    bounds: tuple[tuple[float, float], ...] | None = None,
    frontier_points: int = 200,
    minimize_method: LiteralMinimizeMethods = "SLSQP",
    *,
    tweak: bool = True,
) -> tuple[DataFrame, DataFrame, NDArray[float64]]:
    """Identify an efficient frontier.

    Args:
        eframe: Portfolio data.
        num_ports: Number of possible portfolios to simulate. Defaults to 5000.
        seed: The seed for the random process. Defaults to 71.
        bounds: The range of minimum and maximum allowed allocations for each asset.
        frontier_points: Number of points along frontier to optimize. Defaults to 200.
        minimize_method: The method passed into the scipy.minimize function.
            Defaults to SLSQP.
        tweak: Cutting the frontier to exclude multiple points with almost the
            same risk.
            Defaults to True.

    Returns:
        The efficient frontier data, simulation data and optimal portfolio.
    """
    log_ret, copi = _prepare_returns_for_frontier(eframe)

    simulated = simulate_portfolios(simframe=copi, num_ports=num_ports, seed=seed)

    frontier_min, frontier_max = _calculate_frontier_bounds(
        simulated=simulated,
        log_ret=log_ret,
        periods_in_a_year=copi.periods_in_a_year,
    )

    if not bounds:
        bounds = tuple((0.0, 1.0) for _ in range(eframe.item_count))
    init_guess = array(eframe.weights)

    _check_sum, _get_ret_vol_sr, _neg_sharpe = _create_optimization_functions(
        log_ret=log_ret,
        periods_in_a_year=copi.periods_in_a_year,
    )

    optimal, opt_weights = _optimize_max_sharpe_portfolio(
        init_guess=init_guess,
        bounds=bounds,
        minimize_method=minimize_method,
        _check_sum=_check_sum,
        _get_ret_vol_sr=_get_ret_vol_sr,
        _neg_sharpe=_neg_sharpe,
    )

    frontier_y = linspace(start=frontier_min, stop=frontier_max, num=frontier_points)
    frontier_x, frontier_weights = _build_frontier_line(
        log_ret=log_ret,
        frontier_min=frontier_min,
        frontier_max=frontier_max,
        frontier_points=frontier_points,
        periods_in_a_year=copi.periods_in_a_year,
        init_guess=init_guess,
        bounds=bounds,
        minimize_method=minimize_method,
    )

    line_df = _build_frontier_dataframe(
        frontier_x=frontier_x,
        frontier_y=frontier_y,
        frontier_weights=frontier_weights,
        columns_lvl_zero=eframe.columns_lvl_zero,
    )

    if tweak:
        line_df = _apply_tweak(line_df)

    return line_df, simulated, append(optimal, opt_weights)


def constrain_optimized_portfolios(
    data: OpenFrame,
    serie: OpenTimeSeries,
    portfolioname: str = "Current Portfolio",
    simulations: int = 10000,
    curve_points: int = 200,
    bounds: tuple[tuple[float, float], ...] | None = None,
    minimize_method: LiteralMinimizeMethods = "SLSQP",
) -> tuple[OpenFrame, OpenTimeSeries, OpenFrame, OpenTimeSeries]:
    """Constrain optimized portfolios to those that improve on the current one.

    Args:
        data: Portfolio data.
        serie: A timeseries representing the current portfolio.
        portfolioname: Name of the portfolio. Defaults to "Current Portfolio".
        simulations: Number of possible portfolios to simulate. Defaults to 10000.
        curve_points: Number of optimal portfolios on the efficient frontier.
            Defaults to 200.
        bounds: The range of minimum and maximum allowed allocations for each asset.
        minimize_method: The method passed into the scipy.minimize function.
            Defaults to SLSQP.

    Returns:
        The constrained optimal portfolio data.

    """
    lr_frame = data.from_deepcopy()
    mv_frame = data.from_deepcopy()

    if not bounds:
        bounds = tuple((0.0, 1.0) for _ in range(data.item_count))

    front_frame, _, _ = efficient_frontier(
        eframe=data,
        num_ports=simulations,
        frontier_points=curve_points,
        bounds=bounds,
        minimize_method=minimize_method,
    )

    condition_least_ret = front_frame.ret > serie.arithmetic_ret
    least_ret_frame = front_frame[condition_least_ret].sort_values(by="stdev")
    least_ret_port: Series[float] = least_ret_frame.iloc[0]
    least_ret_port_name = f"Minimize vol & target return of {portfolioname}"
    least_ret_weights: list[float] = [
        least_ret_port.loc[c] for c in lr_frame.columns_lvl_zero
    ]
    lr_frame.weights = least_ret_weights
    resleast = OpenTimeSeries.from_df(lr_frame.make_portfolio(least_ret_port_name))

    condition_most_vol = front_frame.stdev < serie.vol
    most_vol_frame = front_frame[condition_most_vol].sort_values(
        by="ret",
        ascending=False,
    )
    most_vol_port: Series[float] = most_vol_frame.iloc[0]
    most_vol_port_name = f"Maximize return & target risk of {portfolioname}"
    most_vol_weights: list[float] = [
        most_vol_port.loc[c] for c in mv_frame.columns_lvl_zero
    ]
    mv_frame.weights = most_vol_weights
    resmost = OpenTimeSeries.from_df(mv_frame.make_portfolio(most_vol_port_name))

    return lr_frame, resleast, mv_frame, resmost


def prepare_plot_data(
    assets: OpenFrame,
    current: OpenTimeSeries,
    optimized: NDArray[float64],
) -> DataFrame:
    """Prepare data to be used as point_frame in the sharpeplot function.

    Args:
        assets: Portfolio data with individual assets and a weighted portfolio.
        current: The current or initial portfolio based on given weights.
        optimized: Data optimized with the efficient_frontier method.

    Returns:
        The data prepared with mean returns, volatility and weights.
    """
    txt = "<br><br>Weights:<br>" + "<br>".join(
        [
            f"{wgt:.1%}  {nm}"
            for wgt, nm in zip(
                cast("list[float]", assets.weights),
                assets.columns_lvl_zero,
                strict=True,
            )
        ],
    )

    opt_text_list = [
        f"{wgt:.1%}  {nm}"
        for wgt, nm in zip(optimized[3:], assets.columns_lvl_zero, strict=True)
    ]
    opt_text = "<br><br>Weights:<br>" + "<br>".join(opt_text_list)
    plotframe = DataFrame(
        data=[
            assets.arithmetic_ret,
            assets.vol,
            Series(
                data=[""] * assets.item_count,
                index=assets.vol.index,
            ),
        ],
        index=["ret", "stdev", "text"],
    )
    plotframe.columns = plotframe.columns.droplevel(level=1)
    plotframe["Max Sharpe Portfolio"] = [optimized[0], optimized[1], opt_text]
    if current.label is not None:
        plotframe[current.label] = [current.arithmetic_ret, current.vol, txt]  # type: ignore[assignment]

    return plotframe


def _determine_output_directory(directory: DirectoryPath | None) -> Path:
    """Determine output directory for plot file.

    Args:
        directory: Optional directory path.

    Returns:
        Path to output directory.
    """
    if directory:
        return Path(directory).resolve()
    if Path.home().joinpath("Documents").exists():
        return Path.home().joinpath("Documents")
    return Path(stack()[2].filename).parent


def _add_simulated_portfolios_trace(
    figure: Figure,
    sim_frame: DataFrame,
    returns: list[float],
    risk: list[float],
) -> None:
    """Add simulated portfolios trace to figure.

    Args:
        figure: Plotly figure.
        sim_frame: Simulated portfolios DataFrame.
        returns: List to extend with returns.
        risk: List to extend with risk values.
    """
    returns.extend(list(sim_frame.loc[:, "ret"]))
    risk.extend(list(sim_frame.loc[:, "stdev"]))
    figure.add_scatter(
        x=sim_frame.loc[:, "stdev"],
        y=sim_frame.loc[:, "ret"],
        hoverinfo="skip",
        marker={
            "size": 10,
            "opacity": 0.5,
            "color": sim_frame.loc[:, "sharpe"],
            "colorscale": "Jet",
            "reversescale": True,
            "colorbar": {"thickness": 20, "title": "Ratio<br>ret / vol"},
        },
        mode="markers",
        name="simulated portfolios",
    )


def _add_efficient_frontier_trace(
    figure: Figure,
    line_frame: DataFrame,
    returns: list[float],
    risk: list[float],
) -> None:
    """Add efficient frontier trace to figure.

    Args:
        figure: Plotly figure.
        line_frame: Efficient frontier DataFrame.
        returns: List to extend with returns.
        risk: List to extend with risk values.
    """
    returns.extend(list(line_frame.loc[:, "ret"]))
    risk.extend(list(line_frame.loc[:, "stdev"]))
    figure.add_scatter(
        x=line_frame.loc[:, "stdev"],
        y=line_frame.loc[:, "ret"],
        text=line_frame.loc[:, "text"],
        xhoverformat=".2%",
        yhoverformat=".2%",
        hovertemplate="Return %{y}<br>Vol %{x}%{text}",
        hoverlabel_align="right",
        line={"width": 2.5, "dash": "solid"},
        mode="lines",
        name="Efficient frontier",
    )


def _add_point_frame_traces(
    figure: Figure,
    point_frame: DataFrame,
    point_frame_mode: LiteralLinePlotMode,
    fig: dict[str, Any],
    returns: list[float],
    risk: list[float],
) -> None:
    """Add point frame traces to figure.

    Args:
        figure: Plotly figure.
        point_frame: Point frame DataFrame.
        point_frame_mode: Mode for point frame traces.
        fig: Plotly figure dictionary.
        returns: List to extend with returns.
        risk: List to extend with risk values.
    """
    layout_dict = cast(
        "dict[str, str | int | float | bool | list[str]]",
        fig["layout"],
    )
    colorway = cast("list[str]", layout_dict.get("colorway", []))[
        : len(point_frame.columns)
    ]
    for col, clr in zip(point_frame.columns, colorway, strict=True):
        returns.extend([cast("float", point_frame.loc["ret", col])])
        risk.extend([cast("float", point_frame.loc["stdev", col])])
        figure.add_scatter(
            x=[point_frame.loc["stdev", col]],
            y=[point_frame.loc["ret", col]],
            xhoverformat=".2%",
            yhoverformat=".2%",
            hovertext=[point_frame.loc["text", col]],
            hovertemplate="Return %{y}<br>Vol %{x}%{hovertext}",
            hoverlabel_align="right",
            marker={"size": 20, "color": clr},
            mode=point_frame_mode,
            name=col,
            text=col,
            textfont={"size": 14},
            textposition="bottom center",
        )


def _configure_figure_layout(
    figure: Figure,
    titletext: str | None,
    logo: dict[str, Any],
    *,
    title: bool = True,
    add_logo: bool = True,
) -> None:
    """Configure figure layout.

    Args:
        figure: Plotly figure.
        title: Whether to add title.
        titletext: Optional title text.
        add_logo: Whether to add logo.
        logo: Logo dictionary.
    """
    figure.update_layout(
        xaxis={"tickformat": ".1%"},
        xaxis_title="volatility",
        yaxis={
            "tickformat": ".1%",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        yaxis_title="annual return",
        showlegend=False,
    )
    if title:
        if titletext is None:
            titletext = "<b>Risk and Return</b><br>"
        figure.update_layout(title={"text": titletext, "font": {"size": 32}})

    if add_logo:
        figure.add_layout_image(logo)


def _generate_sharpeplot_output(
    figure: Figure,
    plotfile: Path,
    filename: str,
    output_type: LiteralPlotlyOutput,
    include_plotlyjs: LiteralPlotlyJSlib,
    fig: dict[str, Any],
    *,
    auto_open: bool = True,
) -> str:
    """Generate output for sharpeplot.

    Args:
        figure: Plotly figure.
        plotfile: Path to plot file.
        filename: Filename.
        output_type: Output type.
        include_plotlyjs: How to include plotly.js.
        fig: Plotly figure dictionary.
        auto_open: Whether to auto-open.

    Returns:
        Output string.
    """
    if output_type == "file":
        plot(
            figure_or_data=figure,
            filename=str(plotfile),
            auto_open=auto_open,
            auto_play=False,
            link_text="",
            include_plotlyjs=include_plotlyjs,
            config=fig["config"],
            output_type=output_type,
        )
        return str(plotfile)

    div_id = filename.split(sep=".")[0]
    return cast(
        "str",
        to_html(
            fig=figure,
            config=fig["config"],
            auto_play=False,
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            div_id=div_id,
        ),
    )


def sharpeplot(
    sim_frame: DataFrame | None = None,
    line_frame: DataFrame | None = None,
    point_frame: DataFrame | None = None,
    point_frame_mode: LiteralLinePlotMode = "markers",
    filename: str | None = None,
    directory: DirectoryPath | None = None,
    titletext: str | None = None,
    output_type: LiteralPlotlyOutput = "file",
    include_plotlyjs: LiteralPlotlyJSlib = "cdn",
    *,
    title: bool = True,
    add_logo: bool = True,
    auto_open: bool = True,
) -> tuple[Figure, str]:
    """Create scatter plot coloured by Sharpe Ratio.

    Args:
        sim_frame: Data from the simulate_portfolios method.
        line_frame: Data from the efficient_frontier method.
        point_frame: Data to highlight current and efficient portfolios.
        point_frame_mode: Which type of scatter to use. Defaults to markers.
        filename: Name of the Plotly html file.
        directory: Directory where Plotly html file is saved.
        titletext: Text for the plot title.
        output_type: Determines output type. Defaults to "file".
        include_plotlyjs: Determines how the plotly.js library is included
            in the output.
            Defaults to "cdn".
        title: Whether to add standard plot title. Defaults to True.
        add_logo: Whether to add Captor logo. Defaults to True.
        auto_open: Determines whether to open a browser window with the plot.
            Defaults to True.

    Returns:
        The scatter plot with simulated and optimized results.
    """
    if sim_frame is None and line_frame is None and point_frame is None:
        msg = "One of sim_frame, line_frame or point_frame must be provided."
        raise AtLeastOneFrameError(msg)

    returns: list[float] = []
    risk: list[float] = []

    dirpath = _determine_output_directory(directory)
    if not filename:
        filename = "sharpeplot.html"
    plotfile = dirpath.joinpath(filename)

    fig, logo = load_plotly_dict()
    figure = Figure(fig)

    if sim_frame is not None:
        _add_simulated_portfolios_trace(figure, sim_frame, returns, risk)
    if line_frame is not None:
        _add_efficient_frontier_trace(figure, line_frame, returns, risk)
    if point_frame is not None:
        _add_point_frame_traces(
            figure, point_frame, point_frame_mode, fig, returns, risk
        )

    _configure_figure_layout(figure, titletext, logo, title=title, add_logo=add_logo)

    string_output = _generate_sharpeplot_output(
        figure,
        plotfile,
        filename,
        output_type,
        include_plotlyjs,
        fig,
        auto_open=auto_open,
    )

    return figure, string_output
