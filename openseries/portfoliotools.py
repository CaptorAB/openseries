"""Defining the portfolio tools for the OpenFrame class."""

from __future__ import annotations

from inspect import stack
from pathlib import Path
from typing import TYPE_CHECKING, cast

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

    simulated = simulate_portfolios(simframe=copi, num_ports=num_ports, seed=seed)

    frontier_min = simulated.loc[simulated["stdev"].idxmin()]["ret"]

    arithmetic_means = array(log_ret.mean() * copi.periods_in_a_year)
    cleaned_arithmetic_means = arithmetic_means[~isnan(arithmetic_means)]

    frontier_max = cleaned_arithmetic_means.max()

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

    def _neg_sharpe(weights: NDArray[float64]) -> float64:
        return cast(
            "float64",
            _get_ret_vol_sr(
                lg_ret=log_ret,
                weights=weights,
                per_in_yr=copi.periods_in_a_year,
            )[2]
            * -1,
        )

    def _minimize_volatility(
        weights: NDArray[float64],
    ) -> float64:
        return cast(
            "float64",
            _get_ret_vol_sr(
                lg_ret=log_ret,
                weights=weights,
                per_in_yr=copi.periods_in_a_year,
            )[1],
        )

    constraints = {"type": "eq", "fun": _check_sum}
    if not bounds:
        bounds = tuple((0.0, 1.0) for _ in range(eframe.item_count))
    init_guess = array(eframe.weights)

    opt_results = minimize(  # type: ignore[call-overload]
        fun=_neg_sharpe,
        x0=init_guess,
        method=minimize_method,
        bounds=bounds,
        constraints=constraints,
    )

    optimal = _get_ret_vol_sr(
        lg_ret=log_ret,
        weights=opt_results.x,
        per_in_yr=copi.periods_in_a_year,
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
                        per_in_yr=copi.periods_in_a_year,
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

    line_df = concat(
        [
            DataFrame(data=frontier_weights, columns=eframe.columns_lvl_zero),
            DataFrame({"stdev": frontier_x, "ret": frontier_y}),
        ],
        axis="columns",
    )
    line_df["sharpe"] = line_df.ret / line_df.stdev

    limit_small = 0.0001
    line_df = line_df.mask(line_df.abs() < limit_small, 0.0)
    line_df["text"] = line_df.apply(
        lambda c: "<br><br>Weights:<br>"
        + "<br>".join(
            [f"{c[nm]:.1%}  {nm}" for nm in eframe.columns_lvl_zero],
        ),
        axis="columns",
    )

    if tweak:
        limit_tweak = 0.001
        line_df["stdev_diff"] = line_df.stdev.ffill().pct_change()
        line_df = line_df.loc[line_df.stdev_diff.abs() > limit_tweak]
        line_df = line_df.drop(columns="stdev_diff")

    return line_df, simulated, append(optimal, opt_results.x)


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
    plotframe[current.label] = [current.arithmetic_ret, current.vol, txt]

    return plotframe


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
    returns = []
    risk = []

    if directory:
        dirpath = Path(directory).resolve()
    elif Path.home().joinpath("Documents").exists():
        dirpath = Path.home().joinpath("Documents")
    else:
        dirpath = Path(stack()[1].filename).parent

    if not filename:
        filename = "sharpeplot.html"
    plotfile = dirpath.joinpath(filename)

    fig, logo = load_plotly_dict()
    figure = Figure(fig)

    if sim_frame is None and line_frame is None and point_frame is None:
        msg = "One of sim_frame, line_frame or point_frame must be provided."
        raise AtLeastOneFrameError(msg)

    if sim_frame is not None:
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
    if line_frame is not None:
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

    if point_frame is not None:
        colorway = cast(  # type: ignore[index]
            "dict[str, str | int | float | bool | list[str]]",
            fig["layout"],
        ).get("colorway")[: len(point_frame.columns)]
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
        string_output = str(plotfile)
    else:
        div_id = filename.split(sep=".")[0]
        string_output = to_html(
            fig=figure,
            config=fig["config"],
            auto_play=False,
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            div_id=div_id,
        )

    return figure, string_output
