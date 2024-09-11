"""Defining the portfolio tools for the OpenFrame class."""

# mypy: disable-error-code="index,assignment"
from __future__ import annotations

from inspect import stack
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

from numpy import (
    append,
    array,
    float64,
    inf,
    isnan,
    linspace,
    nan,
    sqrt,
    zeros,
)
from numpy import (
    sum as npsum,
)
from numpy.typing import NDArray
from pandas import (
    DataFrame,
    Series,
    concat,
)
from plotly.graph_objs import Figure  # type: ignore[import-untyped,unused-ignore]
from plotly.io import to_html  # type: ignore[import-untyped,unused-ignore]
from plotly.offline import plot  # type: ignore[import-untyped,unused-ignore]
from scipy.optimize import minimize  # type: ignore[import-untyped,unused-ignore]

from .load_plotly import load_plotly_dict
from .series import OpenTimeSeries

# noinspection PyProtectedMember
from .simulation import _random_generator
from .types import (
    LiteralLinePlotMode,
    LiteralMinimizeMethods,
    LiteralPlotlyJSlib,
    LiteralPlotlyOutput,
    ValueType,
)

if TYPE_CHECKING:  # pragma: no cover
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

    Parameters
    ----------
    simframe: OpenFrame
        Return data for portfolio constituents
    num_ports: int
        Number of possible portfolios to simulate
    seed: int
        The seed for the random process

    Returns
    -------
    pandas.DataFrame
        The resulting data

    """
    copi = simframe.from_deepcopy()

    if any(
        x == ValueType.PRICE for x in copi.tsdf.columns.get_level_values(1).to_numpy()
    ):
        copi.value_to_ret()
        log_ret = copi.tsdf.copy()[1:]
    else:
        log_ret = copi.tsdf.copy()

    log_ret.columns = log_ret.columns.droplevel(level=1)

    randomizer = _random_generator(seed=seed)

    all_weights = zeros((num_ports, simframe.item_count))
    ret_arr = zeros(num_ports)
    vol_arr = zeros(num_ports)
    sharpe_arr = zeros(num_ports)

    for x in range(num_ports):
        weights = array(randomizer.random(simframe.item_count))
        weights = weights / npsum(weights)
        all_weights[x, :] = weights

        vol_arr[x] = sqrt(
            weights.T @ (log_ret.cov() * simframe.periods_in_a_year @ weights),
        )

        ret_arr[x] = npsum(log_ret.mean() * weights * simframe.periods_in_a_year)

        sharpe_arr[x] = ret_arr[x] / vol_arr[x]

    simdf = concat(
        [
            DataFrame({"stdev": vol_arr, "ret": ret_arr, "sharpe": sharpe_arr}),
            DataFrame(all_weights, columns=simframe.columns_lvl_zero),
        ],
        axis="columns",
    )
    simdf = simdf.replace([inf, -inf], nan)
    return simdf.dropna()


# noinspection PyUnusedLocal
def efficient_frontier(  # noqa: C901
    eframe: OpenFrame,
    num_ports: int = 5000,
    seed: int = 71,
    bounds: tuple[tuple[float]] | None = None,
    frontier_points: int = 200,
    minimize_method: LiteralMinimizeMethods = "SLSQP",
    *,
    tweak: bool = True,
) -> tuple[DataFrame, DataFrame, NDArray[float64]]:
    """Identify an efficient frontier.

    Parameters
    ----------
    eframe: OpenFrame
        Portfolio data
    num_ports: int, default: 5000
        Number of possible portfolios to simulate
    seed: int, default: 71
        The seed for the random process
    bounds: tuple[tuple[float]], optional
        The range of minumum and maximum allowed allocations for each asset
    frontier_points: int, default: 200
        number of points along frontier to optimize
    minimize_method: LiteralMinimizeMethods, default: SLSQP
        The method passed into the scipy.minimize function
    tweak: bool, default: True
        cutting the frontier to exclude multiple points with almost the same risk

    Returns
    -------
    tuple[DataFrame, DataFrame, NDArray[float]]
        The efficient frontier data, simulation data and optimal portfolio

    """
    if eframe.weights is None:
        eframe.weights = [1.0 / eframe.item_count] * eframe.item_count

    copi = eframe.from_deepcopy()

    if any(
        x == ValueType.PRICE for x in copi.tsdf.columns.get_level_values(1).to_numpy()
    ):
        copi.value_to_ret()
        log_ret = copi.tsdf.copy()[1:]
    else:
        log_ret = copi.tsdf.copy()

    log_ret.columns = log_ret.columns.droplevel(level=1)

    simulated = simulate_portfolios(simframe=copi, num_ports=num_ports, seed=seed)

    frontier_min = simulated.loc[simulated["stdev"].idxmin()]["ret"]

    arithmetic_means = array(log_ret.mean() * copi.periods_in_a_year)
    cleaned_arithmetic_means = arithmetic_means[~isnan(arithmetic_means)]

    frontier_max = cleaned_arithmetic_means.max()

    def _check_sum(weights: NDArray[float64]) -> float64:
        return cast(float64, npsum(weights) - 1)

    def _get_ret_vol_sr(
        lg_ret: DataFrame,
        weights: NDArray[float64],
        per_in_yr: float,
    ) -> NDArray[float64]:
        ret = npsum(lg_ret.mean() * weights) * per_in_yr
        volatility = sqrt(weights.T @ (lg_ret.cov() * per_in_yr @ weights))
        sr = ret / volatility
        return cast(NDArray[float64], array([ret, volatility, sr]))

    def _diff_return(
        lg_ret: DataFrame,
        weights: NDArray[float64],
        per_in_yr: float,
        poss_return: float,
    ) -> float64:
        return cast(
            float64,
            _get_ret_vol_sr(lg_ret=lg_ret, weights=weights, per_in_yr=per_in_yr)[0]
            - poss_return,
        )

    def _neg_sharpe(weights: NDArray[float64]) -> float64:
        return cast(
            float64,
            _get_ret_vol_sr(
                lg_ret=log_ret,
                weights=weights,
                per_in_yr=eframe.periods_in_a_year,
            )[2]
            * -1,
        )

    def _minimize_volatility(
        weights: NDArray[float64],
    ) -> float64:
        return cast(
            float64,
            _get_ret_vol_sr(
                lg_ret=log_ret,
                weights=weights,
                per_in_yr=eframe.periods_in_a_year,
            )[1],
        )

    constraints = {"type": "eq", "fun": _check_sum}
    if not bounds:
        bounds = tuple((0.0, 1.0) for _ in range(eframe.item_count))
    init_guess = array(eframe.weights)

    opt_results = minimize(
        fun=_neg_sharpe,
        x0=init_guess,
        method=minimize_method,
        bounds=bounds,
        constraints=constraints,
    )

    optimal = _get_ret_vol_sr(
        lg_ret=log_ret,
        weights=opt_results.x,
        per_in_yr=eframe.periods_in_a_year,
    )

    frontier_y = linspace(start=frontier_min, stop=frontier_max, num=frontier_points)
    frontier_x = []
    frontier_weights = []

    for possible_return in frontier_y:
        cons = cast(
            dict[str, str | Callable[[float, NDArray[float64]], float64]],
            (
                {"type": "eq", "fun": _check_sum},
                {
                    "type": "eq",
                    "fun": lambda w, poss_return=possible_return: _diff_return(
                        lg_ret=log_ret,
                        weights=w,
                        per_in_yr=eframe.periods_in_a_year,
                        poss_return=poss_return,
                    ),
                },
            ),
        )

        result = minimize(
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
        line_df["stdev_diff"] = line_df.stdev.pct_change()
        line_df = line_df.loc[line_df.stdev_diff.abs() > limit_tweak]
        line_df = line_df.drop(columns="stdev_diff")

    return line_df, simulated, append(optimal, opt_results.x)


def constrain_optimized_portfolios(
    data: OpenFrame,
    serie: OpenTimeSeries,
    portfolioname: str = "Current Portfolio",
    simulations: int = 10000,
    curve_points: int = 200,
    bounds: tuple[tuple[float]] | None = None,
    minimize_method: LiteralMinimizeMethods = "SLSQP",
) -> tuple[OpenFrame, OpenTimeSeries, OpenFrame, OpenTimeSeries]:
    """Constrain optimized portfolios to those that improve on the current one.

    Parameters
    ----------
    data: OpenFrame
        Portfolio data
    serie: OpenTimeSeries
        A
    portfolioname: str, default: "Current Portfolio"
        Name of the portfolio
    simulations: int, default: 10000
        Number of possible portfolios to simulate
    curve_points: int, default: 200
        Number of optimal portfolios on the efficient frontier
    bounds: tuple[tuple[float]], optional
        The range of minumum and maximum allowed allocations for each asset
    minimize_method: LiteralMinimizeMethods, default: SLSQP
        The method passed into the scipy.minimize function

    Returns
    -------
    tuple[OpenFrame, OpenTimeSeries, OpenFrame, OpenTimeSeries]
        The constrained optimal portfolio data

    """
    lr_frame = data.from_deepcopy()
    mv_frame = data.from_deepcopy()

    if not bounds:
        bounds = tuple((0.0, 1.0) for _ in range(data.item_count))

    front_frame, sim_frame, optimal = efficient_frontier(
        eframe=data,
        num_ports=simulations,
        frontier_points=curve_points,
        bounds=bounds,
        minimize_method=minimize_method,
    )

    condition_least_ret = front_frame.ret > serie.arithmetic_ret
    # noinspection PyArgumentList
    least_ret_frame = front_frame[condition_least_ret].sort_values(by="stdev")
    least_ret_port = least_ret_frame.iloc[0]
    least_ret_port_name = f"Minimize vol & target return of {portfolioname}"
    least_ret_weights = [least_ret_port[c] for c in lr_frame.columns_lvl_zero]
    lr_frame.weights = least_ret_weights
    resleast = OpenTimeSeries.from_df(lr_frame.make_portfolio(least_ret_port_name))

    condition_most_vol = front_frame.stdev < serie.vol
    # noinspection PyArgumentList
    most_vol_frame = front_frame[condition_most_vol].sort_values(
        by="ret",
        ascending=False,
    )
    most_vol_port = most_vol_frame.iloc[0]
    most_vol_port_name = f"Maximize return & target risk of {portfolioname}"
    most_vol_weights = [most_vol_port[c] for c in mv_frame.columns_lvl_zero]
    mv_frame.weights = most_vol_weights
    resmost = OpenTimeSeries.from_df(mv_frame.make_portfolio(most_vol_port_name))

    return lr_frame, resleast, mv_frame, resmost


def prepare_plot_data(
    assets: OpenFrame,
    current: OpenTimeSeries,
    optimized: NDArray[float64],
) -> DataFrame:
    """Prepare date to be used as point_frame in the sharpeplot function.

    Parameters
    ----------
    assets: OpenFrame
        Portfolio data with individual assets and a weighted portfolio
    current: OpenTimeSeries
        The current or initial portfolio based on given weights
    optimized: DataFrame
        Data optimized with the efficient_frontier method

    Returns
    -------
    DataFrame
        The data prepared with mean returns, volatility and weights

    """
    txt = "<br><br>Weights:<br>" + "<br>".join(
        [
            f"{wgt:.1%}  {nm}"
            for wgt, nm in zip(
                cast(list[float], assets.weights),
                assets.columns_lvl_zero,
            )
        ],
    )

    opt_text_list = [
        f"{wgt:.1%}  {nm}" for wgt, nm in zip(optimized[3:], assets.columns_lvl_zero)
    ]
    opt_text = "<br><br>Weights:<br>" + "<br>".join(opt_text_list)
    vol: Series[float] = assets.vol
    plotframe = DataFrame(
        data=[
            assets.arithmetic_ret,
            vol,
            Series(
                data=[""] * assets.item_count,
                index=vol.index,
            ),
        ],
        index=["ret", "stdev", "text"],
    )
    plotframe.columns = plotframe.columns.droplevel(level=1)
    plotframe["Max Sharpe Portfolio"] = [optimized[0], optimized[1], opt_text]
    plotframe[current.label] = [current.arithmetic_ret, current.vol, txt]

    return plotframe


def sharpeplot(  # noqa: C901
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

    Parameters
    ----------
    sim_frame: DataFrame, optional
        Data from the simulate_portfolios method.
    line_frame: DataFrame, optional
        Data from the efficient_frontier method.
    point_frame: DataFrame, optional
        Data to highlight current and efficient portfolios.
    point_frame_mode: LiteralLinePlotMode, default: markers
        Which type of scatter to use.
    filename: str, optional
        Name of the Plotly html file
    directory: DirectoryPath, optional
        Directory where Plotly html file is saved
    titletext: str, optional
        Text for the plot title
    output_type: LiteralPlotlyOutput, default: "file"
        Determines output type
    include_plotlyjs: LiteralPlotlyJSlib, default: "cdn"
        Determines how the plotly.js library is included in the output
    title: bool, default: True
        Whether to add standard plot title
    add_logo: bool, default: True
        Whether to add Captor logo
    auto_open: bool, default: True
        Determines whether to open a browser window with the plot

    Returns
    -------
    Figure
        The scatter plot with simulated and optimized results

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
        raise ValueError(msg)

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
        colorway = cast(
            dict[str, str | int | float | bool | list[str]],
            fig["layout"],
        ).get("colorway")[: len(point_frame.columns)]
        for col, clr in zip(point_frame.columns, colorway):
            returns.extend([point_frame.loc["ret", col]])
            risk.extend([point_frame.loc["stdev", col]])
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
            include_plotlyjs=cast(bool, include_plotlyjs),
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
            include_plotlyjs=cast(bool, include_plotlyjs),
            full_html=False,
            div_id=div_id,
        )

    return figure, string_output
