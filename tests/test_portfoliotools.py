"""Test suite for the openseries/portfoliotools.py module."""

# mypy: disable-error-code="arg-type"
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, localcontext
from json import loads
from pathlib import Path
from typing import Optional, cast
from unittest.mock import patch

from openseries.portfoliotools import (
    constrain_optimized_portfolios,
    efficient_frontier,
    prepare_plot_data,
    sharpeplot,
    simulate_portfolios,
)
from openseries.series import OpenTimeSeries
from tests.test_common_sim import CommonTestCase


class TestPortfoliotools(CommonTestCase):
    """class to run tests on the module portfoliotools.py."""

    def test_simulate_portfolios(self: TestPortfoliotools) -> None:
        """Test function simulate_portfolios."""
        simulations = 100

        spframe = self.randomframe.from_deepcopy()

        result_returns = simulate_portfolios(
            simframe=spframe,
            num_ports=simulations,
            seed=self.seed,
        )

        if result_returns.shape != (simulations, spframe.item_count + 3):
            msg = "Function simulate_portfolios not working as intended"
            raise ValueError(msg)

        return_least_vol = f"{result_returns.loc[:, 'stdev'].min():.7f}"
        return_where_least_vol = (
            f"{result_returns.loc[result_returns['stdev'].idxmin()]['ret']:.7f}"
        )

        if (return_least_vol, return_where_least_vol) != ("0.0476781", "0.0558805"):
            msg = (
                "Function simulate_portfolios not working as intended"
                f"\n{(return_least_vol, return_where_least_vol)}"
            )
            raise ValueError(msg)

        spframe.to_cumret()
        result_values = simulate_portfolios(
            simframe=spframe,
            num_ports=simulations,
            seed=self.seed,
        )

        if result_values.shape != (simulations, spframe.item_count + 3):
            msg = "Function simulate_portfolios not working as intended"
            raise ValueError(msg)

        value_least_vol = f"{result_values.loc[:, 'stdev'].min():.7f}"
        value_where_least_vol = (
            f"{result_values.loc[result_values['stdev'].idxmin()]['ret']:.7f}"
        )

        if (value_least_vol, value_where_least_vol) != ("0.0476875", "0.0559028"):
            msg = (
                "Function simulate_portfolios not working as intended"
                f"\n{(value_least_vol, value_where_least_vol)}"
            )
            raise ValueError(msg)

    def test_efficient_frontier(self: TestPortfoliotools) -> None:
        """Test function efficient_frontier."""
        simulations = 100
        points = 20
        with localcontext() as decimal_context:
            decimal_context.rounding = ROUND_HALF_UP

            eframe = self.randomframe.from_deepcopy()

            frnt, _, _ = efficient_frontier(
                eframe=eframe,
                num_ports=simulations,
                seed=self.seed,
                frontier_points=points,
                tweak=False,
            )

            if frnt.shape != (points, eframe.item_count + 4):
                msg = "Function efficient_frontier not working as intended"
                raise ValueError(msg)

            eframe.to_cumret()

            frontier, result, optimal = efficient_frontier(
                eframe=eframe,
                num_ports=simulations,
                seed=self.seed,
                frontier_points=points,
                tweak=False,
            )

            if frontier.shape != (points, eframe.item_count + 4):
                msg = "Function efficient_frontier not working as intended"
                raise ValueError(msg)

            frt_most_sharpe = round(Decimal(frontier.loc[:, "sharpe"].max()), 6)
            frt_return_where_most_sharpe = round(
                Decimal(float(frontier.loc[frontier["sharpe"].idxmax()]["ret"])),
                6,
            )

            if (frt_most_sharpe, frt_return_where_most_sharpe) != (
                Decimal("1.302126"),
                Decimal("0.067698"),
            ):
                msg = (
                    "Function efficient_frontier not working as intended"
                    f"\n{(frt_most_sharpe, frt_return_where_most_sharpe)}"
                )
                raise ValueError(msg)

            sim_least_vol = round(Decimal(result.loc[:, "stdev"].min()), 6)
            sim_return_where_least_vol = round(
                Decimal(float(result.loc[result["stdev"].idxmin()]["ret"])),
                6,
            )

            if (sim_least_vol, sim_return_where_least_vol) != (
                Decimal("0.047678"),
                Decimal("0.055881"),
            ):
                msg = (
                    "Function efficient_frontier not working as intended"
                    f"\n{(sim_least_vol, sim_return_where_least_vol)}"
                )
                raise ValueError(msg)

            optlist = [round(Decimal(wgt), 6) for wgt in cast(list[float], optimal)]
            total = sum(optimal[3:])

            if round(total, 7) != 1.0:
                msg = f"Function efficient_frontier not working as intended\n{total}"
                raise ValueError(msg)

            if optlist != [
                Decimal("0.068444"),
                Decimal("0.052547"),
                Decimal("1.302525"),
                Decimal("0.116616"),
                Decimal("0.140094"),
                Decimal("0.352682"),
                Decimal("0.312324"),
                Decimal("0.078283"),
            ]:
                msg = f"Function efficient_frontier not working as intended\n{optlist}"
                raise ValueError(msg)

    def test_constrain_optimized_portfolios(self: TestPortfoliotools) -> None:
        """Test function constrain_optimized_portfolios."""
        simulations = 100
        curve_points = 20
        org_port_name = "Current Portfolio"

        std_frame = self.randomframe.from_deepcopy()
        std_frame.to_cumret()
        std_frame.weights = [1 / std_frame.item_count] * std_frame.item_count

        bounds = cast(
            Optional[tuple[tuple[float]]],
            tuple((0.0, 1.0) for _ in range(std_frame.item_count)),
        )

        assets_std = OpenTimeSeries.from_df(std_frame.make_portfolio(org_port_name))

        minframe, minseries, maxframe, maxseries = constrain_optimized_portfolios(
            data=std_frame,
            serie=assets_std,
            portfolioname=org_port_name,
            simulations=simulations,
            curve_points=curve_points,
            bounds=bounds,
        )

        minframe_nb, _, _, _ = constrain_optimized_portfolios(
            data=std_frame,
            serie=assets_std,
            portfolioname=org_port_name,
            simulations=simulations,
            curve_points=curve_points,
        )

        if round(sum(minframe.weights), 7) != 1.0:
            msg = (
                "Function constrain_optimized_portfolios not working as "
                f"intended\n{round(sum(minframe.weights), 7)}"
            )
            raise ValueError(msg)

        minframe_weights = [f"{minw:.7f}" for minw in list(minframe.weights)]
        if minframe_weights != [
            "0.1150512",
            "0.2045890",
            "0.2412361",
            "0.2352707",
            "0.2038530",
        ]:
            msg = (
                "Function constrain_optimized_portfolios not "
                f"working as intended\n{minframe_weights}"
            )
            raise ValueError(msg)

        if round(sum(minframe_nb.weights), 7) != 1.0:
            msg = (
                "Function constrain_optimized_portfolios not working as "
                f"intended\n{round(sum(minframe_nb.weights), 7)}"
            )
            raise ValueError(msg)

        minframe_nb_weights = [f"{minw:.7f}" for minw in list(minframe_nb.weights)]
        if minframe_nb_weights != [
            "0.1150512",
            "0.2045890",
            "0.2412361",
            "0.2352707",
            "0.2038530",
        ]:
            msg = (
                "Function constrain_optimized_portfolios not "
                f"working as intended\n{minframe_nb_weights}"
            )
            raise ValueError(msg)

        if (
            f"{minseries.arithmetic_ret - assets_std.arithmetic_ret:.7f}"
            != "0.0016062"
        ):
            msg = (
                "Optimization did not find better return with similar vol\n"
                f"{minseries.arithmetic_ret - assets_std.arithmetic_ret:.7f}"
            )

            raise ValueError(msg)

        if round(sum(maxframe.weights), 7) != 1.0:
            msg = (
                "Function constrain_optimized_portfolios not working as "
                f"intended\n{round(sum(maxframe.weights), 7)}"
            )
            raise ValueError(msg)

        maxframe_weights = [f"{maxw:.7f}" for maxw in list(maxframe.weights)]
        if maxframe_weights != [
            "0.1151792",
            "0.1738639",
            "0.2942018",
            "0.2704667",
            "0.1462884",
        ]:
            msg = (
                "Function constrain_optimized_portfolios not "
                f"working as intended\n{maxframe_weights}"
            )
            raise ValueError(msg)

        if f"{assets_std.vol - maxseries.vol:.7f}" != "0.0001994":
            msg = (
                "Optimization did not find better return with similar vol\n"
                f"{assets_std.vol - maxseries.vol:.7f}"
            )

            raise ValueError(msg)

    def test_sharpeplot(self: TestPortfoliotools) -> None:  # noqa: C901
        """Test function sharpeplot."""
        simulations = 100
        points = 20

        spframe = self.randomframe.from_deepcopy()
        spframe.to_cumret()
        current = OpenTimeSeries.from_df(
            spframe.make_portfolio(
                name="Current Portfolio",
                weight_strat="eq_weights",
            ),
        )

        frontier, simulated, optimum = efficient_frontier(
            eframe=spframe,
            num_ports=simulations,
            seed=self.seed,
            frontier_points=points,
            tweak=False,
        )

        plotframe = prepare_plot_data(
            assets=spframe,
            current=current,
            optimized=optimum,
        )

        figure_title_no_text, _ = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=True,
            auto_open=False,
            output_type="div",
        )

        fig_json_title_no_text = loads(cast(str, figure_title_no_text.to_json()))
        if "Risk and Return" not in fig_json_title_no_text["layout"]["title"]["text"]:
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        figure_title_text, _ = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=True,
            titletext="Awesome title",
            auto_open=False,
            output_type="div",
        )

        fig_json_title_text = loads(cast(str, figure_title_text.to_json()))
        if fig_json_title_text["layout"]["title"]["text"] != "Awesome title":
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        figure, _ = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=False,
            auto_open=False,
            output_type="div",
        )

        fig_json = loads(cast(str, figure.to_json()))

        if "text" in fig_json["layout"]["title"]:
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        names = [item["name"] for item in fig_json["data"]]

        if names != [
            "simulated portfolios",
            "Efficient frontier",
            "Asset_0",
            "Asset_1",
            "Asset_2",
            "Asset_3",
            "Asset_4",
            "Max Sharpe Portfolio",
            "Current Portfolio",
        ]:
            msg = f"Function sharpeplot not working as intended\n{names}"
            raise ValueError(msg)

        directory = Path(__file__).resolve().parent
        _, figfile = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=False,
            auto_open=False,
            output_type="file",
            directory=directory,
        )

        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        if figfile[:5] == "<div>":
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        _, divstring = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=False,
            auto_open=False,
            output_type="div",
        )
        if divstring[:5] != "<div>" or divstring[-6:] != "</div>":
            msg = "Html div section not created"
            raise ValueError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = True
            mockhomefig, _ = sharpeplot(
                sim_frame=simulated,
                line_frame=frontier,
                point_frame=plotframe,
                point_frame_mode="markers+text",
                title=False,
                auto_open=False,
                output_type="div",
            )
            mockhomefig_json = loads(cast(str, mockhomefig.to_json()))

        if mockhomefig_json["data"][0]["name"] != "simulated portfolios":
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = False
            _, mockfile = sharpeplot(
                sim_frame=simulated,
                line_frame=frontier,
                point_frame=plotframe,
                point_frame_mode="markers+text",
                title=False,
                auto_open=False,
                output_type="file",
                filename="seriesfile.html",
            )
            mockfilepath = Path(mockfile).resolve()

        if mockfilepath.parts[-2:] != ("tests", "seriesfile.html"):
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        mockfilepath.unlink()
