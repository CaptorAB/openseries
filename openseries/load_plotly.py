"""Function to load plotly layout and configuration from local json file."""
from __future__ import annotations

from json import load
from pathlib import Path

from openseries.types import CaptorLogoType, PlotlyLayoutType


def load_plotly_dict(
    responsive: bool = True,  # noqa: FBT001, FBT002
) -> tuple[PlotlyLayoutType, CaptorLogoType]:
    """
    Load Plotly defaults.

    Parameters
    ----------
    responsive : bool
        Flag whether to load as responsive

    Returns
    -------
    tuple[PlotlyLayoutType, CaptorLogoType]
        A dictionary with the Plotly config and layout template
    """
    project_root = Path(__file__).resolve().parent.parent
    layoutfile = project_root.joinpath("openseries").joinpath("plotly_layouts.json")
    logofile = project_root.joinpath("openseries").joinpath("plotly_captor_logo.json")

    with Path.open(layoutfile, encoding="utf-8") as layout_file:
        fig = load(layout_file)
    with Path.open(logofile, encoding="utf-8") as logo_file:
        logo = load(logo_file)

    fig["config"].update({"responsive": responsive})

    return fig, logo
