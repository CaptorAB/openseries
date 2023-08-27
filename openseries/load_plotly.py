"""Function to load plotly layout and configuration from local json file."""
from __future__ import annotations

from json import load
from os.path import abspath, dirname, join

from openseries.types import PlotlyLayoutType


def load_plotly_dict(
    responsive: bool = True,
) -> PlotlyLayoutType:
    """
    Load Plotly defaults.

    Parameters
    ----------
    responsive : bool
        Flag whether to load as responsive

    Returns
    -------
    PlotlyLayoutType
        A dictionary with the Plotly config and layout template
    """
    project_root = dirname(dirname(abspath(__file__)))
    layoutfile = join(abspath(project_root), "openseries", "plotly_layouts.json")
    logofile = join(abspath(project_root), "openseries", "plotly_captor_logo.json")

    with open(layoutfile, encoding="utf-8") as layout_file:
        fig = load(layout_file)
    with open(logofile, encoding="utf-8") as logo_file:
        logo = load(logo_file)

    fig["config"].update({"responsive": responsive})

    return fig, logo
