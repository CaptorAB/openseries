"""
Function to load plotly layout and configuration from local json file
"""
from json import load
from os.path import abspath, dirname, join
from typing import Dict, List, Tuple, Union


def load_plotly_dict(
    responsive: bool = True,
) -> Tuple[
    Dict[
        str,
        Union[
            str,
            int,
            float,
            bool,
            List[str],
            Dict[str, str | int | float | bool | List[str]],
        ],
    ],
    Dict[str, Union[str, float]],
]:
    """Function to load the plotly defaults

    Parameters
    ----------
    responsive : bool

    Returns
    -------
    A dictionary with the Plotly config and layout template
    """
    project_root = dirname(dirname(abspath(__file__)))
    layoutfile = join(abspath(project_root), "openseries", "plotly_layouts.json")
    logofile = join(abspath(project_root), "openseries", "plotly_captor_logo.json")

    with open(layoutfile, "r", encoding="utf-8") as layout_file:
        fig = load(layout_file)
    with open(logofile, "r", encoding="utf-8") as logo_file:
        logo = load(logo_file)

    fig["config"].update({"responsive": responsive})

    return fig, logo
