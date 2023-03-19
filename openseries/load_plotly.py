from json import load
from os.path import abspath, dirname, join
from typing import Dict, List, Tuple


def load_plotly_dict(
    responsive: bool = True,
) -> Tuple[
    Dict[
        str,
        str
        | int
        | float
        | bool
        | List[str]
        | Dict[str, str | int | float | bool | List[str]],
    ],
    Dict[str, str | float],
]:
    project_root = dirname(dirname(abspath(__file__)))
    layoutfile = join(abspath(project_root), "openseries", "plotly_layouts.json")
    logofile = join(abspath(project_root), "openseries", "plotly_captor_logo.json")

    with open(layoutfile, "r", encoding="utf-8") as f:
        fig = load(f)
    with open(logofile, "r", encoding="utf-8") as ff:
        logo = load(ff)

    fig["config"].update({"responsive": responsive})

    return fig, logo
