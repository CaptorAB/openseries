import json
import os


def load_plotly_dict() -> (dict, dict):

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layoutfile = os.path.join(
        os.path.abspath(project_root), "openseries", "plotly_layouts.json"
    )
    logofile = os.path.join(
        os.path.abspath(project_root), "openseries", "plotly_captor_logo.json"
    )
    with open(layoutfile, "r", encoding="utf-8") as f:
        fig = json.load(f)
    with open(logofile, "r", encoding="utf-8") as ff:
        logo = json.load(ff)

    return fig, logo
