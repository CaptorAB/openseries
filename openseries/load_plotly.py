"""Function to load plotly layout and configuration from local json file.

Copyright (c) Captor Fund Management AB. This file is part of the openseries project.

Licensed under the BSD 3-Clause License. You may obtain a copy of the License at:
https://github.com/CaptorAB/openseries/blob/master/LICENSE.md
SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

from json import load
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError

if TYPE_CHECKING:
    from .owntypes import CaptorLogoType, PlotlyLayoutType  # pragma: no cover

logger = getLogger(__name__)

__all__ = ["load_plotly_dict"]


def _check_remote_file_existence(url: str) -> bool:
    """Check if remote file exists.

    Parameters
    ----------
    url: str
        Path to remote file

    Returns:
    -------
    bool
        True if url is valid and False otherwise

    """
    ok_code = 200

    try:
        response = requests.head(url, timeout=30)
        if response.status_code != ok_code:
            return False
    except RequestsConnectionError:
        return False
    return True


def load_plotly_dict(
    *,
    responsive: bool = True,
) -> tuple[PlotlyLayoutType, CaptorLogoType]:
    """Load Plotly defaults.

    Parameters
    ----------
    responsive : bool
        Flag whether to load as responsive

    Returns:
    -------
    tuple[PlotlyLayoutType, CaptorLogoType]
        A dictionary with the Plotly config and layout template

    """
    project_root = Path(__file__).parent.parent
    layoutfile = project_root.joinpath("openseries").joinpath("plotly_layouts.json")
    logofile = project_root.joinpath("openseries").joinpath("plotly_captor_logo.json")

    with layoutfile.open(mode="r", encoding="utf-8") as layout_file:
        fig = load(layout_file)
    with logofile.open(mode="r", encoding="utf-8") as logo_file:
        logo = load(logo_file)

    if not _check_remote_file_existence(url=logo["source"]):
        msg = f"Failed to add logo image from URL {logo['source']}"
        logger.warning(msg)
        logo = {}

    fig["config"].update({"responsive": responsive})

    return fig, logo
