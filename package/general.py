import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .gui_utils import Asc, plot_asc


def check_plot_tdoas(
    doaEstTarget: list[float], doaEstAll: list[float], asc: Asc
) -> Figure:
    """
    Generates a plot of the acoustic scenario to check the validity of TDOA
    estimation obtained, e.g., via the frequency-domain MUSIC algorithm.

    Paramters
    ---------
    doaEstTarget : list of floats
        Estimated DOA for target talker(s).
    doaEstAll : list of floats
        All estimated DOAs (target talker(s) and noise source(s)).
    asc : RIRg_GUI class instance
        Acoustic scenario parameters.
    """

    def _get_line(doaRadiants, startCoords, roomDim):
        """
        Get 2D line (x and y coordinates) from DOA.
        """
        x0, y0 = startCoords[0], startCoords[1]
        x1 = x0 + roomDim * np.cos(doaRadiants)
        y1 = y0 + roomDim * np.sin(doaRadiants)
        return [x0, x1], [y0, y1]

    # Check inputs
    if isinstance(doaEstTarget, float):
        doaEstTarget = [doaEstTarget]
    if isinstance(doaEstAll, float):
        doaEstAll = [doaEstAll]  # convert to list

    # Plot room
    fig = plot_asc(asc)
    # Plot DOA estimates
    for ii in range(len(doaEstAll)):
        x, y = _get_line(
            doaRadiants=doaEstAll[ii] + np.pi / 2,  # align to space orientation
            startCoords=np.mean(asc.micPos, axis=0),
            roomDim=asc.roomDim,
        )
        if doaEstAll[ii] in doaEstTarget:
            rayColor = "g"  # indicate selected (target) DOAs with green
        else:
            rayColor = "0.7"
        plt.plot(x, y, "--", color=rayColor)
        plt.text(
            np.mean(x),
            np.mean(y),
            f"$\\hat{{\\theta}}_{ii + 1}$={np.round(doaEstAll[ii] * 180 / np.pi, 2)}$^\\circ$",
            c=rayColor,
        )
    return fig


def select_latest_rir(path: str) -> Path:
    """
    Returns the full path to the latest RIR that was generated (via the
    RIR-generating GUI) in the folder `path`.
    """

    p = Path(path).glob("**/*")
    files = [x for x in p if x.is_file()]
    files.sort(key=os.path.getmtime)
    if str(files[-1])[-3:] == "pkl":
        rirFile = files[-1]
    else:
        rirFile = files[-2]

    return rirFile
