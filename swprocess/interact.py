"""Plot interaction module."""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor


def ginput_session(ax, initial_adjustment=True,
                   initial_adjustment_message=None, npts=1,
                   ask_to_continue=True,
                   ask_to_continue_message=None):
    """Start ginput session using the provided axes object.

    Parameters
    ----------
    ax : Axes
        Axes on which points are to be selected.
    initial_adjustment : bool, optional
        Allow user to pan and zoom prior to the selection of the
        first point, default is `True`.
    initial_adjustment_message : str, optional
        Message to print and display during `initial_adjustment` stage,
        default is `None` so predefined message is displayed.
    npts : int, optional
        Predefine the number of points the user is allowed to
        select, the default is `1`.
    ask_to_continue : bool, optional
        Pause the selection process after each point. This allows
        the user to pan and zoom the figure as well as select when
        to continue, default is `True`.
    ask_to_continue_message : str, optional
        Message to print and display prior to select stage,
        default is `None` so predefined message is displayed.

    Returns
    -------
    tuple
        Of the form `(xs, ys)` where `xs` is a `list` of x
        coordinates and `ys` is a `list` of y coordinates in the
        order in which they were picked.

    """
    # Enable cursor to make precise selection easier.
    cursor = Cursor(ax, color='k', linewidth=1)

    # Permit initial adjustment with blocking call to figure.
    if initial_adjustment:
        if initial_adjustment_message is None:
            initial_adjustment_message = "Adjust view,\nspacebar when ready."
        text = ax.text(0.95, 0.95, initial_adjustment_message,
                       ha="right", va="top", transform=ax.transAxes)
        while True:
            if plt.waitforbuttonpress(timeout=-1):
                text.set_visible(False)
                break

    # Begin selection of npts.
    npt, xs, ys = 0, [], []
    while npt < npts:
        selection_message = "Left click to add,\nright click to remove,\nenter to accept."
        text = ax.text(0.95, 0.95, selection_message,
                       ha="right", va="top", transform=ax.transAxes)
        vals = plt.ginput(n=-1, timeout=0)
        text.set_visible(False)

        if len(vals) > 1:
            msg = "More than one point selected, ignoring all but the last point."
            warnings.warn(msg)

        if len(vals) == 0:
            msg = "No points selected, try again."
            warnings.warn(msg)
            continue

        x, y = vals[-1]
        ax.plot(x, y, "r", marker="+", linestyle="")
        xs.append(x)
        ys.append(y)
        npt += 1

        if ask_to_continue:
            if ask_to_continue_message is None:
                ask_to_continue_message = "Adjust view,\npress spacebar\nonce to contine,\ntwice to exit."
            text = ax.text(0.95, 0.95, ask_to_continue_message,
                           ha="right", va="top",
                           transform=ax.transAxes)
            while True:
                if plt.waitforbuttonpress(timeout=-1):
                    text.set_visible(False)
                    break

        if plt.waitforbuttonpress(timeout=0.3):
            break

    finish_message = "Interactive session complete,\nclose figure(s) when ready."
    text = ax.text(0.95, 0.95, finish_message,
                   ha="right", va="top",
                   transform=ax.transAxes)

    return (xs, ys)
