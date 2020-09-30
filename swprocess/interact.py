"""Plot interaction module."""

# TODO (jpv): Replace hard-coded messaged with user-defined inputs. Generator?
def _ginput_session(ax, initial_adjustment=True, ask_to_continue=True,
                    npts=None):  # pragma: no cover
    """Start ginput session using the provided axes object.

    Parameters
    ----------
    ax : Axes
        Axes on which points are to be selected.
    initial_adjustment : bool, optional
        Allow user to pan and zoom prior to the selection of the
        first point, default is `True`.
    ask_to_continue : bool, optional
        Pause the selection process after each point. This allows
        the user to pan and zoom the figure as well as select when
        to continue, default is `True`.
    npts : int, optional
        Predefine the number of points the user is allowed to
        select, the default is `None` which allows the selection of
        an infinite number of points.

    Returns
    -------
    tuple
        Of the form `(xs, ys)` where `xs` is a `list` of x
        coordinates and `ys` is a `list` of y coordinates in the
        order in which they were picked.

    """
    # Set npts to infinity if npts is None
    if npts is None:
        npts = np.inf

    # Enable cursor to make precise selection easier.
    cursor = Cursor(ax, useblit=True, color='k', linewidth=1)

    # Permit initial adjustment with blocking call to figure.
    if initial_adjustment:
        print("Adjust view, spacebar when ready.")
        while True:
            if plt.waitforbuttonpress(timeout=-1):
                break

    # Begin selection of npts.
    npt, xs, ys = 0, [], []
    while npt < npts:
        print("Left click to add, right click to remove, enter to accept.")
        vals = plt.ginput(n=-1, timeout=0)
        x, y = vals[-1]
        ax.plot(x, y, "r", marker="+", linestyle="")
        xs.append(x)
        ys.append(y)
        npt += 1

        if ask_to_continue:
            print("Press spacebar once to contine, twice to exit)")
            while True:
                if plt.waitforbuttonpress(timeout=-1):
                    break

        if plt.waitforbuttonpress(timeout=0.5):
            print("Exiting ... ")
            break
    print("Close figure when ready.")

    return (xs, ys)
