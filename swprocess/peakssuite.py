"""PeaksSuite class definition."""

import json
import warnings
import logging

import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from .wavefieldtransforms import AbstractWavefieldTransform as AWTransform
from .peaks import Peaks
from .regex import get_all

logger = logging.getLogger("swprocess.peakssuite")


class PeaksSuite():

    def __init__(self, peaks):
        """Instantiate a `PeaksSuite` object from a `Peaks` object.

        Parameters
        ----------
        peaks : Peaks
            A `Peaks` object to include in the suite.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        self._check_input(peaks)
        self.peaks = [peaks]
        self.ids = [peaks.identifier]

    @staticmethod
    def _check_input(peaks):
        if not isinstance(peaks, Peaks):
            msg = f"peaks must be an instance of `Peaks`, not {type(peaks)}."
            raise TypeError(msg)

    def append(self, peaks):
        """Append a `Peaks` object.

        Parameters
        ----------
        peaks : Peaks
            A `Peaks` object to include in the suite.

        Returns
        -------
        None
            Appends `Peaks` to `PeaksSuite`.

        """
        self._check_input(peaks)
        if peaks.identifier in self.ids:
            msg = "There already exists a member object with an "
            msg += f"identifier = {peaks.identifier}."
            raise KeyError(msg)
        self.peaks.append(peaks)
        self.ids.append(peaks.identifier)

    def blitz(self, attribute, limits):
        """Reject peaks outside the stated boundary.

        TODO (jpv): Reference Peaks.blitz for more information.

        """
        for peak in self.peaks:
            peak.blitz(attribute, limits)

    def reject(self, xtype, xlims, ytype, ylims):
        """Reject peaks inside the stated boundary.

        TODO (jpv): Reference Peaks.reject for more information.

        """
        for peak in self.peaks:
            peak.reject(xtype, xlims, ytype, ylims)

    def reject_ids(self, xtype, xlims, ytype, ylims):
        """Reject peaks inside the stated boundary.

        TODO (jpv): Reference Peaks.reject for more information.

        """
        rejection_ids = []
        for peak in self.peaks:
            rejection_ids.append(peak.reject_ids(xtype, xlims, ytype, ylims))
        return rejection_ids

    def _reject(self, reject_ids):
        for _peak, _reject_ids in zip(self.peaks, reject_ids):
            _peak._reject(_reject_ids)

    @staticmethod
    def calc_resolution_limits(xtype, attribute, ytype, limits, xs, ys):
        """Calculate resolution limits for a variety of domains."""
        if xtype == "frequency":
            x1, x2 = xs, xs
            if attribute == "wavelength":
                if ytype == "velocity":
                    y1, y2 = [xs*limit for limit in limits]
                elif ytype == "wavenumber":
                    y1, y2 = [np.ones_like(ys)*2*np.pi /
                              limit for limit in limits]
                elif ytype == "slowness":
                    y1, y2 = [1/(xs*limit) for limit in limits]
                else:
                    raise NotImplementedError
            elif attribute == "wavenumber":
                if ytype == "velocity":
                    y1, y2 = [xs*np.pi*2/limit for limit in limits]
                elif ytype == "wavenumber":
                    y1, y2 = [np.ones_like(ys)*limit for limit in limits]
                elif ytype == "slowness":
                    y1, y2 = [limit/(2*np.pi*xs) for limit in limits]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif xtype == "wavelength":
            y1, y2 = ys, ys
            if attribute == "wavelength":
                if ytype == "velocity":
                    x1, x2 = [np.ones_like(ys)*limit for limit in limits]
                elif ytype == "slowness":
                    x1, x2 = [np.ones_like(ys)*limit for limit in limits]
                else:
                    raise NotImplementedError
            elif attribute == "wavenumber":
                if ytype == "velocity":
                    x1, x2 = [np.ones_like(ys)*2*np.pi /
                              limit for limit in limits]
                elif ytype == "slowness":
                    x1, x2 = [np.ones_like(ys)*2*np.pi /
                              limit for limit in limits]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return ((x1, y1), (x2, y2))

    @staticmethod
    def plot_resolution_limits(ax, xtype, ytype, attribute, limits,
                               plot_kwargs=None):
        """Plot resolution limits on provided `Axes`.

        Parameters
        ----------
        ax : Axes
            `Axes` on which resolution limit is to be plotted.
        xtype : {"frequency", "wavelength"}
            Attribute on x-axis.
        ytype : {"velocity", "slowness", "wavenumber"}
            Attribute on y-axis.
        limits : tuple
            Of the form `(lower limit, upper limit)`.
        plot_kwargs : dict, optional
            Keyword arguments to pass along to `ax.plot`, default is
            `None` indicating the predefined settings should be used.

        Returns
        -------
        None
            Updates Axes with resolution limit (if possible).

        """
        xs = AWTransform._create_vector(*ax.get_xlim(), 50, ax.get_xscale())
        ys = AWTransform._create_vector(*ax.get_ylim(), 50, ax.get_yscale())
        try:
            limits = PeaksSuite.calc_resolution_limits(xtype, attribute,
                                                       ytype, limits, xs, ys)
        except NotImplementedError:
            warnings.warn("Could not calculate resolution limits.")
        else:
            plot_kwargs = {} if plot_kwargs is None else plot_kwargs
            default_kwargs = dict(color="#000000", linestyle="--",
                                  linewidth=0.75, label="limit")
            plot_kwargs = {**default_kwargs, **plot_kwargs}

            for limit_pair in limits:
                ax.plot(*limit_pair, **plot_kwargs)
                plot_kwargs["label"] = None

    def plot(self, xtype="frequency", ytype="velocity", ax=None,
             plot_kwargs=None, indices=None):
        """Plot dispersion data in `Peaks` object.

        Parameters
        ----------
        xtype : {'frequency', 'wavelength'}, optional
            Denote whether the x-axis should be either `frequency` or
            `wavelength`, default is `frequency`.
        ytype : {'velocity', 'slowness'}, optional
            Denote whether the y-axis should be either `velocity` or
            `slowness`, default is `velocity`.
        ax : Axes, optional
            `Axes` object on which to plot the disperison peaks,
            default is `None` so `Axes` will be generated on-the-fly.
        plot_kwargs : dict, optional
            Keyword arguments to pass along to `ax.plot` can be in the
            form `plot_kwargs = {"key":value_allpeaks}` or
            `plot_kwargs = {"key":[value_peaks0, value_peaks1, ... ]}`,
            default is `None` indicating the predefined settings should
            be used.
        indices : list of ndarray, optional
            Indices to plot from each `Peaks` object in the `PeaksSuite`
            , default is `None` so all points will be plotted.

        Returns
        -------
        None or tuple
            `None` if `ax` is provided, otherwise `tuple` of the form
            `(fig, ax)` where `fig` is the figure handle and `ax` is
            the axes handle.


        """
        # Prepare xtype, ytype.
        xtype, ytype = Peaks._prepare_types(xtype=xtype, ytype=ytype)

        # Prepare keyword arguments.
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        plot_kwargs = self._prepare_plot_kwargs(plot_kwargs, len(self))

        # Prepare indices argument.
        if indices is None:
            indices = [None]*len(self)
        else:
            if len(indices) != len(self):
                msg = f"len(indices)={len(indices)} must equal "
                msg += f"len(self)={len(self)}."
                raise IndexError(msg)

        # Plot the first Peaks object from the suite.
        if ax is None:
            ax_was_none = True
            fig, ax = self.peaks[0].plot(xtype=xtype, ytype=ytype,
                                         plot_kwargs=plot_kwargs[0],
                                         indices=indices[0])
        else:
            ax_was_none = False

            if not isinstance(ax, (list, tuple, np.ndarray)):
                ax = [ax]
            if len(ax) != len(xtype):
                msg = f"len(ax)={len(ax)} must equal len(xtype)={len(xtype)}."
                raise IndexError(msg)

            for _ax, _xtype, _ytype in zip(ax, xtype, ytype):
                self.peaks[0]._plot(ax=_ax, xtype=_xtype, ytype=_ytype,
                                    plot_kwargs=plot_kwargs[0],
                                    indices=indices[0])

        # Plot the remaining Peaks from the PeaksSuite (if they exist).
        if len(self.peaks) > 1:
            for _peak, _plot_kwargs, _indices in zip(self.peaks[1:], plot_kwargs[1:], indices[1:]):
                label = _plot_kwargs.get("label", None)
                for _ax, _xtype, _ytype in zip(ax, xtype, ytype):

                    # Only label unique data sets.
                    _, labels = _ax.get_legend_handles_labels()
                    if label in labels:
                        _plot_kwargs["label"] = None

                    _peak._plot(xtype=_xtype, ytype=_ytype, ax=_ax,
                                plot_kwargs=_plot_kwargs, indices=_indices)

        # Configure Axes.
        for _ax, _xtype, _ytype in zip(ax, xtype, ytype):
            Peaks._configure_axes(ax=_ax, xtype=_xtype, ytype=_ytype,
                                  defaults=Peaks.axes_defaults)

        # Return fig, ax if generated on-the-fly.
        if ax_was_none:
            return (fig, ax)

    @staticmethod
    def _prepare_plot_kwargs(plot_kwargs, ncols):
        """Prepare keyword arguments for easy looping.

        Parameters
        ----------
        plot_kwargs : dict
            Keyword arguments to pass along to `ax.plot` can be in the
            form `plot_kwargs = {"key":value_allpeaks}` or
            `plot_kwargs = {"key":[value_peaks0, value_peaks1, ... ]}`,
            default is `None` indicating the predefined settings should
            be used.

        Returns
        -------
        list of dict
            Expands `plot_kwargs` to a `list` of `dict` rather than a
            `dict` of `list`.

        """
        expanded_kwargs = []
        for index in range(ncols):
            new_dict = {}
            for key, value in plot_kwargs.items():
                if isinstance(value, str) or value is None:
                    new_dict[key] = value
                elif isinstance(value, (list, tuple)):
                    new_dict[key] = value[index]
                else:
                    msg = f"kwarg must be a `str` or `iterable` not {type(value)}."
                    raise NotImplementedError(msg)
            expanded_kwargs.append(new_dict)
        return expanded_kwargs

    def interactive_trimming(self, xtype="wavelength", ytype="velocity",
                             plot_kwargs=None, resolution_limits=None,
                             resolution_limits_plot_kwargs=None):
        """Interactively trim experimental dispersion data.

        Parameters
        ----------
        xtype : {'frequency', 'wavelength'}, optional
            Denote whether the x-axis should be either `frequency` or
            `wavelength`, default is `frequency`.
        ytype : {'velocity', 'slowness'}, optional
            Denote whether the y-axis should be either `velocity` or
            `slowness`, default is `velocity`.
        plot_kwargs : dict, optional
            Keyword arguments to pass along to `ax.plot` can be in the
            form `plot_kwargs = {"key":value_allpeaks}` or
            `plot_kwargs = {"key":[value_peaks0, value_peaks1, ... ]}`,
            default is `None` indicating the predefined settings should
            be used.
        resolution_limits : iterable, optional
            Of form `("domain", (min, max))` where `"domain"` is a `str`
            denoting the domain of the limits and `min` and `max` are
            `floats` denoting their value, default is `None` so no
            resolution limits are plotted for reference.
        resolution_limits_plot_kwargs : dict, optional
            Formatting of resolution limits passed to `ax.plot`, default
            is `None` so default settings will be used.

        Returns
        -------
        None
            Updates the `PeaksSuite` state.

        """
        # Prepare xtype, ytype.
        xtype, ytype = Peaks._prepare_types(xtype=xtype, ytype=ytype)

        # Create fig, ax and plot data
        fig, ax = self.plot(xtype=xtype, ytype=ytype, plot_kwargs=plot_kwargs)

        # Store minimum and maximum axes limits
        pxlims, pylims = [], []
        for _ax in ax:
            _ax.autoscale(enable=False)
            pxlims.append(_ax.get_xlim())
            pylims.append(_ax.get_ylim())

        # Plot resolution limits (if desired):
        if resolution_limits is not None:
            attribute, limits = resolution_limits
            for _ax, _xtype, _ytype, in zip(ax, xtype, ytype):
                self.plot_resolution_limits(ax=_ax, xtype=_xtype,
                                            ytype=_ytype, limits=limits,
                                            attribute=attribute,
                                            plot_kwargs=resolution_limits_plot_kwargs)

        # Force display of figure.
        fig.show()

        _continue = 1
        master_indices = [np.array([], dtype=int) for _ in self.peaks]
        err_bar = None
        while _continue:

            # Ask user to draw box.
            (xlims, ylims, axclicked) = self._draw_box(fig)

            # Find all points inside the box.
            rejection_ids = self.reject_ids(xtype[axclicked], xlims,
                                            ytype[axclicked], ylims)

            # Count number of rejected points.
            rejection_count = 0
            for _rejection_id in rejection_ids:
                rejection_count += _rejection_id.size
            logging.debug(f"\trejection_count = {rejection_count}")

            # If latest rejection box has points, store and continue.
            if rejection_count > 0:
                self.plot(xtype=xtype, ytype=ytype, ax=ax,
                          plot_kwargs=dict(color="#bbbbbb", label=None),
                          indices=rejection_ids)

                master_indices = [np.union1d(master, slave) for master, slave in zip(
                    master_indices, rejection_ids)]
            # If latest rejection box is empty, ask user for input.
            else:
                while True:
                    msg = "Enter (0 to quit, 1 to continue, 2 to undo): "
                    _continue = input(msg)

                    # Bad entry, ask again.
                    if _continue not in ["0", "1", "2"]:
                        warnings.warn(f"Entry {_continue}, is not recognized.")
                        continue
                    # Execute decision.
                    else:
                        _continue = int(_continue)
                        break

                # If continue or quit, reject points and reset master_indices.
                if _continue in [0, 1]:
                    self._reject(master_indices)
                    master_indices = [np.array([], dtype=int)
                                      for _ in self.peaks]

                # Clear, set axis limits, and lock axis.
                for _ax, pxlim, pylim in zip(ax, pxlims, pylims):
                    _ax.clear()
                    _ax.set_xlim(pxlim)
                    _ax.set_ylim(pylim)
                    # Note: _ax.clear() re-enables autoscale.
                    _ax.autoscale(enable=False)

                # Replot points (rejected points have been removed).
                self.plot(xtype=xtype, ytype=ytype, ax=ax,
                          plot_kwargs=plot_kwargs)

                # Plot resolution limits (if desired):
                if resolution_limits is not None:
                    attribute, limits = resolution_limits
                    for _ax, _xtype, _ytype, in zip(ax, xtype, ytype):
                        self.plot_resolution_limits(ax=_ax, xtype=_xtype,
                                                    ytype=_ytype, limits=limits,
                                                    attribute=attribute,
                                                    plot_kwargs=resolution_limits_plot_kwargs)

    # TODO (jpv): To be refactored. Place in interact module?

    def _draw_box(self, fig):
        """Prompt user to define a rectangular box on figure.

        Parameters
        ----------
        fig : Figure
            Figure object, on which the user is to draw the box.

        Returns
        -------
        tuple
            Of the form `((xmin, xmax), (ymin,ymax)), axclicked` where
            `xmin` and `xmax` are the minimum and maximum abscissa and
            `ymin` and `ymax` are the minimum and maximum ordinate of
            the user-defined box and `axclicked` determine which `Axes`
            was clicked.

        """
        # print("0")
        cursors = []
        for ax in fig.axes:
            cursors.append(Cursor(ax, useblit=True, color='k', linewidth=1))
        # print("1")

        def on_click(event):
            if event.inaxes is not None:
                axclicked.append(event.inaxes)
        # print("2")

        while True:
            # print("3")
            axclicked = []
            # on_click = lambda event : axclicked.append(event.inaxes) if event.inaxes is not None else None
            session = fig.canvas.mpl_connect('button_press_event', on_click)
            (x1, y1), (x2, y2) = fig.ginput(2, timeout=0)
            xs = (x1, x2)
            ys = (y1, y2)
            # print(xs, ys)
            fig.canvas.mpl_disconnect(session)
            # print("4")

            # print(axclicked)

            if len(axclicked) == 2 and axclicked[0] == axclicked[1]:
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                ax_index = fig.axes.index(axclicked[0])
                logger.debug(f"\tax_index = {ax_index}")
                return ((xmin, xmax), (ymin, ymax), ax_index)
            else:
                msg = "Both clicks must be on the same axes. Please try again."
                warnings.warn(msg)
            # print("5")

    def statistics(self, xtype, ytype, xx, ignore_corr=True):
        """Determine the statistics of the `PeaksSuite`.

        Parameters
        ----------
        xtype : {"frequency","wavelength"}
            Axis along which to calculate statistics.
        ytype : {"velocity", "slowness"}
            Axis along which to define uncertainty.
        xx : ndarray
            Array of values in `xtype` units where statistics are to be
            calculated.
        ignore_corr : bool, optional
            Ignore calculation of data's correlation coefficients,
            default is `True`.

        Returns
        -------
        tuple
            Of the form `(xx, mean, std, corr)` where `mean` and
            `std` are the mean and standard deviation at each point and
            `corr` are the correlation coefficients between every point
            and all other points.

        """
        npeaks = len(self.peaks)
        if npeaks < 3:
            msg = f"Cannot calculate statistics on fewer than 3 `Peaks`."
            raise ValueError(msg)

        data_matrix = np.empty((len(xx), npeaks))

        for col, _peaks in enumerate(self.peaks):
            # TODO (jpv): Allow assume_sorted should improve speed.
            interpfxn = interp1d(getattr(_peaks, xtype),
                                 getattr(_peaks, ytype),
                                 copy=False, bounds_error=False,
                                 fill_value=np.nan)
            data_matrix[:, col] = interpfxn(xx)

        # if missing_data_procedure == "drop":
        #     xx, data_matrix = self._drop(xx, data_matrix)
        # elif missing_data_procedure == "ignore":
        #     pass
        # else:
        #     NotImplementedError

        mean = np.nanmean(data_matrix, axis=1)
        std = np.nanstd(data_matrix, axis=1, ddof=1)
        corr = None if ignore_corr else np.corrcoef(data_matrix)

        return (xx, mean, std, corr)

    def plot_statistics(self, ax, xx, mean, stddev, errorbar_kwargs=None):
        errorbar_kwargs = {} if errorbar_kwargs is None else errorbar_kwargs
        default_kwargs = dict(linestyle="", color="k", label=None,
                              marker="o", markerfacecolor="k",
                              markersize=0.5, capsize=2, zorder=20)
        errorbar_kwargs = {**default_kwargs, **errorbar_kwargs}
        ax.errorbar(xx, mean, yerr=stddev, **errorbar_kwargs)

    @staticmethod
    def _drop(xx, data_matrix):
        drop_cols = []
        for index, column in enumerate(data_matrix.T):
            if np.isnan(column).all():
                drop_cols.append(index)
        drop_cols = np.array(drop_cols, dtype=int)
        data_matrix = np.delete(data_matrix, drop_cols, axis=1)

        drop_rows = []
        for index, row in enumerate(data_matrix):
            if np.isnan(row).any():
                drop_rows.append(index)
        drop_rows = np.array(drop_rows, dtype=int)
        xx = np.delete(xx, drop_rows)
        data_matrix = np.delete(data_matrix, drop_rows, axis=0)

        return (xx, data_matrix)

    def to_json(self, fname):
        """Write `PeaksSuite` to json file.

        Parameters
        ----------
        fnames : str
            Name of the output file, may contain a relative or the full
            path.

        Returns
        -------
        None
            Write `json` to disk.

        """
        append = False
        for peak in self.peaks:
            peak.to_json(fname, append=append)
            append = True

    @classmethod
    def from_dict(cls, dicts):
        """Instantiate `PeaksSuite` from `list` of `dict`s.

        Parameters
        ----------
        dicts : list of dict or dict
            List of `dict` or a single `dict` containing dispersion
            data.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        if isinstance(dicts, dict):
            dicts = [dicts]

        peaks = []
        for _dict in dicts:
            for identifier, data in _dict.items():
                peaks.append(Peaks.from_dict(data, identifier=identifier))

        return cls.from_peaks(peaks)

    @classmethod
    def from_json(cls, fnames):
        """Instantiate `PeaksSuite` from json file(s).

        Parameters
        ----------
        fnames : list of str or str
            File name or list of file names containing dispersion data.
            Names may contain a relative or the full path.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        if isinstance(fnames, str):
            fnames = [fnames]

        dicts = []
        for fname in fnames:
            with open(fname, "r") as f:
                dicts.append(json.load(f))
        return cls.from_dict(dicts)

    @classmethod
    def from_max(cls, fnames, wavetype="rayleigh"):
        peaks = []
        for fname in fnames:
            with open(fname, "r") as f:
                peak_data = f.read()

            regex = get_all(wavetype=wavetype)
            found_times = []
            for found in regex.finditer(peak_data):
                start_time = found.groups()[0]
                if start_time in found_times:
                    continue
                found_times.append(start_time)
                peak = Peaks._parse_peaks(peak_data, wavetype=wavetype,
                                          start_time=start_time)
                peaks.append(peak)

        return cls.from_peaks(peaks)

    @classmethod
    def from_peaks(cls, peaks):
        """Instantiate `PeaksSuite` from iterable of `Peaks`.

        Parameters
        ----------
        peaks : iterable
            Iterable containing `Peaks` objects.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        obj = cls(peaks[0])

        if len(peaks) >= 1:
            for peak in peaks[1:]:
                obj.append(peak)

        return obj

    @classmethod
    def from_peaksuite(cls, peakssuites):
        """Instantiate `PeaksSuite` from iterable of `PeaksSuite`.

        Parameters
        ----------
        peakssuites : iterable
            Iterable containing `PeaksSuite` objects.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        if isinstance(peakssuites, PeaksSuite):
            peakssuites = [peakssuites]

        obj = cls.from_peaks(peakssuites[0].peaks)

        for peaksuite in peakssuites[1:]:
            for peak in peaksuite:
                obj.append(peak)

        return obj

    def __getitem__(self, index):
        return self.peaks[index]

    def __len__(self):
        return len(self.peaks)

    def __eq__(self, other):
        if not isinstance(other, PeaksSuite):
            return False

        if len(self) != len(other):
            return False

        for mypeaks, urpeaks in zip(self.peaks, other.peaks):
            if mypeaks != urpeaks:
                return False

        return True
