"""PeaksSuite class definition."""

import json
import warnings
import logging

import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from .peaks import Peaks
from .regex import get_all

logger = logging.getLogger("swprocess.peakssuite")

_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*100


class PeaksSuite():

    @staticmethod
    def _check_input(peaks):
        if not isinstance(peaks, Peaks):
            msg = f"peaks must be an instance of `Peaks`, not {type(peaks)}."
            raise TypeError(msg)

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

        # TODO (jpv): Dict?
        self.peaks = [peaks]
        self.ids = [peaks.identifier]

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
            msg = f"There already exists a member object with identifiers = {peaks.identifier}."
            raise KeyError(msg)
        self.peaks.append(peaks)
        self.ids.append(peaks.identifier)

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

        iterable = []
        for _dict in dicts:
            for identifier, data in _dict.items():
                iterable.append(Peaks.from_dict(data, identifier=identifier))

        return cls.from_iter(iterable)

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
    def from_json(cls, fnames):
        """Instantiate `PeaksSuite` from json file(s).

        Parameters
        ----------
        fnames : list of str or str
            List of or a single file name containing dispersion data.
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
        iterable = []
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
                iterable.append(peak)

        return cls.from_iter(iterable)

    @classmethod
    def from_iter(cls, iterable):
        """Instantiate `PeaksSuite` from iterable object.

        Parameters
        ----------
        iterable : iterable
            Iterable containing `Peaks` objects.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        obj = cls(iterable[0])

        if len(iterable) >= 1:
            for _iter in iterable[1:]:
                obj.append(_iter)

        return obj

    def blitz(self, attr, limits):
        """Reject peaks outside the stated boundary.

        TODO (jpv): Refence Peaks.blitz for more information.

        """
        for peak in self.peaks:
            peak.blitz(attr, limits)

    def reject(self, xtype, xlims, ytype, ylims):
        """Reject peaks inside the stated boundary.

        TODO (jpv): Reference Peaks.reject for more information.

        """
        for peak in self.peaks:
            peak.reject(xtype, xlims, ytype, ylims)

    def reject_ids(self, xtype, xlims, ytype, ylims):
        """Reject peaks inside the stated boundary.

        TODO (jpv): Refence Peaks.reject for more information.

        """
        rejection = []
        for peak in self.peaks:
            rejection.append(peak.reject_ids(xtype, xlims, ytype, ylims))
        return rejection

    def _reject(self, reject_ids):
        for _peak, _reject_ids in zip(self.peaks, reject_ids):
            _peak._reject(_reject_ids)

    def plot(self, xtype="frequency", ytype="velocity", ax=None,
             plot_kwargs=None, ax_kwargs=None):
        """Create plot of dispersion data.

        TODO (jpv): Reference Peaks.plot for more information.

        plot_kwargs = {"key":value}
        plot_kwargs = {"key":[value1, value2, value3 ... ]}

        """
        if plot_kwargs is None:
            plot_kwargs = {}

        if ax_kwargs is None:
            ax_kwargs = {}

        if "color" not in plot_kwargs:
            plot_kwargs["color"] = _colors

        _plot_kwargs = self._prepare_kwargs(plot_kwargs, 0)
        _ax_kwargs = self._prepare_kwargs(ax_kwargs, 1)
        result = self.peaks[0].plot(xtype, ytype, ax, _plot_kwargs, _ax_kwargs)

        if ax is None:
            ax_was_none = True
            fig, ax = result
        else:
            ax_was_none = False

        if len(self.peaks) > 1:
            for index, peak in enumerate(self.peaks[1:], 1):
                _plot_kwargs = self._prepare_kwargs(plot_kwargs, index)
                _ax_kwargs = self._prepare_kwargs(ax_kwargs, index)
                peak.plot(xtype, ytype, ax, _plot_kwargs, _ax_kwargs)

        if ax_was_none:
            return (fig, ax)

    def plot_subset(self, ax, xtype, ytype, indices, plot_kwargs=None):
        if isinstance(xtype, str):
            ax = [ax]
            xtype = [xtype]
            ytype = [ytype]
            indices = [indices]
        elif len(ax) != len(xtype) or len(ax) != len(ytype):
            msg = f"`ax`, `xtype`, and `ytype` must all be the same size, not {len(ax)}, {len(xtype)}, {len(ytype)}s."
            raise IndexError(msg)

        default_plot_kwargs = dict(linestyle="", marker="x", color="#ababab",
                                   markersize=1, markerfacecolor="none",
                                   label=None)
        plot_kwargs = self._merge_kwargs(default_plot_kwargs, plot_kwargs)

        for _ax, _xtype, _ytype in zip(ax, xtype, ytype):
            for _peaks, _indices in zip(self.peaks, indices):
                _ax.plot(getattr(_peaks, _xtype)[_indices],
                         getattr(_peaks, _ytype)[_indices],
                         **plot_kwargs)

    @staticmethod
    def _prepare_kwargs(kwargs, index):
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float)):
                new_kwargs[key] = value
            # elif len(value) == 1:
            #     new_kwargs[key] = value[0]
            else:
                new_kwargs[key] = value[index]
        return new_kwargs

    @staticmethod
    def create_settings_dict(domains, xdomain="wavelength", ydomain="velocity",
                             xmin=3, xmax=100, nx=30, xspace="log",
                             stat_kwargs=None, limits=None):
        """Helper to define settings for `interactive_trimming`.

        Parameters
        ----------
        domains : list of lists
            Define the domains on which to plot the dispersion data, of
            the form `[ [x0, y0], [x1, y1] ... ]`.
        xdomain, ydomain : {"frequency", "velocity", "wavelength", "slowness"}, optional
            X and Y on which to calculate statistics, default is
            "wavelength" and "velocity", respectively.
        xmin, xmax : float, optional
            Minimum and maximum values in the xdomain for statistical
            calculations, default is 3 and 100, respectively.
        nx : int, optional
            Number of samples between `xmin` and `xmax` for statistical
            calculations, default is 30.
        xspace : {"log", "linear"}, optional
            Space along which `nx` points are selected, default is
            "log".
        limits : dict
            Define upper and lower limits in any domain of the form:
            `{"domain1": [d1min, d1max], "domain2":[d2min, d2max]}`.

        Returns
        -------
        dict
            Formatted correctly to be accepted by
            `interactive_trimming`.

        """
        stat_kwargs = {} if stat_kwargs is None else stat_kwargs
        settings = {
            "domains": domains,
            "statistics": {
                "xtype": xdomain,
                "ytype": ydomain,
                "type": xspace,
                "start": xmin,
                "stop": xmax,
                "num": nx,
                **stat_kwargs
            },
            "limits": limits
        }
        return settings

    @staticmethod
    def create_setting_file(fname, domains, xdomain="wavelength",
                            ydomain="velocity", xmin=3, xmax=100, nx=30,
                            xspace="log", stat_kwargs=None, limits=None):
        """Write settings file for `interactive_trimming` to disk.

        Parameters
        ----------
        domains : list of lists
            Define the domains on which to plot the dispersion data, of
            the form `[ [x0, y0], [x1, y1] ... ]`.
        xdomain, ydomain : {"frequency", "velocity", "wavelength", "slowness"}, optional
            X and Y on which to calculate statistics, default is
            "wavelength" and "velocity", respectively.
        xmin, xmax : float, optional
            Minimum and maximum values in the xdomain for statistical
            calculations, default is 3 and 100, respectively.
        nx : int, optional
            Number of samples between `xmin` and `xmax` for statistical
            calculations, default is 30.
        xspace : {"log", "linear"}, optional
            Space along which `nx` points are selected, default is
            "log".
        limits : dict
            Define upper and lower limits in any domain of the form:
            `{"domain1": [d1min, d1max], "domain2":[d2min, d2max]}`.

        Returns
        -------
        None
            Writes settings file for `interactive_trimming` to disk.

        """
        settings = PeaksSuite.create_settings_dict(
            domains=domains, xdomain=xdomain, ydomain=ydomain, xmin=xmin,
            xmax=xmax, nx=nx, xspace=xspace, stat_kwargs=stat_kwargs,
            limits=limits)
        with open(fname, "w") as f:
            json.dump(settings, f)

    def interactive_trimming(self, settings_file):
        with open(settings_file, "r") as f:
            settings = json.load(f)

        for key, value in settings.get("limits", {}).items():
            self.blitz(key, value)

        xtype = [pair[0] for pair in settings["domains"]]
        ytype = [pair[1] for pair in settings["domains"]]

        stat_settings = settings.get("statistics")
        if stat_settings is not None:
            if stat_settings["type"] == "log":
                stat_settings["xx"] = np.geomspace(stat_settings["start"],
                                                   stat_settings["stop"],
                                                   stat_settings["num"])
            elif stat_settings["type"] == "linear":
                stat_settings["xx"] = np.linspace(stat_settings["start"],
                                                  stat_settings["stop"],
                                                  stat_settings["num"])
            else:
                raise NotImplementedError
            keys = ["xtype", "ytype", "xx"]
            stat_settings = {key: stat_settings[key] for key in keys}

            for stat_ax_index, (_xtype, _ytype) in enumerate(zip(xtype, ytype)):
                if _xtype == stat_settings["xtype"] and _ytype == stat_settings["ytype"]:
                    break
            else:
                msg = f"Can only calculate statistics on a displayed domain."
                raise ValueError(msg)

        fig, ax = self.plot(xtype, ytype)
        for _ax in ax:
            _ax.autoscale(False)
        fig.show()

        _continue = 1
        master_indices = [np.array([]) for _ in self.peaks]
        err_bar = None
        while _continue:
            # if stat_settings is not None:
            #     if err_bar is not None:
            #         del err_bar
            #     statistics = self.statistics(**stat_settings)
            #     self.plot_statistics(statistics, ax=ax[stat_ax_index])

            (xlims, ylims, axclicked) = self._draw_box(fig)

            rejection_ids = self.reject_ids(xtype[axclicked], xlims,
                                            ytype[axclicked], ylims)
            rejection_count = 0
            for _rejection_id in rejection_ids:
                rejection_count += _rejection_id.size
            logging.debug(f"\trejection_count = {rejection_count}")

            if rejection_count > 0:
                self.plot_subset(ax, xtype, ytype, rejection_ids)

                master_indices = [np.union1d(master, slave) for master, slave in zip(
                    master_indices, rejection_ids)]
            else:
                while True:
                    msg = "Enter 1 to continue, 0 to quit, 2 to undo): "
                    _continue = input(msg)
                    if _continue not in ["0", "1", "2"]:
                        warnings.warn(f"Entry {_continue}, is not recognized.")
                        continue
                    else:
                        _continue = int(_continue)

                        if _continue in [0, 1]:
                            self._reject(master_indices)
                            master_indices = [np.array([]) for _ in self.peaks]

                        for _ax in ax:
                            _ax.clear()
                        self.plot(xtype, ytype, ax=ax)

                        for _ax in ax:
                            _ax.autoscale(False)
                        break

    def statistics(self, xx, xtype, ytype, missing_data_procedure="drop",
                   ignore=None):
        """Determine the statistiscs of the `PeaksSuite`.

        Parameters
        ----------
        xx : ndarray
        xtype : {"frequency","wavelength"}
            Axis along which to calculate statistics.
        ytype : {"velocity", "slowness"}
            Axis along which to define uncertainty.

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
                                 copy=False, bounds_error=False)
            data_matrix[:, col] = interpfxn(xx)

        if missing_data_procedure == "drop":
            xx, data_matrix = self._drop(xx, data_matrix)
        elif missing_data_procedure == "ignore":
            pass
        else:
            NotImplementedError

        if ignore is None:
            ignore = []

        if "mean" not in ignore:
            mean = np.nanmean(data_matrix, axis=1)
        else:
            mean = None

        if "stddev" not in ignore:
            std = np.nanstd(data_matrix, axis=1, ddof=1)
        else:
            std = None

        if "corr" not in ignore:
            corr = np.corrcoef(data_matrix)
        else:
            corr = None

        return (xx, mean, std, corr)

    @staticmethod
    def _merge_kwargs(default, custom):
        custom = {} if custom is None else custom
        return {**default, **custom}

    def plot_statistics(self, statistics=None, ax=None,
                        statistics_kwargs=None, plot_kwargs=None):
        if ax is None:
            raise NotImplementedError

        if statistics is None:
            raise NotImplementedError

        default_plot_kwargs = dict(linestyle="", color="k", label=None,
                                   marker="o", markerfacecolor="k",
                                   markersize=0.5, capsize=2, zorder=20)
        plot_kwargs = self._merge_kwargs(default_plot_kwargs, plot_kwargs)
        xx, mean, stddev, corr = statistics
        ax.errorbar(xx, mean, yerr=stddev, **plot_kwargs)

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
        cursors = []
        for ax in fig.axes:
            cursors.append(Cursor(ax, useblit=True, color='k', linewidth=1))

        def on_click(event):
            if event.inaxes is not None:
                axclicked.append(event.inaxes)

        while True:
            axclicked = []
            session = fig.canvas.mpl_connect('button_press_event', on_click)
            (x1, y1), (x2, y2) = fig.ginput(2, timeout=0)
            xs = (x1, x2)
            ys = (y1, y2)
            fig.canvas.mpl_disconnect(session)

            if len(axclicked) == 2 and axclicked[0] == axclicked[1]:
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                ax_index = fig.axes.index(axclicked[0])
                logger.debug(f"\tax_index = {ax_index}")
                return ((xmin, xmax), (ymin, ymax), ax_index)
            else:
                msg = "Both clicks must be on the same axes. Please try again."
                warnings.warn(msg)

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
