# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020-2024 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""PeaksSuite class definition."""

import json
import warnings
import logging

import numpy as np
from scipy.interpolate import interp1d
from matplotlib.widgets import Cursor

from .wavefieldtransforms import AbstractWavefieldTransform as AWTransform
from .peaks import Peaks
from .regex import get_nmaxima, get_peak_from_max, get_geopsy_version, get_wavetype
from .meta import check_geopsy_version

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
        """Append a `Peaks` object to `PeaksSuite`.

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

    def reject_limits_outside(self, attribute, limits):
        """Reject peaks outside the stated limits.

        Parameters
        ----------
        attr : {"frequency", "velocity", "slowness", "wavelength"}
            Parameter domain in which the limits are defined.
        limits : tuple
            Tuple with the lower and upper limits. `None` may be used to
            perform one-sided rejections. For example `limits=(None, 5)`
            will reject all values above `5` and `limits=(5, None)` will
            reject all values below `5`.

        Returns
        -------
        None
            Updates the `PeaksSuite` internal state.

        """
        for peak in self.peaks:
            peak.reject_limits_outside(attribute, limits)

    def reject_box_inside(self, xtype, xlims, ytype, ylims):
        """Reject peaks inside the stated limits.

        Parameters
        ----------
        xtype, ytype : {"frequency", "velocity", "slowness", "wavelength"}
            Parameter domain in which the limits are defined.
        xlims, ylims : tuple
            Tuple with the lower and upper limits for each of the
            boundaries.

        Returns
        -------
        None
            Updates the `PeaksSuite` internal state.

        """
        for peak in self.peaks:
            peak.reject_box_inside(xtype, xlims, ytype, ylims)

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
             plot_kwargs=None, mask=None):
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
        mask : list of ndarray, optional
            Boolean array mask for each `Peaks` object in the
            `PeaksSuite` to control which points will be plotted,
            default is `None` so no mask is applied.

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
        if mask is None:
            mask = [None]*len(self)
        else:
            if len(mask) != len(self):
                msg = f"len(mask)={len(mask)} must equal "
                msg += f"len(self)={len(self)}."
                raise IndexError(msg)

        # Plot the first Peaks object from the suite.
        if ax is None:
            ax_was_none = True
            fig, ax = self.peaks[0].plot(xtype=xtype, ytype=ytype,
                                         plot_kwargs=plot_kwargs[0],
                                         mask=mask[0])
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
                                    mask=mask[0])

        # Plot the remaining Peaks from the PeaksSuite (if they exist).
        if len(self.peaks) > 1:
            for _peak, _plot_kwargs, _mask in zip(self.peaks[1:], plot_kwargs[1:], mask[1:]):
                label = _plot_kwargs.get("label", None)
                for _ax, _xtype, _ytype in zip(ax, xtype, ytype):

                    # Only label unique data sets.
                    _, labels = _ax.get_legend_handles_labels()
                    if label in labels:
                        _plot_kwargs["label"] = None

                    _peak._plot(xtype=_xtype, ytype=_ytype, ax=_ax,
                                plot_kwargs=_plot_kwargs, mask=_mask)

        # Configure Axes.
        for _ax, _xtype, _ytype in zip(ax, xtype, ytype):
            Peaks._configure_axes(ax=_ax, xtype=_xtype, ytype=_ytype,
                                  defaults=Peaks.axes_defaults)

        # Return (fig, ax) if generated on-the-fly.
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
                             resolution_limits_plot_kwargs=None,
                             margins=0.1):
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

        # Create (fig, ax) and plot data
        fig, ax = self.plot(xtype=xtype, ytype=ytype, plot_kwargs=plot_kwargs)

        # Store minimum and maximum axes limits
        pxlims, pylims = [], []
        for _ax in ax:
            _ax.margins(margins)
            _ax.autoscale()
            pxlims.append(_ax.get_xlim())
            pylims.append(_ax.get_ylim())
            _ax.autoscale(enable=False)

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
        rejection_bool_arrays = [np.zeros_like(peak._valid, dtype=bool) for peak in self.peaks]

        while _continue:
            # Instruct user to select a bounding box.
            ax[0].text(0.95, 0.95, "Select two points that\nbound data to be removed.\nDouble click to pause trimming.",
                       ha="right", va="top", transform=ax[0].transAxes)

            # User draws box.
            (xlims, ylims, axclicked) = self._draw_box(fig)

            # Find all points inside the box.
            rejection_count = 0
            for index, peak in enumerate(self.peaks):
                rejection_mask = peak._reject_box_inside_bool_array(xtype[axclicked], xlims, ytype[axclicked], ylims)
                rejection_count += np.sum(rejection_mask)
                rejection_bool_arrays[index][rejection_mask] = True
            logging.debug(f"\trejection_count = {rejection_count}")

            # If latest rejection box has points, store and continue.
            if rejection_count > 0:
                self.plot(xtype=xtype, ytype=ytype, ax=ax,
                          plot_kwargs=dict(color="#bbbbbb", label=None),
                          mask=rejection_bool_arrays)

            # If latest rejection box is empty, ask user for input.
            else:
                # Clear canvas and tell user to go to Jupyter
                for _ax, pxlim, pylim in zip(ax, pxlims, pylims):
                    _ax.clear()
                    _ax.set_xlim(pxlim)
                    _ax.set_ylim(pylim)
                    # Note: _ax.clear() re-enables autoscale.
                    _ax.autoscale(enable=False)
                    _ax.text(0.5, 0.7, "Interactive trimming paused.\nDo not close window.\nUse alt+tab to go back\nto Jupyter to quit, continue, or undo.",
                            ha="center", va="top", transform=_ax.transAxes)
                fig.canvas.draw()
                # session = fig.canvas.mpl_connect('button_press_event', lambda x:None)
                _ = fig.ginput(0, timeout=0.01)
                # fig.canvas.mpl_disconnect(session)

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

                # If continue or quit, reject points.
                if _continue in [0, 1]:
                    for peak, bool_array in zip(self.peaks, rejection_bool_arrays):
                        peak._reject(bool_array)

                # If continue, quit, or undo, reset boolean arrays.
                rejection_bool_arrays = [np.zeros_like(peak._valid, dtype=bool) for peak in self.peaks]

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
        cursors = []
        for ax in fig.axes:
            cursors.append(Cursor(ax, useblit=False, color='k', linewidth=1))

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

    def statistics(self, xtype, ytype, xx, ignore_corr=True,
                   drop_sample_if_fewer_count=3, mean_substitution=False):
        """Determine the statistics of the `PeaksSuite`.

        Parameters
        ----------
        xtype : {"frequency","wavelength"}
            Axis along which to calculate statistics.
        ytype : {"velocity", "slowness"}
            Axis along which to define uncertainty.
        xx : iterable
            Values in `xtype` units where statistics are to be
            calculated.
        ignore_corr : bool, optional
            Ignore calculation of data's correlation coefficients,
            default is `True`.
        drop_sample_if_fewer_count : int, optional
            Remove statistic sample if the number of valid entries
            is fewer than the specified number, default is 3.

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
            msg = "Cannot calculate statistics on fewer than 3 `Peaks`."
            raise ValueError(msg)

        xx, data_matrix = self.to_array(xtype, ytype, xx)

        xx, data_matrix = self._drop(xx, data_matrix,
                                     drop_sample_if_fewer_percent=0.01,
                                     drop_observation_if_fewer_percent=0.01,
                                     drop_sample_if_fewer_count=drop_sample_if_fewer_count)

        mean = np.nanmean(data_matrix, axis=0)
        std = np.nanstd(data_matrix, axis=0, ddof=1)

        if mean_substitution:
            for cid, values in enumerate(data_matrix.T):
                mean_value = np.nanmean(values)
                rbools = np.isnan(values)
                data_matrix[rbools, cid] = mean_value

        corr = None if ignore_corr else np.corrcoef(data_matrix.T)

        return (xx, mean, std, corr)

    def to_array(self, xtype, ytype, xx):
        """Create an array representation of the `PeaksSuite`.

        Parameters
        ----------
        xtype : {"frequency","wavelength"}
            Axis along which to define samples.
        ytype : {"velocity", "slowness"}
            Axis along which to define values.
        xx : iterable
            Values, in the units of `xtype`, where `PeaksSuite` is to
            be discretized.

        Returns
        -------
        tuple
            Of the form `(xx, array)` where `xx` is the discretized
            values and `array` is a two-dimensional array with one row
            per `Peaks` in the `PeaksSuite` and one column for each
            entry of `xx`. Missing values are denoted with `np.nan`.

        """
        xx = np.array(xx)
        npeaks = len(self.peaks)
        array = np.empty((npeaks, len(xx)))

        for row, _peaks in enumerate(self.peaks):
            # TODO (jpv): Allow assume_sorted should improve speed.
            x = _peaks.simplify_mpeaks(xtype)
            y = _peaks.simplify_mpeaks(ytype)
            if np.sum(np.isnan(x)) + np.sum(np.isnan(y)):
                raise ValueError("NaN in x and or y!")
            try:
                interpfxn = interp1d(x, y,
                                     copy=True, bounds_error=False,
                                     fill_value=np.nan)
            except ValueError:
                array[row, :] = np.nan
            else:
                array[row, :] = interpfxn(xx)

        return (xx, array)

    @staticmethod
    def plot_statistics(ax, xx, mean, stddev, errorbar_kwargs=None):
        errorbar_kwargs = {} if errorbar_kwargs is None else errorbar_kwargs
        default_kwargs = dict(linestyle="", color="k", label=None,
                              marker="o", markerfacecolor="k",
                              markersize=0.5, capsize=2, zorder=20)
        errorbar_kwargs = {**default_kwargs, **errorbar_kwargs}
        ax.errorbar(xx, mean, yerr=stddev, **errorbar_kwargs)

    @staticmethod
    def _drop(xx, data_matrix,
              drop_observation_if_fewer_percent=0.8,
              drop_sample_if_fewer_percent=0.4,
              drop_sample_if_fewer_count=3):
        """Procedure for removing problematic observations/samplings.

        Parameters
        ----------
        xx : ndarray
            Statistic sampling locations.
        data_matrix : ndarray
            Of shape `(# observations, # samples)` each entry's
            value indicates the parameters value (e.g., velocity)
            the presence of `np.nan` indicates missing data.
        drop_observation_if_fewer_percent : {0. - 1.}, optional
            Remove observations if the number of valid entries is
            fewer than the specified fraction times the total
            possible, default is 0.8.
        drop_sample_if_fewer_percent : {0. - 1.}, optional
            Remove statistic sample if the number of valid entries
            is fewer than the specified fraction times the total
            possible, default is 0.4.
        drop_sample_if_fewer_count : int, optional
            Remove statistic sample if the number of valid entries
            is fewer than the specified number, default is 3.

        Returns
        -------
        tuple
            Of the form `(xx, data_matrix)` where `xx` and
            `data_matrix` are the permuted inputs.

        """
        # Initial
        i_nans = np.sum(np.isnan(data_matrix))
        i_nums = data_matrix.size - i_nans

        # Option 1: Drop columns then rows.
        drop_cols = PeaksSuite._drop_indices(data_matrix.T,
                                             drop_if_fewer_percent=drop_sample_if_fewer_percent,
                                             drop_if_fewer_count=drop_sample_if_fewer_count)
        xx_1 = np.delete(xx, drop_cols)
        data_matrix_1 = np.delete(data_matrix, drop_cols, axis=1)
        drop_rows = PeaksSuite._drop_indices(data_matrix_1,
                                             drop_if_fewer_percent=drop_observation_if_fewer_percent,
                                             drop_if_fewer_count=0)
        data_matrix_1 = np.delete(data_matrix_1, drop_rows, axis=0)
        r_nans = np.sum(np.isnan(data_matrix_1))
        r_nums = data_matrix_1.size - r_nans
        utility_option_1 = (i_nans - r_nans) / (i_nums - r_nums + 1)

        # Option 2: Drop rows then columns.
        drop_rows = PeaksSuite._drop_indices(data_matrix,
                                             drop_if_fewer_percent=drop_observation_if_fewer_percent,
                                             drop_if_fewer_count=0)
        data_matrix_2 = np.delete(data_matrix, drop_rows, axis=0)
        drop_cols = PeaksSuite._drop_indices(data_matrix_2.T,
                                             drop_if_fewer_percent=drop_sample_if_fewer_percent,
                                             drop_if_fewer_count=drop_sample_if_fewer_count)
        xx_2 = np.delete(xx, drop_cols)
        data_matrix_2 = np.delete(data_matrix_2, drop_cols, axis=1)
        r_nans = np.sum(np.isnan(data_matrix_2))
        r_nums = data_matrix_2.size - r_nans
        utility_option_2 = (i_nans - r_nans) / (i_nums - r_nums + 1)

        logger.debug(
            f"utility_option_1={utility_option_1}, utility_option_2={utility_option_2}")
        if utility_option_1 > utility_option_2:
            return (xx_1, data_matrix_1)
        else:
            return (xx_2, data_matrix_2)

    @staticmethod
    def _drop_indices(data_matrix, drop_if_fewer_percent, drop_if_fewer_count):
        """Iterate by row, return rejection indices."""
        if data_matrix.size == 0:
            return np.array([], dtype=int)

        drop_indices = []
        for index, row in enumerate(data_matrix):
            n_nan = np.sum(np.isnan(row))
            n_tot = len(row)
            n_num = n_tot - n_nan
            p_num = n_num/n_tot
            if n_num < drop_if_fewer_count or p_num < drop_if_fewer_percent:
                drop_indices.append(index)
        return np.array(drop_indices, dtype=int)

    def to_json(self, fname):
        """Write `PeaksSuite` to json file.

        Parameters
        ----------
        fname : str
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
        """Instantiate `PeaksSuite` from `list` of `dict`.

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

        try:
            iter(fnames)
        except TypeError:
            fnames = [str(fnames)]

        dicts = []
        for fname in fnames:
            with open(fname, "r") as f:
                dicts.append(json.load(f))
        return cls.from_dict(dicts)

    @classmethod
    def from_max(cls, fnames, wavetype="rayleigh"):
        """Instantiate `PeaksSuite` from .max file(s).

        Parameters
        ----------
        fnames : list of str or str
            File name or list of file names containing dispersion data.
            Names may contain a relative or the full path.
        wavetype : {'rayleigh', 'love'}, optional
            Wavetype to extract from file, default is 'rayleigh'.

        Returns
        -------
        Peaks
            Initialized `PeaksSuite` object.

        """
        if isinstance(fnames, str):
            fnames = [fnames]

        try:
            iter(fnames)
        except TypeError:
            fnames = [str(fnames)]

        peaks = []
        for fname in fnames:
            with open(fname, "r") as f:
                peak_data = f.read()

            regex = get_geopsy_version()
            major, minor, micro = regex.search(peak_data).groups()
            version = f"{major}.{minor}.{micro}"
            check_geopsy_version(version)

            regex = get_wavetype()
            wavetype_from_file = regex.search(peak_data).groups()[0]
            if wavetype == "rayleigh" and wavetype_from_file == "Vertical":
                wavetype = "vertical"

            regex = get_nmaxima()
            nmaxima = int(regex.search(peak_data).groups()[0])
            nmaxima = max(1, nmaxima)
            nmaxima = min(100, nmaxima)

            regex = get_peak_from_max(wavetype=wavetype)
            frequencies = []
            for match in regex.finditer(peak_data):
                _, f, *_ = match.groups()
                if f in frequencies:
                    continue
                else:
                    frequencies.append(f)

            regex = get_peak_from_max(wavetype=wavetype)
            found_times = []
            for found in regex.finditer(peak_data):
                start_time = found.groups()[0]
                if start_time in found_times:
                    continue
                found_times.append(start_time)
                peak = Peaks._parse_peaks(peak_data,
                                          wavetype=wavetype,
                                          start_time=start_time,
                                          frequencies=frequencies,
                                          nmaxima=nmaxima)
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
    def from_peakssuite(cls, peakssuites):
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
        """Define equality between self and other."""
        if not isinstance(other, PeaksSuite):
            return False

        if len(self) != len(other):
            return False

        for mypeaks, urpeaks in zip(self.peaks, other.peaks):
            if mypeaks != urpeaks:
                return False

        return True
