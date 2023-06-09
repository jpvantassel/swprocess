# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Peaks class definition."""

import json
import warnings
import logging

import numpy as np
import matplotlib.pyplot as plt

from .regex import get_wavetype, get_process_type, get_peak_from_max, get_all, get_nmaxima, get_geopsy_version
from .meta import check_geopsy_version

logger = logging.getLogger("swprocess.peaks")


class Peaks():
    """Class for handling dispersion peaks.

    Attributes
    ----------
    frequency : ndarray
        Frequency associate with each peak.
    velocity : ndarray
        Velocity associate with each peak.
    identifier : str
        Used to uniquely identify the `Peaks` object.
    attrs : list
        List of strings describing Peak attributes.

    """
    axes_defaults = {"frequency": {"label": "Frequency (Hz)",
                                   "scale": "log"},
                     "wavelength": {"label": "Wavelength (m)",
                                    "scale": "log"},
                     "velocity": {"label": "Velocity (m/s)",
                                  "scale": "linear"},
                     "slowness": {"label": "Slowness (s/m)",
                                  "scale": "log"},
                     "azimuth": {"label": "Azimuth (deg)",
                                 "scale": "linear"}
                     }

    def __init__(self, frequency, velocity, identifier="0", **kwargs):
        """Create `Peaks` from a iterable of frequencies and velocities.

        Parameters
        ----------
        frequency, velocity : iterable of floats
            Frequency and velocity (one per peak), respectively.
        identifier : str, optional
            String to uniquely identify the provided `Peaks`,
            default is "0".
        **kwargs : kwargs
            Optional keyword argument(s) these may include
            additional information about the dispersion peaks such as:
            azimuth, ellipticity, power, and noise. Will generally not
            be entered directly.

        Returns
        -------
        Peaks
            Instantiated `Peaks` object.

        """
        self._frequency = np.array(frequency, dtype=float)
        self._velocity = np.array(velocity, dtype=float)
        self._valid = ~np.isnan(self._velocity)
        self.identifier = str(identifier)
        self.attrs = ["frequency", "velocity"] + list(kwargs.keys())

        logger.debug(f"Creating {self}")
        logger.debug(f"  {self}.attrs={self.attrs}")

        for key, val in kwargs.items():
            setattr(self, f"_{key}", np.array(val, dtype=float))

    @property
    def frequency(self):
        return self._frequency[self._valid]

    @property
    def velocity(self):
        return self._velocity[self._valid]

    @property
    def azimuth(self):
        return self._azimuth[self._valid]

    @property
    def ellipticity(self):
        return self._ellipticity[self._valid]

    @property
    def noise(self):
        return self._noise[self._valid]

    @property
    def power(self):
        return self._power[self._valid]

    @property
    def slowness(self):
        return 1/self.velocity

    @property
    def _slowness(self):
        return 1/self._velocity

    @property
    def wavelength(self):
        return self.velocity/self.frequency

    @property
    def _wavelength(self):
        return self._velocity/self._frequency

    @property
    def wavenumber(self):
        return 2*np.pi*self.frequency/self.velocity

    @property
    def _wavenumber(self):
        return 2*np.pi*self._frequency/self._velocity

    @property
    def extended_attrs(self):
        """List of available Peaks attributes, including calculated."""
        others = ["wavelength", "slowness", "wavenumber"]
        return self.attrs + others

    @classmethod
    def _parse_peaks(cls, peak_data, wavetype="rayleigh", start_time=None, frequencies=None, nmaxima=None):
        """Parse data for a given blockset."""
        regex = get_wavetype()
        wavetype_from_file = regex.search(peak_data).groups()[0]
        if wavetype == "rayleigh" and wavetype_from_file == "Vertical":
            wavetype = "vertical"

        regex = get_process_type()
        process_type = regex.search(peak_data).groups()[0]
        if process_type.lower() == "rtbf" and wavetype_from_file == "Vertical":
            process_type = "capon"
        else:
            process_type = process_type.lower()

        if start_time is None:
            regex = get_peak_from_max(wavetype=wavetype)
            start_time, *_ = regex.search(peak_data).groups()

        if frequencies is None:
            regex = get_peak_from_max(time=start_time, wavetype=wavetype)
            frequencies = []
            for match in regex.finditer(peak_data):
                _, f, *_ = match.groups()
                if f in frequencies:
                    continue
                else:
                    frequencies.append(f)
        nfrequencies = len(frequencies)

        if nmaxima is None:
            regex = get_nmaxima()
            nmaxima = int(regex.search(peak_data).groups()[0])
            nmaxima = max(1, nmaxima)
            nmaxima = min(100, nmaxima)

        frqs = np.full((nmaxima, nfrequencies), fill_value=np.nan, dtype=float)
        vels = np.full_like(frqs, fill_value=np.nan, dtype=float)
        azis = np.full_like(frqs, fill_value=np.nan, dtype=float)
        ells = np.full_like(frqs, fill_value=np.nan, dtype=float)
        nois = np.full_like(frqs, fill_value=np.nan, dtype=float)
        pwrs = np.full_like(frqs, fill_value=np.nan, dtype=float)

        for col, frequency in enumerate(frequencies):
            getpeak = get_peak_from_max(time=start_time, frequency=frequency,
                                        wavetype=wavetype)

            for row, match in enumerate(getpeak.finditer(peak_data)):
                _, _frq, _slo, _azi, _ell, _noi, _pwr = match.groups()

                frqs[row, col] = float(_frq)
                vels[row, col] = 1/float(_slo)
                azis[row, col] = float(_azi)
                ells[row, col] = float(_ell)
                nois[row, col] = float(_noi)
                pwrs[row, col] = float(_pwr)

        # Include for "belt and suspenders".
        getall = get_all(time=start_time, wavetype=wavetype)
        count = len(getall.findall(peak_data))
        if np.sum(~np.isnan(frqs)) != count:  # pragma: no cover
            msg = f"Missing {count - len(frqs)} dispersion peaks."
            raise ValueError(msg)

        return cls(frqs, vels, identifier=f"{start_time}-{process_type}", azimuth=azis,
                   ellipticity=ells, noise=nois, power=pwrs)

    def plot(self, xtype="frequency", ytype="velocity", plot_kwargs=None,
             mask=None):
        """Plot dispersion data in `Peaks` object.

        Parameters
        ----------
        xtype : {'frequency', 'wavelength'}, optional
            Denote whether the x-axis should be either `frequency` or
            `wavelength`, default is `frequency`.
        ytype : {'velocity', 'slowness'}, optional
            Denote whether the y-axis should be either `velocity` or
            `slowness`, default is `velocity`.
        plot_kwargs : dict, optional
            Keyword arguments to pass along to `ax.plot`, default is
            `None` indicating the predefined settings should be used.
        mask : ndarray, optional
            Boolean array mask to determine which points are to be
            plotted, default is `None` so all valid points will be
            plotted.

        Returns
        -------
        tuple
            Of the form `(fig, ax)` where `fig` and `ax` are the
            `Figure` and `Axes` objects which were generated on-the-fly.

        """
        # Prepare xtype, ytype.
        xtype, ytype = self._prepare_types(xtype=xtype, ytype=ytype)

        # Generate fig, ax on-the-fly.
        ncols = len(xtype)
        fig, ax = plt.subplots(ncols=ncols, figsize=(3*ncols, 3), dpi=150)
        ax = [ax] if ncols == 1 else ax

        # Loop across Axes(s).
        for _ax, _xtype, _ytype in zip(ax, xtype, ytype):
            # Plot Peaks.
            self._plot(ax=_ax, xtype=_xtype, ytype=_ytype,
                       plot_kwargs=plot_kwargs, mask=mask)
            # Configure Axes
            self._configure_axes(ax=_ax, xtype=_xtype, ytype=_ytype,
                                 defaults=self.axes_defaults)

        # Return fig, ax.
        fig.tight_layout()
        return (fig, ax)

    @staticmethod
    def _prepare_types(**kwargs):
        """Handle `xtype` and `ytype` to ensure they are acceptable.

        Accept xtype and ytype as kwargs. If any is `str` cast to
        `list`, if `list` or `tuple` pass, otherwise raise `TypeError`.

        Parameters
        ----------
        **kwargs : kwargs
            `dict` of the form `dict(xtype=xtype, ytype=ytype)`.

        Returns
        -------
        dict_values
            Containing the handles `kwargs` in the order in which they
            were provided.

        Raises
        ------
        TypeError
            If any in `kwargs.values()` is not `str`, `list`, or
            `tuple`.

        """
        # Check type, cast if necessary.
        for key, value in kwargs.items():
            if isinstance(value, str):
                kwargs[key] = [value]
            elif isinstance(value, (list, tuple)):
                pass
            else:
                msg = f"{key} must be a str or iterable not, {type(key)}."
                raise TypeError(msg)

        # Ensure lengths are consistant.
        reference_key = key
        reference_length = len(kwargs[reference_key])
        for key, value in kwargs.items():
            if len(value) != reference_length:
                msg = f"len({reference_key}) != len({key}). "
                msg += "All entries must have consistent length."
                raise IndexError(msg)

        return kwargs.values()

    def _plot(self, ax, xtype, ytype, plot_kwargs=None, mask=None):
        """Plot `Peaks` data to provided `Axes`.

        Parameters
        ----------
        ax : Axes
            `Axes` on which to plot the `Peaks` data.
        xtype : {"frequency", "wavelength"}
            Attribute to plot along the x-axis.
        ytype : {"velocity", "slowness"}
            Attribute to plot along the y-axis.
        plot_kwargs : kwargs, optional
            Keyword arguments to pass along to `ax.plot`, default is
            `None` indicating the predefined settings should be used.
        mask : ndarray, optional
            Boolean array mask to determine which points are to be
            plotted, default is `None` so all valid points will be
            plotted.

        Returns
        -------
        None
            Updates `Axes` object with the data from `Peaks`.

        Raises
        ------
        AttributeError
            If `xtype` and/or `ytype` are invalid attributes.

        """
        # Organize plot kwargs.
        default_plot_kwargs = dict(linestyle="", marker="o", color="b",
                                   markersize=1, markerfacecolor="none",
                                   label=self.identifier)
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

        # Handle presense or absence of mask.
        mask = self._valid if mask is None else mask

        try:
            ax.plot(getattr(self, f"_{xtype}")[mask],
                    getattr(self, f"_{ytype}")[mask],
                    **plot_kwargs)
        except AttributeError as e:
            msg = f"{xtype} and/or {ytype} is/are not attribute(s). "
            msg += f"Available attributes are: {self.extended_attrs}"
            raise AttributeError(msg) from e

    @staticmethod
    def _configure_axes(ax, xtype, ytype, defaults):
        """Prepare `Axes` with user-friendly defaults."""
        # x-axis
        for key, value in defaults.get(xtype, {}).items():
            getattr(ax, f"set_x{key}")(value)

        # y-axis
        for key, value in defaults.get(ytype, {}).items():
            getattr(ax, f"set_y{key}")(value)

    def reject_limits_outside(self, attr, limits):
        """Reject peaks outside the stated bounds.

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
            Updates the `Peaks` object's state.

        Notes
        -----
        This method is somewhat similar to
        :meth:`swprocess.Peaks.reject_inside`, but is more
        computationally expensive.

        """
        _attr = getattr(self, f"_{attr}")
        _min, _max = limits

        bool_array = np.zeros_like(_attr, dtype=bool)
        if _min is None and _max is None:
            msg = "`reject_outside` called, but limits were both `None`, "
            msg += "therefore no values were rejected."
            warnings.warn(msg)
        elif _min is None:
            np.greater(_attr, _max, out=bool_array, where=self._valid)
        elif _max is None:
            np.less(_attr, _min, out=bool_array, where=self._valid)
        else:
            np.greater(_attr, _max, out=bool_array, where=self._valid)
            np.less(_attr, _min, out=bool_array,
                    where=np.logical_and(self._valid, ~bool_array))

        self._reject(bool_array)

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
            Updates the `Peaks` object's state.

        """
        bool_array = self._reject_box_inside_bool_array(xtype, xlims,
                                                        ytype, ylims)
        self._reject(bool_array)

    def _reject_box_inside_bool_array(self, xtype, xlims, ytype, ylims):
        """Boolean array to describe which peaks should be rejected.

        Parameters
        ----------
        xtype, ytype : {"frequency", "velocity", "slowness", "wavelength"}
            Parameter domain in which the limits are defined.
        xlims, ylims : tuple
            Tuple with the lower and upper limits for each of the
            boundaries.

        Returns
        -------
        ndarray
            Containing the indices for rejection.

        """
        xs = getattr(self, f"_{xtype}")
        x_min, x_max = min(xlims), max(xlims)

        ys = getattr(self, f"_{ytype}")
        y_min, y_max = min(ylims), max(ylims)

        bool_array = np.zeros_like(xs, dtype=bool)
        np.greater(xs, x_min, out=bool_array, where=self._valid)
        np.less(xs, x_max, out=bool_array, where=bool_array)
        np.greater(ys, y_min, out=bool_array, where=bool_array)
        np.less(ys, y_max, out=bool_array, where=bool_array)

        return bool_array

    def _reject(self, bool_array):
        """Reject peaks according to the provided boolean array."""
        self._valid[bool_array] = False

    def simplify_mpeaks(self, attr):
        """Produce desired attribute with multiple peaks removed.

        Parameters
        ----------
        attr : {"frequency", "velocity", "azimuth", "power", "ellipticity", "noise"}
            Attribute of interest.

        Returns
        -------
        ndarray
            With the attribute of interest simplified to remove
            duplicate peaks.

        """
        # If 1D, return valid.
        if getattr(self, f"_{attr}").ndim == 1:
            return getattr(self, f"{attr}")

        # If 2D and has power, return values based on maximum.
        elif hasattr(self, "_power"):
            attr = getattr(self, f"_{attr}")
            decide = self._power

        # If 2D and missing power, return first available value.
        else:
            attr = getattr(self, f"_{attr}")
            decide = self._frequency

        values = []
        for cindex, (colvals, valids) in enumerate(zip(decide.T, self._valid.T)):
            mindex = 0
            mval = -np.inf
            for rindex, (rowval, valid) in enumerate(zip(colvals, valids)):
                if not valid:
                    continue
                if rowval > mval:
                    mindex = rindex
                    mval = rowval

            if mval != -np.inf:
                values.append(attr[mindex, cindex])

        return values

    def to_json(self, fname, append=False):
        """Write `Peaks` to json file.

        Parameters
        ----------
        fname : str
            Output file name, can include a relative or the full path.
        append : bool, optional
            Controls whether `fname` (if it exists) should be appended
            to or overwritten, default is `False` indicating `fname`
            will be overwritten.

        Returns
        -------
        None
            Instead writes file to disk.

        """
        data = {}
        for attr in self.attrs:
            data[attr] = getattr(self, f"_{attr}").tolist()
        data["valid"] = self._valid.tolist()

        # Assumes dict of the form {"id0":{data}, "id1":{data} ... }
        if append:
            with open(fname, "r") as f:
                data_to_update = json.load(f)
            if self.identifier in data_to_update.keys():
                msg = "Data already exists in file with identifier "
                msg += f"{self.identifier}, file left unmodified."
                raise KeyError(msg)
            else:
                data_to_update[self.identifier] = data
                with open(fname, "w") as f:
                    json.dump(data_to_update, f)
        else:
            data = {self.identifier: data}
            with open(fname, "w") as f:
                json.dump(data, f)

    @classmethod
    def from_json(cls, fname):
        """Read `Peaks` from json file.

        Parameters
        ----------
        fnames : str
            Name of the input file, may contain a relative or the full
            path.

        Returns
        -------
        Peaks
            Initialized `Peaks` object.

        """
        with open(fname, "r") as f:
            data = json.load(f)

        key_list = list(data.keys())
        if len(key_list) > 1:
            msg = f"More than one dataset in {fname}, taking only the first! "
            msg += "If you want all see `PeaksSuite.from_json`."
            warnings.warn(msg)

        return cls.from_dict(data[key_list[0]], identifier=key_list[0])

    @classmethod
    def from_dict(cls, data_dict, identifier="0"):
        """Initialize `Peaks` from `dict`.

        Parameters
        ----------
        data_dict: dict
            Of the form
            `{"frequency":freq, "velocity":vel, "kwarg1": kwarg1}`
            where `freq` is a list of floats denoting frequency values.
            `vel` is a list of floats denoting velocity values.
            `kwarg1` is an optional keyword argument denoting some
            additional parameter (may include more than one).
        identifiers : str
            String to uniquely identify the provided `Peaks` object.

        Returns
        -------
        Peaks
            Initialized `Peaks` instance.

        """
        valid = data_dict.get("valid")
        try:
            del data_dict["valid"]
        except KeyError:
            pass

        obj = cls(identifier=identifier, **data_dict)

        if valid is not None:
            obj._valid = np.array(valid, dtype=bool)

        return obj

    @classmethod
    def from_max(cls, fname, wavetype="rayleigh"):
        """Initialize a `Peaks` object from a `.max` file.

        Parameters
        ----------
        fname : str
            Denotes the filename for the .max file, may include a
            relative or the full path.
        wavetype : {'rayleigh', 'love'}, optional
            Wavetype to extract from file, default is 'rayleigh'.

        Returns
        -------
        Peaks
            Initialized `Peaks` object.

        Notes
        -----
        If the results from multiple time windows are in the same .max
        file, as is most often the case, this method ignores all but the
        first instance found.

        """
        with open(fname, "r") as f:
            peak_data = f.read()

        regex = get_geopsy_version()
        major, minor, micro = regex.search(peak_data).groups()
        version = f"{major}.{minor}.{micro}"
        check_geopsy_version(version)

        return cls._parse_peaks(peak_data, wavetype=wavetype)

    def __eq__(self, other):
        if not isinstance(other, Peaks):
            return False

        # Check non-iterable attributes
        for key in ["identifier"]:
            try:
                if getattr(self, key) != getattr(other, key):
                    return False
            except AttributeError:
                return False

        # Check attributes
        if len(self.attrs) != len(other.attrs):
            return False

        # Check iterable attributes
        for attr in self.attrs:
            try:
                myattr = getattr(self, attr)
                urattr = getattr(other, attr)
            except AttributeError:
                return False
            else:
                try:
                    if not np.allclose(myattr, urattr, equal_nan=True):
                        return False
                except ValueError:
                    return False

        return True

    def __str__(self):
        """Human-readable representation of the `Peaks` object."""
        return f"Peaks with identifier={self.identifier}"
