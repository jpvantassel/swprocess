"""Peaks class definition."""

import json
import warnings
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from .regex import get_peak_from_max, get_nmaxima

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
                     "azimuth" : {"label": "Azimuth (deg)",
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
            azimuth (azi), ellipticity (ell), power (pwr), and noise
            (pwr). Will generally not be entered directly.

        Returns
        -------
        Peaks
            Instantiated `Peaks` object.

        """
        self.frequency = np.atleast_2d(np.array(frequency, dtype=float))
        self.velocity = np.atleast_2d(np.array(velocity, dtype=float))
        self.identifier = str(identifier)
        self.attrs = ["frequency", "velocity"] + list(kwargs.keys())

        logger.debug(f"Creating {self}")
        logger.debug(f"  {self}.attrs={self.attrs}")

        for key, val in kwargs.items():
            setattr(self, key, np.atleast_2d(np.array(val, dtype=float)))

    @property
    def slowness(self):
        return 1/self.velocity

    @property
    def wavelength(self):
        return self.velocity/self.frequency

    @property
    def wavenumber(self):
        return 2*np.pi*self.frequency/self.velocity

    @property
    def extended_attrs(self):
        """List of available Peaks attributes, including calculated."""
        others = ["wavelength", "slowness", "wavenumber"]
        return self.attrs + others

    @classmethod
    def _parse_peaks(cls, peak_data, wavetype="rayleigh", start_time=None, frequencies=None, nmaxima=None):
        """Parse data for a given time block."""
        if start_time is None:
            regex = get_peak_from_max(wavetype=wavetype)
            start_time, *_ = regex.search(peak_data).groups()

        getpeak = get_peak_from_max(time=start_time, wavetype=wavetype)

        if frequencies is None:
            frequencies = []
            for match in getpeak.finditer(peak_data):
                _, f, *_ = match.groups()
                if f in frequencies:
                    continue
                else:
                    frequencies.append(f)
            nfrequencies = len(frequencies)

        if nmaxima is None:
            nmaxima = int(get_nmaxima().findall(peak_data)[0])

        frqs = np.full((nmaxima, nfrequencies), fill_value=np.nan, dtype=float)
        vels = np.full_like(frqs, fill_value=np.nan, dtype=float)
        azis = np.full_like(frqs, fill_value=np.nan, dtype=float)
        ells = np.full_like(frqs, fill_value=np.nan, dtype=float)
        nois = np.full_like(frqs, fill_value=np.nan, dtype=float)
        pwrs = np.full_like(frqs, fill_value=np.nan, dtype=float)

        for col, frequency in enumerate(frequencies):
            getpeak = get_peak_from_max(time=start_time, wavetype=wavetype, frequency=frequency)
            
            for row, match in enumerate(getpeak.finditer(peak_data)):
                _, _frq, _slo, _azi, _ell, _noi, _pwr = match.groups()

                frqs[row, col] = float(_frq)
                vels[row, col] = 1/float(_slo)
                azis[row, col] = float(_azi)
                ells[row, col] = float(_ell)
                nois[row, col] = float(_noi)
                pwrs[row, col] = float(_pwr)

        # TODO (jpv): Belt & Suspenders.
        # # Include for "belt and suspenders".
        # getall = get_all(time=start_time, wavetype=wavetype)
        # count = len(getall.findall(peak_data))
        # if len(frqs) != count:  # pragma: no cover
        #     msg = f"Missing {count - len(frqs)} dispersion peaks."
        #     raise ValueError(msg)

        args = (frqs, vels, start_time)
        kwargs = dict(azimuth=azis, ellipticity=ells, noise=nois, power=pwrs)
        return cls(*args, **kwargs)

    def plot(self, xtype="frequency", ytype="velocity", plot_kwargs=None,
             indices=None):
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
        indices : ndarray, optional
            Indices to plot, default is `None` so all points will be
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
                       plot_kwargs=plot_kwargs, indices=indices)
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

    def _plot(self, ax, xtype, ytype, plot_kwargs=None, indices=None):
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
        indices : ndarray, optional
            Indices to plot, default is `None` so all points will be
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
        default_plot_kwargs = dict(linestyle="", marker="o", color="b",
                                   markersize=1, markerfacecolor="none",
                                   label=self.identifier)
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

        indices = Ellipsis if indices is None else indices

        try:
            ax.plot(getattr(self, xtype)[indices],
                    getattr(self, ytype)[indices],
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

    def blitz(self, attr, limits):
        """Reject peaks outside the stated boundary.

        This method is similar to `reject`, however it is more expensive
        and therefore should only be called once on a dataset.

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

        """
        values = getattr(self, attr)
        _min, _max = limits
        reject_ids = self._reject_outside_ids(values, _min, _max)
        self._reject(reject_ids)

    def reject(self, xtype, xlims, ytype, ylims):
        """Reject peaks inside the stated boundaries.

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
        reject_ids = self.reject_ids(xtype, xlims, ytype, ylims)
        self._reject(reject_ids)

    def reject_ids(self, xtype, xlims, ytype, ylims):
        """Determine rejection ids.

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
        xs = getattr(self, xtype)
        x_min, x_max = min(xlims), max(xlims)

        ys = getattr(self, ytype)
        y_min, y_max = min(ylims), max(ylims)

        return self._reject_inside_ids(xs, x_min, x_max,
                                       ys, y_min, y_max)

    @staticmethod
    def _reject_inside_ids(d1, d1_min, d1_max, d2, d2_min, d2_max):
        condition1 = np.logical_and(d1 > d1_min, d1 < d1_max)
        condition2 = np.logical_and(d2 > d2_min, d2 < d2_max)
        return np.flatnonzero(np.logical_and(condition1, condition2))

    @staticmethod
    def _reject_outside_ids(values, _min, _max):
        if _min is None and _max is None:
            msg = "blitz called, but limits are `None`, so no values rejected."
            warnings.warn(msg)
            condition = np.zeros_like(values, dtype=int)
        elif _min is None:
            condition = values > _max
        elif _max is None:
            condition = values < _min
        else:
            condition = np.logical_or(values > _max, values < _min)
        return np.flatnonzero(condition)

    def _reject(self, reject_ids):
        """Reject peaks with the given ids."""
        for attr in self.attrs:
            setattr(self, attr, np.delete(getattr(self, attr), reject_ids))

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
        return cls(identifier=identifier, **data_dict)

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
            data[attr] = getattr(self, attr).tolist()

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

    def write_peak_json(self, fname):
        msg = ".write_peak_json is deprecated use .to_json instead."
        warnings.warn(msg, DeprecationWarning)
        self.to_json(fname)

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
        None
            Reads `Peaks` object from disk.

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
    def from_max(cls, fname, wavetype="rayleigh"):
        """Initialize a `Peaks` object from a `.max` file.

        If the results from multiple time windows are in the same .max
        file, as is most often the case, this method ignores all but the
        first instance found.

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

        """
        with open(fname, "r") as f:
            peak_data = f.read()

        return cls._parse_peaks(peak_data, wavetype=wavetype, start_time=None)

    def __eq__(self, other):
        if not isinstance(other, Peaks):
            return False

        # Check non-iterable attributes
        for key in ["identifier"]:
            try:
                if getattr(self, key) != getattr(other, key):
                    return False
            except AttributeError as e:
                return False

        # Check attributes
        if len(self.attrs) != len(other.attrs):
            return False

        # Check iterable attributes
        for attr in self.attrs:
            try:
                myattr = getattr(self, attr)
                urattr = getattr(other, attr)
            except AttributeError as e:
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
