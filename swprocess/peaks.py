"""Peaks class definition."""

import json
import re
import warnings
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from swprocess import plot_tools
from swprocess.regex import *

logger = logging.getLogger(__name__)


class Peaks():
    """Class for handling peak dispersion data.

    Attributes
    ----------
        TODO (jpv): Finish documentation.

    """

    def __init__(self, frequency, velocity, identifier="0", **kwargs):
        """Initialize an instance of Peaks from a list of frequency
        and velocity values.

        Parameters
        ----------
        frequency, velocity : list
            Frequency and velocity (one per peak), respectively.
        identifiers : str
            String to uniquely identify the provided frequency-velocity
            pair.
        **kwargs : kwargs
            Optional keyword argument(s) these may include
            additional details about the dispersion peaks such as:
            azimuth (azi), ellipticity (ell), power (pwr), and noise
            (pwr). Will generally not be used directly.

        Returns
        -------
        Peaks
            Instantiated `Peaks` object.

        """
        self.frequency = np.array(frequency, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.ids = str(identifier)
        logging.debug(f"**kwargs = {kwargs}")
        self.attrs = ["frequency", "velocity"] + list(kwargs.keys())
        for key, val in kwargs.items():
            setattr(self, key, np.array(val, dtype=float))

    @property
    def identifiers(self):
        return self.ids

    @property
    def wav(self):
        msg = "Use of wav is deprecated use wavelength instead."
        warnings.warn(msg, DeprecationWarning)
        return self.wavelength

    @property
    def wavelength(self):
        return self.velocity/self.frequency

    @property
    def extended_attrs(self):
        others = ["wavelength", "slowness"]
        return self.attrs + others

    @property
    def slowness(self):
        return 1/self.velocity

    def from_dicts(self, *args, **kwargs):
        msg = "Peaks.from_dicts has been deprecated, use Peaks.from_dict or PeaksSuite.from_dicts() instead."
        raise DeprecationWarning(msg)

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
            `kwarg1` is an optional keyword argument, may include as
            many or none.
        identifiers : str
            String to uniquely identify the provided frequency-velocity
            pair.

        Returns
        -------
        Peaks
            Initialized `Peaks` instance.

        """
        for key in ["frequency", "velocity"]:
            if key not in data_dict.keys():
                msg = f"`frequency` and `velocity` keys are not optional."
                raise ValueError(msg)

        return cls(identifier=identifier, **data_dict)

    @classmethod
    def from_jsons(self, *args, **kwargs):
        msg = "Peaks.from_jsons has been deprecated, use Peaks.from_json() or PeaksSuite.from_jsons() instead."
        raise DeprecationWarning(msg)

    @classmethod
    def from_json(cls, fname):
        with open(fname, "r") as f:
            data = json.load(f)

        key_list = list(data.keys())
        if len(key_list) > 1:
            msg = f"More than one dataset in {fname}, taking only the first! If you want all see `PeaksSuite.from_jsons`."
            warnings.warn(msg)

        return cls.from_dict(data[key_list[0]], identifier=key_list[0])

    @classmethod
    def from_maxs(cls, *args, **kwargs):
        msg = "Peaks.from_maxs has been deprecated, use PeaksPassive.from_max() or PeaksSuite.from_maxs() instead."
        raise DeprecationWarning(msg)

    @classmethod
    def from_max(cls, fname, identifier="0", rayleigh=True, love=False):
        """Initialize `Peaks` from `.max` file(s).

        Parameters
        ----------
        fnames : str
            Denotes the filename(s) for the .max file, may include a
            relative or the full path.
        identifier : str
            Uniquely identifying the dispersion data from each file.
        rayleigh : bool, optional
            Denote if Rayleigh data should be extracted, default is
            `True`.
        love : bool, optional
            Denote if Love data should be extracted, default is
            `False`.

        Returns
        -------
        PeaksPassive
            Initialized `PeaksPassive` object.

        Raises
        ------
        ValueError
            If neither or both `rayleigh` and `love` are equal to
            `True`.

        """
        if not isinstance(rayleigh, bool) and not isinstance(love, bool):
            msg = f"`rayleigh` and `love` must both be of type `bool`, not {type(rayleigh)} and {type(love)}."
            raise TypeError(msg)
        if rayleigh == True and love == True:
            raise ValueError("`rayleigh` and `love` cannot both be `True`.")
        if rayleigh == False and love == False:
            raise ValueError("`rayleigh` and `love` cannot both be `False`.")

        with open(fname, "r") as f:
            lines = f.read()

        getpeak = getpeak_rayleigh if rayleigh else getpeak_love

        tims, frqs, vels, azis, ells, nois, pwrs = [], [], [], [], [], [], []
        for lineinfo in getpeak.finditer(lines):
            _tim, _frq, _slo, _azi, _ell, _noi, _pwr = lineinfo.groups()

            tims.append(float(_tim))
            frqs.append(float(_frq))
            vels.append(1/float(_slo))
            azis.append(float(_azi))
            ells.append(float(_ell))
            nois.append(float(_noi))
            pwrs.append(float(_pwr))

        # Check got everything
        getall = getall_rayleigh if rayleigh else getall_love
        if len(tims) != len(getall.findall(lines)):
            msg = f"Missing {len(getall.findall(lines)) - len(times)} dispersion peaks."
            raise ValueError(msg)

        args = (frqs, vels, identifier)
        kwargs = dict(azimuth=azis, ellipticity=ells,
                      noise=nois, power=pwrs, time=tims)
        return cls(*args, **kwargs)

    def _plot(self, xtype, ytype, ax, plot_kwargs=None, ax_kwargs=None):
        """Plot requested `Peaks` data to provided `Axes`."""
        for _type, value in zip(["xtype", "ytype"], [xtype, ytype]):
            if value not in self.extended_attrs:
                msg = f"{_type} = {value} is not an attribute. Attributes are: {self.attrs}"
                raise ValueError(msg)

        default_plot_kwargs = dict(linestyle="", marker="o", color="b",
                                   markersize=1, markerfacecolor="none",
                                   label=self.ids)
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

        # TODO (jpv): I don't need this information here, this get run
        # on each loop iteration.
        pot_ax_kwargs = {"frequency": {"set_xlabel": "Frequency (Hz)",
                                       "set_xscale": "log"},
                         "wavelength": {"set_xlabel": "Wavelength (m)",
                                        "set_xscale": "log"},
                         "velocity": {"set_ylabel": "Velocity (m/s)",
                                      "set_yscale": "linear"},
                         "slowness": {"set_ylabel": "Slowness (s/m)",
                                      "set_yscale": "log"}
                         }
        ax_kwargs = {} if ax_kwargs is None else ax_kwargs
        ax_kwargs = {**pot_ax_kwargs.get(xtype, {}),
                     **pot_ax_kwargs.get(ytype, {}),
                     **ax_kwargs}

        ax.plot(getattr(self, xtype), getattr(self, ytype), **plot_kwargs)
        for key, value in ax_kwargs.items():
            getattr(ax, key)(value)

    def plot(self, xtype="frequency", ytype="velocity", ax=None,
             plot_kwargs=None, ax_kwargs=None):
        """Create plot of dispersion data.

        Parameters
        ----------
        xtype : {'frequency', 'wavelength'}, optional
            Denote whether the x-axis should be either `frequency` or
            `wavelength`, default is `frequency`.
        ytype : {'velocity', 'slowness'}, optional
            Denote whether the y-axis should be either `velocity` or
            `slowness`, default is `velocity`.
        ax : Axes, optional
            Pass an `Axes` on which to plot, default is `None` meaning
            a `Axes` will be generated on-the-fly.
        plot_kwargs : dict, optional
            Optional keyword arguements to pass to plot.
        ax_kwargs : dict, optional
            Optional keyword arguements to control plotting `Axes`.

        Returns
        -------
        None or tuple
            `None` if `ax` is provided, otherwise `tuple` of the form
            `(fig, ax)` where `fig` is the figure handle and `ax` is
            the axes handle.

        """
        values = self._check_plot(xtype, ytype, ax, plot_kwargs, ax_kwargs)
        xtype, ytype, plot_kwargs, ax_kwargs, ax_was_none = values

        if ax_was_none:
            ncols = len(xtype)
            fig, ax = plt.subplots(nrows=1, ncols=ncols,
                                   figsize=(3*ncols, 3), dpi=150)
            if ncols == 1:
                ax = [ax]

        for _ax, _xtype, _ytype, _plot_kwargs, _ax_kwargs in zip(ax, xtype, ytype, plot_kwargs, ax_kwargs):
            self._plot(xtype=_xtype, ytype=_ytype, ax=_ax,
                       plot_kwargs=_plot_kwargs, ax_kwargs=_ax_kwargs)
        _ax.legend()

        if ax_was_none:
            fig.tight_layout()
            return (fig, ax)

    @staticmethod
    def _check_plot(xtype, ytype, ax, plot_kwargs, ax_kwargs):
        if isinstance(xtype, str):
            xtype = [xtype]
        if isinstance(ytype, str):
            ytype = [ytype]
        ncols = len(xtype)

        plot_kwargs = Peaks._check_kwargs(plot_kwargs, ncols)
        ax_kwargs = Peaks._check_kwargs(ax_kwargs, ncols)

        if ax is None:
            ax_was_none = True
        else:
            ax_was_none = False
            if len(xtype) != len(ax):
                msg = f"len(xtype) must equal len(ax), {len(xtype)} != {len(ax)}"
                raise ValueError(msg)

        return (xtype, ytype, plot_kwargs, ax_kwargs, ax_was_none)

    @staticmethod
    def _check_kwargs(kwargs, ncols):
        if kwargs is None or isinstance(kwargs, dict):
            return [kwargs]*ncols
        elif isinstance(kwargs, list) and len(kwargs) == ncols:
            return kwargs
        else:
            msg = f"`kwargs` must be `None` or `dict`, not {type(kwargs)}."
            raise TypeError(msg)

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
            Instead updates the `Peaks` object's state.

        """
        values = getattr(self, attr)
        _min, _max = limits

        reject_ids = self._reject_outside_ids(values, _min, _max)

        self._reject(reject_ids)

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
            condition = np.logical_or(values > _max,
                                      values < _min)
        return np.flatnonzero(condition)

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
            Instead updates the `Peaks` object's state.

        """
        reject_ids = self.reject_ids(xtype, xlims, ytype, ylims)
        self._reject(reject_ids)

    def reject_ids(self, xtype, xlims, ytype, ylims):
        xs = getattr(self, xtype)
        x_min, x_max = min(xlims), max(xlims)

        ys = getattr(self, ytype)
        y_min, y_max = min(ylims), max(ylims)

        return self._reject_inside_ids(xs, x_min, x_max,
                                       ys, y_min, y_max)

    @staticmethod
    def _reject_inside_ids(d1, d1_min, d1_max, d2, d2_min, d2_max):
        condition1 = np.logical_and(d1 > d1_min,
                                    d1 < d1_max)
        condition2 = np.logical_and(d2 > d2_min,
                                    d2 < d2_max)
        return np.flatnonzero(np.logical_and(condition1, condition2))

    def _reject(self, reject_ids):
        for attr in self.attrs:
            setattr(self, attr, np.delete(getattr(self, attr), reject_ids))

    def write_peak_json(self, fname):
        msg = ".write_peak_json is deprecated use .to_json instead."
        warnings.warn(msg, DeprecationWarning)
        self.to_json(fname)

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
            if self.ids in data_to_update.keys():
                msg = f"Data already exists in file with identifier {self.ids}, file left unmodified."
                raise KeyError(msg)
            else:
                existing_data[self.ids] = data
                with open(fname, "w") as f:
                    json.dump(data_to_update, f)
        else:
            data = {self.ids: data}
            with open(fname, "w") as f:
                json.dump(data, f)

    def __eq__(self, other):
        # Check other is a Peak object
        if not isinstance(other, Peaks):
            return False

        # Check non-iterable attributes
        for key in ["ids"]:
            try:
                if getattr(self, key) != getattr(other, key):
                    return False
            except AttributeError as e:
                warnings.warn(f"self or other missing attribute {key}.")
                return False

        # Check iterable attributes
        for attr in self.attrs:
            try:
                myattr = getattr(self, attr)
                urattr = getattr(other, attr)
                for selfval, otherval in zip(myattr, urattr):
                    if selfval != otherval:
                        return False
            except AttributeError as e:
                warnings.warn(f"self or other missing attribute {attr}.")
                return False

        return True
