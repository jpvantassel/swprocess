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

_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5
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
        wav = []
        for frq, vel in zip(self.frequency, self.velocity):
            wav.append(vel/frq)
        return wave

    @property
    def slowness(self):
        slo = []
        for vel in self.velocity:
            slo.append(1/vel)
        return slo

    @property
    def mean_disp(self, **kwargs):
        return self.compute_dc_stats(self.frequency, self.velocity, **kwargs)

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
        kwargs = dict(azi=azis, ell=ells, noi=nois, pwr=pwrs, tim=tims)
        return cls(*args, **kwargs)

    def plot(self, xtype="frequency", ax=None):
        """Create plot of dispersion data.

        Parameters
        ----------
        xtype : {'frequency', 'wavelength'}, optional
            Denote whether the x-axis should be either `frequency` or
            `wavelength`, default is `frequency`.
        ytype : {'velocity', 'slowness', '}
        ax : Axes, optional
            Pass an `Axes` on which to plot, default is `None` meaning
            a `Axes` will be generated on-the-fly.

        Returns
        -------
        None or tuple
            `None` if `ax` is provided, otherwise `tuple` of the form
            `(fig, ax)` where `fig` is the figure handle and `ax` is
            the axes handle.

        """
        ax_was_none = False
        if ax is None:
            ax_was_none = True
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3), dpi=150)

        if xtype.lower() == "wavelength":
            x = self.wavelength
            xlabel = "Wavelength (m)"
        elif xtype.lower() == "frequency":
            x = self.frequency
            xlabel = "Frequency (Hz)"
        else:
            raise ValueError(f"xtype = {xtype}, not recognized.")

        for x, v, color, label in zip(x, self.vel, _colors, self.ids):
            ax.plot(x, v, color=color, linestyle="none",
                    marker="o", markerfacecolor="none", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Phase Velocity (m/s)")
        ax.set_xscale("log")
        ax.legend()

        if ax_was_none:
            return (fig, ax)

    def plot_2pannel(self):
        fig, (axf, axw) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))
        for f, v, i, color in zip(self.frequency, self.velocity, self.ids, _colors):
            axf.plot(f, v, color=color, linestyle="none",
                     marker="o", markerfacecolor="none", label=i)
            axw.plot(v/f, v, color=color, linestyle="none",
                     marker="o", markerfacecolor="none")
        axf.set_xlabel("Frequency (Hz)")
        axw.set_xlabel("Wavelength (m)")
        axf.set_ylabel("Phase Velocity (m/s)")
        axw.set_ylabel("Phase Velocity (m/s)")
        axw.set_xscale("log")
        return (fig, axf, axw)

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
        xs = getattr(self, xtype)
        x_min, x_max = min(xlims), max(xlims)

        ys = getattr(self, ytype)
        y_min, y_max = min(ylims), max(ylims)

        reject_ids = self._reject_inside_ids(xs, x_min, x_max,
                                             ys, y_min, y_max)

        self._reject(reject_ids)

    @staticmethod
    def _reject_inside_ids(d1, d1_min, d1_max, d2, d2_min, d2_max):
        condition1 = np.logical_and(d1 > d1_min,
                                    d1 < d1_max)
        condition2 = np.logical_and(d2 > d2_min,
                                    d1 < d2_max)
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
