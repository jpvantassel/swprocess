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

"""ActiveTimeSeries class definition."""

import warnings
import logging

import numpy as np
from scipy.signal import correlate
from sigpropy import TimeSeries

logger = logging.getLogger("swprocess.activetimeseries")


class ActiveTimeSeries(TimeSeries):
    """A class for working with active-source `TimeSeries`.

    Attributes
    ----------
    amplitude : ndarray
        Recording's amplitude, one per sample.
    dt : float
        Time step between samples in seconds.

    """

    @staticmethod
    def _check_input(nstacks, delay):
        """Check inputs have correct type and values.

        Specifically:
        1. `nstacks` is an `int` greater than zero.
        2. `delay` is a `float` less than or equal to zero.

        """
        nstacks = int(nstacks)
        if nstacks <= 0:
            msg = f"nstacks must be greater than zero, not {nstacks}."
            raise ValueError(msg)

        delay = float(delay)
        if delay > 0:
            msg = f"delay must be less than or equal to zero, not {delay}."
            raise ValueError(msg)

        return (nstacks, delay)

    def __init__(self, amplitude, dt, nstacks=1, delay=0):
        """Initialize an `ActiveTimeSeries` object.

        Parameters
        ----------
        amplitude : array-like
            Recording's amplitude, one per sample. The first
            value is associated with `time=0` seconds and the last is
            associate with `time=(len(amplitude)-1)*dt` seconds.
        dt : float
            Time step between samples in seconds.
        nstacks : int, optional
            Number of stacks used to produce `amplitude`, default is 1.
        delay : float, optional
            Delay to the start of the record in seconds, default is 0.

        Returns
        -------
        ActiveTimeSeries
            Intialized `ActiveTimeSeries` object.

        """
        super().__init__(amplitude=amplitude, dt=dt)
        self._nstacks, self._delay = self._check_input(nstacks, delay)
        self._multiple = 1

    @property
    def n_stacks(self):
        warnings.warn("`n_stacks` is deprecated, use `nstacks` instead",
                      DeprecationWarning)
        return self._nstacks

    @property
    def nstacks(self):
        return self._nstacks

    @property
    def delay(self):
        return self._delay

    @property
    def _df(self):
        # Internal (i.e., real) df.
        return super().df

    @property
    def df(self):
        # User facing df.
        return self._df * self.multiple

    @property
    def multiple(self):
        return self._multiple

    @property
    def time(self):
        """Time vector for `ActiveTimeSeries`."""
        return super().time + self._delay

    def stack_append(self, timeseries):
        """Stack (i.e., average) a new timeseries onto the current one.

        Parameters
        ----------
        timeseries : ActiveTimeSeries
            `ActiveTimeSeries` to be stacked onto the current object.

        Returns
        -------
        None
            Updates the attributes `amplitude` and `nstacks`.

        Raises
        ------
        ValueError
            If `timeseries` is not an `ActiveTimeSeries` or
            it cannot be stacked to the current object (i.e., the two
            are dissimilar).

        """
        if not self._is_similar(timeseries, exclude=["nstacks"]):
            msg = "`timeseries` is incompatible and cannot be stacked."
            raise ValueError(msg)

        self._amp = (self._amp*self.nstacks +
                     timeseries._amp*timeseries.nstacks)
        self._amp /= (self.nstacks + timeseries.nstacks)
        self._nstacks += timeseries.nstacks

    @classmethod
    def from_activetimeseries(cls, activetimeseries):
        return cls(activetimeseries.amplitude,
                   activetimeseries.dt,
                   nstacks=activetimeseries.nstacks,
                   delay=activetimeseries.delay)

    @classmethod
    def from_trace_seg2(cls, trace):
        """Initialize from a SEG2 `Trace` object.

        This method is similar to :meth:`ActiveTimeSeries.from_trace`
        except that it extracts additional information from the `Trace`
        header. So only use this method if you have a seg2 file and the
        header information is correct.

        Parameters
        ----------
        trace: Trace
            `Trace` object from a correctly written seg2 file.

        Returns
        -------
        ActiveTimeSeries
            Instantiated with seg2 file.

        """
        return cls.from_trace(trace=trace,
                              nstacks=int(trace.stats.seg2.STACK),
                              delay=float(trace.stats.seg2.DELAY))

    @classmethod
    def from_trace(cls, trace, nstacks=1, delay=0):
        """Create `ActiveTimeSeries` from a `Trace` object.

        This method is more general than
        :meth:`ActiveTimeSeries.from_trace_seg2`,
        as it does not attempt to extract any metadata from the `Trace`
        object.

        Parameters
        ----------
        trace : Trace
            Refer to
            `obspy documentation <https://github.com/obspy/obspy/wiki>`_
            for more information.
        nstacks : int, optional
            Number of stacks the time series represents, default is
            1, signifying a single unstacked time record.
        delay : float {<=0.}, optional
            Denotes the pre-event delay, default is zero, meaning no
            pre-event noise was recorded.

        Returns
        -------
        ActiveTimeSeries
            Initialized with information from `trace`.

        """
        return cls(amplitude=trace.data, dt=trace.stats.delta,
                   nstacks=nstacks, delay=delay)

    def trim(self, start_time, end_time):
        """Trim in the interval [`start_time`, `end_time`].

        For more information see :meth:`sigpropy.TimeSeries.trim`.

        Parameters
        ----------
        start_time : float
            New time-zero in seconds.
        end_time : float
            New end-time in seconds.

        Returns
        -------
        None
            Updates the attributes `nsamples` and `delay`.

        """
        super().trim(start_time, end_time)
        self._delay = start_time

    def zero_pad(self, df):
        """Append zeros to `amp` to achieve a desired frequency step.

        Note for exact results, `1/(df*dt)` must be an integer,
        otherwise a `df` close to the desired `df` will be returned.

        Parameters
        ----------
        df : float
            Desired frequency step in Hertz.

        Returns
        -------
        None
            Instead modifies attributes: `amp`, `nsamples`, `multiple`.

        Raises
        ------
        ValueError
            If `df` < 0 (i.e., non-positive).

        """
        df = float(df)
        logger.info(f"zero_pad(df={df})")
        self._multiple = 1
        if df <= 0:
            raise ValueError(f"df must be positive, currently {df}.")

        new_nsamples_float = 1/(df*self.dt)
        new_nsamples = int(round(new_nsamples_float))
        if new_nsamples_float != new_nsamples:
            msg = "  `1/(df*dt)` is not an integer results will be approximate."
            logger.warning(msg)

        # If new_nsamples > nsamples, pad zeros.
        if new_nsamples > self.nsamples:
            padding = new_nsamples - self.nsamples
        # If new_nsamples < nsamples, pad zeros to achieve a multiple
        # of new_nsamples (i.e.,  a fraction of df). After processing,
        # extract the results at the frequencies of interest.
        elif new_nsamples < self.nsamples:
            # If df_old is already an integer of df_new
            if self.nsamples % new_nsamples == 0:
                self._multiple = int(self.nsamples/new_nsamples)
                return
            # If df_old is not already an integer of df_new, pad existing series.
            else:
                padding = new_nsamples - (self.nsamples % new_nsamples)
                self._multiple = int((self.nsamples + padding) / new_nsamples)
        # If new_samples == nsamples, do nothing.
        else:
            return

        self._amp = np.concatenate((self._amp, np.zeros((1, padding))), axis=1)

    @staticmethod
    def crosscorr(a, b, correlate_kwargs=None, exclude=("nsamples")):
        """Cross correlation of two `ActiveTimeSeries` objects.

        Parameters
        ----------
        a : ActiveTimeSeries
            Base `ActiveTimeSeries` to which `b` is correlated.
        b: ActiveTimeSeries
            `ActiveTimeSeries` correlated to `a`.
        correlate_kwargs : dict, optional
            `dict` of keyword argument for the correlate function, see
            `scipy.signal.correlate <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html>`_
            for details.
        exclude : tuple, optional
            `tuple` of attributes to exclude in an is_similar
            comparison, default is `('nsamples')`.

        Returns
        -------
        ndarray
            Containing the cross correlation.

        """
        if not a._is_similar(b, exclude=exclude):
            raise ValueError("`a` and `b` must be similar.")

        if correlate_kwargs is None:
            correlate_kwargs = {}

        return correlate(a.amplitude, b.amplitude, **correlate_kwargs)

    @staticmethod
    def crosscorr_shift(a, b, exclude=None):
        """Shift `b` so that it is maximally corrlated with `a`.

        Parameters
        ----------
        a : ActiveTimeSeries
            `ActiveTimeSeries` to which `b` will be correlated. `a`
            should be similar to `b`.
        b : ActiveTimeSeries
            `ActiveTimeSeries` which will be shifted so that it is
            maximally correlated with `a`. `b` should be similar to `a`.
        exclude : tuple, optional
            `tuple` of attributes to exclude in an is_similar
            comparison, default is `('nsamples')`.

        Returns
        -------
        ndarray
            Which represents the stack of the correlated and padded `b`
            onto `a`.

        """
        corr = ActiveTimeSeries.crosscorr(a, b, exclude=exclude)
        maxcorr_location = np.argmax(corr)
        shifts = (maxcorr_location+1)-b.nsamples
        if shifts > 0:
            return (shifts, np.concatenate((np.zeros(shifts), b.amplitude[:-shifts])))
        elif shifts < 0:
            return (shifts, np.concatenate((b.amplitude[abs(shifts):], np.zeros(abs(shifts)))))
        else:
            return (shifts, np.array(b.amplitude))

    @classmethod
    def from_cross_stack(cls, a, b):
        """Create `ActiveTimeSeries` from cross-correlation.

        Cross-correlate `b` to `a` and shift `b` such that it is
        maximally correlated with `a`. Then stack the shifted version
        of `b` onto `a`.

        Parameters
        ----------
        a : ActiveTimeSeries
            `ActiveTimeSeries` to which `b` will be correlated and
            stacked. `a` should be similar to `b`.
        b : ActiveTimeSeries
            `ActiveTimeSeries` which will be correlated with and stacked
            onto `a`. `b` should be similar to `a`.

        Return
        ------
        ActiveTimeSeries
            Which represents the correlated and potentially zero-padded
            `b` stacked onto `a`.

        """
        obj = cls.from_activetimeseries(a)
        b_copy = cls.from_activetimeseries(b)
        _, shifted_amplitude = cls.crosscorr_shift(
            a, b_copy, exclude=("nsamples", "nstacks"))
        b_copy._amp = np.expand_dims(shifted_amplitude, axis=0)
        obj.stack_append(b_copy)
        return obj

    def _is_similar(self, other, exclude=None):
        """Check if `other` is similar to `self`."""
        if exclude is None:
            exclude = []

        if not isinstance(other, ActiveTimeSeries):
            return False

        for attr in ["dt", "nstacks", "delay", "nsamples"]:
            if attr in exclude:
                continue
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __eq__(self, other):
        """Check if `other` is equal to `self`."""
        if not self._is_similar(other):
            return False

        if not np.allclose(self.amplitude, other.amplitude):
            return False

        return True

    def __repr__(self):
        """Unambiguous representation of the object."""
        return f"ActiveTimeSeries(dt={self.dt}, amplitude={self.amplitude}, nstacks={self._nstacks}, delay={self._delay})"

    def __str__(self):
        """Human-readable representation of the object."""
        return f"ActiveTimeSeries with:\n\tdt={self.dt}\n\tnsamples={self.nsamples}\n\tnstacks={self._nstacks}\n\tdelay={self._delay}"
