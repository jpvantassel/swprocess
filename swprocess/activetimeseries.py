"""ActiveTimeSeries class definition."""

import warnings
import logging

import numpy as np
import scipy.signal as signal
from sigpropy import TimeSeries

logger = logging.getLogger(__name__)


class ActiveTimeSeries(TimeSeries):
    """A class for working with active-source `TimeSeries`.

    Attributes
    ----------
    amp : ndarray
        Recording's amplitude, one per sample.
    dt : float
        Time step between samples in seconds.

    """

    @staticmethod
    def _check_input(nstacks, delay):
        """Check inputs have correct type and values.

        Specificially:
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
        """Initialize a `TimeSeries` object.

        Parameters
        ----------
        amplitude : array-like
            Recording's amplitude, one per sample. The first 
            value is associated with time=0 seconds and the last is 
            associate with (len(amplitude)-1)*dt seconds.
        dt : float
            Time step between samples in seconds.
        n_stacks : int, optional
            Number of stacks used to produce `amplitude`, default is 1.
        delay : float
            Delay to the start of the record in seconds.

        Returns
        -------
        TimeSeries
            Intialized `TimeSeries` object.

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
    def df(self):
        return super().df * self._multiple

    @property
    def multiple(self):
        return self._multiple

    @property
    def time(self):
        """Time vector for `ActiveTimeSeries`."""
        return super().time + self._delay

    def stack_append(self, timeseries):
        """Stack (i.e., average) a new time series into the current one.

        Parameters
        ----------
        timeseries : ActiveTimeSeries
            Active time series to be stacked only the current object.

        Returns
        -------
        None
            Updates the attribute `amp` and `nstacks`.

        Raises
        ------
        ValueError
            If the provided `timeseries` is not an `ActiveTimeSeries` or
            it cannot be stacked to the current object (i.e., time
            series are dissimilar). 

        """
        if not self._is_similar(timeseries, exclude=["_nstacks"]):
            msg = f"The provided `timeseries` object is incompatable, and cannot be stacked."
            raise ValueError(msg)

        namp = timeseries.amp
        nstk = timeseries._nstacks
        self.amp = (self.amp*self._nstacks + namp*nstk)/(self._nstacks + nstk)
        self._nstacks += nstk

    @classmethod
    def from_activetimeseries(cls, activetimeseries):
        return cls(activetimeseries.amplitude,
                   activetimeseries.dt,
                   nstacks=activetimeseries.nstacks,
                   delay=activetimeseries.delay)

    @classmethod
    def from_trace_seg2(cls, trace):
        """Initialize a `TimeSeries` object from a SEG2 `Trace` object.

        This method is similar to meth:`from_trace` except that it
        extracts additional information from the `Trace` header. So only
        use this method if you have a SEG2 file and the header
        information is correct.

        Parameters
        ----------
        trace: Trace 
            `Trace` object from a correctly written seg2 file.

        Returns
        -------
        ActiveTimeSeries
            Instantiated with seg2 file information.
        """
        return cls.from_trace(trace=trace,
                              nstacks=int(trace.stats.seg2.STACK),
                              delay=float(trace.stats.seg2.DELAY))

    @classmethod
    def from_trace(cls, trace, nstacks=1, delay=0):
        """Initialize an `ActiveTimeSeries` object from a trace object.

        This method is more general method than `from_trace_seg2`, 
        as it does not attempt to extract any metadata from the `Trace` 
        object.

        Parameters
        ----------
        trace : Trace
            Refer to
            `obspy documentation <https://github.com/obspy/obspy/wiki>`_
            for more information
        n_stacks : int, optional
            Number of stacks the time series represents, (default is
            1, signifying a single unstacked time record).
        delay : float {<=0.}, optional
            Denotes the pre-event delay, (default is zero, 
            meaning no pre-event noise was recorded).

        Returns
        -------
        TimeSeries
            Initialized with information from `trace`.

        """
        return cls(amplitude=trace.data, dt=trace.stats.delta,
                   nstacks=nstacks, delay=delay)

    def trim(self, start_time, end_time):
        """Trim `ActiveTimeSeries` in the interval [`start_time`, `end_time`].

        Parameters
        ----------
        start_time : float
            New time zero in seconds.
        end_time : float
            New end time in seconds.

        Returns
        -------
        None
            Updates the attributes `nsamples` and `delay`.

        Raises
        ------
        IndexError
            If the `start_time` and `end_time` is illogical.
            For example, `start_time` is before the start of the
            `delay` or after `end_time`, or the `end_time` is
            after the end of the record.

        """
        super().trim(start_time, end_time)
        self._delay = start_time

    def zero_pad(self, df):
        """Append zeros to `amp` to achieve a desired frequency step.

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
        if df <= 0:
            raise ValueError(f"df must be positive, currently {df}.")

        new_nsamples = int(round(1/(df*self.dt)))

        logging.info(f"df={self.df} --> df={df}")
        logging.debug(f"  new_nsamples = {new_nsamples}")

        # If new_nsamples > nsamples, padd zeros.
        if new_nsamples > self.nsamples:
            padding = new_nsamples - self.nsamples
            self._multiple = 1

        # If new_nsamples <= nsamples, padd zeros to achieve a multiple
        # of new_nsamples (i.e.,  a fraction of df). After processing,
        # extract the results at the frequencies of interest.
        else:
            multiple = 1
            for _ in range(10):
                trial_nsamples = new_nsamples*multiple
                if trial_nsamples >= self.nsamples:
                    self._multiple = multiple
                    break
                multiple *= 2
            else:
                msg = f"Could not find an acceptable multiple, after 10 attempts."
                raise ValueError(msg)

            padding = new_nsamples*self.multiple - self.nsamples
            logging.debug(f"  trial_nsamples = {trial_nsamples}")
            logging.debug(f"  multiple = {self.multiple}")

        self.amp = np.concatenate((self.amp, np.zeros(padding)))
        logging.info(f"  nsamples = {self.nsamples}")

    @staticmethod
    def crosscorr(timeseries_a, timeseries_b):
        """Return cross correlation of two timeseries objects."""
        if not timeseries_a._is_similar(timeseries_b, exclude=["nsamples"]):
            msg = "`timeseries_a` and `timeseries_b` must be similar."
            raise ValueError(msg)
        return signal.correlate(timeseries_a.amp, timeseries_b.amp)

    @staticmethod
    def crosscorr_shift(timeseries_a, timeseries_b):
        """Return shifted timeseries_b so that it is maximally
        corrlated with timeseries_a."""
        corr = ActiveTimeSeries.crosscorr(timeseries_a, timeseries_b)
        maxcorr_location = np.where(corr == max(corr))[0][0]
        shifts = (maxcorr_location+1)-timeseries_b.nsamples
        if shifts > 0:
            return np.concatenate((np.zeros(shifts), timeseries_b.amp[:-shifts]))
        elif shifts < 0:
            return np.concatenate((timeseries_b.amp[abs(shifts):], np.zeros(abs(shifts))))
        else:
            return timeseries_b.amp

    @classmethod
    def from_cross_stack(cls, timeseries_a, timeseries_b):
        """Return a single trace that is the result of stacking two 
        time series, which are aligned using cross-correlation."""
        obj = cls(timeseries_a.amp, timeseries_a.dt)
        shifted_amp = cls.crosscorr_shift(timeseries_a, timeseries_b)
        timeseries_b.amp = shifted_amp
        obj.stack_append(timeseries_b)
        return obj

    def _is_similar(self, other, exclude=[]):
        """Check if `other` is similar to `self` though not equal."""

        if not isinstance(other, ActiveTimeSeries):
            return False

        for attr in ["dt", "_nstacks", "_delay", "nsamples"]:
            if attr in exclude:
                continue
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __eq__(self, other):
        """Check if `other` is equal to the `ActiveTimeSeries`."""

        if not self._is_similar(other):
            return False

        for my_val, ur_val in zip(self.amp, other.amp):
            if my_val != ur_val:
                return False

        return True

    def __repr__(self):
        """Unambiguous representation of an `ActiveTimeSeries` object."""
        return f"ActiveTimeSeries(dt={self.dt}, amplitude={self.amp}, nstacks={self._nstacks}, delay={self._delay})"

    def __str__(self):
        """Human-readable representation of an `ActiveTimeSeries` object."""
        return f"ActiveTimeSeries with:\n\tdt={self.dt}\n\tnsamples={self.nsamples}\n\tnstacks={self._nstacks}\n\tdelay={self._delay}"
