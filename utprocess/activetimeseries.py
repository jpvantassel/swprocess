"""This file contains the class ActiveTimeSeries."""

import numpy as np
import scipy.signal as signal
from sigpropy import TimeSeries
import warnings
import logging
logger = logging.getLogger(__name__)


class ActiveTimeSeries(TimeSeries):
    """A class for working with active-source TimeSeries.

    Attributes
    ----------
    amp : ndarray
        Recording's amplitude, one per sample.
    dt : float
        Time step between samples in seconds.

    """
    # @staticmethod
    # def check_input(name, values):
    #     """Check 'values' is `ndarray`, `list`, or `tuple`.
    #     If it is a list or tuple convert it to a np.ndarray.
    #     Ensure 'values' is one-dimensional np.ndarray.
    #     'name' is only used to raise easily understood exceptions.
    #     """
    #     if type(values) not in [np.ndarray, list, tuple]:
    #         raise TypeError("{} must be of type np.array, list, or tuple not {}".format(
    #             name, type(values)))

    #     if isinstance(values, (list, tuple)):
    #         values = np.array(values)

    #     if len(values.shape) > 1:
    #         raise TypeError("{} must be 1-dimensional, not {}-dimensional".format(
    #             name, values.shape))
    #     return values

    def __init__(self, amplitude, dt, n_stacks=1, delay=0):
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

        self.n_stacks = n_stacks
        self._nstacks = n_stacks

        if delay > 0:
            raise ValueEroror()
        # assert(delay <= 0)
        self.delay = delay

        # TODO (jpv): Dont think this is needed, but wavefieldtransform is broken.
        self.multiple = 1
        
    @property
    def time(self):
        """Return time vector for `ActiveTimeSeries` object."""
        # if self.n_windows == 1:
        return np.arange(0, self.n_samples*self.dt, self.dt) + self.delay
        # else:
        #     samples_per_window = (self.n_samples//self.n_windows)+1
        #     time = np.zeros((self.n_windows, samples_per_window))
        #     for cwindow in range(self.n_windows):
        #         start_time = cwindow*(samples_per_window-1)*self.dt
        #         stop_time = start_time + (samples_per_window-1)*self.dt
        #         time[cwindow] = np.linspace(
        #             start_time, stop_time, samples_per_window)
        #     return time

    def trim(self, start_time, end_time):
        """Trim time series in the interval [`start_time`, `end_time`].

        Parameters
        ----------
        start_time : float
            New time zero in seconds.
        end_time : float
            New end time in seconds.

        Returns
        -------
        None
            Updates the attributes `n_samples` and `delay`.

        Raises
        ------
        IndexError
            If the `start_time` and `end_time` is illogical.
            For example, `start_time` is before the start of the
            `delay` or after `end_time`, or the `end_time` is
            after the end of the record.

        """
        super().trim(start_time, end_time)

        if start_time < self.delay:
            raise ValueError(f"`start_time` must be >= `delay`={self.delay}.")
        else:
            self.delay = start_time

    def zero_pad(self, df=0.2):
        """Append zeros to `amp` to achieve a desired frequency step.

        Parameters
        ----------
        df : float
            Desired frequency step in Hertz.

        Returns
        -------
        None
            Instead modifies attributes: `amp`, `n_samples`, `multiple`.

        Raises
        ------
        ValueError
            If `df` < 0 (i.e., non-positive).
        """
        raise NotImplementedError
        if isinstance(df, (float, int)):
            if df <= 0:
                raise ValueError(f"df must be positive.")
        else:
            raise TypeError(f"df must be `float` or `int`, not {type(df)}.")

        nreq = int(np.round(1/(df*self.dt)))

        logging.info(f"nreq = {nreq}")

        # If nreq > n_samples, padd zeros to achieve n_samples = nreq
        if nreq > self.n_samples:
            self.amp = np.concatenate((self.amp,
                                       np.zeros(nreq - self.n_samples)))
            self.n_samples = nreq

        # If nreq <= n_samples, padd zeros to achieve a fraction of df (e.g., df/2)
        # After processing, extract the results at the frequencies of interest
        else:
            # Multiples of df and nreq
            multiples = np.array([1, 2, 4, 8, 16, 32], int)
            trial_n = nreq*multiples

            logging.debug(f"trial_n = {trial_n}")

            # Find first trial_n > nreq
            self.multiple = multiples[np.where(self.n_samples < trial_n)[0][0]]
            logging.debug(f"multiple = {self.multiple}")

            self.amp = np.concatenate((self.amp,
                                       np.zeros(nreq*self.multiple - self.n_samples)))
            self.n_samples = nreq*self.multiple

            logging.debug(f"n_samples = {self.n_samples}")

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
                              n_stacks=int(trace.stats.seg2.STACK),
                              delay=float(trace.stats.seg2.DELAY))

    def stack_append(self, amplitude, dt, n_stacks=1):
        """Stack (i.e., average) a new time series into the current oen.

        Parameters
        ----------
        amplitude : array-like
            New amplitiude to be stacked (i.e., averaged with the
            current time series).
        dt : float
            Time step of the new time series in seconds. Only used for
            comparison with the current time series.
        n_stacks : int
            Number of stacks used to produce `amplitude`, default is 1.

        Returns
        -------
        None
            Instead updates the attribute `amp`.

        Raises:
            TypeError: If amplitude is not an np.array or list.

            IndexError: If the length of amplitude does not match the
                length of the current time series.
        """
        amplitude = ActiveTimeSeries._check_input("amplitude", amplitude)

        if len(amplitude) != len(self.amp):
            raise IndexError("Length of two waveforms must be the same.")

        if type(amplitude) is list:
            amplitude = np.array(amplitude)

        self.amp = (self.amp*self._nstacks + amplitude*n_stacks) / \
            (self._nstacks+n_stacks)
        self._nstacks += n_stacks

    @classmethod
    def from_trace(cls, trace, n_stacks=1, delay=0):
        """Initialize a `TimeSeries` object from a trace object.

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
                   n_stacks=n_stacks, delay=delay)

    @staticmethod
    def crosscorr(timeseries_a, timeseries_b):
        """Return cross correlation of two timeseries objects."""
        # TODO (jpv): Create method for comparing tseriesa == tseriesb
        return signal.correlate(timeseries_a.amp, timeseries_b.amp)

    @staticmethod
    def crosscorr_shift(timeseries_a, timeseries_b):
        """Return shifted timeseries_b so that it is maximally
        corrlated with tiemseries_a."""
        corr = ActiveTimeSeries.crosscorr(timeseries_a, timeseries_b)
        # print(corr)
        # print(np.where(abs(corr) == max(abs(corr))))
        maxcorr_location = np.where(corr == max(corr))[0][0]
        # print(maxcorr_location)
        shifts = (maxcorr_location+1)-timeseries_b.n_samples
        # print(shifts)
        # print(type(shifts))
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
        obj.stack_append(shifted_amp, timeseries_b.dt)
        return obj

    # TODO (jpv): Implement equal comparison.
        # def __eq__(self, other):
        # my = self.amp
        # ur = other.amp

        # if my.size != ur.size:
        #     return False

        # for my_val, ur_val in zip(my, ur):
        #     if my_val != ur_val:
        #         return False

        # for attr in ["dt", "n_stacks", "delay"]:
        #     if getattr(self, attr) != getattr(other, attr):
        #         return False
        # return True

    def __repr__(self):
        return f"ActiveTimeSeries(dt={self.dt}, amplitude={str(self.amp[0:3])[:-1]} ... {str(self.amp[-3:])[1:]})"
