"""This file contains the class TimeSeries for creating and working with
time series objects.
"""

import numpy as np
import scipy.signal as signal
import logging
logger = logging.getLogger(__name__)


class TimeSeries():
    """A class for editing and manipulating time series.

    Attributes:
        amp: np.array denoting the recordings amplitude.

        dt: Float denoting the time step between samples in seconds.
    """
    @staticmethod
    def check_input(name, values):
        """Check 'values' is np.ndarray, list, or tuple. 
        If it is a list or tuple convert it to a np.ndarray. 
        Ensure 'values' is one-dimensional np.ndarray.
        'name' is only used to raise easily understood exceptions.
        """
        if type(values) not in [np.ndarray, list, tuple]:
            raise TypeError("{} must be of type np.array, list, or tuple not {}".format(
                name, type(values)))

        if isinstance(values, (list, tuple)):
            values = np.array(values)

        if len(values.shape) > 1:
            raise TypeError("{} must be 1-dimensional, not {}-dimensional".format(
                name, values.shape))
        return values

    def __init__(self, amplitude, dt, nstacks=1, delay=0):
        """Initialize a TimeSeries object.

        Args:
            amplitude: Any type that can be transformed into an np.array
                denoting the records amplitude with time. The first 
                value is associated with time=0 seconds and the last is 
                associate with (len(amplitude)-1)*dt seconds.

            dt: Float denoting the time step between samples in seconds.

            nstacks: Number of stacks used to produce the amplitude.
                The default value is 1.

            delay: Float indicating the delay to the start of the record
                in seconds.

        Returns:
            Intialized TimeSeries object.

        Exceptions:
            This method raises no exceptions.
        """
        amplitude = TimeSeries.check_input("amplitude", amplitude)

        if type(amplitude) == np.ndarray:
            self.amp = amplitude
        else:
            self.amp = np.array(amplitude)
        self.nsamples = len(self.amp)
        self.dt = dt
        self.fs = 1/self.dt
        self.fnyq = 0.5*self.fs
        self.df = self.fs/self.nsamples
        self._nstack = 1
        assert(delay <= 0)
        self.delay = delay
        self.multiple = 1

        logging.info(f"Initialize a TimeSeries object.")
        logging.info(f"\tdt = {dt}")
        logging.info(f"\tfs = {self.fs}")
        logging.info(f"\tdelay = {delay}")
        logging.info(f"\tnsamples = {self.nsamples}")

    @property
    def time(self):
        return np.arange(0, self.nsamples)*self.dt

    def trim(self, start_time, end_time):
        """Trim excess off of TimeSeries object in the half open
        interval [start_time, end_time).

        Args: 
            start_time: Float denoting the desired start time in seconds 
                from the point the acquisition was triggered.

            end_time: Float denoting the desired end time in seconds 
                from the point the acquisition was triggered. Note that 
                the interval is half-open.

        Returns:
            Returns no value, but may update the state of attributes:
                nsamples
                delay
                df

        Raises:
            IndexError if the start_time and end_time is illogical.
                For example, start_time is before the start of the
                pretrigger delay or after end_time, or the end_time is
                before the start_time or after the end of the record.
        """
        current_time = np.arange(self.delay,
                                 self.nsamples * self.dt + self.delay,
                                 self.dt)
        start = min(current_time)
        end = max(current_time)
        if start_time < start or start_time > end_time:
            logger.debug(
                f"This must be true: {start} < {start_time} < {end_time}")
            raise IndexError("Illogical start_time, see doctring")
        if end_time > end or end_time < start_time:
            logger.debug(
                f"This must be true: {start_time} < {end_time} < {end}")
            raise IndexError("Illogical end_time, see doctring")

        logger.info(f"start = {start}, moving to start_time = {start_time}")
        logger.info(f"start = {end}, moving to end_time = {end_time}")

        start_index = np.argmin(np.absolute(current_time - start_time))
        end_index = np.argmin(np.absolute(current_time - end_time))

        logger.debug(f"start_index = {start_index}")
        logger.debug(f"start_index = {end_index}")

        self.amp = self.amp[start_index:end_index]
        self.nsamples = len(self.amp)
        self.df = self.fs/self.nsamples
        self.delay = 0 if start_time >= 0 else start_time

        logger.info(f"nsamples = {self.nsamples}")
        logger.info(f"df = {self.df}")
        logger.info(f"delay = {self.delay}")

    def zero_pad(self, df=0.2):
        """Append zeros to the end of the TimeSeries object to achieve a
        desired frequency step.

        Args:
            df: Positive float that denotes the desired frequency step
                in Hertz.

        Returns:
            Returns no value, instead modifies the TimeSeries object
            attributes:
                amp
                nsamples
                multiple

        Raises:
            Raises a value error if df is not a positive number.
        """
        if isinstance(df, (float, int)):
            if df <= 0:
                raise ValueError(f"df must be positive.")
        else:
            raise TypeError(f"df must be `float` or `int`, not {type(df)}.")

        nreq = int(np.round(1/(df*self.dt)))

        logging.info(f"nreq = {nreq}")

        # If nreq > nsamples, padd zeros to achieve nsamples = nreq
        if nreq > self.nsamples:
            self.amp = np.concatenate((self.amp,
                                       np.zeros(nreq - self.nsamples)))
            self.nsamples = nreq

        # If nreq <= nsamples, padd zeros to achieve a fraction of df (e.g., df/2)
        # After processing, extract the results at the frequencies of interest
        else:
            # Multiples of df and nreq
            multiples = np.array([1, 2, 4, 8, 16, 32], int)
            trial_n = nreq*multiples

            logging.debug(f"trial_n = {trial_n}")

            # Find first trial_n > nreq
            self.multiple = multiples[np.where(self.nsamples < trial_n)[0][0]]
            logging.debug(f"multiple = {self.multiple}")

            self.amp = np.concatenate((self.amp,
                                       np.zeros(nreq*self.multiple - self.nsamples)))
            self.nsamples = nreq*self.multiple

            logging.debug(f"nsamples = {self.nsamples}")

    @classmethod
    def from_trace_seg2(cls, trace):
        """Initialize a TimeSeries object from a seg2 trace object.

        This method is similar to from_trace except that it extracts
        additional information from the trace header. So only use this 
        method if you have a seg2 file and the header information is 
        correct.

        Args:
            trace: Trace object from a correctly written seg2 file.

        Returns:
            Initialized TimeSeries object.

        Raises:
            This method raises no exceptions.
        """
        return cls(amplitude=trace.data,
                   dt=trace.stats.delta,
                   nstacks=int(trace.stats.seg2.STACK),
                   delay=float(trace.stats.seg2.DELAY))

    @classmethod
    def from_trace(cls, trace, nstacks=1, delay=0):
        """Initialize a TimeSeries object from a trace object.

        This method is a more general method than from_trace_seg2, which
        does not attempt to extract anything from the trace object 
        except for its data and sampling step.

        Args:
            trace: Trace object.

            nstacks: Integer representing the number of stacks. Default 
                value is one (i.e., only a single recording was made).

            delay: Number that is less than or equal to zero, denoting
                the pre-trigger delay. Default value is zero (i.e., no
                pre-trigger delay was recorded).

        Returns:
            Initialized TimeSeries object.

        Raises:
            This method raises no exceptions.
        """
        return cls(amplitude=trace.data,
                   dt=trace.stats.delta,
                   nstacks=nstacks,
                   delay=delay)

    def stack_append(self, amplitude, dt, nstacks=1):
        """Stack (i.e., average) a new time series to the current time
        series.

        Args:
            amplitude: This new amplitiude will be stacked (i.e.,
                averaged with the current time series).

            dt: Time step of the new time series. Only used for
                comparison with the current time series.

            nstacks: Number of stacks used to produce the
                amplitude. The default value is 1.

        Returns:
            This method returns no value, but rather updates the state
            of the attribute amp.

        Raises:
            TypeError: If amplitude is not an np.array or list.

            IndexError: If the length of amplitude does not match the
                length of the current time series.
        """
        amplitude = TimeSeries.check_input("amplitude", amplitude)

        if len(amplitude) != len(self.amp):
            raise IndexError("Length of two waveforms must be the same.")

        if type(amplitude) is list:
            amplitude = np.array(amplitude)

        self.amp = (self.amp*self._nstack + amplitude*nstacks) / \
            (self._nstack+nstacks)
        self._nstack += nstacks

    @staticmethod
    def crosscorr(timeseries_a, timeseries_b):
        """Return cross correlation of two timeseries objects."""
        # TODO (jpv): Create method for comparing tseriesa == tseriesb
        return signal.correlate(timeseries_a.amp, timeseries_b.amp)

    @staticmethod
    def crosscorr_shift(timeseries_a, timeseries_b):
        """Return shifted timeseries_b so that it is maximally
        corrlated with tiemseries_a."""
        corr = TimeSeries.crosscorr(timeseries_a, timeseries_b)
        # print(corr)
        # print(np.where(abs(corr) == max(abs(corr))))
        maxcorr_location = np.where(corr == max(corr))[0][0]
        # print(maxcorr_location)
        shifts = (maxcorr_location+1)-timeseries_b.nsamples
        print(shifts)
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

    def __repr__(self):
        # """Valid python expression to reproduce the object"""
        return "TimeSeries(dt, amplitude)"

    def __str__(self):
        # """Informal representation of the object."""
        return f"TimerSeries object\namp = {self.amp}\ndt = {self.dt}"
