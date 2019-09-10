"""This file contains the class TimeSeries for creating and working with
time series objects.
"""

import numpy as np
import obspy


class TimeSeries():
    """A class for editing and manipulating time series.

    Attributes:
        amp: np.array denoting the recordings amplitude.

        dt: Float denoting the time step between samples in seconds.
    """
    @staticmethod
    def __check_input(name, values):
        if type(values) not in [np.ndarray, list, tuple]:
            raise TypeError("{} must be of type np.array, list, or tuple not {}".format(
                name, type(values)))

        if isinstance(values, (list, tuple)):
            values = np.array(values)

        if len(values.shape) > 1:
            raise TypeError("{} must be 1-dimensional, not {}-dimensional".format(
                name, values.shape))

    def __init__(self, amplitude, dt, nstacks=1):
        """Initialize a TimeSeries object.

        Args:
            amplitude: Any type that can be transformed into an np.array
                denoting the records amplitude with time. The first value
                is associated with time=0 seconds and the last is associate
                with (len(amplitude)-1)*dt seconds.

            dt: Float denoting the time step between samples in seconds.

            nstacks: Number of stacks used to produce the amplitude.
                The default value is 1.

        Returns:
            Intialized TimeSeries object.

        Exceptions:
            This method raises no exceptions.
        """
        TimeSeries.__check_input("amplitude", amplitude)

        if type(amplitude) == np.ndarray:
            self.amp = amplitude
        else:
            self.amp = np.array(amplitude)
        self.dt = dt
        self.fs = 1/self.dt
        self.fnyq = 0.5*self.fs
        self._nstack = 1
        self.nsamples = len(self.amp)

    @classmethod
    def from_trace(cls, trace):
        return cls(amplitude=trace.data,
                   dt=trace.stats.delta,
                   nstacks=int(trace.stats.seg2.STACK))

    @classmethod
    def from_miniseed(cls, filename):
        pass

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

            IndexError: If length of the amplitude does not match
                the length of the current time series.
        """
        TimeSeries.__check_input("amplitude", amplitude)

        if len(amplitude) != len(self.amp):
            raise IndexError("Length of two waveforms must be the same.")

        if type(amplitude) is list:
            amplitude = np.array(amplitude)

        self.amp = (self.amp*self._nstack + amplitude *
                    nstacks)/(self._nstack+nstacks)
        # print(self.amp)
        self._nstack += nstacks
        # print(self._nstack)

    def __repr__(self):
        # """Valid python expression to reproduce the object"""
        return "TimeSeries(dt, amplitude)"

    def __str__(self):
        # """Informal representation of the object."""
        return f"TimerSeries object\namp = {self.amp}\ndt = {self.dt}"
