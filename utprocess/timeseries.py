"""
This file contains the class TimeSeries for working with time series
objects.
"""

import numpy as np


class TimeSeries():
    """A class for editing and manipulating time series.

    Attributes:

        dt: Float denoting the time step between samples in seconds.

        amp: np.array denoting the recordings amplitude.
    """

    def __init__(self, dt, amplitude):
        """Initialize a TimeSeries object.

        Args:
            dt: Float denoting the time step between samples in seconds.

            amplitude: Any type that can be transformed into an np.array
             denoting the recordings amplitude with time, where the
             first value is associated with time=0 seconds and the last
             is associate with (len(amplitude)-1)*dt seconds.

        Returns:
            Intialized TimeSeries object.

        Exceptions:
            This method raises no exceptions.
        """
        self.amp = np.array(amplitude)
        self.dt = dt

    def fft(self, normalize=False):
        """Compute the Fast Fourier Transform for TimeSeries.

        Args:
            normalize: Boolean to control whether the amplitude of the 
                fft is normalized.

        Returns:
            Returns a tuple of the form (freq, pos_fft) where freq is a 
            numpy array of frequencies and pos_fft is a numpy array of
            magniutde of the fft over its postive frequencies, which may
            or may not be normalized.

        Raises:
            This method raises no exceptions.
        """
        # TODO (jpv): Build fft class.

        # Ensure an even number of samples.
        if (len(self.amp) % 2) != 0:
            amp = np.append(self.dt, 0)
        else:
            amp = self.amp

        complete_fft = np.fft.fft(amp)
        n = len(complete_fft)

        pos_fft = np.abs(complete_fft[0:int(n/2)+1])

        freq = np.linspace(0, 1/(2*self.dt), int(n/2)+1)

        if normalize:
            return (freq, pos_fft/np.max(pos_fft))
        else:
            return (freq, pos_fft)

    def __repr__(self):
        """ Valid python expression to reproduce the object"""
        return "TimeSeries(dt, amplitude)"

    def __str__(self):
        """Informal representation of the object."""
        pass