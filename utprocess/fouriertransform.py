"""This file contains the class FourierTransform for creating and 
working with fourier transform objects."""

import numpy as np
# from timeseries import TimeSeries

class FourierTransform():
    """A class for editing and manipulating fourier transforms.

    Attributes:
        freq: np.array denoting the frequency vector of the transform in
            Hertz.

        amp: np.array denoting the transforms amplitude.
    """

    def __init__(self, ):
        """Compute the Fast Fourier Transform from a TimeSeries object.

        Args:
            timeseries: TimeSeries object to be transformed using the 
                fast fourier transform.

        Returns:
            An initialized FourierTransform object.

        Raises:
            This method raises no exceptions.
        """
        pass

    @staticmethod
    def fft(amplitude, dt):
        """Compute the fast-fourier transform of time series.

        Args:
            amplitude: np.array of amplitudes (one per time step).

            dt: Float indicating the time step in seconds.

        Returns:
            Tuple of the form (freq, fft) where:
                freq is the frequency vector in a np.array in units of 
                    Hertz.
                fft is the complex amplitude in a np.array with units 
                    of the input ampltiude/second 
                    TODO (jpv): Check this.

        Raises:
            This method raises no exceptions.
        """
        # # Ensure an even number of samples.
        # if (len(amplitude) % 2) != 0:
        #     amplitude = np.append(dt, 0)

        # complete_fft = np.fft.fft(amp)
        # n = len(complete_fft)
        # fft = complete_fft[0:int(n/2)+1]
        # freq = np.linspace(0, 1/(2*self.dt), int(n/2)+1)
        # return (freq, fft)
        pass

    @property
    def mag(self):
        """Magnitude of real and complex components of fft."""
        pass

    @property
    def phase(self):
        """Phase of real and complex components of fft in Radians."""
        pass

    @property
    def cmplx(self):
        """Amplitude of the complex component of the fft."""
        pass

    @property
    def real(self):
        """Amplitude of the real component of the fft."""
        pass

    def __repr__(self):
        return "FourierTransform(amplitude, dt)"
