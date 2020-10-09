"""Signal-to-Noise class definition."""

import numpy as np

from .register import WavefieldTransformRegistry


class SignaltoNoiseRatio():

    def __init__(self, frequencies, snr):
        self.frequencies = np.array(frequencies)
        self.snr = np.array(snr)

    @classmethod
    def from_array1ds(cls, noise, signal, fmin=3, fmax=75, pad_snr=False, df_snr=None):
        # Pad
        if pad_snr:
            noise.zero_pad(snr["pad"]["df"])
            signal.zero_pad(snr["pad"]["df"])

        # Check signal and noise windows are indeed the same length.
        if noise[0].nsamples != signal[0].nsamples:
            msg = f"Signal and noise windows must be of equal length, or set 'pad_snr' to 'True'."
            raise IndexError(msg)

        # Frequency vector
        sensor = noise[0]
        frqs = np.arange(sensor.nsamples) * sensor.df
        Empty = WavefieldTransformRegistry.create_class("empty")
        keep_ids = Empty._frequency_keep_ids(frqs, fmin, fmax, sensor.multiple)
        snr_frequencies = frqs[keep_ids]

        # Compute SNR
        snr = np.mean(np.abs(np.fft.fft(signal.timeseriesmatrix())[:, keep_ids]), axis=0)
        snr /= np.mean(np.abs(np.fft.fft(noise.timeseriesmatrix())[:, keep_ids]), axis=0)

        return cls(snr_frequencies, snr)
