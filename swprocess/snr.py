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

"""SignaltoNoiseRatio class definition."""

import numpy as np

from .register import WavefieldTransformRegistry


class SignaltoNoiseRatio():

    def __init__(self, frequencies, snr):
        self.frequencies = np.array(frequencies)
        self.snr = np.array(snr)

    @classmethod
    def from_array1ds(cls, signal, noise, fmin=3, fmax=75,
                      pad_snr=False, df_snr=None):
        # Pad
        if pad_snr:
            noise.zero_pad(df_snr)
            signal.zero_pad(df_snr)

        # Check signal and noise windows are indeed the same length.
        if noise[0].nsamples != signal[0].nsamples:
            msg = "Signal and noise windows must be of equal length, or set 'pad_snr' to 'True'."
            raise IndexError(msg)

        # Frequency vector
        sensor = noise[0]
        frqs = np.arange(sensor.nsamples) * sensor.df
        Empty = WavefieldTransformRegistry.create_class("empty")
        keep_ids = Empty._frequency_keep_ids(frqs, fmin, fmax, sensor.multiple)
        frequencies = frqs[keep_ids]

        # Compute SNR
        s_tseries = signal.timeseriesmatrix()
        snr = np.mean(np.abs(np.fft.fft(s_tseries)[:, keep_ids]), axis=0)
        n_tseries = noise.timeseriesmatrix()
        snr /= np.mean(np.abs(np.fft.fft(n_tseries)[:, keep_ids]), axis=0)

        return cls(frequencies, snr)
