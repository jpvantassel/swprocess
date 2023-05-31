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

"""Tests for SignaltoNoiseRatio class."""

import numpy as np

import swprocess
from testtools import unittest, TestCase


class Test_SignaltoNoiseRatio(TestCase):
    def test_init(self):
        frequencies = [1, 3, 5]
        snr_values = [1, 5, 10]
        snr = swprocess.snr.SignaltoNoiseRatio(frequencies, snr_values)

        self.assertArrayEqual(np.array(frequencies), snr.frequencies)
        self.assertArrayEqual(np.array(snr_values), snr.snr)

    def test_from_array1ds(self):
        # Define source.
        source = swprocess.Source(x=-5, y=0, z=0)

        # Create signal array.
        s_sensors = []
        for n in range(5):
            amp = 5*np.random.random(100)
            s_sensors.append(swprocess.Sensor1C(amp, dt=0.01, x=2*n, y=0, z=0))
        signal = swprocess.Array1D(s_sensors, source)

        # Create noise array.
        n_sensors = []
        for n in range(5):
            amp = np.random.random(100)
            n_sensors.append(swprocess.Sensor1C(amp, dt=0.01, x=2*n, y=0, z=0))
        noise = swprocess.Array1D(n_sensors, source)

        # Calculate SNR
        fmin, fmax = 5, 50
        snr = swprocess.snr.SignaltoNoiseRatio.from_array1ds(
            signal, noise, fmin=fmin, fmax=fmax)
        expected = np.arange(fmin, fmax+signal[0].df, signal[0].df)
        returned = snr.frequencies
        self.assertArrayAlmostEqual(expected, returned)

        # Fail due to unequal samples
        noise.trim(0, 0.25)
        self.assertRaises(IndexError,
                          swprocess.snr.SignaltoNoiseRatio.from_array1ds,
                          signal, noise, fmin=fmin, fmax=fmax)

        # Pass after padding
        snr = swprocess.snr.SignaltoNoiseRatio.from_array1ds(
            signal, noise, fmin=fmin, fmax=fmax, pad_snr=True, df_snr=signal[0].df)
        expected = np.arange(fmin, fmax+signal[0].df, signal[0].df)
        returned = snr.frequencies
        self.assertArrayAlmostEqual(expected, returned)



if __name__ == "__main__":
    unittest.main()
