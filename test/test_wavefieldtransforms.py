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

"""Tests for class wavefieldtransforms module."""

import logging
import warnings

import numpy as np

from swprocess.wavefieldtransforms import EmptyWavefieldTransform
import swprocess
import matplotlib.pyplot as plt
from testtools import TestCase, unittest

logger = logging.getLogger("swprocess.wavefieldtransforms")
logger.setLevel(logging.WARNING)


class Test_AbstractWavefieldTransform(TestCase):

    def test_create_vector(self):
        pmin, pmax, pn = 100, 400, 100

        # linspace
        approaches = ["linear", "lin"]
        expected = np.linspace(pmin, pmax, pn)
        for pspace in approaches:
            returned = EmptyWavefieldTransform._create_vector(
                pmin, pmax, pn, pspace)
            self.assertArrayEqual(expected, returned)

        # geomspace
        approaches = ["log", "logarithmic"]
        expected = np.geomspace(pmin, pmax, pn)
        for pspace in approaches:
            returned = EmptyWavefieldTransform._create_vector(
                pmin, pmax, pn, pspace)
            self.assertArrayEqual(expected, returned)

    def test_from_array(self):
        class Example(swprocess.wavefieldtransforms.AbstractWavefieldTransform):

            def __init__(self, *args, **kwargs):
                pass

            @classmethod
            def _create_vector(cls, *args, **kwargs):
                cls.vector_args = args
                cls.vector_kwargs = kwargs
                return None

            @classmethod
            def transform(cls, *args, **kwargs):
                cls.transform_args = args
                cls.transform_kwargs = kwargs
                return (None, None)

        settings = dict(vmin=100, vmax=300, nvel=50, vspace="log")
        example = Example.from_array("array", settings)

        self.assertTupleEqual((100, 300, 50, "log"), Example.vector_args)
        self.assertDictEqual({}, Example.vector_kwargs)

        self.assertTupleEqual(("array", None, settings),
                              Example.transform_args)
        self.assertDictEqual({}, Example.transform_kwargs)

    def test_frequency_keep_ids(self):
        class Mock(swprocess.wavefieldtransforms.AbstractWavefieldTransform):

            def __init__(self, *args, **kwargs):
                pass

            def transform(self, *args, **kwargs):
                pass

        mock = Mock()

        # df = 1 -> Find 5 to 30 by 1 Hz steps
        frequencies = np.arange(100)
        fmin, fmax, multiple = 5, 30, 1
        expected = np.arange(fmin, fmax+1)
        keep_ids = mock._frequency_keep_ids(frequencies, fmin, fmax, multiple)
        returned = frequencies[keep_ids]
        self.assertArrayEqual(expected, returned)

        # df = 0.5 -> Find 5 to 30 by 1 Hz steps
        frequencies = np.arange(100)*0.5
        fmin, fmax, multiple = 5, 30, 2
        expected = np.arange(fmin, fmax+1)
        keep_ids = mock._frequency_keep_ids(frequencies, fmin, fmax, multiple)
        returned = frequencies[keep_ids]
        self.assertArrayEqual(expected, returned)

    def test_normalize_power(self):
        nfrqs, nvels = 4, 3
        power = np.arange(nfrqs*nvels, dtype=float).reshape(nvels, nfrqs)
        frqs = np.arange(nfrqs)
        vels = np.arange(nvels)

        approaches = ["none", "absolute-maximum", "frequency-maximum"]
        divisors = [1, (nfrqs*nvels)-1, np.max(power, axis=0)]
        for approach, divisor in zip(approaches, divisors):
            transform = EmptyWavefieldTransform(frqs, vels, power)
            transform.normalize(by=approach)
            returned = transform.power
            expected = np.array(power)/divisor
            self.assertArrayEqual(expected, returned)

    def test_plot_snr(self):
        # transform example file.
        frequencies = np.arange(5, 100, 5)
        velocities = np.linspace(100, 300, 50)
        power = np.random.random((len(velocities), len(frequencies)))
        Empty = swprocess.wavefieldtransforms.EmptyWavefieldTransform
        transform = Empty(frequencies, velocities, power)

        # Try plotting without snr.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returned = transform.plot_snr()
        self.assertTrue(returned is None)

        # Default.
        transform.snr = np.ones(95)
        transform.snr_frequencies = np.linspace(5, 100, 95)
        transform.plot_snr()

        plt.show(block=False)
        plt.close("all")

    def test_plot(self):
        # transform example file.
        frequencies = np.arange(5, 100, 5)
        velocities = np.linspace(100, 300, 50)
        power = np.random.random((len(velocities), len(frequencies)))
        Empty = swprocess.wavefieldtransforms.EmptyWavefieldTransform
        transform = Empty(frequencies, velocities, power)

        # plot default.
        transform.plot()

        # plot with figure, figure axes, and colorbar axis.
        fig, (ax, cax) = plt.subplots(ncols=2)
        transform.plot(fig=fig, ax=ax, cax=cax)

        # plot with figure axes but no figure.
        fig, ax = plt.subplots()
        self.assertRaises(ValueError, transform.plot, ax=ax)

        plt.show(block=False)
        plt.close("all")


class test_EmptyWavefieldTransform(TestCase):

    def test_init(self):
        f = np.arange(5, 50, 1)
        v = np.linspace(100, 500, 100)
        p = np.random.random((len(v), len(v)))
        empty = swprocess.wavefieldtransforms.EmptyWavefieldTransform(f, v, p)
        self.assertArrayEqual(empty.frequencies, f)
        self.assertArrayEqual(empty.velocities, v)
        self.assertArrayEqual(empty.power, p)
        self.assertEqual(0, empty.n)

    def test_from_array(self):
        class SubEmpty(swprocess.wavefieldtransforms.EmptyWavefieldTransform):

            @classmethod
            def _create_vector(cls, *args, **kwargs):
                cls._create_vector_args = args
                cls._create_vector_kwargs = kwargs

        sensors = [swprocess.Sensor1C([0]*100, 0.01, 2*n, 0, 0)
                   for n in range(5)]
        source = swprocess.Source(-5, 0, 0)
        array = swprocess.Array1D(sensors, source)
        settings = dict(fmin=5, fmax=50, vmin=75,
                        vmax=300, nvel=30, vspace="lin")
        empty = SubEmpty.from_array(array, settings)

        self.assertArrayEqual(np.arange(5., 50+1, 1), empty.frequencies)
        expected = (settings["vmin"], settings["vmax"], settings["nvel"], settings["vspace"])
        returned = empty._create_vector_args
        self.assertTupleEqual(expected, returned)
        self.assertDictEqual({}, empty._create_vector_kwargs)

    def test_stack(self):
        f = np.array([5., 10.])
        v = np.array([100., 200])
        p = np.array([[3., 5.], [7., 9.]])
        a = swprocess.wavefieldtransforms.EmptyWavefieldTransform(f, v, p)
        self.assertEqual(0, a.n)

        p = np.array([[1., 2.], [3., 4.]])
        b = swprocess.wavefieldtransforms.EmptyWavefieldTransform(f, v, p)
        b.n = 1

        # Stack b on a.
        a.stack(b)
        self.assertArrayEqual(p, a.power)

        # Missing power attribute
        delattr(b, "power")
        self.assertRaises(AttributeError, a.stack, b)


if __name__ == "__main__":
    unittest.main()
