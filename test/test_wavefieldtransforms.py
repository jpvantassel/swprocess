"""Tests for class wavefieldtransforms module."""

import logging
from unittest.mock import patch, MagicMock
import warnings

import numpy as np

from swprocess.wavefieldtransforms import EmptyWavefieldTransform
import swprocess
import matplotlib.pyplot as plt
from testtools import TestCase, unittest, get_full_path

logger = logging.getLogger("swprocess.wavefieldtransforms")
logger.setLevel(logging.WARNING)


class Test_WavefieldTransforms(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)
        cls.wghs_path = cls.full_path + "../examples/sample_data/wghs/"

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


if __name__ == "__main__":
    unittest.main()
