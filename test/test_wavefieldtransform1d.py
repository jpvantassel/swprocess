"""Tests for class WavefieldTransform1D."""

import numpy as np

from swprocess.wavefieldtransforms import EmptyWavefieldTransform
import swprocess
import matplotlib.pyplot as plt
from testtools import TestCase, unittest, get_full_path


class Test_WaveTransform1d(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)
        cls.vuws_path = cls.full_path + "../examples/sample_data/vuws/"
        cls.wghs_path = cls.full_path + "../examples/sample_data/wghs/"

    def test_normalize_power(self):
        nfrqs, nvels = 4, 3
        power = np.arange(nfrqs*nvels).reshape(nvels, nfrqs)
        frqs = np.arange(nfrqs)
        vels = np.arange(nvels)

        approaches = ["none", "absolute-maximum", "frequency-maximum"]
        divisors = [1, (nfrqs*nvels)-1, np.max(power, axis=0)]
        for approach, divisor in zip(approaches, divisors):
            transform = EmptyWavefieldTransform(frqs, vels, power)
            transform.normalize(by=approach)
            returned = transform.power
            expected = power/divisor
            self.assertArrayEqual(expected, returned)

if __name__ == "__main__":
    unittest.main()
