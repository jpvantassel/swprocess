"""Tests for class WavefieldTransform1D."""

import unittest
import swprocess
import matplotlib.pyplot as plt
from testtools import TestCase, unittest, get_full_path


class Test_WaveTransform1d(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_init(self):
        array = swprocess.Array1D.from_files(fnames=self.full_path+"data/vuws/22.dat")
        fk = swprocess.WavefieldTransform1D(array=array,
                                            settings=self.full_path+"settings/fk.json")
        # fk.plot_spectra(stype="fv")
        # plt.close()
        plt.show()


if __name__ == "__main__":
    unittest.main()
