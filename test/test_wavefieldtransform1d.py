"""Tests for class WavefieldTransform1D."""

import unittest
import swprocess
import matplotlib.pyplot as plt
from testtools import TestCase, unittest, get_full_path


class Test_WaveTransform1d(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)
        cls.vuws_path = cls.full_path + "../examples/sample_data/vuws/"
        cls.wghs_path = cls.full_path + "../examples/sample_data/wghs/"

    def test_init(self):
        array = swprocess.Array1D.from_files(fnames=self.vuws_path+"22.dat")
        fk = swprocess.WavefieldTransform1D(array=array,
                                            settings=self.full_path+"settings/settings_fk.json")
        fk.plot_spectra(stype="fv")

        array = swprocess.Array1D.from_files(fnames=self.vuws_path+"22.dat")
        phase_shift = swprocess.WavefieldTransform1D(array=array,
                                                     settings=self.full_path+"settings/settings_phase-shift.json")
        phase_shift.plot_spectra(stype="fv")

        array = swprocess.Array1D.from_files(fnames=self.vuws_path+"22.dat")
        slant_stack = swprocess.WavefieldTransform1D(array=array,
                                                     settings=self.full_path+"settings/settings_slant-stack.json")
        slant_stack.plot_spectra(stype="fv")

        array = swprocess.Array1D.from_files(fnames=self.vuws_path+"22.dat")
        fdbf = swprocess.WavefieldTransform1D(array=array,
                                              settings=self.full_path+"settings/settings_fdbf.json")
        fdbf.plot_spectra(stype="fv")
        plt.show()


if __name__ == "__main__":
    unittest.main()
