"""Tests for class WavefieldTransform1D."""

import unittest
import utprocess
import matplotlib.pyplot as plt


class TestWV1D(unittest.TestCase):
    def test_init(self):
        array = utprocess.Array1D.from_seg2s(fnames="test/data/vuws/4.dat")
        fk = utprocess.WavefieldTransform1D(array=array,
                                            settings_file="test/test_fksettings.json")
        fk.disp_power.plot_spec(plot_type="fv", plot_limit=[5, 100, 0, 500])
        plt.show()


if __name__ == "__main__":
    unittest.main()
