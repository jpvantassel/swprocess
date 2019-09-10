"""Tests for class Array1D """
import unittest
import utprocess
import obspy
import numpy as np
import matplotlib.pyplot as plt


class Test_Array1D(unittest.TestCase):
    def test_init(self):
        recordings = {"amp": [1, 2, 3], "dt": 1, "nstacks": 1}
        pos1 = {'x': 0, 'y': 0, 'z': 0}
        rec1 = utprocess.Receiver1D(recordings=recordings, position=pos1)
        pos2 = {'x': 1, 'y': 0, 'z': 0}
        rec2 = utprocess.Receiver1D(recordings=recordings, position=pos2)
        arr = utprocess.Array1D([rec1, rec2])
        self.assertEqual(arr.nchannels, 2)
        self.assertEqual(arr.nsamples, 3)
        self.assertEqual(arr.dt, 1)
        self.assertListEqual(arr.timeseriesmatrix.tolist(),
                             [[1, 1], [2, 2], [3, 3]])

    def test_fromseg2(self):
        fname = "test/data/vuws/1.dat"
        known = obspy.read(fname)
        test = utprocess.Array1D.from_seg2(fname)
        self.assertListEqual(known.traces[0].data.tolist(),
                             test.timeseriesmatrix[:, 0].tolist())

    def test_plot_waterfall(self):
        fname = "test/data/vuws/1.dat"
        utprocess.Array1D.from_seg2(fname).plot_waterfall()
        plt.show()


if __name__ == '__main__':
    unittest.main()
