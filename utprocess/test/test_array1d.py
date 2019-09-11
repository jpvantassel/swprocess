"""Tests for Array1D class."""
import unittest
import utprocess
import obspy
import numpy as np
import matplotlib.pyplot as plt


class Test_Array1D(unittest.TestCase):
    def test_init(self):
        recording = utprocess.TimeSeries(amplitude=[1, 2, 3],
                                         dt=1,
                                         nstacks=1,
                                         delay=0)
        pos1 = {'x': 0, 'y': 0, 'z': 0}
        rec1 = utprocess.Receiver1D(timeseries=recording, position=pos1)
        pos2 = {'x': 1, 'y': 0, 'z': 0}
        rec2 = utprocess.Receiver1D(timeseries=recording, position=pos2)
        source = utprocess.Source(position = {"x": -5, "y": 0, "z": 0})
        arr = utprocess.Array1D(receivers=[rec1, rec2],
                                source=source)
        self.assertEqual(arr.nchannels, 2)
        self.assertEqual(arr.nsamples, 3)
        self.assertEqual(arr.dt, 1)
        self.assertEqual(arr.delay, 0)
        self.assertListEqual(arr.timeseriesmatrix.tolist(),
                             [[1, 1], [2, 2], [3, 3]])

    def test_fromseg2s(self):
        fname = "test/data/vuws/1.dat"
        known = obspy.read(fname)
        test = utprocess.Array1D.from_seg2s(fname)
        self.assertListEqual(known.traces[0].data.tolist(),
                             test.timeseriesmatrix[:, 0].tolist())

    def test_plot_waterfall(self):
        fname = "test/data/vuws/1.dat"
        array1 = utprocess.Array1D.from_seg2s(fname)
        array1.plot_waterfall()

        fnames = [f"test/data/vuws/{x}.dat" for x in range(1, 6)]
        array2 = utprocess.Array1D.from_seg2s(fnames)
        array2.plot_waterfall()

    def test_plot_array(self):
        fname = "test/data/vuws/1.dat"
        utprocess.Array1D.from_seg2s(fname).plot_array()

        recording = utprocess.TimeSeries(amplitude=[1, 2, 3],
                                         dt=1,
                                         nstacks=1,
                                         delay=0)
        pos1 = {'x': 0, 'y': 0, 'z': 0}
        rec1 = utprocess.Receiver1D(timeseries=recording, position=pos1)
        pos2 = {'x': 1, 'y': 0, 'z': 0}
        rec2 = utprocess.Receiver1D(timeseries=recording, position=pos2)
        pos3 = {'x': 3, 'y': 0, 'z': 0}
        rec3 = utprocess.Receiver1D(timeseries=recording, position=pos3)
        source = utprocess.Source(position = {"x": -5, "y": 0, "z": 0})
        arr = utprocess.Array1D(receivers=[rec1, rec2, rec3],
                                source=source)
        arr.plot_array()
        plt.show()


if __name__ == '__main__':
    unittest.main()
