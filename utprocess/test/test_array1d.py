"""Tests for Array1D class."""
import unittest
import utprocess
import obspy
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
logging.basicConfig(level=logging.CRITICAL)


def make_dummy_array(timeseries, nreceivers, spacing, source_location):
    """Make simple dummy array from timeseries for testing."""
    recs = []
    for receiever_number in range(nreceivers):
        pos = {'x': receiever_number*spacing, 'y': 0, 'z': 0}
        rec = utprocess.Receiver1D(timeseries=timeseries, position=pos)
        recs.append(rec)
    source = utprocess.Source(position={"x": source_location, "y": 0, "z": 0})
    return utprocess.Array1D(receivers=recs, source=source)


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
        source = utprocess.Source(position={"x": -5, "y": 0, "z": 0})
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            known = obspy.read(fname)
        test = utprocess.Array1D.from_seg2s(fname)
        self.assertListEqual(known.traces[0].data.tolist(),
                             test.timeseriesmatrix[:, 0].tolist())

    def test_plot_waterfall(self):
        # Single shot (near-side)
        fname = "test/data/vuws/1.dat"
        array1 = utprocess.Array1D.from_seg2s(fname)
        array1.plot_waterfall()

        # Multiple shots (near-side)
        fnames = [f"test/data/vuws/{x}.dat" for x in range(1, 6)]
        array2 = utprocess.Array1D.from_seg2s(fnames)
        array2.plot_waterfall()

        # Single shot (far-side)
        fname = "test/data/vuws/16.dat"
        array3 = utprocess.Array1D.from_seg2s(fname)
        array3.plot_waterfall()

        # Multiple shots (near-side)
        fnames = [f"test/data/vuws/{x}.dat" for x in range(16, 20)]
        array4 = utprocess.Array1D.from_seg2s(fnames)
        array4.plot_waterfall()
        plt.show()

    def test_plot_array(self):
        # Basic case (near-side, 2m spacing)
        fname = "test/data/vuws/1.dat"
        utprocess.Array1D.from_seg2s(fname).plot_array()

        # Non-linear spacing
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
        source = utprocess.Source(position={"x": -5, "y": 0, "z": 0})
        arr = utprocess.Array1D(receivers=[rec1, rec2, rec3],
                                source=source)
        arr.plot_array()

        # Basic case (far-side, 2m spacing)
        fname = "test/data/vuws/20.dat"
        utprocess.Array1D.from_seg2s(fname).plot_array()
        plt.show()

    def test_trim_timeseries(self):
        # Standard case (0.5s delay, 1s record -> 0.5s record)
        recording = utprocess.TimeSeries(amplitude=np.sin(2*np.pi*1*np.arange(-1, 1, 0.01)).tolist(),
                                         dt=0.01,
                                         nstacks=1,
                                         delay=-1)
        arr = make_dummy_array(timeseries=recording,
                               nreceivers=2,
                               spacing=2,
                               source_location=-5)
        self.assertEqual(arr.delay, -1)
        self.assertEqual(arr.nsamples, 200)
        arr.trim_timeseries(0, 0.5)
        self.assertEqual(arr.delay, 0)
        self.assertEqual(arr.nsamples, 51)

        # Long record (-1s delay, 1s record -> 1s record)
        recording = utprocess.TimeSeries(amplitude=np.sin(2*np.pi*1*np.arange(-1, 2, 0.01)).tolist(),
                                         dt=0.01,
                                         nstacks=1,
                                         delay=-1)
        arr = make_dummy_array(timeseries=recording,
                               nreceivers=2,
                               spacing=2,
                               source_location=-5)
        self.assertEqual(arr.delay, -1)
        self.assertEqual(arr.nsamples, 300)
        arr.trim_timeseries(0, 1)
        self.assertEqual(arr.delay, 0)
        self.assertEqual(arr.nsamples, 101)

        # Bad trigger (-0.5s delay, 0.5s record -> 0.2s record)
        recording = utprocess.TimeSeries(amplitude=np.sin(2*np.pi*1*np.arange(-0.5, 0.5, 0.01)).tolist(),
                                         dt=0.01,
                                         nstacks=1,
                                         delay=-0.5)
        arr = make_dummy_array(timeseries=recording,
                               nreceivers=2,
                               spacing=2,
                               source_location=-5)
        self.assertEqual(arr.delay, -0.5)
        self.assertEqual(arr.nsamples, 100)
        arr.trim_timeseries(-0.1, 0.1)
        self.assertEqual(arr.delay, -0.1)
        self.assertEqual(arr.nsamples, 21)


if __name__ == '__main__':
    unittest.main()
