"""Tests for Receiver1D class. """

import unittest
import utprocess
import obspy
import logging
import numpy as np
import warnings
logging.basicConfig(level=logging.DEBUG)


class TestReceiver1D(unittest.TestCase):

    def test_init(self):
        timeseries = utprocess.TimeSeries(amplitude=np.arange(0, 1, 0.01),
                                          dt=0.01)
        position = {"x": 0, "y": 0, "z": 0}
        rec1d = utprocess.Receiver1D(timeseries=timeseries,
                                     position=position)
        self.assertListEqual(rec1d.amp.tolist(), np.arange(0, 1, 0.01).tolist())
        self.assertEqual(rec1d.nsamples, 100)
        self.assertDictEqual(rec1d.position, position)

    def test_from_trace(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            stream = obspy.read("test/data/vuws/1.dat")
        known = stream[0]
        test = utprocess.Receiver1D.from_trace(known)
        print(type(test))
        self.assertListEqual(known.data.tolist(),
                             test.amp.tolist())

    def test_trim(self):
        timeseries = utprocess.TimeSeries(amplitude=np.arange(0, 1, 0.01),
                                          dt=0.01)
        position = {"x": 0, "y": 0, "z": 0}
        rec1d = utprocess.Receiver1D(timeseries=timeseries,
                                     position=position)
        rec1d.timeseries.trim(start_time = 0.1, end_time = 0.2)
        self.assertEqual(rec1d.timeseries.nsamples, 10)



if __name__ == '__main__':
    unittest.main()
