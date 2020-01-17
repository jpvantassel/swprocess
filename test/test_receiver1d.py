"""Tests for Receiver1D class. """

from unittest_extensions import TestCase, unittest
import utprocess
import obspy
import logging
import numpy as np
import warnings
logging.basicConfig(level=logging.WARNING)


class TestReceiver1D(TestCase):

    def test_init(self):
        timeseries = utprocess.TimeSeries(amplitude=np.arange(0, 1, 0.01),
                                          dt=0.01)
        expected_position = {"x": 0, "y": 0, "z": 0}
        rec1d = utprocess.Receiver1D(timeseries=timeseries,
                                     position=expected_position)
        self.assertArrayEqual(timeseries.amp, rec1d.timeseries.amp)
        self.assertEqual(100, rec1d.timeseries.nsamples)
        self.assertDictEqual(expected_position, rec1d.position)

    def test_from_trace(self):
        # TODO (jpv): Extend these tests to check header information.
        # SEG2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            stream = obspy.read("test/data/vuws/1.dat")
        trace = stream[0]
        expected = trace.data
        received = utprocess.Receiver1D.from_trace(trace).timeseries.amp
        self.assertArrayEqual(expected, received)

        # SU
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            stream = obspy.read("test/data/denise/v1.2_x.su.shot1")
        trace = stream[0]
        expected = trace.data
        received = utprocess.Receiver1D.from_trace(trace).timeseries.amp
        self.assertArrayEqual(expected, received)
        
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
