"""Tests for Sensor1C."""

import warnings
import logging

import numpy as np
import obspy

from swprocess import ActiveTimeSeries, Sensor1C
from testtools import unittest, TestCase, get_full_path

logging.basicConfig(level=logging.ERROR)


class Test_Sensor1C(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

        cls.a_amp = np.array([0, 1, 2, 1, 0, 1], dtype=np.double)
        cls.a_dt = 1.
        cls.tseries_a = ActiveTimeSeries(cls.a_amp, cls.a_dt)

        cls.b_amp = np.array([0, 1, 0, 1, 0, 1], dtype=np.double)
        cls.b_dt = 1.
        cls.tseries_b = ActiveTimeSeries(cls.b_amp, cls.b_dt)

    def test_init(self):
        # __init__
        sensor_1 = Sensor1C(self.a_amp, self.a_dt, 0, 0, 0)

        self.assertArrayEqual(self.a_amp, sensor_1.amp)
        self.assertEqual(self.a_dt, sensor_1.dt)

        # from_activetimeseries
        sensor_2 = Sensor1C.from_activetimeseries(self.tseries_a, 0, 0, 0)
        self.assertEqual(sensor_1, sensor_2)

    def test_from_trace(self):
        # seg2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traces = obspy.read(self.full_path+"../examples/sample_data/vuws/6.dat")
        trace = traces[0]
        sensor = Sensor1C.from_trace(trace)
        self.assertArrayEqual(trace.data, sensor.amp)
        self.assertEqual(trace.stats.delta, sensor.dt)
        x = float(trace.stats.seg2.RECEIVER_LOCATION)
        self.assertTupleEqual((x, 0., 0.),
                              (sensor.x, sensor.y, sensor.z))
        self.assertEqual(float(trace.stats.seg2.DELAY), sensor.delay)
        self.assertEqual(int(trace.stats.seg2.STACK), sensor.nstacks)

        # su
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traces = obspy.read(self.full_path+"data/denise/v1.2_x.su.shot1")
        trace = traces[0]
        sensor = Sensor1C.from_trace(trace)
        self.assertArrayEqual(trace.data, sensor.amp)
        self.assertEqual(trace.stats.delta, sensor.dt)
        header = trace.stats.su.trace_header
        x, y = [
            float(header[key])/1000 for key in [f"group_coordinate_{c}" for c in ["x", "y"]]]
        self.assertTupleEqual((x, y, 0.),
                              (sensor.x, sensor.y, sensor.z))
        self.assertEqual(float(header["delay_recording_time"]), sensor.delay)
        nstack_key = "number_of_horizontally_stacked_traces_yielding_this_trace"
        self.assertEqual(int(header[nstack_key])+1, sensor.nstacks)

        # read_header=False
        for cpath in ["../../examples/sample_data/vuws/11.dat", "denise/v1.2_x.su.shot1"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traces = obspy.read(self.full_path+"data/"+cpath)
            sensor = Sensor1C.from_trace(
                trace, read_header=False, nstacks=15, delay=-2, x=3, y=6, z=12)
            self.assertArrayEqual(trace.data, sensor.amp)
            self.assertEqual(trace.stats.delta, sensor.dt)
            self.assertEqual(15, sensor.nstacks)
            self.assertEqual(-2., sensor.delay)
            self.assertListEqual([3, 6, 12],
                                [getattr(sensor, c) for c in ["x", "y", "z"]])


if __name__ == "__main__":
    unittest.main()
