"""Tests for Sensor1C."""

import logging

import numpy as np

from utprocess import ActiveTimeSeries, Sensor1C
from testtools import unittest, TestCase

logging.basicConfig(level=logging.ERROR)


class Test_Sensor1C(TestCase):

    @classmethod
    def setUpClass(cls):
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
        sensor_2 = Sensor1C.from_activetimeseries(self.tseries_a, 0,0,0)
        self.assertEqual(sensor_1, sensor_2)


if __name__ == "__main__":
    unittest.main()
