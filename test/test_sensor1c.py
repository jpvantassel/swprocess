# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Tests for Sensor1C."""

import warnings
import logging
from unittest.mock import MagicMock

import numpy as np
import obspy

from swprocess import ActiveTimeSeries, Sensor1C
from testtools import unittest, TestCase, get_path

logging.basicConfig(level=logging.ERROR)


class Test_Sensor1C(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = get_path(__file__)
        cls.wghs_path = cls.path / "../examples/masw/data/wghs/"

        cls.a_amp = np.array([0, 1, 2, 1, 0, 1], dtype=np.double)
        cls.a_dt = 1.
        cls.tseries_a = ActiveTimeSeries(cls.a_amp, cls.a_dt)

        cls.b_amp = np.array([0, 1, 0, 1, 0, 1], dtype=np.double)
        cls.b_dt = 1.
        cls.tseries_b = ActiveTimeSeries(cls.b_amp, cls.b_dt)

    def test_init(self):
        # __init__
        sensor_1 = Sensor1C(self.a_amp, self.a_dt, 0, 0, 0)

        self.assertArrayEqual(self.a_amp, sensor_1.amplitude)
        self.assertEqual(self.a_dt, sensor_1.dt)

        # from_activetimeseries
        sensor_2 = Sensor1C.from_activetimeseries(self.tseries_a, 0, 0, 0)
        self.assertEqual(sensor_1, sensor_2)

    def test_from_trace(self):
        # seg2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traces = obspy.read(str(self.wghs_path / "6.dat"))
            trace = traces[0]
            sensor = Sensor1C.from_trace(trace)
        self.assertArrayEqual(trace.data, sensor.amplitude)
        self.assertEqual(trace.stats.delta, sensor.dt)
        x = float(trace.stats.seg2.RECEIVER_LOCATION)
        self.assertTupleEqual((x, 0., 0.),
                              (sensor.x, sensor.y, sensor.z))
        self.assertEqual(float(trace.stats.seg2.DELAY), sensor.delay)
        self.assertEqual(int(trace.stats.seg2.STACK), sensor.nstacks)

        # su
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traces = obspy.read(str(self.path / "data/custom/shot1.su"))
            trace = traces[0]
            sensor = Sensor1C.from_trace(trace)
        self.assertArrayEqual(trace.data, sensor.amplitude)
        self.assertEqual(trace.stats.delta, sensor.dt)
        header = trace.stats.su.trace_header
        scaleco = int(header["scalar_to_be_applied_to_all_coordinates"])
        scaleco = abs(1/scaleco) if scaleco < 0 else scaleco
        x, y = [int(header[key]) *scaleco for key in [f"group_coordinate_{c}" for c in ["x", "y"]]]
        self.assertTupleEqual((x, y, 0.),
                              (sensor.x, sensor.y, sensor.z))
        self.assertEqual(float(header["delay_recording_time"]), sensor.delay)
        nstack_key = "number_of_horizontally_stacked_traces_yielding_this_trace"
        self.assertEqual(int(header[nstack_key])+1, sensor.nstacks)

        # read_header=False
        for cpath in [self.wghs_path / "11.dat", self.path /"data/custom/shot1.su"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traces = obspy.read(str(cpath))
            sensor = Sensor1C.from_trace(
                trace, read_header=False, nstacks=15, delay=-2, x=3, y=6, z=12)
            self.assertArrayEqual(trace.data, sensor.amplitude)
            self.assertEqual(trace.stats.delta, sensor.dt)
            self.assertEqual(15, sensor.nstacks)
            self.assertEqual(-2., sensor.delay)
            self.assertListEqual([3, 6, 12],
                                [getattr(sensor, c) for c in ["x", "y", "z"]])

        # set trace.stats._format to integer so it raises ValueError
        mock_trace = MagicMock()
        mock_trace.stats._format = 1
        self.assertRaises(ValueError, Sensor1C.from_trace, mock_trace)

    def test_is_similar(self):
        a = Sensor1C(amplitude=[1.,2,3], dt=1., x=0, y=0, z=0, nstacks=1, delay=0)

        b = "Not a Sensor1C"
        c = Sensor1C(amplitude=[1.,2], dt=1., x=0, y=0, z=0, nstacks=1, delay=0)
        d = Sensor1C(amplitude=[1.,2,3], dt=2., x=0, y=0, z=0, nstacks=1, delay=0)
        e = Sensor1C(amplitude=[1.,2,3], dt=1., x=1, y=0, z=0, nstacks=1, delay=0)
        f = Sensor1C(amplitude=[1.,2,3], dt=1., x=0, y=1, z=0, nstacks=1, delay=0)
        g = Sensor1C(amplitude=[1.,2,3], dt=1., x=0, y=0, z=1, nstacks=1, delay=0)
        h = Sensor1C(amplitude=[1.,2,3], dt=1., x=0, y=0, z=0, nstacks=2, delay=0)
        i = Sensor1C(amplitude=[1.,2,3], dt=1., x=0, y=0, z=0, nstacks=1, delay=-0.5)

        j = Sensor1C(amplitude=[1.,2,3], dt=1., x=0, y=0, z=0, nstacks=1, delay=0)

        self.assertFalse(a._is_similar(b))
        self.assertFalse(a._is_similar(c))
        self.assertFalse(a._is_similar(d))
        self.assertFalse(a._is_similar(e))
        self.assertFalse(a._is_similar(f))
        self.assertFalse(a._is_similar(g))

        self.assertTrue(a._is_similar(c, exclude=["nsamples"]))
        self.assertTrue(a._is_similar(d, exclude=["dt"]))
        self.assertTrue(a._is_similar(e, exclude=["x"]))
        self.assertTrue(a._is_similar(f, exclude=["y"]))
        self.assertTrue(a._is_similar(g, exclude=["z"]))
        self.assertTrue(a._is_similar(h, exclude=["nstacks"]))
        self.assertTrue(a._is_similar(i, exclude=["delay"]))
        self.assertTrue(a._is_similar(j))


if __name__ == "__main__":
    unittest.main()
