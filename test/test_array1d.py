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

"""Tests for Array1D class."""

import os
import warnings
import logging

import obspy
import numpy as np
import matplotlib.pyplot as plt

from unittest.mock import patch
from testtools import TestCase, unittest, get_path
import swprocess

logging.basicConfig(level=logging.ERROR)


class Test_Array1D(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = get_path(__file__)
        cls.wghs_path = cls.path.parent / "examples/masw/data/wghs/"

        cls.sensor_0 = swprocess.Sensor1C(amplitude=[-.1]*9 + [-0.11] + [-.1]*9,
                                          dt=1, x=0, y=0, z=0, nstacks=1,
                                          delay=0)
        cls.sensor_1 = swprocess.Sensor1C(amplitude=[0.2]*9 + [+0.25] + [0.2]*9,
                                          dt=1, x=1, y=0, z=0, nstacks=1,
                                          delay=0)
        cls.sensor_5 = swprocess.Sensor1C(amplitude=[1.0]*9 + [+1.05] + [1.0]*9,
                                          dt=1, x=5, y=0, z=0, nstacks=1,
                                          delay=0)
        cls.sensor_6 = swprocess.Sensor1C(amplitude=[2.0]*9 + [+2.02] + [2.0]*9,
                                          dt=1, x=6, y=0, z=0, nstacks=1,
                                          delay=0)

    @staticmethod
    def dummy_array(amp, dt, nstacks, delay, nsensors, spacing, source_x):
        """Make simple dummy array from timeseries for testing."""
        sensors = []
        for i in range(nsensors):
            sensor = swprocess.Sensor1C(amp, dt, x=i*spacing, y=0, z=0,
                                        nstacks=nstacks, delay=delay)
            sensors.append(sensor)
        source = swprocess.Source(x=source_x, y=0, z=0)
        return swprocess.Array1D(sensors=sensors, source=source)

    def test_init(self):
        # Basic
        source = swprocess.Source(x=-5, y=0, z=0)
        sensors = [self.sensor_0, self.sensor_1]
        array = swprocess.Array1D(sensors=sensors, source=source)
        self.assertEqual(array.source, source)
        self.assertListEqual(array.sensors, sensors)

        # Bad: Invalid sensors
        self.assertRaises(ValueError, swprocess.Array1D, sensors=[self.sensor_5, self.sensor_5],
                          source=source)

        # Bad: Incompatable sensors
        sensor_bad = swprocess.Sensor1C(amplitude=[1, 2, 3, 4], dt=1, x=7, y=0, z=0,
                                        nstacks=1, delay=0)
        self.assertRaises(ValueError, swprocess.Array1D, sensors=[self.sensor_5, sensor_bad],
                          source=source)

    def test_timeseriesmatrix(self):
        source = swprocess.Source(x=-5, y=0, z=0)
        sensors = [self.sensor_0, self.sensor_1, self.sensor_5, self.sensor_6]
        array = swprocess.Array1D(sensors=sensors, source=source)
        base = np.array([[-.1]*9 + [-0.11] + [-.1]*9,
                         [0.2]*9 + [+0.25] + [0.2]*9,
                         [1.0]*9 + [+1.05] + [1.0]*9,
                         [2.0]*9 + [+2.02] + [2.0]*9])

        # detrend=False, normalize="none"
        expected = base
        returned = array.timeseriesmatrix(detrend=False, normalize="none")
        self.assertArrayEqual(expected, returned)

        # detrend=False, normalize="each"
        expected = base / np.array([0.11, 0.25, 1.05, 2.02]).reshape(4, 1)
        returned = array.timeseriesmatrix(detrend=False, normalize="each")
        self.assertArrayEqual(expected, returned)

        # detrend=False, normalize="all"
        expected = base / 2.02
        returned = array.timeseriesmatrix(detrend=False, normalize="all")
        self.assertArrayEqual(expected, returned)

        # detrend=True, normalize="none"
        expected = base - np.array([-.1, 0.2, 1, 2]).reshape(4, 1)
        returned = array.timeseriesmatrix(detrend=True, normalize="none")
        self.assertArrayAlmostEqual(expected, returned, places=2)

    def test_position(self):
        source = swprocess.Source(x=-5, y=0, z=0)
        sensors = [self.sensor_1, self.sensor_5, self.sensor_6]
        array = swprocess.Array1D(sensors=sensors, source=source)

        # normalize=False
        returned = array.position(normalize=False)
        expected = [1, 5, 6]
        self.assertListEqual(expected, returned)

        # normalize=True
        returned = array.position(normalize=True)
        expected = [0, 4, 5]
        self.assertListEqual(expected, returned)

    def test_offsets(self):
        source = swprocess.Source(x=-5, y=0, z=0)
        sensors = [self.sensor_1, self.sensor_5, self.sensor_6]
        array = swprocess.Array1D(sensors=sensors, source=source)

        # simple
        returned = array.offsets
        expected = [5+1, 5+5, 5+6]
        self.assertListEqual(expected, returned)

    def test_kres(self):
        for spacing in [1, 2.2, 5.5]:
            array = self.dummy_array(amp=[0, 0, 0], dt=1, nstacks=1, delay=0,
                                     nsensors=5, spacing=spacing, source_x=-5)
            self.assertEqual(np.pi/spacing, array.kres)

    def test_nchannels(self):
        for nsensors in [1, 3, 5]:
            array = self.dummy_array(amp=[0, 0, 0], dt=1, nstacks=1, delay=0,
                                     nsensors=nsensors, spacing=1, source_x=-5)
            self.assertEqual(nsensors, array.nchannels)

    def test_array_center_distance(self):
        spacing = 2
        source_offset = 5
        for nsensors in [24, 48, 96]:
            array = self.dummy_array(amp=[0, 0, 0], dt=1, nstacks=1, delay=0,
                                     nsensors=nsensors, spacing=spacing, source_x=-source_offset)
            expected = source_offset + (nsensors-1)*spacing/2
            self.assertEqual(expected, array.array_center_distance)

    def test_spacing(self):
        # constant spacing
        for spacing in [1, 2, 5.5]:
            array = self.dummy_array(amp=[0, 0, 0], dt=1, nstacks=1, delay=0,
                                     nsensors=3, spacing=spacing, source_x=-5)
            self.assertEqual(spacing, array.spacing)

        # non-constant spacing
        source = swprocess.Source(x=-10, y=0, z=0)
        sensors = [self.sensor_0, self.sensor_5, self.sensor_6]
        array = swprocess.Array1D(sensors=sensors, source=source)
        try:
            array.spacing
        except ValueError:
            raised_error = True
        else:
            raised_error = False
        finally:
            self.assertTrue(raised_error)

    def test_source_inside(self):
        sensors = [self.sensor_0, self.sensor_6]

        # _source_inside -> True
        source = swprocess.Source(x=3, y=0, z=0)
        array = swprocess.Array1D(sensors, source)
        self.assertTrue(array._source_inside)

        # _source_inside -> False
        source = swprocess.Source(x=-10, y=0, z=0)
        array = swprocess.Array1D(sensors, source)
        self.assertFalse(array._source_inside)

    def test_trim(self):
        # Standard case (1s delay, 1s record -> 0.5s record)
        array = self.dummy_array(amp=np.sin(2*np.pi*1*np.arange(-1, 1, 0.01)),
                                 dt=0.01, nstacks=1, delay=-1, nsensors=2,
                                 spacing=2, source_x=-5)
        self.assertEqual(-1, array.sensors[0].delay)
        self.assertEqual(200, array.sensors[0].nsamples)
        array.trim(0, 0.5)
        self.assertEqual(0, array.sensors[0].delay)
        self.assertEqual(51, array.sensors[0].nsamples)

        # Long record (-1s delay, 2s record -> 1s record)
        array = self.dummy_array(amp=np.sin(2*np.pi*1*np.arange(-1, 2, 0.01)),
                                 dt=0.01, nstacks=1, delay=-1, nsensors=2,
                                 spacing=2, source_x=-5)
        self.assertEqual(-1, array.sensors[0].delay)
        self.assertEqual(300, array.sensors[0].nsamples)
        array.trim(0, 1)
        self.assertEqual(0, array.sensors[0].delay)
        self.assertEqual(101, array.sensors[0].nsamples)

        # Bad trigger (-0.5s delay, 0.5s record -> 0.2s record)
        array = self.dummy_array(amp=np.sin(2*np.pi*1*np.arange(-0.5, 0.5, 0.01)),
                                 dt=0.01, nstacks=1, delay=-0.5, nsensors=2,
                                 spacing=2, source_x=-5)
        self.assertEqual(-0.5, array.sensors[0].delay)
        self.assertEqual(100, array.sensors[0].nsamples)
        array.trim(-0.1, 0.1)
        self.assertEqual(-0.1, array.sensors[0].delay)
        self.assertEqual(21, array.sensors[0].nsamples)

    def test_zero_pad(self):
        # No change: df=0.1
        nsamples = 10
        array = self.dummy_array(amp=[0]*nsamples, dt=1, nstacks=1, delay=0,
                                 nsensors=2, spacing=1, source_x=-1)
        array.zero_pad(df=0.1)
        for sensor in array:
            self.assertEqual(nsamples, sensor.nsamples)

        # Pad zeros: df=0.01
        nsamples = 10
        array = self.dummy_array(amp=[0]*nsamples, dt=1, nstacks=1, delay=0,
                                 nsensors=2, spacing=1, source_x=-1)
        array.zero_pad(df=0.01)
        for sensor in array:
            self.assertEqual(nsamples*10, sensor.nsamples)

        # Select subset: df=1
        nsamples = 10
        array = self.dummy_array(amp=[0]*nsamples, dt=1, nstacks=1, delay=0,
                                 nsensors=2, spacing=1, source_x=-1)
        array.zero_pad(df=1)
        for sensor in array:
            self.assertEqual(nsamples, sensor.nsamples)
            self.assertEqual(10, sensor._multiple)

    def test_flip_required(self):
        sensors = [self.sensor_0, self.sensor_1]

        # _flip_required -> True
        source = swprocess.Source(x=3, y=0, z=0)
        array = swprocess.Array1D(sensors, source)
        self.assertTrue(array._flip_required)

        # _flip_required -> False
        source = swprocess.Source(x=-5, y=0, z=0)
        array = swprocess.Array1D(sensors, source)
        self.assertFalse(array._flip_required)

    def test_waterfall(self):
        # Single shot (near-side)
        fname = self.wghs_path / "1.dat"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            array1 = swprocess.Array1D.from_files(fname)
        array1.waterfall()

        # Multiple shots (near-side)
        fnames = [self.wghs_path / f"{x}.dat" for x in range(1, 6)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            array2 = swprocess.Array1D.from_files(fnames)
        array2.waterfall()

        # Single shot (far-side)
        fname = self.wghs_path / "16.dat"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            array3 = swprocess.Array1D.from_files(fname)
        array3.waterfall()

        # Multiple shots (near-side)
        fnames = [self.wghs_path / f"{x}.dat" for x in range(16, 20)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            array4 = swprocess.Array1D.from_files(fnames)
        array4.waterfall()
        array4.waterfall(time_ax="x")

        # Bad : time_along
        self.assertRaises(ValueError, array4.waterfall, time_ax="z")

        plt.show(block=False)
        plt.close("all")

    def test_plot(self):
        # Basic case (near-side, 2m spacing)
        fname = self.wghs_path / "1.dat"
        swprocess.Array1D.from_files(fname).plot()

        # Non-linear spacing
        sensors = [swprocess.Sensor1C(
            [1, 2, 3], dt=1, x=x, y=0, z=0,) for x in [0, 1, 3]]
        source = swprocess.Source(x=-5, y=0, z=0)
        array = swprocess.Array1D(sensors=sensors, source=source)
        array.plot()

        # Basic case (far-side, 2m spacing)
        fname = self.wghs_path / "20.dat"
        swprocess.Array1D.from_files(fname).plot()

        plt.show(block=False)
        plt.close("all")

    def test_auto_pick_first_arrivals(self):
        s1 = swprocess.Sensor1C(np.concatenate((np.zeros(100),
                                                np.array([0.1, 0, 0]),
                                                np.zeros(100))),
                                dt=1, x=1, y=0, z=0)
        s2 = swprocess.Sensor1C(np.concatenate((np.zeros(100),
                                                np.array([0, 0.2, 0]),
                                                np.zeros(100))),
                                dt=1, x=2, y=0, z=0)
        source = swprocess.Source(0, 0, 0)
        array = swprocess.Array1D([s1, s2], source)

        # algorithm = "threshold"
        position, times = array.auto_pick_first_arrivals(algorithm="threshold")
        self.assertListEqual(array.position(), position)
        self.assertListEqual([100, 101], times)

        # algorithm = "bad"
        self.assertRaises(NotImplementedError, array.auto_pick_first_arrivals,
                          algorithm="bad")

    def test_manual_pick_first_arrivals(self):
        # Replace self._ginput_session
        edist, etime = [0, 1, 2], [2, 1, 0]

        class DummyArray1D(swprocess.Array1D):
            def _ginput_session(ax, initial_adjustment=True,
                                ask_to_continue=True, npts=None):
                return (edist, etime)
        array = DummyArray1D([self.sensor_0, self.sensor_5, self.sensor_6],
                             swprocess.Source(x=-5, y=0, z=0))

        # time_ax = "y"
        for waterfall_kwargs in [dict(time_ax="y"), None]:
            rdist, rtime = array.manual_pick_first_arrivals(
                waterfall_kwargs=waterfall_kwargs)
        self.assertListEqual(edist, rdist)
        self.assertListEqual(etime, rtime)

        # time_ax = "x"
        rtime, rdist = array.manual_pick_first_arrivals(
            waterfall_kwargs=dict(time_ax="x"))
        self.assertListEqual(edist, rdist)
        self.assertListEqual(etime, rtime)

    @patch('matplotlib.pyplot.ginput', return_value=[(0, 1)])
    @patch('matplotlib.pyplot.waitforbuttonpress', return_value=True)
    def test_ginput_session(self, input_a, input_b):
        _, ax = plt.subplots()
        x, y = swprocess.Array1D._ginput_session(ax,
                                                 npts=1,
                                                 initial_adjustment=False,
                                                 ask_to_continue=False)
        returned = [(x[-1], y[-1])]
        expected = [(0, 1)]
        self.assertEqual(expected[0], returned[0])

    def test_interactive_mute(self):
        # Replace self._ginput_session
        class DummyArray1D(swprocess.Array1D):

            def set_xy_before(self, xs, ys):
                self.xs_before = xs
                self.ys_before = ys

            def set_xy_after(self, xs, ys):
                self.xs_after = xs
                self.ys_after = ys

            def _my_generator(self):
                xs = [self.xs_before, self.xs_after]
                ys = [self.ys_before, self.ys_after]
                for x, y in zip(xs, ys):
                    yield (x, y)

            def _opt1_ginput_session(self, *args, **kwargs):
                if not getattr(self, "mygen", False):
                    self.mygen = self._my_generator()
                return next(self.mygen)

            def _opt2_ginput_session(self, *args, **kwargs):
                return (self.xs_after, self.ys_after)

            def interactive_mute(self, mute_location="both",
                                 window_kwargs=None, waterfall_kwargs=None):
                if mute_location == "after":
                    self._ginput_session = self._opt2_ginput_session
                else:
                    self._ginput_session = self._opt1_ginput_session

                return super().interactive_mute(mute_location=mute_location,
                                                window_kwargs=window_kwargs,
                                                waterfall_kwargs=waterfall_kwargs)

        # Create dummy array
        source = swprocess.Source(x=-5, y=0, z=0)
        sensors = [self.sensor_0, self.sensor_5, self.sensor_6]
        sensors = [swprocess.Sensor1C.from_sensor1c(x) for x in sensors]
        array = DummyArray1D(sensors=sensors, source=source)

        # Set mute locations -> time_ax = "y"
        xs_b, ys_b = (1, 5), (1, 3)
        array.set_xy_before(xs_b, ys_b)
        xs_a, ys_a = (1, 6), (4, 7)
        array.set_xy_after(xs_a, ys_a)

        # mute_location = "before" & time_ax = "y"
        start, end = array.interactive_mute(mute_location="before")
        self.assertTupleEqual(((xs_b[0], ys_b[0]), (xs_b[1], ys_b[1])), start)
        self.assertTrue(end is None)
        delattr(array, "mygen")

        # mute_location = "after" & time_ax = "y"
        start, end = array.interactive_mute(mute_location="after")
        self.assertTrue(start is None)
        self.assertTupleEqual(((xs_a[0], ys_a[0]), (xs_a[1], ys_a[1])), end)

        # mute_location = "both" & time_ax = "y"
        start, end = array.interactive_mute(mute_location="both")
        self.assertTupleEqual(((xs_b[0], ys_b[0]), (xs_b[1], ys_b[1])), start)
        self.assertTupleEqual(((xs_a[0], ys_a[0]), (xs_a[1], ys_a[1])), end)
        delattr(array, "mygen")

        # Set mute locations -> time_ax = "x"
        array.set_xy_before(ys_b, xs_b)
        array.set_xy_after(ys_a, xs_a)

        # mute_location = "before" & time_ax = "x"
        start, end = array.interactive_mute(mute_location="before",
                                            waterfall_kwargs=dict(time_ax="x"))
        self.assertTupleEqual(((xs_b[0], ys_b[0]), (xs_b[1], ys_b[1])), start)
        self.assertTrue(end is None)
        delattr(array, "mygen")

        # mute_location = "after" & time_ax = "x"
        start, end = array.interactive_mute(mute_location="after",
                                            waterfall_kwargs=dict(time_ax="x"))
        self.assertTrue(start is None)
        self.assertTupleEqual(((xs_a[0], ys_a[0]), (xs_a[1], ys_a[1])), end)

        # mute_location = "both" & time_ax = "x"
        start, end = array.interactive_mute(mute_location="both",
                                            waterfall_kwargs=dict(time_ax="x"))
        self.assertTupleEqual(((xs_b[0], ys_b[0]), (xs_b[1], ys_b[1])), start)
        self.assertTupleEqual(((xs_a[0], ys_a[0]), (xs_a[1], ys_a[1])), end)
        delattr(array, "mygen")

    def test_mute(self):
        array = self.dummy_array(amp=[1]*100, dt=1, nstacks=1, delay=0,
                                 nsensors=5, spacing=2, source_x=-5)

        # Rectangular mute
        array.mute(signal_start=((0, 0), (2, 3)),
                   signal_end=((0, 6), (4, 12)),
                   window_kwargs=dict(alpha=0))
        returned = array.timeseriesmatrix()
        expected = np.array([[0]*0 + [1]*6 + [0]*94,
                             [0]*3 + [1]*6 + [0]*91,
                             [0]*6 + [1]*6 + [0]*88,
                             [0]*9 + [1]*6 + [0]*85,
                             [0]*12 + [1]*6 + [0]*82], dtype=float)
        self.assertArrayEqual(expected, returned)

    def test_flipped_tseries_and_offsets(self):
        # Near source offset -> No flip required
        near_array = self.dummy_array(amp=[0, 0, 0, 0], dt=1, nstacks=1,
                                      delay=0, nsensors=5, spacing=2,
                                      source_x=-5)
        for index, sensor in enumerate(near_array):
            sensor._amp += index

        expected_tmatrix = np.array([[0]*4, [1]*4, [2]*4, [3]*4, [4]*4])
        expected_offsets = np.array([5, 7, 9, 11, 13])
        returned_tmatrix, returned_offsets = near_array._flipped_tseries_and_offsets()
        self.assertArrayEqual(expected_tmatrix, returned_tmatrix)
        self.assertArrayEqual(expected_offsets, returned_offsets)

        # Far source offset -> Flip required
        far_array = swprocess.Array1D.from_array1d(near_array)
        far_array.source._x = 18

        expected_tmatrix = np.array([[4]*4, [3]*4, [2]*4, [1]*4, [0]*4])
        expected_offsets = np.array([10, 12, 14, 16, 18])
        returned_tmatrix, returned_offsets = far_array._flipped_tseries_and_offsets()
        self.assertArrayEqual(expected_tmatrix, returned_tmatrix)
        self.assertArrayEqual(expected_offsets, returned_offsets)

    def test_from_files(self):
        # Single File : SEG2
        fname = self.wghs_path / "1.dat"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            known = obspy.read(str(fname))
            array = swprocess.Array1D.from_files(fname)
        self.assertArrayEqual(np.arange(0, 48, 2), np.array(array.position()))
        self.assertEqual(-2, array.source.x)
        for expected, returned in zip(known.traces, array.timeseriesmatrix()):
            self.assertArrayEqual(expected.data, returned)

        # Single File : SU
        fname = self.path / "data/custom/shot1.su"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            known = obspy.read(str(fname))
            array = swprocess.Array1D.from_files(fname)
        self.assertArrayEqual(np.arange(7, 55, 2), np.array(array.position()))
        self.assertEqual(6, array.source.x)
        for expected, returned in zip(known.traces, array.timeseriesmatrix()):
            self.assertArrayEqual(expected.data, returned)

        # Multiple Files
        fnames = [self.wghs_path / f"{x}.dat" for x in range(1, 5)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp_stream = obspy.read(str(fnames[0]))
            expected = np.zeros(tmp_stream.traces[0].data.size)
            for fname in fnames:
                tmp = obspy.read(str(fname)).traces[0]
                expected += tmp.data
            expected /= len(fnames)
            returned = swprocess.Array1D.from_files(fnames)[0].amplitude
        self.assertArrayAlmostEqual(expected, returned, places=2)

        # Bad : coordinate units
        fname = self.path / "data/custom/shot1_badcu.su"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertRaises(ValueError, swprocess.Array1D.from_files, fname)

        # Bad : incompatible sources
        fnames = [self.wghs_path / f"{x}.dat" for x in range(1, 10)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertRaises(ValueError, swprocess.Array1D.from_files, fnames)

        # Bad : miniseed
        fname = self.path / "data/custom/0101010.miniseed"
        self.assertRaises(NotImplementedError,
                          swprocess.Array1D.from_files, fname)

        # Bad : format
        fname = self.path / "data/custom/0101010.sac"
        self.assertRaises(NotImplementedError,
                          swprocess.Array1D.from_files, fname)

    def test_to_and_from_su(self):
        spacing = 2
        sensors = []
        nsamples = 1000
        time = np.arange(0, 1, 1/nsamples)
        for n in range(24):
            sensor = swprocess.Sensor1C(amplitude=np.sin(2*np.pi*n*time),
                                        dt=1/nsamples, x=spacing*n, y=0, z=0,
                                        nstacks=1, delay=0)
            sensors.append(sensor)
        source = swprocess.Source(x=-10, y=0, z=0)

        expected = swprocess.Array1D(sensors, source)

        # To and from : SU
        fname = "to_file.su"
        expected.to_file(fname)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returned = swprocess.Array1D.from_files(fname)
        self.assertEqual(expected, returned)
        os.remove(fname)

        # Bad : format
        self.assertRaises(ValueError, expected.to_file, fname, ftype="seg2")

    def test_from_array1d(self):
        source = swprocess.Source(1, 0, 0)

        # Non-normalized
        sensors = [self.sensor_1, self.sensor_5]
        expected = swprocess.Array1D(sensors, source)
        returned = swprocess.Array1D.from_array1d(expected)
        self.assertEqual(expected, returned)

    def test_eq(self):
        source_0 = swprocess.Source(0, 0, 0)
        source_1 = swprocess.Source(1, 0, 0)
        array_a = swprocess.Array1D([self.sensor_0, self.sensor_1], source_0)
        array_b = "array1d"
        array_c = swprocess.Array1D([self.sensor_0], source_0)
        array_d = swprocess.Array1D([self.sensor_0, self.sensor_1], source_1)
        array_e = swprocess.Array1D([self.sensor_5, self.sensor_6], source_0)
        array_f = swprocess.Array1D([self.sensor_0, self.sensor_1], source_0)

        self.assertFalse(array_a == array_b)
        self.assertFalse(array_a == array_c)
        self.assertFalse(array_a == array_d)
        self.assertFalse(array_a == array_e)
        self.assertFalse(array_a != array_f)


if __name__ == '__main__':
    unittest.main()
