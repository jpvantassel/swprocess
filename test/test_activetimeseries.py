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

"""Tests for ActiveTimeSeries class."""

import warnings
import logging

import obspy
import numpy as np

import swprocess
from testtools import unittest, TestCase, get_path

logger = logging.getLogger("swprocess")
logger.setLevel(level=logging.CRITICAL)


class Test_ActiveTimeSeries(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = get_path(__file__)
        cls.wghs_path = cls.path / "../examples/masw/data/wghs/"

    def test_check(self):
        good_nstacks = 1
        good_delay = 0

        bad_type_nstacks = [["nstacks"]]
        bad_type_delays = [["delay"]]
        for nstack, delay in zip(bad_type_nstacks, bad_type_delays):
            self.assertRaises(TypeError,
                              swprocess.ActiveTimeSeries._check_input,
                              nstacks=good_nstacks,
                              delay=delay)
            self.assertRaises(TypeError,
                              swprocess.ActiveTimeSeries._check_input,
                              nstacks=nstack,
                              delay=good_delay)

        bad_value_nstacks = [-1, 0, "nstacks"]
        bad_value_delays = [0.1, 0.1, "delay"]
        for nstack, delay in zip(bad_value_nstacks, bad_value_delays):
            self.assertRaises(ValueError,
                              swprocess.ActiveTimeSeries._check_input,
                              nstacks=good_nstacks,
                              delay=delay)
            self.assertRaises(ValueError,
                              swprocess.ActiveTimeSeries._check_input,
                              nstacks=nstack,
                              delay=good_delay)

    def test_init(self):
        dt = 1
        amplitude = [0., 1, 0, -1]
        returned = swprocess.ActiveTimeSeries(amplitude, dt)
        self.assertArrayEqual(np.array(amplitude), returned.amplitude)
        self.assertEqual(dt, returned.dt)

    def test_time(self):
        dt = 0.5
        amplitude = [0., 1, 2, 3]
        obj = swprocess.ActiveTimeSeries(amplitude, dt)

        expected = np.array([0., 0.5, 1., 1.5])
        returned = obj.time
        self.assertArrayEqual(expected, returned)

        # With pre-event delay
        dt = 0.5
        amplitude = [-1., 0, 1, 0, -1]
        true_time = [-0.5, 0., 0.5, 1., 1.5]
        test = swprocess.ActiveTimeSeries(amplitude, dt, delay=-0.5)
        self.assertListEqual(test.time.tolist(), true_time)

    def test_from_trace_seg2(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read(str(self.wghs_path / "1.dat"))[0]
        returned = swprocess.ActiveTimeSeries.from_trace_seg2(trace)
        self.assertArrayEqual(trace.data, returned.amplitude)
        self.assertEqual(trace.stats.delta, returned.dt)
        self.assertEqual(int(trace.stats.seg2.STACK), returned._nstacks)
        self.assertEqual(float(trace.stats.seg2.DELAY), returned.delay)

    def test_from_trace(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read(str(self.wghs_path / "1.dat"))[0]
        tseries = swprocess.ActiveTimeSeries.from_trace(trace, delay=-0.5)
        self.assertArrayEqual(tseries.amplitude, trace.data)
        self.assertEqual(trace.stats.delta, tseries.dt)
        self.assertEqual(int(trace.stats.seg2.STACK), tseries._nstacks)
        self.assertEqual(float(trace.stats.seg2.DELAY), tseries.delay)

    def test_stack_append(self):
        # Append trace with 1 stack.
        tseries = swprocess.ActiveTimeSeries(amplitude=[0, 1, 0, -1], dt=1)
        nseries = swprocess.ActiveTimeSeries(amplitude=[0, -1, 0, 1], dt=1)
        tseries.stack_append(nseries)
        expected = np.array([0, 0, 0, 0])
        returned = tseries.amplitude
        self.assertArrayEqual(expected, returned)

        # Append trace with 5 stacks.
        tseries = swprocess.ActiveTimeSeries(amplitude=[10], dt=1,
                                             nstacks=3)
        nseries = swprocess.ActiveTimeSeries(amplitude=[5], dt=1,
                                             nstacks=5)
        tseries.stack_append(nseries)
        returned = tseries.amplitude
        expected = (10*3 + 5*5)/(3+5)
        self.assertEqual(expected, returned)

        # Bad stack
        tseries = swprocess.ActiveTimeSeries([1, 2, 3], dt=1)
        nseries = swprocess.ActiveTimeSeries([0, 0, 0], dt=2)
        self.assertRaises(ValueError, tseries.stack_append, nseries)

    def test_zero_pad(self):
        dt = 0.01
        thist = swprocess.ActiveTimeSeries(amplitude=np.arange(0, 2, dt),
                                           dt=dt)
        self.assertEqual(0.5, thist.df)
        self.assertEqual(200, thist.nsamples)

        # Request df = 1*df
        thist.zero_pad(df=thist.df)
        self.assertEqual(0.5, thist._df)
        self.assertEqual(0.5, thist.df)
        self.assertEqual(200, thist.nsamples)
        self.assertEqual(1, thist.multiple)

        # Request df = 2*df
        thist.zero_pad(df=1)
        self.assertEqual(0.5, thist._df)
        self.assertEqual(1, thist.df)
        self.assertEqual(200, thist.nsamples)
        self.assertEqual(2, thist.multiple)

        # Request df = 0.5*df
        thist.zero_pad(df=0.5)
        self.assertEqual(0.5, thist._df)
        self.assertEqual(0.5, thist.df)
        self.assertEqual(200, thist.nsamples)
        self.assertEqual(1, thist.multiple)

        # Request df = 0.1*df
        thist.zero_pad(df=0.05)
        self.assertEqual(0.05, thist._df)
        self.assertEqual(0.05, thist.df)
        self.assertEqual(2000, thist.nsamples)
        self.assertEqual(1, thist.multiple)

        # Request df = 20*df
        thist.zero_pad(df=1)
        self.assertEqual(0.05, thist._df)
        self.assertEqual(1, thist.df)
        self.assertEqual(2000, thist.nsamples)
        self.assertEqual(20, thist.multiple)

        # Request df = 1.5 -> Approximate solution
        desired_df = 1.5
        thist.zero_pad(df=desired_df)
        approx_n = int(round(1/(desired_df*dt)))
        approx_df = 1/(approx_n*dt)
        pad = approx_n - (2000 % approx_n)
        multiple = (2000 + pad) / approx_n

        self.assertAlmostEqual(1/((2000+pad) * dt), thist._df)
        self.assertAlmostEqual(approx_df, thist.df)
        self.assertEqual(2000 + pad, thist.nsamples)
        self.assertEqual(multiple, thist.multiple)

        # Invalid df
        self.assertRaises(ValueError, thist.zero_pad, df=0)

    def test_trim(self):
        # First Example: no delay
        thist = swprocess.ActiveTimeSeries(amplitude=np.arange(0, 2, 0.001),
                                           dt=0.001)
        self.assertEqual(len(thist.amplitude), 2000)
        thist.trim(0, 1)
        self.assertEqual(len(thist.amplitude), 1001)
        self.assertEqual(thist.nsamples, 1001)

        # With pre-trigger delay
        thist = swprocess.ActiveTimeSeries(amplitude=np.arange(0, 2, 0.001),
                                           dt=0.001,
                                           delay=-.5)

        # Remove part of pre-trigger
        thist.trim(-0.25, 0.25)
        self.assertEqual(501, thist.nsamples)
        self.assertEqual(-0.25, thist.delay)

        # Remove all of pre-trigger
        thist.trim(0, 0.2)
        self.assertEqual(201, thist.nsamples)
        self.assertEqual(0, thist.delay)

        # Second Example: with delay
        dt = 0.001
        thist = swprocess.ActiveTimeSeries(amplitude=np.arange(0, 2, dt),
                                           dt=dt,
                                           delay=-.5)

        # Remove part of pre-trigger
        thist.trim(-0.25, 0.25)
        self.assertEqual(thist.nsamples, 501)
        self.assertEqual(thist.delay, -0.25)
        self.assertAlmostEqual(min(thist.time), -0.25, delta=dt)
        self.assertAlmostEqual(max(thist.time), +0.25, delta=dt)

        # Remove all of pre-trigger
        thist.trim(0, 0.25)
        self.assertEqual(thist.nsamples, 251)
        self.assertAlmostEqual(thist.delay, 0.001, delta=dt)
        self.assertAlmostEqual(min(thist.time), 0, delta=dt)
        self.assertAlmostEqual(max(thist.time), 0.25, delta=dt)

        # Bad: Trim before trigger
        self.assertRaises(IndexError, thist.trim, -1, 0.1)

    def test_eq(self):
        thist_a = swprocess.ActiveTimeSeries([0, 1, 0, -1, 0], dt=1)
        thist_b = swprocess.ActiveTimeSeries([0, 1, 0, -1, 0], dt=1)
        thist_c = swprocess.ActiveTimeSeries([0, 0, 0, +1, 0], dt=1)
        thist_d = swprocess.ActiveTimeSeries([0, 1, 0, -1, 0], dt=0.5)
        thist_e = swprocess.ActiveTimeSeries([0, 1, 0, -1, 0], dt=1, delay=-1)
        thist_f = swprocess.ActiveTimeSeries([0, 1, 0, -1, 0], dt=1, nstacks=2)
        thist_g = np.array([0, 1, 0, -1, 0])

        self.assertEqual(thist_a, thist_b)
        self.assertNotEqual(thist_a, thist_c)
        self.assertNotEqual(thist_a, thist_d)
        self.assertNotEqual(thist_a, thist_e)
        self.assertNotEqual(thist_a, thist_f)
        self.assertNotEqual(thist_a, thist_g)

        self.assertTrue(thist_a._is_similar(thist_b))
        self.assertTrue(thist_a._is_similar(thist_c))
        self.assertFalse(thist_a._is_similar(thist_d))
        self.assertFalse(thist_a._is_similar(thist_e))
        self.assertFalse(thist_a._is_similar(thist_f))
        self.assertFalse(thist_a._is_similar(thist_g))

    def test_crosscorr(self):
        # TODO (jpv): Timeseries of length 1 get reduced to float.
        # ampa = [0., 1, 0]
        # ampb = [1.]
        # dt = 1
        # tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        # tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        # correlate = swprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        # self.assertListEqual([0., 1, 0], correlate.tolist())

        # ampa = [0, 0, 1, 0]
        # ampb = [1]
        # dt = 1
        # tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        # tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        # correlate = swprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        # self.assertListEqual([0, 0, 1, 0], correlate.tolist())

        ampa = [0, 0, 1, 0]
        ampb = [1, 0]
        dt = 1
        tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        correlate = swprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        self.assertListEqual([0, 0, 0, 1, 0], correlate.tolist())

        ampa = [0, 0, 1, 0]
        ampb = [0, 1, 0]
        dt = 1
        tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        correlate = swprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        self.assertListEqual([0, 0, 0, 1, 0, 0], correlate.tolist())

        ampa = [0, 0, -1, 0]
        ampb = [0, -1, 0, 0]
        dt = 1
        tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        correlate = swprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        self.assertListEqual([0, 0, 0, 0, 1, 0, 0], correlate.tolist())

        # Bad example
        self.assertRaises(ValueError, swprocess.ActiveTimeSeries.crosscorr,
                          swprocess.ActiveTimeSeries([1, 2], dt=1),
                          swprocess.ActiveTimeSeries([1, 2], dt=2))

    def test_crosscorr_shift(self):
        # Simple Pulse
        ampa = [0, 0, 1, 0]
        ampb = [0, 1, 0, 0]
        dt = 1
        tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        shifts, shifted = swprocess.ActiveTimeSeries.crosscorr_shift(tseries_a,
                                                                     tseries_b)
        self.assertEqual(1, shifts)
        self.assertListEqual(ampa, shifted.tolist())

        # Simple Pulse
        ampa = [0, 0, 2, 3, 4, 5, 0, 2, 3, 0]
        ampb = [0, 2, 3, 4, 5, 0, 0, 0, 0, 0]
        dt = 1
        tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        shifts, shifted = swprocess.ActiveTimeSeries.crosscorr_shift(tseries_a,
                                                                     tseries_b)
        self.assertEqual(1, shifts)
        self.assertListEqual([0, 0, 2, 3, 4, 5, 0, 0, 0, 0], shifted.tolist())

        # Sinusoidal Pulse
        ampa = [0, -1, 0, 1, 0, -1, 0, 0]
        ampb = [0, 0, 0, -1, 0, 1, 0, 0]
        dt = 0.1
        tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        shifts, shifted = swprocess.ActiveTimeSeries.crosscorr_shift(tseries_a,
                                                                     tseries_b)
        self.assertEqual(-2, shifts)
        self.assertListEqual([0, -1, 0, 1, 0, 0, 0, 0], shifted.tolist())

        # Sinusoid
        dt = 0.01
        time = np.arange(0, 2, dt)
        f = 5
        ampa = np.concatenate((np.zeros(20), np.sin(2*np.pi*f*time)))
        ampb = np.concatenate((np.sin(2*np.pi*f*time), np.zeros(20)))
        tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        shifts, shifted = swprocess.ActiveTimeSeries.crosscorr_shift(tseries_a,
                                                                     tseries_b)
        self.assertEqual(20, shifts)
        self.assertArrayEqual(ampa, shifted)

        # No shift
        dt = 0.1
        f = 10
        time = np.arange(0, 5, dt)
        amplitude = np.concatenate((np.zeros(15), np.sin(2*np.pi*f*time)))
        tseries_a = swprocess.ActiveTimeSeries(amplitude, dt)
        tseries_b = swprocess.ActiveTimeSeries.from_activetimeseries(tseries_a)
        shifts, shifted = swprocess.ActiveTimeSeries.crosscorr_shift(tseries_a,
                                                                     tseries_b)
        self.assertEqual(0, shifts)
        self.assertArrayEqual(tseries_a.amplitude, shifted)

    def test_from_cross_stack(self):
        # Simple pulse
        ampa = [0, 0, 1, 0]
        ampb = [0, 1, 0, 0]
        dt = 1
        tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        stacked = swprocess.ActiveTimeSeries.from_cross_stack(tseries_a,
                                                              tseries_b)
        self.assertListEqual(ampa, stacked.amplitude.tolist())

        # Simple Sinusoid
        ampa = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        ampb = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
        dt = 1
        tseries_a = swprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = swprocess.ActiveTimeSeries(ampb, dt)
        stacked = swprocess.ActiveTimeSeries.from_cross_stack(tseries_a,
                                                              tseries_b)
        self.assertListEqual(ampa, stacked.amplitude.tolist())

    def test_properties(self):
        amplitude = [0, 1, 0, 1, 0, 1, 0, 1]
        dt = 0.5
        tseries = swprocess.ActiveTimeSeries(amplitude,
                                             dt,
                                             nstacks=3,
                                             delay=-1)

        # nstacks
        self.assertEqual(3, tseries.nstacks)

        # n_stacks (Deprecated)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(3, tseries.n_stacks)

    def test_str_and_repr(self):
        amplitude = [0, 1, 2, 3, 4, 5]
        dt = 0.25
        delay = -0.5
        nstacks = 5
        tseries = swprocess.ActiveTimeSeries(amplitude,
                                             dt,
                                             nstacks=nstacks,
                                             delay=delay)

        # __repr__
        expected = f"ActiveTimeSeries(dt={tseries.dt}, amplitude={tseries.amplitude}, nstacks={tseries._nstacks}, delay={tseries._delay})"
        returned = tseries.__repr__()
        self.assertEqual(expected, returned)

        # __str__
        expected = f"ActiveTimeSeries with:\n\tdt={tseries.dt}\n\tnsamples={tseries.nsamples}\n\tnstacks={tseries._nstacks}\n\tdelay={tseries._delay}"
        returned = tseries.__str__()
        self.assertEqual(expected, returned)


if __name__ == '__main__':
    unittest.main()
