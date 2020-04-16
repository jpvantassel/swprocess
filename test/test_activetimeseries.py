"""Tests for ActiveTimeSeries class. """

import matplotlib.pyplot as plt
from testtools import unittest, TestCase, get_full_path
import utprocess
import obspy
import numpy as np
import warnings
import logging
logging.basicConfig(level=logging.WARNING)


class Test_ActiveTimeSeries(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_check(self):
        for value in ["values", ["a", "b", "c"]]:
            self.assertRaises(TypeError,
                              utprocess.ActiveTimeSeries._check_input,
                              name="blah",
                              values=value)
        for value in [[1, 2, 3], (1, 2, 3)]:
            value = utprocess.ActiveTimeSeries._check_input(name="blah",
                                                            values=value)
            self.assertTrue(isinstance(value, np.ndarray))

    def test_init(self):
        dt = 1
        amp = [0, 1, 0, -1]
        returned = utprocess.ActiveTimeSeries(amp, dt)
        self.assertArrayEqual(np.array(amp), returned.amp)
        self.assertEqual(dt, returned.dt)

    def test_time(self):
        dt = 0.5
        amp = [0, 1, 2, 3]
        obj = utprocess.ActiveTimeSeries(amp, dt)

        expected = np.array([0., 0.5, 1., 1.5])
        returned = obj.time
        self.assertArrayEqual(expected, returned)

        # With pre-event delay
        dt = 0.5
        amp = [-1, 0, 1, 0, -1]
        true_time = [-0.5, 0., 0.5, 1., 1.5]
        test = utprocess.ActiveTimeSeries(amp, dt, delay=-0.5)
        self.assertListEqual(test.time.tolist(), true_time)

    def test_from_trace_seg2(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read(self.full_path + "data/vuws/1.dat")[0]
        returned = utprocess.ActiveTimeSeries.from_trace_seg2(trace)
        self.assertArrayEqual(trace.data, returned.amp)
        self.assertEqual(trace.stats.delta, returned.dt)
        self.assertEqual(int(trace.stats.seg2.STACK), returned._nstacks)
        self.assertEqual(float(trace.stats.seg2.DELAY), returned.delay)

    def test_from_trace(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read(self.full_path + "data/vuws/1.dat")[0]
        tseries = utprocess.ActiveTimeSeries.from_trace(trace, delay=-0.5)
        self.assertListEqual(tseries.amp.tolist(), trace.data.tolist())
        self.assertEqual(trace.stats.delta, tseries.dt)
        self.assertEqual(int(trace.stats.seg2.STACK), tseries._nstacks)
        self.assertEqual(float(trace.stats.seg2.DELAY), tseries.delay)

    def test_stack_append(self):
        # Append trace with 1 stack.
        tseries = utprocess.ActiveTimeSeries(amplitude=[0, 1, 0, -1], dt=1)
        tseries.stack_append(amplitude=[0, -1, 0, 1], dt=1)
        returned = tseries.amp
        expected = np.array([0, 0, 0, 0])
        self.assertArrayEqual(expected, returned)

        # Append trace with 5 stacks.
        tseries = utprocess.ActiveTimeSeries(amplitude=[10], dt=1)
        tseries.stack_append(amplitude=[5], dt=1, n_stacks=5)
        returned = tseries.amp
        expected = (10*1 + 5*5)/(5+1)
        self.assertEqual(expected, returned)

    # def test_zero_pad(self):
    #     thist = utprocess.ActiveTimeSeries(amplitude=list(np.arange(0, 2, 0.01)),
    #                                        dt=0.01)
    #     self.assertEqual(len(thist.amp), 200)
    #     thist.zero_pad(df=0.1)
    #     self.assertEqual(len(thist.amp), 1000)
    #     thist.zero_pad(df=0.5)
    #     self.assertEqual(len(thist.amp)/thist.multiple, 1/(0.01*0.5))

    #     thist = utprocess.ActiveTimeSeries(amplitude=list(np.arange(0, 2, 0.02)),
    #                                        dt=0.02)
    #     self.assertEqual(len(thist.amp), 100)
    #     thist.zero_pad(df=1)
    #     self.assertEqual(len(thist.amp), 200)
    #     self.assertEqual(thist.multiple, 4)
    #     self.assertEqual(len(thist.amp)/thist.multiple, 1/(0.02*1))

    def test_trim(self):
        # Standard
        thist = utprocess.ActiveTimeSeries(amplitude=np.arange(0, 2, 0.001),
                                           dt=0.001)
        self.assertEqual(len(thist.amp), 2000)
        thist.trim(0, 1)
        self.assertEqual(len(thist.amp), 1001)
        self.assertEqual(thist.n_samples, 1001)

        # With pre-trigger delay
        thist = utprocess.ActiveTimeSeries(amplitude=np.arange(0, 2, 0.001),
                                           dt=0.001,
                                           delay=-.5)
        # Remove part of pre-trigger
        thist.trim(-0.25, 0.25)
        self.assertEqual(501, thist.n_samples)
        self.assertEqual(-0.25, thist.delay)
        # Remove all of pre-trigger
        thist.trim(0, 0.2)
        self.assertEqual(201, thist.n_samples)
        self.assertEqual(0, thist.delay)

        # With pre-trigger delay
        dt = 0.001
        thist = utprocess.ActiveTimeSeries(amplitude=np.arange(0, 2, dt),
                                           dt=dt,
                                           delay=-.5)
        # Remove part of pre-trigger
        thist.trim(-0.25, 0.25)
        self.assertEqual(thist.n_samples, 501)
        self.assertEqual(thist.delay, -0.25)
        self.assertAlmostEqual(min(thist.time), -0.25, delta=dt)
        self.assertAlmostEqual(max(thist.time), +0.25, delta=dt)
        # Remove all of pre-trigger
        thist.trim(0, 0.25)
        self.assertEqual(thist.n_samples, 251)
        self.assertAlmostEqual(thist.delay, 0.001, delta=dt)
        self.assertAlmostEqual(min(thist.time), 0, delta=dt)
        self.assertAlmostEqual(max(thist.time), 0.25, delta=dt)

    def test_crosscorr(self):
        ampa = [0, 1, 0]
        ampb = [1]
        dt = 1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        correlate = utprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        self.assertListEqual(correlate.tolist(), [0, 1, 0])

        ampa = [0, 0, 1, 0]
        ampb = [1]
        dt = 1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        correlate = utprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        self.assertListEqual(correlate.tolist(), [0, 0, 1, 0])

        ampa = [0, 0, 1, 0]
        ampb = [1, 0]
        dt = 1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        correlate = utprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        self.assertListEqual(correlate.tolist(), [0, 0, 0, 1, 0])

        ampa = [0, 0, 1, 0]
        ampb = [0, 1, 0]
        dt = 1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        correlate = utprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        self.assertListEqual(correlate.tolist(), [0, 0, 0, 1, 0, 0])

        ampa = [0, 0, -1, 0]
        ampb = [0, -1, 0, 0]
        dt = 1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        correlate = utprocess.ActiveTimeSeries.crosscorr(tseries_a, tseries_b)
        self.assertListEqual(correlate.tolist(), [0, 0, 0, 0, 1, 0, 0])

    def test_crosscorr_shift(self):
        # Simple Pulse
        ampa = [0, 0, 1, 0]
        ampb = [0, 1, 0, 0]
        dt = 1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        shifted = utprocess.ActiveTimeSeries.crosscorr_shift(
            tseries_a, tseries_b)
        self.assertListEqual(shifted.tolist(), ampa)

        # Simple Pulse
        ampa = [0, 0, 2, 3, 4, 5, 0, 2, 3, 0]
        ampb = [0, 2, 3, 4, 5, 0, 0, 0, 0, 0]
        dt = 1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        shifted = utprocess.ActiveTimeSeries.crosscorr_shift(
            tseries_a, tseries_b)
        self.assertListEqual(shifted.tolist(), [0, 0, 2, 3, 4, 5, 0, 0, 0, 0])

        # Sinusoidal Pulse
        ampa = [0, -1, 0, 1, 0, -1, 0, 0]
        ampb = [0, 0, 0, -1, 0, 1, 0, 0]
        dt = 0.1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        shifted = utprocess.ActiveTimeSeries.crosscorr_shift(
            tseries_a, tseries_b)
        self.assertListEqual(shifted.tolist(), [0, -1, 0, 1, 0, 0, 0, 0])

        # Sinusoid
        dt = 0.01
        time = np.arange(0, 2, dt)
        f = 5
        ampa = np.concatenate((np.zeros(20), np.sin(2*np.pi*f*time)))
        ampb = np.concatenate((np.sin(2*np.pi*f*time), np.zeros(20)))
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        shifted = utprocess.ActiveTimeSeries.crosscorr_shift(
            tseries_a, tseries_b)
        self.assertListEqual(shifted.tolist(), ampa.tolist())

    def test_cross_stack(self):
        # Simple pulse
        ampa = [0, 0, 1, 0]
        ampb = [0, 1, 0, 0]
        dt = 1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        stacked = utprocess.ActiveTimeSeries.from_cross_stack(
            tseries_a, tseries_b)
        self.assertListEqual(stacked.amp.tolist(), ampa)

        # Simple Sinusoid
        ampa = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        ampb = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
        dt = 1
        tseries_a = utprocess.ActiveTimeSeries(ampa, dt)
        tseries_b = utprocess.ActiveTimeSeries(ampb, dt)
        stacked = utprocess.ActiveTimeSeries.from_cross_stack(
            tseries_a, tseries_b)
        self.assertListEqual(stacked.amp.tolist(), ampa)


if __name__ == '__main__':
    unittest.main()
