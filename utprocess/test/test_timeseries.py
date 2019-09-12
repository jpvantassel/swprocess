"""Tests for TimeSeries class. """
import unittest
import utprocess
import obspy
import numpy as np
import warnings
import logging
logging.basicConfig(level=logging.DEBUG)


class TestTimeSeries(unittest.TestCase):

    def test_check(self):
        for value in [True, "values", 1, 1.57]:
            self.assertRaises(TypeError,
                              utprocess.TimeSeries.check_input,
                              name="blah",
                              values=value)
        for value in [[1, 2, 3], (1, 2, 3)]:
            value = utprocess.TimeSeries.check_input(name="blah", values=value)
            self.assertTrue(isinstance(value, np.ndarray))

        for value in [[[1, 2], [3, 4]], ((1, 2), (3, 4)), np.array([[1, 2], [3, 4]])]:
            self.assertRaises(TypeError,
                              utprocess.TimeSeries.check_input,
                              name="blah",
                              values=value)

    def test_init(self):
        dt = 1
        amp = [0, 1, 0, -1]
        test = utprocess.TimeSeries(amp, dt)
        self.assertListEqual(amp, test.amp.tolist())
        self.assertEqual(dt, test.dt)

        amp = np.array(amp)
        test = utprocess.TimeSeries(amp, dt)
        self.assertListEqual(amp.tolist(), test.amp.tolist())

    def test_from_trace_seg2(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read("test/data/vuws/1.dat")[0]
        tseries = utprocess.TimeSeries.from_trace_seg2(trace)
        self.assertListEqual(tseries.amp.tolist(), trace.data.tolist())
        self.assertEqual(tseries.dt, trace.stats.delta)
        self.assertEqual(tseries._nstack, int(trace.stats.seg2.STACK))
        self.assertEqual(tseries.delay, float(trace.stats.seg2.DELAY))

    def test_from_trace(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read("test/data/vuws/1.dat")[0]
        tseries = utprocess.TimeSeries.from_trace(trace, delay=-0.5)
        self.assertListEqual(tseries.amp.tolist(), trace.data.tolist())
        self.assertEqual(tseries.dt, trace.stats.delta)
        self.assertEqual(tseries._nstack, int(trace.stats.seg2.STACK))
        self.assertEqual(tseries.delay, float(trace.stats.seg2.DELAY))

        # TODO: add miniseed

    def test_stack_append(self):
        # Append trace with 1 stack.
        test = utprocess.TimeSeries(amplitude=[0, 1, 0, -1], dt=1)
        test.stack_append(amplitude=[0, -1, 0, 1], dt=1)
        self.assertListEqual(test.amp.tolist(), [0, 0, 0, 0])

        # Append trace with 5 stacks.
        test = utprocess.TimeSeries(amplitude=[10], dt=1)
        test.stack_append(amplitude=[5], dt=1, nstacks=5)
        true = (10*1 + 5*5)/(5+1)
        self.assertEqual(true, test.amp)

    def test_zero_pad(self):
        thist = utprocess.TimeSeries(amplitude=list(np.arange(0, 2, 0.01)),
                                     dt=0.01)
        self.assertEqual(len(thist.amp), 200)
        thist.zero_pad(df=0.1)
        self.assertEqual(len(thist.amp), 1000)
        thist.zero_pad(df=0.5)
        self.assertEqual(len(thist.amp)/thist.multiple, 1/(0.01*0.5))

        thist = utprocess.TimeSeries(amplitude=list(np.arange(0, 2, 0.02)),
                                     dt=0.02)
        self.assertEqual(len(thist.amp), 100)
        thist.zero_pad(df=1)
        self.assertEqual(len(thist.amp), 200)
        self.assertEqual(thist.multiple, 4)
        self.assertEqual(len(thist.amp)/thist.multiple, 1/(0.02*1))

    def test_trim(self):
        # Standard
        thist = utprocess.TimeSeries(amplitude=list(np.arange(0, 2, 0.01)),
                                     dt=0.01)
        self.assertEqual(len(thist.amp), 200)
        thist.trim(0, 1)
        self.assertEqual(len(thist.amp), 100)
        self.assertEqual(thist.nsamples, 100)

        # With pre-trigger delay
        thist = utprocess.TimeSeries(amplitude=list(np.arange(0, 2, 0.01)),
                                     dt=0.01,
                                     delay=-.5)
        # Remove part of pre-trigger
        thist.trim(-0.25, 0.25)
        self.assertEqual(thist.nsamples, 50)
        self.assertEqual(thist.delay, -0.25)
        # Remove all of pre-trigger
        thist.trim(0, 0.2)
        self.assertEqual(thist.nsamples, 20)
        self.assertEqual(thist.delay, 0)



if __name__ == '__main__':
    unittest.main()
