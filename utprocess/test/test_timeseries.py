"""Tests for TimeSeries class. """
import unittest
import utprocess
import obspy
import numpy as np

class TestTimeSeries(unittest.TestCase):

    def test_init(self):
        dt = 1
        amp = [0, 1, 0, -1]
        test = utprocess.TimeSeries(amp, dt)
        self.assertListEqual(amp, test.amp.tolist())
        self.assertEqual(dt, test.dt)
        
        amp = np.array(amp)
        test = utprocess.TimeSeries(amp, dt)
        self.assertListEqual(amp.tolist(), test.amp.tolist())

    def test_from_seg2(self):
        pass

    def test_from_miniseed(self):
        pass

    def test_stack_append_1trace(self):
        amp = [0,1,0,-1]
        dt = 1
        test = utprocess.TimeSeries(amp, dt)
        new_amp = [0,-1,0,1]
        dt = 1
        test.stack_append(new_amp, dt)
        self.assertListEqual(test.amp.tolist(), [0,0,0,0])

    def test_stack_5trace(self):
        amp = [10]
        dt = 1
        test = utprocess.TimeSeries(amp, dt)
        new_amp = [5]
        dt = 1
        nstacks = 5
        test.stack_append(new_amp, dt, nstacks=nstacks)
        true = (10*1 + 5*5)/(5+1)
        self.assertEqual(true, test.amp)

    # def test_

if __name__ == '__main__':
    unittest.main()
