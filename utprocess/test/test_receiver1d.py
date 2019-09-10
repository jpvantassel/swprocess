"""Tests for Receiver1D class. """

import unittest
import utprocess
import obspy


class TestReceiver1D(unittest.TestCase):

    def test_init(self):
        pass

    def test_from_trace(self):
        stream = obspy.read("test/data/vuws/1.dat")
        known = stream[0]
        test = utprocess.Receiver1D.from_trace(known)
        print(type(test))
        self.assertListEqual(known.data.tolist(),
                             test.amp.tolist())


if __name__ == '__main__':
    unittest.main()
