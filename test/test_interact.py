"""Test for interact module."""

import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt

from swprocess.interact import ginput_session
from testtools import unittest, TestCase


class Test_Interact(TestCase):

    @patch('matplotlib.pyplot.ginput', return_value=[(0.5, 0.5), (0, 1)])
    @patch('matplotlib.pyplot.waitforbuttonpress', return_value=True)
    def test_ginput_session(self, input_a, input_b):
        fig, ax = plt.subplots()        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xs, ys = ginput_session(ax=ax, npts=2, initial_adjustment=True,
                                    ask_to_continue=True)

        self.assertListEqual([0], xs)
        self.assertListEqual([1], ys)


if __name__ == "__main__":
    unittest.main()
