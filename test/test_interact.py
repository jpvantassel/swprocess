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
        _, ax = plt.subplots()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xs, ys = ginput_session(ax=ax, npts=2, initial_adjustment=True,
                                    ask_to_continue=True)

        self.assertListEqual([0], xs)
        self.assertListEqual([1], ys)


if __name__ == "__main__":
    unittest.main()
