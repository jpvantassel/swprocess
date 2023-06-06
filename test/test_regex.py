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

"""Tests for PeaksSuite class."""

import logging

from testtools import unittest, TestCase

logger = logging.getLogger("swprocess")
logger.setLevel(logging.ERROR)


class Test_PeaksSuite(TestCase):

    # TODO (jpv): Add this to the test cases.
    def test_get_peak_from_max(self):
        pass
        # txt = "20201021184000.000000 0.88707185499315710508 Rayleigh 0.0052274209500743342924 146.16012705150347983 -9.7624928074594592431 4.1036102507755423119 2.4140442228149637278e-05 1"

if __name__ == "__main__":
    unittest.main()
