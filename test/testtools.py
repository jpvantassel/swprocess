# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (jvantassel@utexas.edu)
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

"""Testing tools."""

import unittest

import numpy as np

def get_full_path(path):
    if path.count("/") > 1:
        file_name = path.split(r"/")[-1]
        full_path = path[:-len(file_name)]
    else:
        file_name = path.split(r"\\")[-1]
        full_path = path[:-len(file_name)]
    return full_path

class TestCase(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, **kwargs):
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, **kwargs)

    def assertArrayEqual(self, array1, array2):
        try:
            self.assertTrue(np.equal(array1, array2, casting='safe').all())
        except AssertionError as e:
            msg = f"\nExpected:\n{array1}\nReturned:\n{array2})"
            raise AssertionError(msg) from e

    def assertArrayAlmostEqual(self, array1, array2, **kwargs):
        if kwargs.get("places", False):
            kwargs["atol"] = 1/(10**kwargs["places"])
            del kwargs["places"]

        try:
            self.assertTrue(np.allclose(array1, array2, **kwargs))
        except AssertionError as e:
            msg = f"\nExpected:\n{array1}\nReturned:\n{array2}"
            raise AssertionError(msg) from e