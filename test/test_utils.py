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

"""Test for utils module."""

import os
import shutil

import obspy
import numpy as np

from testtools import get_path, unittest, TestCase
from swprocess import utils


class Test_Utils(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = get_path(__file__)

    def test_extract_mseed(self):
        startend_fname = self.path / "data/utils/extract_startandend.csv"
        utils.extract_mseed(startend_fname=startend_fname,
                            network="NW",
                            data_dir=self.path / "data/utils/data_dir/",
                            output_dir=self.path / "data/utils/")

        # Assert directories exist.
        arrays = ["ex_array_0", "ex_array_1", "ex_array_2"]
        for array in arrays:
            self.assertTrue(os.path.isdir(self.path / f"data/utils/{array}"))

        # Open files and assert connets are correct.
        files = [self.path / f"data/utils/{array}/NW.STN01.{array[3:]}.mseed" for array in arrays]
        start_seconds = [0, 4*3600, 23*3600 + 30*60]
        duration_seconds = [1800, 7200, 3600]
        for _file, start, duration in zip(files, start_seconds, duration_seconds):
            returned = obspy.read(str(_file))[0].data
            expected = np.arange(start, start+duration+1)
            self.assertArrayEqual(expected, returned)

        # Clean up.
        for array in arrays:
            shutil.rmtree(self.path / "data/utils/" / array)

        # Bad start and end time
        startend_fname_bad = self.path / "data/utils/extract_startandend_bad.csv"
        self.assertRaises(ValueError, utils.extract_mseed,
                          startend_fname=startend_fname_bad, network="NW",
                          data_dir=self.path / "data/utils/data_dir/",
                          output_dir=self.path / "data/utils/")

        # Bad file type
        startend_fname_csv = self.path / "data/utils/extract_startandend.xlsx"
        self.assertRaises(NotImplementedError, utils.extract_mseed,
                          startend_fname=startend_fname_csv, network="NW",
                          data_dir=self.path / "data/utils/data_dir/",
                          output_dir=self.path / "data/utils/")



if __name__ == "__main__":
    unittest.main()
