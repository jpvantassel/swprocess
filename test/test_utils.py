"""Test for utils module."""

import os
import shutil

import obspy
import numpy as np

from testtools import get_full_path, unittest, TestCase
from swprocess import utils


class Test_Utils(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_extract_mseed(self):
        startend_fname = self.full_path + "data/utils/extract_startandend.csv"
        utils.extract_mseed(startend_fname=startend_fname,
                            network="NW",
                            data_dir=self.full_path + "data/utils/data_dir/",
                            output_dir=self.full_path + "data/utils/")

        # Assert directories exist.
        arrays = ["ex_array_0", "ex_array_1", "ex_array_2"]
        for array in arrays:
            self.assertTrue(os.path.isdir(
                f"{self.full_path}data/utils/{array}"))

        # Open files and assert connets are correct.
        files = [
            f"{self.full_path}data/utils/{array}/NW.STN01.{array[3:]}.mseed" for array in arrays]
        start_seconds = [0, 4*3600, 23*3600 + 30*60]
        duration_seconds = [1800, 7200, 3600]
        for _file, start, duration in zip(files, start_seconds, duration_seconds):
            returned = obspy.read(_file)[0].data
            expected = np.arange(start, start+duration+1)
            self.assertArrayEqual(expected, returned)

        # Clean up.
        for array in arrays:
            shutil.rmtree(self.full_path + "data/utils/" + array)

        # Bad start and end time
        startend_fname_bad = self.full_path + "data/utils/extract_startandend_bad.csv"
        self.assertRaises(ValueError, utils.extract_mseed,
                          startend_fname=startend_fname_bad, network="NW",
                          data_dir=self.full_path + "data/utils/data_dir/",
                          output_dir=self.full_path + "data/utils/")

        # Bad file type
        startend_fname_csv = self.full_path + "data/utils/extract_startandend.xlsx"
        self.assertRaises(NotImplementedError, utils.extract_mseed,
                          startend_fname=startend_fname_csv, network="NW",
                          data_dir=self.full_path + "data/utils/data_dir/",
                          output_dir=self.full_path + "data/utils/")



if __name__ == "__main__":
    unittest.main()
