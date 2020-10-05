"""Test for utils module."""

import os
import shutil

from testtools import get_full_path, unittest, TestCase
from swprocess import utils

class Test_Utils(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_extract_mseed(self):
        startend_fname = self.full_path + "/data/utils/extract_startandend.xlsx"
        utils.extract_mseed(startend_fname=startend_fname,
                            network="NW",
                            data_dir=self.full_path + "data/utils/data_dir/",
                            output_dir=self.full_path + "data/utils/")

        # Assert directories exist.
        arrays = ["ex_array_0", "ex_array_1"]
        for array in arrays:
            self.assertTrue(os.path.isdir(self.full_path + "data/utils/" + array))

        # # Open files and assert connets are correct.
        # files = []
        # for file in files:
        #     trace = obspy.read(file)



        # # Clean up.
        # for array in arrays:
        #     shutil.rmtree(self.full_path + "data/utils/" + array)



if __name__ == "__main__":
    unittest.main()
