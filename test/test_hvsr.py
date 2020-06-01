"""Tests for HorVertSpecRatio class."""

import swprocess
import json
from testtools import unittest, TestCase, get_full_path
import logging
logging.basicConfig(level=logging.CRITICAL)


class Test_Hvsr(TestCase):

    def setUp(self):
        self.fpath = get_full_path(__file__)

    def test_from_geopsy_file(self):
        fname = self.fpath + "data/myanmar/test_ZM_STN01.hv"
        test = swprocess.Hvsr.from_geopsy_file(fname=fname,
                                                           identifier="TaDa")
        frq = [[0.1, 0.101224, 0.102462, 50]]
        amp = [[4.26219, 4.24461, 4.20394, 0.723993]]
        idn = "TaDa"

        self.assertListEqual(test.frq, frq)
        self.assertListEqual(test.amp, amp)
        self.assertTrue(test.idn, idn)

    def test_from_geopsy_folder(self):
        dirname = self.fpath + "data/myanmar/test_dir"
        hv = swprocess.Hvsr.from_geopsy_folder(dirname=dirname,
                                                           identifier="TADA")
        with open(self.fpath + "data/myanmar/test_dir_data.json", "r") as f:
            known = json.load(f)

        for test_frq, test_amp, known in zip(hv.frq, hv.amp, known):
            self.assertListEqual(test_frq, known["frequency"])
            self.assertListEqual(test_amp, known["amplitude"])


if __name__ == "__main__":
    unittest.main()
