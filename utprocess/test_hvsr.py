"""Tests for HorVertSpecRatio class."""

import utprocess
import json
import unittest
import logging
logging.basicConfig(level=logging.DEBUG)


class Test(unittest.TestCase):

    def test_from_geopsy_file(self):
        fname = "test/data/myanmar/test_ZM_STN01.hv"
        test = utprocess.HorVertSpecRatio.from_geopsy_file(fname=fname,
                                                           identifier="TaDa")
        frq = [[0.1, 0.101224, 0.102462, 50]]
        amp = [[4.26219, 4.24461, 4.20394, 0.723993]]
        idn = "TaDa"

        self.assertListEqual(test.frq, frq)
        self.assertListEqual(test.amp, amp)
        self.assertTrue(test.idn, idn)

    def test_from_geopsy_folder(self):
        dirname = "test/data/myanmar/test_dir"
        hv = utprocess.HorVertSpecRatio.from_geopsy_folder(dirname=dirname,
                                                           identifier="TADA")
        with open("test/data/myanmar/test_dir_data.json", "r") as f:
            known = json.load(f)

        for test_frq, test_amp, known in zip(hv.frq, hv.amp, known):
            self.assertListEqual(test_frq, known["frequency"])
            self.assertListEqual(test_amp, known["amplitude"])


if __name__ == "__main__":
    unittest.main()
