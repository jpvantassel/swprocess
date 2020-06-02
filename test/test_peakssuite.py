"""Tests for PeaksSuite class."""

import json
import os
import logging

import numpy as np

import swprocess
from testtools import unittest, TestCase, get_full_path

logging.basicConfig(level=logging.WARN)


class Test_PeaksSuite(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_from_jsons(self):
        # Advanced Case: Two keyword arguements
        frequency = [100., 50, 30, 10, 5, 3]
        velocity = [100., 120, 130, 140, 145, 150]
        azi = [10., 15, 20, 35, 10, 20]
        pwr = [10., 15, 20, 35, 11, 20]

        peak_list = []
        fnames = []
        for num in range(0, 3):
            fname = f"{self.full_path}data/tmp_id{num}.json"
            peaks = swprocess.Peaks(
                frequency, velocity, str(num), azi=azi, pwr=pwr)
            peaks.to_json(fname)
            fnames.append(fname)
            peak_list.append(peaks)

        returned = swprocess.PeaksSuite.from_jsons(fnames=fnames)
        expected = swprocess.PeaksSuite.from_iter(peak_list)
        self.assertEqual(expected, returned)

        for fname in fnames:
            os.remove(fname)

    def test_from_maxs(self):
        # Check Rayleigh (2 files, 2 lines per file)
        fnames = [self.full_path +
                  f"data/mm/test_hfk_line2_{x}.max" for x in range(2)]
        returned = swprocess.PeaksSuite.from_maxs(fnames=fnames,
                                                  identifiers=["f1", "f2"],
                                                  rayleigh=True,
                                                  love=False)

        ray_f1 = {"f1": {"frequency": [20.000000000000106581, 19.282217609815102577],
                         "velocity": [1/0.0068859013683322750979, 1/0.0074117944332218188563],
                         "azi": [144.53791572557310019, 143.1083743693494057],
                         "ell": [1.0214647665926679387, 1.022287917081338593],
                         "noi": [8.9773778053801098764, 7.3044365524672443257],
                         "pwr": [2092111.2367646128405, 2074967.9391639579553],
                         "tim": [16200, 16200]
                         }}

        ray_f2 = {"f2": {"frequency": [5.0000000000000266454, 5.0000000000000266454],
                         "velocity": [1/0.0062049454559321261596, 1/0.005256207300441733711],
                         "azi": [336.6469768619966203, 77.740031601074633727],
                         "ell": [-1.6318692462645167929, -1.8969709764459561363],
                         "noi": [0, 0],
                         "pwr": [6040073199.9762010574, 17030488659.287288666],
                         "tim": [20700, 20760.040000000000873]
                         }}

        ray_dicts = [ray_f1, ray_f2]
        expected = swprocess.PeaksSuite.from_dicts(ray_dicts)

        self.assertEqual(expected, returned)

        # Check Love (2 files, 2 lines per file)
        fnames = [self.full_path +
                  f"data/mm/test_hfk_line2_{x}.max" for x in range(2)]
        returned = swprocess.PeaksSuite.from_maxs(fnames=fnames,
                                                  identifiers=["f0", "f1"],
                                                  rayleigh=False,
                                                  love=True)

        lov_f0 = {"f0": {"frequency": [20.000000000000106581, 19.282217609815102577],
                         "velocity": [1/0.0088200863560403078983, 1/0.0089530611050798007688],
                         "azi": [252.05441718438927978, 99.345595852002077208],
                         "ell": [0, 0],
                         "noi": [0, 0],
                         "pwr": [3832630.8840260845609, 4039408.6602126094513],
                         "tim": [16200, 16200]
                         }}

        lov_f1 = {"f1": {"frequency": [5.0000000000000266454, 5.0000000000000266454],
                         "velocity": [1/0.001522605285544077541, 1/0.004102623000517897911],
                         "azi": [92.499974198256211366, 174.33483725071613435],
                         "ell": [0, 0],
                         "noi": [0, 0],
                         "pwr": [232295.4010567846417, 422413.79929310601437],
                         "tim": [20700, 20760.040000000000873]
                         }}

        lov_dicts = [lov_f0, lov_f1]
        expected = swprocess.PeaksSuite.from_dicts(lov_dicts)

        self.assertEqual(expected, returned)


if __name__ == "__main__":
    unittest.main()
