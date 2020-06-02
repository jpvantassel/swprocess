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

        fnames = []
        for num in range(0, 3):
            identifier = str(num)
            write_data = {identifier: {"frequency": frequency,
                                       "velocity": velocity,
                                       "azi": azi,
                                       "pwr": pwr}}
            fnames.append(f"{self.full_path}data/temp_dict{num}.json")
            with open(fnames[-1], "w") as f:
                json.dump(write_data, f)

        my_peaks = swprocess.Peaks.from_jsons(fnames=fnames)

        for num in range(0, 3):
            self.assertListEqual(my_peaks.ext["azi"][num].tolist(), azi)
            self.assertListEqual(my_peaks.ext["pwr"][num].tolist(), pwr)

        for fname in fnames:
            os.remove(fname)

    # def test_from_max_2files(self):
    #     # Check Rayleigh (2 files, 2 lines per file)
    #     ray = swprocess.Peaks.from_maxs(fnames=[self.full_path + "data/myanmar/test_hfk_ds_short.max",
    #                                             self.full_path + "data/myanmar/test_hfk_pr_short_2.max"],
    #                                     identifiers=["test_short",
    #                                                  "test_short_2"],
    #                                     rayleigh=True,
    #                                     love=False)
    #     self.assertListEqual(ray.frq[0].tolist(), [20.000000000000106581,
    #                                                19.282217609815102577])
    #     self.assertListEqual(ray.frq[1].tolist(), [5.0000000000000266454,
    #                                                5.0000000000000266454])

    #     self.assertListEqual(ray.vel[0].tolist(), [1/0.0068859013683322750979,
    #                                                1/0.0074117944332218188563])
    #     self.assertListEqual(ray.vel[1].tolist(), [1/0.0062049454559321261596,
    #                                                1/0.005256207300441733711])

    #     self.assertListEqual(ray.ext["azi"][0].tolist(), [144.53791572557310019,
    #                                                       143.1083743693494057])
    #     self.assertListEqual(ray.ext["azi"][1].tolist(), [336.6469768619966203,
    #                                                       77.740031601074633727])

    #     self.assertListEqual(ray.ext["ell"][0].tolist(), [1.0214647665926679387,
    #                                                       1.022287917081338593])
    #     self.assertListEqual(ray.ext["ell"][1].tolist(), [-1.6318692462645167929,
    #                                                       -1.8969709764459561363])

    #     self.assertListEqual(ray.ext["noi"][0].tolist(), [8.9773778053801098764,
    #                                                       7.3044365524672443257])
    #     self.assertListEqual(ray.ext["noi"][1].tolist(), [np.inf,
    #                                                       np.inf])

    #     self.assertListEqual(ray.ext["pwr"][0].tolist(), [2092111.2367646128405,
    #                                                       2074967.9391639579553])
    #     self.assertListEqual(ray.ext["pwr"][1].tolist(), [6040073199.9762010574,
    #                                                       17030488659.287288666])

    #     # Check Love (2 files, 2 lines per file)
    #     lov = swprocess.Peaks.from_maxs(fnames=[self.full_path + "data/myanmar/test_hfk_ds_short.max",
    #                                             self.full_path + "data/myanmar/test_hfk_pr_short_2.max"],
    #                                     identifiers=["test_short",
    #                                                  "test_short_2"],
    #                                     rayleigh=False,
    #                                     love=True)
    #     self.assertListEqual(lov.frq[0].tolist(), [20.000000000000106581,
    #                                                19.282217609815102577])
    #     self.assertListEqual(lov.frq[1].tolist(), [5.0000000000000266454,
    #                                                5.0000000000000266454])

    #     self.assertListEqual(lov.vel[0].tolist(), [1/0.0088200863560403078983,
    #                                                1/0.0089530611050798007688])
    #     self.assertListEqual(lov.vel[1].tolist(), [1/0.001522605285544077541,
    #                                                1/0.004102623000517897911])

    #     self.assertListEqual(lov.ext["azi"][0].tolist(), [252.05441718438927978,
    #                                                       99.345595852002077208])
    #     self.assertListEqual(lov.ext["azi"][1].tolist(), [92.499974198256211366,
    #                                                       174.33483725071613435])

    #     self.assertListEqual(lov.ext["ell"][0].tolist(), [0, 0])
    #     self.assertListEqual(lov.ext["ell"][1].tolist(), [0, 0])

    #     self.assertListEqual(lov.ext["noi"][0].tolist(), [0, 0])
    #     self.assertListEqual(lov.ext["noi"][1].tolist(), [0, 0])

    #     self.assertListEqual(lov.ext["pwr"][0].tolist(), [3832630.8840260845609,
    #                                                       4039408.6602126094513])
    #     self.assertListEqual(lov.ext["pwr"][1].tolist(), [232295.4010567846417,
    #                                                       422413.79929310601437])
