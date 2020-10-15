"""Tests for PeaksSuite class."""

import json
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import swprocess
from swprocess.peaks import Peaks
from testtools import unittest, TestCase, get_full_path

logger = logging.getLogger("swprocess")
logger.setLevel(logging.ERROR)

class Test_PeaksSuite(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_init(self):
        # Basic
        p0 = Peaks([1, 2, 3], [4, 5, 6], "p0")
        p1 = Peaks([7, 8, 9], [10, 11, 12], "p1")
        suite = swprocess.PeaksSuite(p0)
        suite.append(p1)

        for returned, expected in zip(suite, [p0, p1]):
            self.assertEqual(expected, returned)

        # Bad: replicated identifier
        self.assertRaises(KeyError, suite.append, p1)

        # Bad: append non-Peaks object
        self.assertRaises(TypeError, suite.append, "not a Peaks object")

    def test_from_dict(self):
        # Simple Case: Single dictionary
        data = {"test": {"frequency": [1, 2, 3], "velocity": [4, 5, 6]}}
        suite = swprocess.PeaksSuite.from_dict(data)
        peaks = Peaks.from_dict(data["test"], "test")
        self.assertEqual(peaks, suite[0])

    def test_from_json(self):
        # Advanced Case: Two keyword arguements
        frequency = [100., 50, 30, 10, 5, 3]
        velocity = [100., 120, 130, 140, 145, 150]
        azi = [10., 15, 20, 35, 10, 20]
        pwr = [10., 15, 20, 35, 11, 20]

        peak_list = []
        fnames = []
        for num in range(0, 3):
            fname = f"{self.full_path}data/tmp_id{num}.json"
            peaks = Peaks(
                frequency, velocity, str(num), azi=azi, pwr=pwr)
            peaks.to_json(fname)
            fnames.append(fname)
            peak_list.append(peaks)

        returned = swprocess.PeaksSuite.from_json(fnames=fnames)
        expected = swprocess.PeaksSuite.from_iter(peak_list)
        self.assertEqual(expected, returned)

        for fname in fnames:
            os.remove(fname)

    def test_from_max(self):
        # Check Rayleigh (2 files, 2 lines per file)
        fnames = [self.full_path +
                  f"data/mm/test_hfk_line2_{x}.max" for x in range(2)]
        returned = swprocess.PeaksSuite.from_max(fnames=fnames,
                                                  wavetype="rayleigh")

        r0 = {"16200": {"frequency": [20.000000000000106581, 19.282217609815102577],
                        "velocity": [1/0.0068859013683322750979, 1/0.0074117944332218188563],
                        "azimuth": [144.53791572557310019, 143.1083743693494057],
                        "ellipticity": [1.0214647665926679387, 1.022287917081338593],
                        "noise": [8.9773778053801098764, 7.3044365524672443257],
                        "power": [2092111.2367646128405, 2074967.9391639579553],
                        }}

        r1 = {"20700": {"frequency": [5.0000000000000266454],
                        "velocity": [1/0.0062049454559321261596],
                        "azimuth": [336.6469768619966203],
                        "ellipticity": [-1.6318692462645167929],
                        "noise": [0],
                        "power": [6040073199.9762010574],
                        }}

        r2 = {"20760.040000000000873": {"frequency": [5.0000000000000266454],
                                        "velocity": [1/0.005256207300441733711],
                                        "azimuth": [77.740031601074633727],
                                        "ellipticity": [-1.8969709764459561363],
                                        "noise": [0],
                                        "power": [17030488659.287288666],
                                        }}

        ray_dicts = [r0, r1, r2]
        expected = swprocess.PeaksSuite.from_dict(ray_dicts)

        self.assertEqual(expected, returned)

        # Check Love (2 files, 2 lines per file)
        fnames = [self.full_path +
                  f"data/mm/test_hfk_line2_{x}.max" for x in range(2)]
        returned = swprocess.PeaksSuite.from_max(fnames=fnames,
                                                  wavetype="love")

        l0 = {"16200": {"frequency": [20.000000000000106581, 19.282217609815102577],
                        "velocity": [1/0.0088200863560403078983, 1/0.0089530611050798007688],
                        "azimuth": [252.05441718438927978, 99.345595852002077208],
                        "ellipticity": [0, 0],
                        "noise": [0, 0],
                        "power": [3832630.8840260845609, 4039408.6602126094513],
                        }}

        l1 = {"20700": {"frequency": [5.0000000000000266454],
                        "velocity": [1/0.001522605285544077541],
                        "azimuth": [92.499974198256211366],
                        "ellipticity": [0],
                        "noise": [0],
                        "power": [232295.4010567846417],
                        }}

        l2 = {"20760.040000000000873": {"frequency": [5.0000000000000266454],
                                        "velocity": [1/0.004102623000517897911],
                                        "azimuth": [174.33483725071613435],
                                        "ellipticity": [0],
                                        "noise": [0],
                                        "power": [422413.79929310601437],
                                        }}

        lov_dicts = [l0, l1, l2]
        expected = swprocess.PeaksSuite.from_dict(lov_dicts)

        self.assertEqual(expected, returned)

    def test_plot(self):
        fname = self.full_path + "data/peak/suite_raw.json"
        suite = swprocess.PeaksSuite.from_json(fname)
        suite.blitz("velocity", (None, 500))
        fig, ax = suite.plot(xtype=["frequency", "wavelength", "frequency"],
                             ytype=["velocity", "velocity", "slowness"])
        plt.show(block=False)
        plt.pause(0.5)
        plt.close("all")

    # def test_interactive_trimming(self):
    #     fname = self.full_path + "data/peak/suite_raw.json"
    #     suite = swprocess.PeaksSuite.from_json(fname)
    #     suite.blitz("velocity", (None, 500))
    #     settings = self.full_path + "settings/settings_post.json"
    #     suite.interactive_trimming(settings)

    def test_statistics(self):
        # No missing data
        values = np.array([[1, 2, 3, 4, 5],
                           [4, 5, 7, 8, 9],
                           [4, 3, 6, 4, 2]])
        frq = [1, 2, 3, 4, 5]
        peaks = [Peaks(frq, values[k], str(k)) for k in range(3)]
        suite = swprocess.PeaksSuite.from_iter(peaks)
        rfrq, rmean, rstd, rcorr = suite.statistics(frq,
                                                    xtype="frequency",
                                                    ytype="velocity")
        self.assertArrayEqual(np.array(frq), rfrq)
        self.assertArrayEqual(np.mean(values, axis=0), rmean)
        self.assertArrayEqual(np.std(values, axis=0, ddof=1), rstd)
        self.assertArrayEqual(np.corrcoef(values.T), rcorr)

        # missing_data_procedure="drop"
        values = np.array([[np.nan]*6,
                           [np.nan, 1, 2, 3, 4, 5],
                           [0, 4, 5, 7, 8, 9],
                           [0, 4, 3, 6, 4, 2]])
        frq = [0.2, 1, 2, 3, 4, 5]

        valid = np.array([[1, 2, 3, 4, 5],
                          [4, 5, 7, 8, 9],
                          [4, 3, 6, 4, 2]])
        valid_frq = frq[1:]
        peaks = [Peaks(frq, values[k], str(k)) for k in range(4)]
        suite = swprocess.PeaksSuite.from_iter(peaks)
        rfrq, rmean, rstd, rcorr = suite.statistics(frq,
                                                    xtype="frequency",
                                                    ytype="velocity",
                                                    missing_data_procedure="drop")
        self.assertArrayEqual(np.array(valid_frq), rfrq)
        self.assertArrayEqual(np.mean(valid, axis=0), rmean)
        self.assertArrayEqual(np.std(valid, axis=0, ddof=1), rstd)
        self.assertArrayEqual(np.corrcoef(valid.T), rcorr)

    def test_eq(self):
        p0 = Peaks([1, 2, 3], [4, 5, 6], "0")
        p1 = Peaks([1, 2, 3], [7, 8, 9], "1")
        p2 = Peaks([1, 2, 3], [0, 1, 2], "2")

        suite_a = swprocess.PeaksSuite.from_iter([p0, p1, p2])
        suite_b = "I am not a PeakSuite"
        suite_c = swprocess.PeaksSuite.from_iter([p1, p2])
        suite_d = swprocess.PeaksSuite.from_iter([p1])
        suite_e = swprocess.PeaksSuite.from_iter([p1, p0, p2])
        suite_f = swprocess.PeaksSuite.from_iter([p0, p1, p2])

        self.assertTrue(suite_a == suite_a)
        self.assertFalse(suite_a == suite_b)
        self.assertFalse(suite_a == suite_c)
        self.assertFalse(suite_a == suite_d)
        self.assertFalse(suite_a == suite_e)
        self.assertTrue(suite_a == suite_f)


if __name__ == "__main__":
    unittest.main()
