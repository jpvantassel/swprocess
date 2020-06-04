"""Tests for PeaksSuite class."""

import json
import os
import logging

import numpy as np

import swprocess
from testtools import unittest, TestCase, get_full_path

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)


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
                         "azimuth": [144.53791572557310019, 143.1083743693494057],
                         "ellipticity": [1.0214647665926679387, 1.022287917081338593],
                         "noise": [8.9773778053801098764, 7.3044365524672443257],
                         "power": [2092111.2367646128405, 2074967.9391639579553],
                         "time": [16200, 16200]
                         }}

        ray_f2 = {"f2": {"frequency": [5.0000000000000266454, 5.0000000000000266454],
                         "velocity": [1/0.0062049454559321261596, 1/0.005256207300441733711],
                         "azimuth": [336.6469768619966203, 77.740031601074633727],
                         "ellipticity": [-1.6318692462645167929, -1.8969709764459561363],
                         "noise": [0, 0],
                         "power": [6040073199.9762010574, 17030488659.287288666],
                         "time": [20700, 20760.040000000000873]
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
                         "azimuth": [252.05441718438927978, 99.345595852002077208],
                         "ellipticity": [0, 0],
                         "noise": [0, 0],
                         "power": [3832630.8840260845609, 4039408.6602126094513],
                         "time": [16200, 16200]
                         }}

        lov_f1 = {"f1": {"frequency": [5.0000000000000266454, 5.0000000000000266454],
                         "velocity": [1/0.001522605285544077541, 1/0.004102623000517897911],
                         "azimuth": [92.499974198256211366, 174.33483725071613435],
                         "ellipticity": [0, 0],
                         "noise": [0, 0],
                         "power": [232295.4010567846417, 422413.79929310601437],
                         "time": [20700, 20760.040000000000873]
                         }}

        lov_dicts = [lov_f0, lov_f1]
        expected = swprocess.PeaksSuite.from_dicts(lov_dicts)

        self.assertEqual(expected, returned)

    # def test_plot(self):
    #     suite = swprocess.PeaksSuite.from_jsons(self.full_path + "data/denise/peaks_raw.json")
    #     import matplotlib.pyplot as plt
    #     suite.blitz("velocity", (None, 500))
    #     fig, ax = suite.plot(xtype=["frequency", "wavelength", "frequency"], ytype=["velocity", "velocity", "slowness"])
    #     plt.show()

    def test_plot(self):
        suite = swprocess.PeaksSuite.from_jsons(self.full_path + "data/denise/peaks_raw.json")
        suite.blitz("velocity", (None, 500))
        suite.interactive_trimming(self.full_path + "settings/settings_post.json")

    # # def test_partytime_real(self):
    # #     lov = swprocess.Peaks.from_maxs(fnames=self.full_path + "data/mm/test_hfk_ds_full.max",
    # #                                     identifiers="test_lov",
    # #                                     rayleigh=False,
    # #                                     love=True)
    # #     lov.party_time(settings_file=self.full_path + "settings/test_ptimesettings.json")

    # #     ray = swprocess.Peaks.from_maxs(fnames=self.full_path + "data/mm/test_hfk_ds_full.max",
    # #                                     identifiers="test_ray",
    # #                                     rayleigh=True,
    # #                                     love=False)
    # #     ray.party_time(settings_file=self.full_path + "settings/test_ptimesettings.json")

    # # def test_partytime_fake(self):
    # #     # Standard With Extras (Remove Center Point [3,150])
    # #     frq = [1, 5, 5, 1, 3]
    # #     vel = [100, 100, 200, 200, 150]
    # #     ell = [5, 5, 5, 5, 10]
    # #     ids = "Failure is always an option"
    # #     peak = swprocess.Peaks(frequency=frq,
    # #                            velocity=vel,
    # #                            identifier=ids,
    # #                            ell=ell)
    # #     peak.party_time(settings_file=self.full_path + "settings/test_ptimesettings_fake.json")
    # #     self.assertListEqual(peak.frequency[0].tolist(), frq[:-1])
    # #     self.assertListEqual(peak.velocity[0].tolist(), vel[:-1])
    # #     self.assertListEqual(peak.ext["ell"][0].tolist(), ell[:-1])

    # def test_compute_dc_stats(self):
    #     # Two bins, arrayweights=1
    #     frq = [np.array([2.25, 2.75, 4.25, 4.75])]
    #     vel = [np.array([100, 120, 115, 120])]
    #     for arrayweight in [[1], None]:
    #         mdisp = swprocess.Peaks.compute_dc_stats(frq,
    #                                                  vel,
    #                                                  minp=1,
    #                                                  maxp=5,
    #                                                  numbins=2,
    #                                                  binscale="linear",
    #                                                  bintype="frequency",
    #                                                  arrayweights=arrayweight)
    #         self.assertListEqual(mdisp["mean"]["frq"].tolist(), [2.5, 4.5])
    #         self.assertListEqual(mdisp["mean"]["vel"].tolist(), [110., 117.5])

    #         for test, known in zip(mdisp["mean"]["slo"].tolist(), [0.009166667, 0.008514493]):
    #             self.assertAlmostEqual(test, known)
    #         for test, known in zip(mdisp["mean"]["wav"].tolist(), [44.04040404, 26.16099071]):
    #             self.assertAlmostEqual(test, known)
    #         for test, known in zip(mdisp["std"]["vel"].tolist(), [14.14213562, 3.535533906]):
    #             self.assertAlmostEqual(test, known)
    #         for test, known in zip(mdisp["std"]["slo"].tolist(), [0.001178511, 0.000256198]):
    #             self.assertAlmostEqual(test, known)

    #     # Two bins, two arrays of different weight
    #     frq = [np.array([2.3, 2.8, 4.2, 4.7]), np.array([2.6, 2.4, 4.5, 4.5])]
    #     vel = [np.array([100, 120, 115, 120]), np.array([110, 150, 100, 100])]
    #     mdisp = swprocess.Peaks.compute_dc_stats(frq,
    #                                              vel,
    #                                              minp=1,
    #                                              maxp=5,
    #                                              numbins=2,
    #                                              binscale="linear",
    #                                              bintype="frequency",
    #                                              arrayweights=[1, 2])

    #     for test, known in zip(mdisp["mean"]["frq"].tolist(), [2.516666667, 4.483333333]):
    #         self.assertAlmostEqual(test, known)
    #     for test, known in zip(mdisp["mean"]["vel"].tolist(), [123.3333333, 105.8333333]):
    #         self.assertAlmostEqual(test, known)
    #     for test, known in zip(mdisp["mean"]["slo"].tolist(), [0.008308081, 0.009504831]):
    #         self.assertAlmostEqual(test, known)
    #     for test, known in zip(mdisp["mean"]["wav"].tolist(), [49.32513139, 23.63362603]):
    #         self.assertAlmostEqual(test, known)
    #     for test, known in zip(mdisp["std"]["vel"].tolist(), [22.7710017, 9.670497325]):
    #         self.assertAlmostEqual(test, known)
    #     for test, known in zip(mdisp["std"]["slo"].tolist(), [0.001451233, 0.000817577]):
    #         self.assertAlmostEqual(test, known)


if __name__ == "__main__":
    unittest.main()
