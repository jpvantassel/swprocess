"""Tests for Peaks class."""

import json
import os
import logging

import numpy as np

import swprocess
from testtools import unittest, TestCase, get_full_path

logging.basicConfig(level=logging.WARN)


class Test_Peaks(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_init(self):
        # Basic Case: No keyword arguments
        frequency = [100., 50, 30, 10, 5, 3]
        velocity = [100., 120, 130, 140, 145, 150]
        identifier = "-5m"
        peaks = swprocess.Peaks(frequency=frequency,
                                velocity=velocity,
                                identifier=identifier)
        self.assertArrayEqual(np.array(frequency), peaks.frequency)
        self.assertArrayEqual(np.array(velocity), peaks.velocity)
        self.assertEqual(identifier, peaks.ids)

        # Advanced Case: Four keyword arguments
        frequency = [100., 50, 30, 10, 5, 3]
        velocity = [100., 120, 130, 140, 145, 150]
        azi = [10., 15, 20, 35, 10, 20]
        ell = [10., 15, 20, 35, 11, 20]
        noi = [10., 15, 20, 35, 12, 20]
        pwr = [10., 15, 20, 35, 13, 20]
        identifier = "-5m"
        my_peaks = swprocess.Peaks(frequency=frequency,
                                   velocity=velocity,
                                   identifier=identifier,
                                   azi=azi,
                                   ell=ell,
                                   noi=noi,
                                   pwr=pwr)
        self.assertArrayEqual(np.array(azi), my_peaks.azi)
        self.assertArrayEqual(np.array(ell), my_peaks.ell)
        self.assertArrayEqual(np.array(noi), my_peaks.noi)
        self.assertArrayEqual(np.array(pwr), my_peaks.pwr)

    def test_from_dict(self):
        # Basic Case: No keyword arguments
        frequency = [100., 50, 30, 10, 5, 3]
        velocity = [100., 120, 130, 140, 145, 150]
        identifier = "-5m"
        data = {"frequency": frequency, "velocity": velocity}
        peaks = swprocess.Peaks.from_dict(data, identifier=identifier)
        self.assertArrayEqual(np.array(frequency), peaks.frequency)
        self.assertArrayEqual(np.array(velocity), peaks.velocity)
        self.assertEqual(identifier, peaks.ids)

        # Advanced Case: Four keyword arguments
        frequency = [100., 50, 30, 10, 5, 3]
        velocity = [100., 120, 130, 140, 145, 150]
        azi = [10., 15, 20, 35, 10, 20]
        ell = [10., 15, 20, 35, 11, 20]
        noi = [10., 15, 20, 35, 12, 20]
        pwr = [10., 15, 20, 35, 13, 20]
        identifier = "-5m"
        data = {"frequency": frequency, "velocity": velocity,
                "azi": azi, "ell": ell, "noi": noi, "pwr": pwr}
        peaks = swprocess.Peaks.from_dict(data, identifier=identifier)
        self.assertArrayEqual(np.array(azi), peaks.azi)
        self.assertArrayEqual(np.array(ell), peaks.ell)
        self.assertArrayEqual(np.array(noi), peaks.noi)
        self.assertArrayEqual(np.array(pwr), peaks.pwr)

        # Advanced Case: Two keyword arguments
        frequency = [100, 50, 30, 10, 5, 3]
        velocity = [100, 120, 130, 140, 145, 150]
        azi = [10, 15, 20, 35, 10, 20]
        pwr = [10, 15, 20, 35, 11, 20]
        identifier = "-5m"
        data = {"frequency": frequency, "velocity": velocity,
                "azi": azi, "pwr": pwr}
        peaks = swprocess.Peaks.from_dict(data, identifier=identifier)
        self.assertArrayEqual(np.array(azi), peaks.azi)
        self.assertArrayEqual(np.array(pwr), peaks.pwr)

    def test_to_and_from_jsons(self):
        # Advanced Case: Two keywosrd arguements
        frequency = [100., 50, 30, 10, 5, 3]
        velocity = [100., 120, 130, 140, 145, 150]
        azi = [10., 15, 20, 35, 10, 20]
        pwr = [10., 15, 20, 35, 11, 20]
        identifer = "test"
        fname = "test.json"

        expected = swprocess.Peaks(
            frequency, velocity, identifer, azi=azi, pwr=pwr)
        expected.to_json(fname)
        returned = swprocess.Peaks.from_json(fname)
        self.assertEqual(expected, returned)
        os.remove(fname)

    def test_from_max(self):
        # Check rayleigh (2 lines)
        ray = swprocess.Peaks.from_max(fname=self.full_path + "data/mm/test_hfk_line2_0.max",
                                       identifier="test_short_rayleigh",
                                       rayleigh=True,
                                       love=False)
        self.assertArrayEqual(ray.frq, np.array([20.000000000000106581,
                                                 19.282217609815102577]))
        self.assertArrayEqual(ray.vel, 1/np.array([0.0068859013683322750979,
                                                   0.0074117944332218188563]))
        self.assertArrayEqual(ray.azi, np.array([144.53791572557310019,
                                                 143.1083743693494057]))
        self.assertArrayEqual(ray.ell, np.array([1.0214647665926679387,
                                                 1.022287917081338593]))
        self.assertArrayEqual(ray.noi, np.array([8.9773778053801098764,
                                                 7.3044365524672443257]))
        self.assertArrayEqual(ray.pwr, np.array([2092111.2367646128405,
                                                 2074967.9391639579553]))
        self.assertArrayEqual(ray.tim, np.array([16200,
                                                 16200]))

        # Check love (2 lines)
        lov = swprocess.Peaks.from_max(fname=self.full_path+"data/mm/test_hfk_line2_0.max",
                                       identifier="test_short_love",
                                       rayleigh=False,
                                       love=True)
        self.assertArrayEqual(lov.frq, np.array([20.000000000000106581,
                                                 19.282217609815102577]))
        self.assertArrayEqual(lov.vel, np.array([1/0.0088200863560403078983,
                                                 1/0.0089530611050798007688]))
        self.assertArrayEqual(lov.azi, np.array([252.05441718438927978,
                                                 99.345595852002077208]))
        self.assertArrayEqual(lov.ell, np.array([0, 0]))
        self.assertArrayEqual(lov.noi, np.array([0, 0]))
        self.assertArrayEqual(lov.pwr, np.array([3832630.8840260845609,
                                                 4039408.6602126094513]))
        self.assertArrayEqual(ray.tim, np.array([16200,
                                                 16200]))


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
    # #     self.assertListEqual(peak.frq[0].tolist(), frq[:-1])
    # #     self.assertListEqual(peak.vel[0].tolist(), vel[:-1])
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
