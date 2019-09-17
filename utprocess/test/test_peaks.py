"""Tests for Peaks abstract class."""
import unittest
import utprocess
import numpy as np
import json
import os
import logging
logging.basicConfig(level=logging.DEBUG)


class TestPeaks(unittest.TestCase):
    def test_init(self):
        frequency = [np.array([100, 50, 30, 10, 5, 3])]
        velocity = [np.array([100, 120, 130, 140, 145, 150])]
        identifier = ['-5m']
        my_peaks = utprocess.Peaks(frequency_list=frequency,
                                   velocity_list=velocity,
                                   identifiers=identifier)
        self.assertListEqual(my_peaks.frq[0].tolist(), frequency[0].tolist())
        self.assertListEqual(my_peaks.vel[0].tolist(), velocity[0].tolist())
        self.assertListEqual(my_peaks.ids, identifier)

        frequency = [100, 50, 30, 10, 5, 3]
        velocity = [100, 120, 130, 140, 145, 150]
        identifier = '-5m'
        my_peaks = utprocess.Peaks(frequency_list=frequency,
                                   velocity_list=velocity,
                                   identifiers=identifier)
        self.assertListEqual(my_peaks.frq[0].tolist(), frequency)
        self.assertListEqual(my_peaks.vel[0].tolist(), velocity)
        self.assertEqual(my_peaks.ids[0], identifier)

    def test_from_peak_data_dictss(self):
        frequency = [100, 50, 30, 10, 5, 3]
        velocity = [100, 120, 130, 140, 145, 150]
        identifier = '-5m'
        data = {identifier: {"frequency": frequency, "velocity": velocity}}
        my_peaks = utprocess.Peaks.from_peak_data_dicts(data)
        self.assertListEqual(my_peaks.frq[0].tolist(), frequency)
        self.assertListEqual(my_peaks.vel[0].tolist(), velocity)
        self.assertEqual(my_peaks.ids, [identifier])

    def test_from_hfk(self):
        # Check rayleigh (2 lines)
        ray = utprocess.Peaks.from_hfks(fnames="test\\data\\myanmar\\test_hfk_ds_short.max",
                                        array_names="test_short_rayleigh",
                                        rayleigh=True,
                                        love=False)
        self.assertListEqual(ray.frq[0].tolist(), [20.000000000000106581,
                                                   19.282217609815102577])
        self.assertListEqual(ray.vel[0].tolist(), [1/0.0068859013683322750979,
                                                   1/0.0074117944332218188563])

        # Check love (2 lines)
        lov = utprocess.Peaks.from_hfks(fnames="test\\data\\myanmar\\test_hfk_ds_short.max",
                                        array_names="test_short_love",
                                        rayleigh=False,
                                        love=True)
        self.assertListEqual(lov.frq[0].tolist(), [20.000000000000106581,
                                                   19.282217609815102577])
        self.assertListEqual(lov.vel[0].tolist(), [1/0.0088200863560403078983,
                                                   1/0.0089530611050798007688])

    def test_partytime(self):
        pass
        # lov = utprocess.Peaks.from_hfks(fnames="test\\data\\myanmar\\test_hfk_ds_full.max",
        #                                 array_names="test_lov",
        #                                 rayleigh=False,
        #                                 love=True)
        # lov.party_time(settings_file="test\\test_ptimesettings.json")

        # ray = utprocess.Peaks.from_hfks(fnames="test\\data\\myanmar\\test_hfk_ds_full.max",
        #                                 array_names="test_ray",
        #                                 rayleigh=True,
        #                                 love=False)
        # ray.party_time(settings_file="test\\test_ptimesettings.json")

    def test_write_peak_json(self):
        fname = "test//test_write_peak_json"
        known_freq = [1, 2, 3]
        known_vel = [4, 5, 6]
        peaks = utprocess.Peaks(frequency_list=known_freq,
                                velocity_list=known_vel,
                                identifiers=fname)
        peaks.write_peak_json(fname)
        with open(fname+".json", "r") as f:
            test_data = json.load(f)
        self.assertListEqual(known_freq, test_data[fname]["frequency"])
        self.assertListEqual(known_vel, test_data[fname]["velocity"])
        os.remove(fname+".json")

    def test_compute_dc_stats(self):
        # Two bins, arrayweights=1
        frq = [np.array([2.25, 2.75, 4.25, 4.75])]
        vel = [np.array([100, 120, 115, 120])]
        for arrayweight in [[1], None]:
            mdisp = utprocess.Peaks.compute_dc_stats(frq,
                                                    vel,
                                                    minp=1,
                                                    maxp=5,
                                                    numbins=2,
                                                    binscale="linear",
                                                    bintype="frequency",
                                                    arrayweights=arrayweight)
            self.assertListEqual(mdisp["mean"]["frq"].tolist(), [2.5, 4.5])
            self.assertListEqual(mdisp["mean"]["vel"].tolist(), [110., 117.5])

            for test, known in zip(mdisp["mean"]["slo"].tolist(), [0.009166667, 0.008514493]):
                self.assertAlmostEqual(test, known)
            for test, known in zip(mdisp["mean"]["wav"].tolist(), [44.04040404, 26.16099071]):
                self.assertAlmostEqual(test, known)
            for test, known in zip(mdisp["std"]["vel"].tolist(), [14.14213562, 3.535533906]):
                self.assertAlmostEqual(test, known)
            for test, known in zip(mdisp["std"]["slo"].tolist(), [0.001178511, 0.000256198]):
                self.assertAlmostEqual(test, known)

        # Two bins, two arrays of different weight
        frq = [np.array([2.3, 2.8, 4.2, 4.7]), np.array([2.6, 2.4, 4.5, 4.5])]
        vel = [np.array([100, 120, 115, 120]), np.array([110, 150, 100, 100])]
        mdisp = utprocess.Peaks.compute_dc_stats(frq,
                                                 vel,
                                                 minp=1,
                                                 maxp=5,
                                                 numbins=2,
                                                 binscale="linear",
                                                 bintype="frequency",
                                                 arrayweights=[1, 2])

        for test, known in zip(mdisp["mean"]["frq"].tolist(), [2.516666667, 4.483333333]):
            self.assertAlmostEqual(test, known)
        for test, known in zip(mdisp["mean"]["vel"].tolist(), [123.3333333, 105.8333333]):
            self.assertAlmostEqual(test, known)
        for test, known in zip(mdisp["mean"]["slo"].tolist(), [0.008308081, 0.009504831]):
            self.assertAlmostEqual(test, known)
        for test, known in zip(mdisp["mean"]["wav"].tolist(), [49.32513139, 23.63362603]):
            self.assertAlmostEqual(test, known)
        for test, known in zip(mdisp["std"]["vel"].tolist(), [22.7710017, 9.670497325]):
            self.assertAlmostEqual(test, known)
        for test, known in zip(mdisp["std"]["slo"].tolist(), [0.001451233, 0.000817577]):
            self.assertAlmostEqual(test, known)


if __name__ == "__main__":
    unittest.main()
