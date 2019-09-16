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


if __name__ == "__main__":
    unittest.main()
