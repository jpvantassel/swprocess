"""Tests for Peaks abstract class."""
import unittest
import utprocess
import numpy as np
import json
import os
import logging
logging.basicConfig(level=logging.WARN)


class TestPeaks(unittest.TestCase):
    def test_init(self):
        # Basic Case: No keyword arugments
        frequency = [100, 50, 30, 10, 5, 3]
        velocity = [100, 120, 130, 140, 145, 150]
        identifier = '-5m'
        my_peaks = utprocess.Peaks(frequency=frequency,
                                   velocity=velocity,
                                   identifier=identifier)
        self.assertListEqual(my_peaks.frq[0].tolist(), frequency)
        self.assertListEqual(my_peaks.vel[0].tolist(), velocity)
        self.assertEqual(my_peaks.ids[0], identifier)

        # Advanced Case: Four keyword arguements
        frequency = [100, 50, 30, 10, 5, 3]
        velocity = [100, 120, 130, 140, 145, 150]
        azi = [10, 15, 20, 35, 10, 20]
        ell = [10, 15, 20, 35, 10, 20]
        noi = [10, 15, 20, 35, 10, 20]
        pwr = [10, 15, 20, 35, 10, 20]
        identifier = '-5m'
        my_peaks = utprocess.Peaks(frequency=frequency,
                                   velocity=velocity,
                                   identifier=identifier,
                                   azi=azi,
                                   ell=ell,
                                   noi=noi,
                                   pwr=pwr)
        self.assertListEqual(my_peaks.ext["azi"][0].tolist(), azi)
        self.assertListEqual(my_peaks.ext["ell"][0].tolist(), ell)
        self.assertListEqual(my_peaks.ext["noi"][0].tolist(), noi)
        self.assertListEqual(my_peaks.ext["pwr"][0].tolist(), pwr)

    def test_from_peak_data_dictss(self):
        # Basic Case: No keyword arugments
        frequency = [100, 50, 30, 10, 5, 3]
        velocity = [100, 120, 130, 140, 145, 150]
        identifier = '-5m'
        data = {identifier: {"frequency": frequency, "velocity": velocity}}
        my_peaks = utprocess.Peaks.from_dict(data)
        self.assertListEqual(my_peaks.frq[0].tolist(), frequency)
        self.assertListEqual(my_peaks.vel[0].tolist(), velocity)
        self.assertEqual(my_peaks.ids, [identifier])

        # Advanced Case: Four keyword arguements
        frequency = [100, 50, 30, 10, 5, 3]
        velocity = [100, 120, 130, 140, 145, 150]
        azi = [10, 15, 20, 35, 10, 20]
        ell = [10, 15, 20, 35, 10, 20]
        noi = [10, 15, 20, 35, 10, 20]
        pwr = [10, 15, 20, 35, 10, 20]
        identifier = '-5m'
        data = {identifier: {"frequency": frequency,
                             "velocity": velocity,
                             "azi": azi,
                             "ell": ell,
                             "noi": noi,
                             "pwr": pwr}}
        my_peaks = utprocess.Peaks.from_dict(data)
        self.assertListEqual(my_peaks.ext["azi"][0].tolist(), azi)
        self.assertListEqual(my_peaks.ext["ell"][0].tolist(), ell)
        self.assertListEqual(my_peaks.ext["noi"][0].tolist(), noi)
        self.assertListEqual(my_peaks.ext["pwr"][0].tolist(), pwr)

        # Advanced Case: Two keyword arguements
        frequency = [100, 50, 30, 10, 5, 3]
        velocity = [100, 120, 130, 140, 145, 150]
        azi = [10, 15, 20, 35, 10, 20]
        pwr = [10, 15, 20, 35, 10, 20]
        identifier = '-5m'
        data = {identifier: {"frequency": frequency,
                             "velocity": velocity,
                             "azi": azi,
                             "pwr": pwr}}
        my_peaks = utprocess.Peaks.from_dict(data)
        self.assertListEqual(my_peaks.ext["azi"][0].tolist(), azi)
        self.assertListEqual(my_peaks.ext["pwr"][0].tolist(), pwr)

    def test_from_maxs_1file(self):
        # Check rayleigh (2 lines)
        ray = utprocess.Peaks.from_maxs(fnames="test\\data\\myanmar\\test_hfk_ds_short.max",
                                        identifiers="test_short_rayleigh",
                                        rayleigh=True,
                                        love=False)
        self.assertListEqual(ray.frq[0].tolist(), [20.000000000000106581,
                                                   19.282217609815102577])
        self.assertListEqual(ray.vel[0].tolist(), [1/0.0068859013683322750979,
                                                   1/0.0074117944332218188563])
        self.assertListEqual(ray.ext["azi"][0].tolist(), [144.53791572557310019,
                                                          143.1083743693494057])
        self.assertListEqual(ray.ext["ell"][0].tolist(), [1.0214647665926679387,
                                                          1.022287917081338593])
        self.assertListEqual(ray.ext["noi"][0].tolist(), [8.9773778053801098764,
                                                          7.3044365524672443257])
        self.assertListEqual(ray.ext["pwr"][0].tolist(), [2092111.2367646128405,
                                                          2074967.9391639579553])

        # Check love (2 lines)
        lov = utprocess.Peaks.from_maxs(fnames="test\\data\\myanmar\\test_hfk_ds_short.max",
                                        identifiers="test_short_love",
                                        rayleigh=False,
                                        love=True)
        self.assertListEqual(lov.frq[0].tolist(), [20.000000000000106581,
                                                   19.282217609815102577])
        self.assertListEqual(lov.vel[0].tolist(), [1/0.0088200863560403078983,
                                                   1/0.0089530611050798007688])
        self.assertListEqual(lov.ext["azi"][0].tolist(), [252.05441718438927978,
                                                          99.345595852002077208])
        self.assertListEqual(lov.ext["ell"][0].tolist(), [0, 0])
        self.assertListEqual(lov.ext["noi"][0].tolist(), [0, 0])
        self.assertListEqual(lov.ext["pwr"][0].tolist(), [3832630.8840260845609,
                                                          4039408.6602126094513])

    def test_from_max_2files(self):
        # Check Rayleigh (2 files, 2 lines per file)
        ray = utprocess.Peaks.from_maxs(fnames=["test\\data\\myanmar\\test_hfk_ds_short.max",
                                                "test\\data\\myanmar\\test_hfk_pr_short_2.max"],
                                        identifiers=["test_short",
                                                     "test_short_2"],
                                        rayleigh=True,
                                        love=False)
        self.assertListEqual(ray.frq[0].tolist(), [20.000000000000106581,
                                                   19.282217609815102577])
        self.assertListEqual(ray.frq[1].tolist(), [5.0000000000000266454,
                                                   5.0000000000000266454])

        self.assertListEqual(ray.vel[0].tolist(), [1/0.0068859013683322750979,
                                                   1/0.0074117944332218188563])
        self.assertListEqual(ray.vel[1].tolist(), [1/0.0062049454559321261596,
                                                   1/0.005256207300441733711])

        self.assertListEqual(ray.ext["azi"][0].tolist(), [144.53791572557310019,
                                                          143.1083743693494057])
        self.assertListEqual(ray.ext["azi"][1].tolist(), [336.6469768619966203,
                                                          77.740031601074633727])

        self.assertListEqual(ray.ext["ell"][0].tolist(), [1.0214647665926679387,
                                                          1.022287917081338593])
        self.assertListEqual(ray.ext["ell"][1].tolist(), [-1.6318692462645167929,
                                                          -1.8969709764459561363])

        self.assertListEqual(ray.ext["noi"][0].tolist(), [8.9773778053801098764,
                                                          7.3044365524672443257])
        self.assertListEqual(ray.ext["noi"][1].tolist(), [np.inf,
                                                          np.inf])

        self.assertListEqual(ray.ext["pwr"][0].tolist(), [2092111.2367646128405,
                                                          2074967.9391639579553])
        self.assertListEqual(ray.ext["pwr"][1].tolist(), [6040073199.9762010574,
                                                          17030488659.287288666])

        # Check Love (2 files, 2 lines per file)
        lov = utprocess.Peaks.from_maxs(fnames=["test\\data\\myanmar\\test_hfk_ds_short.max",
                                                "test\\data\\myanmar\\test_hfk_pr_short_2.max"],
                                        identifiers=["test_short",
                                                     "test_short_2"],
                                        rayleigh=False,
                                        love=True)
        self.assertListEqual(lov.frq[0].tolist(), [20.000000000000106581,
                                                   19.282217609815102577])
        self.assertListEqual(lov.frq[1].tolist(), [5.0000000000000266454,
                                                   5.0000000000000266454])

        self.assertListEqual(lov.vel[0].tolist(), [1/0.0088200863560403078983,
                                                   1/0.0089530611050798007688])
        self.assertListEqual(lov.vel[1].tolist(), [1/0.001522605285544077541,
                                                   1/0.004102623000517897911])

        self.assertListEqual(lov.ext["azi"][0].tolist(), [252.05441718438927978,
                                                          99.345595852002077208])
        self.assertListEqual(lov.ext["azi"][1].tolist(), [92.499974198256211366,
                                                          174.33483725071613435])

        self.assertListEqual(lov.ext["ell"][0].tolist(), [0, 0])
        self.assertListEqual(lov.ext["ell"][1].tolist(), [0, 0])

        self.assertListEqual(lov.ext["noi"][0].tolist(), [0, 0])
        self.assertListEqual(lov.ext["noi"][1].tolist(), [0, 0])

        self.assertListEqual(lov.ext["pwr"][0].tolist(), [3832630.8840260845609,
                                                          4039408.6602126094513])
        self.assertListEqual(lov.ext["pwr"][1].tolist(), [232295.4010567846417,
                                                          422413.79929310601437])

    # def test_partytime_real(self):
    #     lov = utprocess.Peaks.from_maxs(fnames="test\\data\\myanmar\\test_hfk_ds_full.max",
    #                                     identifiers="test_lov",
    #                                     rayleigh=False,
    #                                     love=True)
    #     lov.party_time(settings_file="test\\test_ptimesettings.json")

    #     ray = utprocess.Peaks.from_maxs(fnames="test\\data\\myanmar\\test_hfk_ds_full.max",
    #                                     identifiers="test_ray",
    #                                     rayleigh=True,
    #                                     love=False)
    #     ray.party_time(settings_file="test\\test_ptimesettings.json")

    def test_partytime_fake(self):
        # Standard With Extras (Remove Center Point [3,150])
        frq = [1, 5, 5, 1, 3]
        vel = [100, 100, 200, 200, 150]
        ell = [5, 5, 5, 5, 10]
        ids = "Failure is always an option"
        peak = utprocess.Peaks(frequency=frq,
                               velocity=vel,
                               identifier=ids,
                               ell=ell)
        peak.party_time(settings_file="test\\test_ptimesettings_fake.json")
        self.assertListEqual(peak.frq[0].tolist(), frq[:-1])
        self.assertListEqual(peak.vel[0].tolist(), vel[:-1])
        self.assertListEqual(peak.ext["ell"][0].tolist(), ell[:-1])

    def test_write_peak_json(self):
        # Basic Case (Only Frequency and Velocity)
        fname = "test//test_write_peak_json"
        known_freq = [1, 2, 3]
        known_vel = [4, 5, 6]
        peaks = utprocess.Peaks(frequency=known_freq,
                                velocity=known_vel,
                                identifier=fname)
        peaks.write_peak_json(fname)
        with open(fname+".json", "r") as f:
            test_data = json.load(f)
        self.assertListEqual(known_freq, test_data[fname]["frequency"])
        self.assertListEqual(known_vel, test_data[fname]["velocity"])
        os.remove(fname+".json")

        # Complex Case (Frequency, Velocity, and Extra)
        fname = "test//test_write_peak_json"
        known_freq = [1, 2, 3]
        known_vel = [4, 5, 6]
        known_ell = [1, 2, 3]
        known_azi = [5, 6, 7]
        known_noi = [9, 10, 11]
        known_pwr = [12, 13, 14]
        peaks = utprocess.Peaks(frequency=known_freq,
                                velocity=known_vel,
                                identifier=fname,
                                ell=known_ell,
                                azi=known_azi,
                                noi=known_noi,
                                pwr=known_pwr)
        peaks.write_peak_json(fname)
        with open(fname+".json", "r") as f:
            test_data = json.load(f)
        self.assertListEqual(known_freq, test_data[fname]["frequency"])
        self.assertListEqual(known_vel, test_data[fname]["velocity"])
        self.assertListEqual(known_ell, test_data[fname]["ell"])
        self.assertListEqual(known_azi, test_data[fname]["azi"])
        self.assertListEqual(known_noi, test_data[fname]["noi"])
        self.assertListEqual(known_pwr, test_data[fname]["pwr"])
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
