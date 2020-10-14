"""Tests for Peaks class."""

import json
import os
import warnings
import logging

import numpy as np

from swprocess.peaks import Peaks
import swprocess
from testtools import unittest, TestCase, get_full_path


class Test_Peaks(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)
        cls._id = "-5m"
        cls.frq = [100., 50, 30, 10, 5, 3]
        cls.vel = [100., 120, 130, 140, 145, 150]
        cls.azi = [10., 15, 20, 35, 10, 20]
        cls.ell = [10., 15, 20, 35, 11, 20]
        cls.noi = [10., 15, 20, 35, 12, 20]
        cls.pwr = [10., 15, 20, 35, 13, 20]

    def test_init(self):
        # Basic Case: No keyword arguments
        peaks = Peaks(frequency=self.frq,
                      velocity=self.vel,
                      identifier=self._id)
        self.assertArrayEqual(np.array(self.frq), peaks.frequency)
        self.assertArrayEqual(np.array(self.vel), peaks.velocity)
        self.assertEqual(self._id, peaks.identifier)

        # Advanced Case: Four keyword arguments
        identifier = "-5m"
        my_peaks = Peaks(frequency=self.frq,
                         velocity=self.vel,
                         identifier=self._id,
                         azi=self.azi,
                         ell=self.ell,
                         noi=self.noi,
                         pwr=self.pwr)
        self.assertArrayEqual(np.array(self.azi), my_peaks.azi)
        self.assertArrayEqual(np.array(self.ell), my_peaks.ell)
        self.assertArrayEqual(np.array(self.noi), my_peaks.noi)
        self.assertArrayEqual(np.array(self.pwr), my_peaks.pwr)

    def test_from_dict(self):
        # Basic Case: No keyword arguments
        data = {"frequency": self.frq, "velocity": self.vel}
        peaks = Peaks.from_dict(data, identifier=self._id)
        self.assertArrayEqual(np.array(self.frq), peaks.frequency)
        self.assertArrayEqual(np.array(self.vel), peaks.velocity)
        self.assertEqual(self._id, peaks.identifier)

        # Advanced Case: Four keyword arguments
        data = {"frequency": self.frq, "velocity": self.vel,
                "azi": self.azi, "ell": self.ell, "noi": self.noi,
                "pwr": self.pwr}
        peaks = Peaks.from_dict(data, identifier=self._id)
        self.assertArrayEqual(np.array(self.azi), peaks.azi)
        self.assertArrayEqual(np.array(self.ell), peaks.ell)
        self.assertArrayEqual(np.array(self.noi), peaks.noi)
        self.assertArrayEqual(np.array(self.pwr), peaks.pwr)

        # Advanced Case: Two keyword arguments
        data = {"frequency": self.frq, "velocity": self.vel,
                "azi": self.azi, "pwr": self.pwr}
        peaks = Peaks.from_dict(data, identifier=self._id)
        self.assertArrayEqual(np.array(self.azi), peaks.azi)
        self.assertArrayEqual(np.array(self.pwr), peaks.pwr)

        # Bad: reference to deprecated from_dicts
        self.assertRaises(DeprecationWarning, Peaks.from_dicts)

        # Bad: missing frequency or velocity
        del data["frequency"]
        self.assertRaises(TypeError, Peaks.from_dict, data)

    def test_to_and_from_jsons(self):
        # Standard to_json and from_json
        fname = "test.json"
        expected = Peaks(self.frq, self.vel, self._id,
                         azi=self.azi, pwr=self.pwr)
        expected.to_json(fname)
        returned = Peaks.from_json(fname)
        self.assertEqual(expected, returned)
        os.remove(fname)

        # Deprecated write_peak_json
        expected = Peaks(self.frq, self.vel, self._id,
                         azi=self.azi, pwr=self.pwr)
        fname = "test_1.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected.write_peak_json(fname)
        returned = Peaks.from_json(fname)
        self.assertEqual(expected, returned)
        os.remove(fname)

        # to_json append data does not exist
        fname = "test_2.json"
        peaks = Peaks(self.frq, self.vel, "org")
        peaks.to_json(fname)
        peaks.identifier = "app"
        peaks.to_json(fname, append=True)
        suite = swprocess.PeaksSuite.from_jsons(fname)
        for _peaks, _id in zip(suite, suite.ids):
            peaks.identifier = _id
            self.assertEqual(peaks, _peaks)

        # to_json append data already exists
        self.assertRaises(KeyError, peaks.to_json, fname, append=True)
        os.remove(fname)

        # to_json overwrite
        fname = "test_3.json"
        peaks = Peaks(self.frq, self.vel, "org")
        peaks.to_json(fname)
        peaks.identifier = "app"
        peaks.to_json(fname, append=False)
        returned = Peaks.from_json(fname)
        self.assertEqual(peaks, returned)
        os.remove(fname)

        # from_json ignore multiple data
        fname = self.full_path + "data/peak/peaks_c2.json"
        peak_suite = swprocess.PeaksSuite.from_jsons(fname)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returned = Peaks.from_json(fname)
        expected = peak_suite[peak_suite.ids.index(returned.identifier)]
        self.assertEqual(expected, returned)

        # Bad: reference to deprecated from_jsons
        self.assertRaises(DeprecationWarning, Peaks.from_jsons)

    def test_from_max(self):
        # Check rayleigh (2 lines)
        fname = self.full_path + "data/mm/test_hfk_line2_0.max"
        peaks = Peaks.from_max(fname=fname, wavetype="rayleigh")
        self.assertArrayEqual(peaks.frequency, np.array([20.000000000000106581,
                                                         19.282217609815102577]))
        self.assertArrayEqual(peaks.velocity, 1/np.array([0.0068859013683322750979,
                                                          0.0074117944332218188563]))
        self.assertArrayEqual(peaks.azimuth, np.array([144.53791572557310019,
                                                       143.1083743693494057]))
        self.assertArrayEqual(peaks.ellipticity, np.array([1.0214647665926679387,
                                                           1.022287917081338593]))
        self.assertArrayEqual(peaks.noise, np.array([8.9773778053801098764,
                                                     7.3044365524672443257]))
        self.assertArrayEqual(peaks.power, np.array([2092111.2367646128405,
                                                     2074967.9391639579553]))
        self.assertEqual("16200", peaks.identifier)

        # Check love (2 lines)
        fname = self.full_path+"data/mm/test_hfk_line2_0.max"
        peaks = Peaks.from_max(fname=fname, wavetype="love")
        self.assertArrayEqual(peaks.frequency, np.array([20.000000000000106581,
                                                         19.282217609815102577]))
        self.assertArrayEqual(peaks.velocity, np.array([1/0.0088200863560403078983,
                                                        1/0.0089530611050798007688]))
        self.assertArrayEqual(peaks.azimuth, np.array([252.05441718438927978,
                                                       99.345595852002077208]))
        self.assertArrayEqual(peaks.ellipticity, np.array([0, 0]))
        self.assertArrayEqual(peaks.noise, np.array([0, 0]))
        self.assertArrayEqual(peaks.power, np.array([3832630.8840260845609,
                                                     4039408.6602126094513]))
        self.assertEqual("16200", peaks.identifier)

        # Bad reference to deprecated from_maxs
        self.assertRaises(DeprecationWarning, Peaks.from_maxs)

        # Bad wavetype
        self.assertRaises(ValueError, Peaks.from_max,
                          fname, wavetype="incorrect")

    def test_reject(self):
        xs = np.array([1, 2, 4, 5, 1, 2, 6, 4, 9, 4])
        ys = np.array([1, 5, 8, 6, 4, 7, 7, 1, 3, 5])

        xlims = (4.5, 7.5)
        xmin, xmax = xlims
        ylims = (3.5, 8.5)
        ymin, ymax = ylims

        # Helper function
        returned = Peaks._reject_inside_ids(xs, xmin, xmax,
                                                      ys, ymin, ymax)
        expected = np.array([3, 6])
        self.assertArrayEqual(expected, returned)

        # Method -> Reject on frequency and velocity
        other = np.arange(10)
        peaks = Peaks(xs, ys, other=other)
        peaks.reject(xtype="frequency", xlims=xlims,
                     ytype="velocity", ylims=ylims)

        keep_ids = [0, 1, 2, 4, 5, 7, 8, 9]
        attrs = dict(frequency=xs, velocity=ys, other=other)
        for attr, value in attrs.items():
            self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

        # Method -> Reject on frequency and other
        other = np.arange(10)
        peaks = Peaks(xs, ys, other=other)
        peaks.reject(xtype="frequency", xlims=xlims,
                     ytype="other", ylims=ylims)

        keep_ids = [0, 1, 2, 3, 4, 5, 7, 8, 9]
        attrs = dict(frequency=xs, velocity=ys, other=other)
        for attr, value in attrs.items():
            self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

        # Method -> Reject on slowness and velocity
        xs = np.array([1, 5, 8, 6, 4, 7, 7, 1, 3, 5])
        ys = 1/np.array([1, 5, 9, 4, 5, 6, 7, 8, 1, 7])

        peaks = Peaks(xs, ys)
        peaks.reject(xtype="frequency", xlims=(0, 10),
                     ytype="slowness", ylims=(4.5, 7.5))

        keep_ids = [0, 2, 3, 7, 8]
        attrs = dict(frequency=xs, velocity=ys)
        for attr, value in attrs.items():
            self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

    def test_blitz(self):
        xs = np.array([1, 5, 9, 3, 5, 7, 4, 6, 2, 8])
        ys = np.array([1, 9, 4, 5, 6, 8, 5, 2, 1, 4])

        # Helper function

        # Remove all above 4.5
        _min, _max = None, 4.5
        expected = np.array([1, 2, 4, 5, 7, 9])
        returned = Peaks._reject_outside_ids(xs, _min, _max)
        self.assertArrayEqual(expected, returned)

        # Remove all below 4.5
        _min, _max = 4.5, None
        expected = np.array([0, 3, 6, 8])
        returned = Peaks._reject_outside_ids(xs, _min, _max)
        self.assertArrayEqual(expected, returned)

        # Remove all below 1.5 and above 6.5
        _min, _max = 1.5, 6.5
        expected = np.array([0, 2, 5, 9])
        returned = Peaks._reject_outside_ids(xs, _min, _max)
        self.assertArrayEqual(expected, returned)

        # None
        _min, _max = None, None
        expected = np.array([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returned = Peaks._reject_outside_ids(xs, _min, _max)
        self.assertArrayEqual(expected, returned)

        # Method

        # Remove all below 0.5 and above 6.5
        limits = (0.5, 6.5)

        other = np.arange(10)
        peaks = Peaks(frequency=xs, velocity=ys, other=other)
        peaks.blitz("frequency", limits)

        keep_ids = [0, 1, 3, 4, 6, 7, 8]
        attrs = dict(frequency=xs, velocity=ys, other=other)
        for attr, value in attrs.items():
            self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

    # def test_plot(self):
    #     import matplotlib.pyplot as plt
    #     fname = self.full_path + "data/mm/test_hfk_full.max"
    #     peaks = Peaks.from_max(fname)
    #     fig, ax = peaks.plot(xtype=["frequency", "wavelength", "azimuth"],
    #                          ytype=["velocity", "velocity", "velocity"],
    #                          plot_kwargs=dict(color="g"))
    #     plt.close()

    def test_properties(self):
        peaks = Peaks(self.frq, self.vel, self._id, azi=self.azi)

        # Deprecated properties
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ids = peaks.ids
            wav = peaks.wav
        self.assertEqual(self._id, ids)
        self.assertArrayEqual(np.array(self.vel)/np.array(self.frq), wav)

        # Wavelength
        self.assertArrayEqual(np.array(self.vel)/np.array(self.frq),
                              peaks.wavelength)

        # Extended Attrs
        expected = ["frequency", "velocity", "azi", "wavelength", "slowness"]
        returned = peaks.extended_attrs
        self.assertListEqual(expected, returned)

    def test__eq__(self):
        peaks_a = Peaks(self.frq, self.vel, self._id,
                        azimuth=self.azi)
        peaks_b = "I am not even a Peaks object"
        peaks_c = Peaks(self.frq, self.vel, "diff", azimuth=self.azi)
        peaks_d = Peaks(self.frq[:-2], self.vel[:-2], self._id,
                        azimuth=self.azi[:-2])
        peaks_e = Peaks(np.zeros_like(self.frq), self.vel, self._id,
                        azimuth=self.azi)
        peaks_f = Peaks(np.zeros_like(self.frq), self.vel, self._id)
        peaks_g = Peaks(self.frq, self.vel, self._id,
                        azimuth=self.azi)
        del peaks_g.identifier
        peaks_h = Peaks(self.frq, self.vel, self._id,
                        noise=self.noi)
        peaks_i = Peaks(self.frq, self.vel, self._id,
                        azimuth=self.azi)

        self.assertTrue(peaks_a == peaks_a)
        self.assertFalse(peaks_a == peaks_b)
        self.assertFalse(peaks_a == peaks_c)
        self.assertFalse(peaks_a == peaks_d)
        self.assertFalse(peaks_a == peaks_e)
        self.assertFalse(peaks_a == peaks_f)
        self.assertFalse(peaks_a == peaks_g)
        self.assertFalse(peaks_a == peaks_h)
        self.assertTrue(peaks_a == peaks_i)


if __name__ == "__main__":
    unittest.main()
