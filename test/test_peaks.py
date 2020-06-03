"""Tests for Peaks class."""

import json
import os
import warnings
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
        self.assertArrayEqual(ray.frequency, np.array([20.000000000000106581,
                                                       19.282217609815102577]))
        self.assertArrayEqual(ray.velocity, 1/np.array([0.0068859013683322750979,
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
        self.assertArrayEqual(lov.frequency, np.array([20.000000000000106581,
                                                       19.282217609815102577]))
        self.assertArrayEqual(lov.velocity, np.array([1/0.0088200863560403078983,
                                                      1/0.0089530611050798007688]))
        self.assertArrayEqual(lov.azi, np.array([252.05441718438927978,
                                                 99.345595852002077208]))
        self.assertArrayEqual(lov.ell, np.array([0, 0]))
        self.assertArrayEqual(lov.noi, np.array([0, 0]))
        self.assertArrayEqual(lov.pwr, np.array([3832630.8840260845609,
                                                 4039408.6602126094513]))
        self.assertArrayEqual(ray.tim, np.array([16200,
                                                 16200]))

    def test_reject(self):
        xs = np.array([1, 2, 4, 5, 1, 2, 6, 4, 9, 4])
        ys = np.array([1, 5, 8, 6, 4, 7, 7, 1, 3, 5])

        xlims = (4.5, 7.5)
        xmin, xmax = xlims
        ylims = (3.5, 8.5)
        ymin, ymax = ylims

        # Helper function
        returned = swprocess.Peaks._reject_inside_ids(xs, xmin, xmax,
                                                      ys, ymin, ymax)
        expected = np.array([3, 6])
        self.assertArrayEqual(expected, returned)

        # Method -> Reject on frequency and velocity
        other = np.arange(10)
        peaks = swprocess.Peaks(xs, ys, other=other)
        peaks.reject(xtype="frequency", xlims=xlims,
                     ytype="velocity", ylims=ylims)

        keep_ids = [0, 1, 2, 4, 5, 7, 8, 9]
        attrs = dict(frequency=xs, velocity=ys, other=other)
        for attr, value in attrs.items():
            self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

        # Method -> Reject on frequency and other
        other = np.arange(10)
        peaks = swprocess.Peaks(xs, ys, other=other)
        peaks.reject(xtype="frequency", xlims=xlims,
                     ytype="other", ylims=ylims)

        keep_ids = [0, 1, 2, 3, 4, 5, 7, 8, 9]
        attrs = dict(frequency=xs, velocity=ys, other=other)
        for attr, value in attrs.items():
            self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

    def test_blitz(self):
        xs = np.array([1, 5, 9, 3, 5, 7, 4, 6, 2, 8])
        ys = np.array([1, 9, 4, 5, 6, 8, 5, 2, 1, 4])

        ## Helper function

        # Remove all above 4.5
        _min, _max = None, 4.5
        expected = np.array([1, 2, 4, 5, 7, 9])
        returned = swprocess.Peaks._reject_outside_ids(xs, _min, _max)
        self.assertArrayEqual(expected, returned)

        # Remove all below 4.5
        _min, _max = 4.5, None
        expected = np.array([0, 3, 6, 8])
        returned = swprocess.Peaks._reject_outside_ids(xs, _min, _max)
        self.assertArrayEqual(expected, returned)

        # Remove all below 1.5 and above 6.5
        _min, _max = 1.5, 6.5
        expected = np.array([0, 2, 5, 9])
        returned = swprocess.Peaks._reject_outside_ids(xs, _min, _max)
        self.assertArrayEqual(expected, returned)

        # None
        _min, _max = None, None
        expected = np.array([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returned = swprocess.Peaks._reject_outside_ids(xs, _min, _max)
        self.assertArrayEqual(expected, returned)

        ## Method

        # Remove all below 0.5 and above 6.5
        limits = (0.5, 6.5)

        other = np.arange(10)
        peaks = swprocess.Peaks(frequency=xs, velocity=ys, other=other)
        peaks.blitz("frequency", limits)

        keep_ids = [0, 1, 3, 4, 6, 7, 8]
        attrs = dict(frequency=xs, velocity=ys, other=other)
        for attr, value in attrs.items():
            self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])


if __name__ == "__main__":
    unittest.main()
