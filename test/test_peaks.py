"""Tests for Peaks class."""

import json
import os
import warnings
import logging
from unittest.mock import MagicMock

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

        cls._id2 = "10m"
        cls.frq2 = [[80., 60, 20],
                    [75., 65, 30]]
        cls.vel2 = [[50., 70, 85],
                    [55., 80, 75]]
        cls.azi2 = [[10., 15, 20],
                    [35, 10, 20]]
        cls.ell2 = [[11., 15, 20],
                    [35, 11, 20]]
        cls.noi2 = [[12., 15, 20],
                    [35, 12, 20]]
        cls.pwr2 = [[13., 15, 20],
                    [35, 13, 20]]

        cls._id2n = "nan"
        cls.frq2n = [[np.nan, 60, 20],
                     [75., 65, np.nan]]
        cls.vel2n = [[np.nan, 70, 85],
                     [55., 80, np.nan]]
        cls.azi2n = [[np.nan, 15, 20],
                     [35, 10, np.nan]]
        cls.ell2n = [[np.nan, 15, 20],
                     [35, 11, np.nan]]
        cls.noi2n = [[np.nan, 15, 20],
                     [35, 12, np.nan]]
        cls.pwr2n = [[np.nan, 15, 20],
                     [35, 13, np.nan]]
        cls.valid = np.array([[False, True, True],[True, True, False]])

    def test_init(self):
        # 1D: No keyword arguments
        peaks = Peaks(self.frq, self.vel, identifier=self._id)
        self.assertArrayEqual(np.array(self.frq), peaks.frequency)
        self.assertArrayEqual(np.array(self.vel), peaks.velocity)
        self.assertEqual(self._id, peaks.identifier)

        # 1D: Four keyword arguments
        peaks = Peaks(self.frq, self.vel, identifier=self._id, azi=self.azi,
                      ell=self.ell, noi=self.noi, pwr=self.pwr)
        self.assertArrayEqual(np.array(self.frq), peaks.frequency)
        self.assertArrayEqual(np.array(self.vel), peaks.velocity)
        self.assertEqual(self._id, peaks.identifier)
        self.assertArrayEqual(np.array(self.azi), peaks.azi)
        self.assertArrayEqual(np.array(self.ell), peaks.ell)
        self.assertArrayEqual(np.array(self.noi), peaks.noi)
        self.assertArrayEqual(np.array(self.pwr), peaks.pwr)

        # 2D: Four keyword arguments
        peaks = Peaks(self.frq2, self.vel2, identifier=self._id2,
                      azi=self.azi2, ell=self.ell2, noi=self.noi2,
                      pwr=self.pwr2)
        self.assertArrayEqual(np.array(self.frq2).flatten(), peaks.frequency)
        self.assertArrayEqual(np.array(self.vel2).flatten(), peaks.velocity)
        self.assertEqual(self._id2, peaks.identifier)
        self.assertArrayEqual(np.array(self.azi2).flatten(), peaks.azi)
        self.assertArrayEqual(np.array(self.ell2).flatten(), peaks.ell)
        self.assertArrayEqual(np.array(self.noi2).flatten(), peaks.noi)
        self.assertArrayEqual(np.array(self.pwr2).flatten(), peaks.pwr)

        # 2D: Four keyword arguments with nans
        peaks = Peaks(self.frq2n, self.vel2n, identifier=self._id2n,
                      azi=self.azi2n, ell=self.ell2n, noi=self.noi2n,
                      pwr=self.pwr2n)
        self.assertArrayEqual(np.array(self.frq2n)[self.valid], peaks.frequency)
        self.assertArrayEqual(np.array(self.vel2n)[self.valid], peaks.velocity)
        self.assertEqual(self._id2n, peaks.identifier)
        self.assertArrayEqual(np.array(self.azi2n)[self.valid], peaks.azi)
        self.assertArrayEqual(np.array(self.ell2n)[self.valid], peaks.ell)
        self.assertArrayEqual(np.array(self.noi2n)[self.valid], peaks.noi)
        self.assertArrayEqual(np.array(self.pwr2n)[self.valid], peaks.pwr)

    def test_properties(self):
        peaks = Peaks(self.frq, self.vel, self._id, azimuth=self.azi)

        # Wavelength
        self.assertArrayEqual(np.array(self.vel)/np.array(self.frq),
                              peaks.wavelength)

        # Extended Attrs
        expected = ["frequency", "velocity", "azimuth",
                    "wavelength", "slowness", "wavenumber"]
        returned = peaks.extended_attrs
        self.assertListEqual(expected, returned)

        # Wavenumber
        self.assertArrayAlmostEqual(2*np.pi/peaks.wavelength,
                                    peaks.wavenumber)

    def test_prepare_types(self):
        # Acceptable (will cast)
        kwargs = dict(xtype="frequency", ytype="velocity")
        returned_xtype, returned_ytype = Peaks._prepare_types(**kwargs)
        self.assertListEqual(["frequency"], returned_xtype)
        self.assertListEqual(["velocity"], returned_ytype)

        # Acceptable (no cast)
        kwargs = dict(xtype=["frequency"], ytype=["velocity"])
        returned_xtype, returned_ytype = Peaks._prepare_types(**kwargs)
        self.assertListEqual(["frequency"], returned_xtype)
        self.assertListEqual(["velocity"], returned_ytype)

        # Unacceptable (raise TypeError)
        kwargs = dict(xtype=5, ytype="velocity")
        self.assertRaises(TypeError, Peaks._prepare_types, **kwargs)

        # Unacceptable (raise IndexError)
        kwargs = dict(xtype=["frequency", "wavelength"], ytype="velocity")
        self.assertRaises(IndexError, Peaks._prepare_types, **kwargs)

    def test__plot(self):
        peaks = Peaks(self.frq, self.vel, self._id)

        # Standard
        fig, ax = plt.subplots()
        peaks._plot(ax=ax, xtype="frequency", ytype="velocity")

        # Bad Attribute
        fig, ax = plt.subplots()
        self.assertRaises(AttributeError, peaks._plot, ax=ax,
                          xtype="magic", ytype="size_of_unicorn")

        plt.show(block=False)
        plt.close("all")

    def test_configure_axes(self):
        peaks = Peaks(self.frq, self.vel, self._id)
        defaults = {"frequency": {"label": "frq",
                                  "scale": "linear"},
                    "velocity": {"label": "vel",
                                 "scale": "log"}}

        # Standard calls.
        ax = MagicMock()
        peaks._configure_axes(ax, xtype="frequency", ytype="velocity",
                              defaults=defaults)
        ax.set_xlabel.assert_called_with("frq")
        ax.set_xscale.assert_called_with("linear")
        ax.set_ylabel.assert_called_with("vel")
        ax.set_yscale.assert_called_with("log")

        # Non-standard calls to unknown attribute.
        ax = MagicMock()
        peaks._configure_axes(ax, xtype="frequency", ytype="azimuth",
                              defaults=defaults)
        ax.set_xlabel.assert_called_with("frq")
        ax.set_xscale.assert_called_with("linear")
        ax.set_ylabel.assert_not_called()
        ax.set_yscale.assert_not_called()

    # def test_blitz(self):
    #     xs = np.array([1, 5, 9, 3, 5, 7, 4, 6, 2, 8])
    #     ys = np.array([1, 9, 4, 5, 6, 8, 5, 2, 1, 4])

    #     # Remove all above 4.5
    #     _min, _max = None, 4.5
    #     expected = np.array([1, 2, 4, 5, 7, 9])
    #     returned = Peaks._reject_outside_ids(xs, _min, _max)
    #     self.assertArrayEqual(expected, returned)

    #     # Remove all below 4.5
    #     _min, _max = 4.5, None
    #     expected = np.array([0, 3, 6, 8])
    #     returned = Peaks._reject_outside_ids(xs, _min, _max)
    #     self.assertArrayEqual(expected, returned)

    #     # Remove all below 1.5 and above 6.5
    #     _min, _max = 1.5, 6.5
    #     expected = np.array([0, 2, 5, 9])
    #     returned = Peaks._reject_outside_ids(xs, _min, _max)
    #     self.assertArrayEqual(expected, returned)

    #     # None
    #     _min, _max = None, None
    #     expected = np.array([])
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         returned = Peaks._reject_outside_ids(xs, _min, _max)
    #     self.assertArrayEqual(expected, returned)

    #     # Remove all below 0.5 and above 6.5
    #     limits = (0.5, 6.5)

    #     other = np.arange(10)
    #     peaks = Peaks(frequency=xs, velocity=ys, other=other)
    #     peaks.blitz("frequency", limits)

    #     keep_ids = [0, 1, 3, 4, 6, 7, 8]
    #     attrs = dict(frequency=xs, velocity=ys, other=other)
    #     for attr, value in attrs.items():
    #         self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

    # def test_reject(self):
    #     xs = np.array([1, 2, 4, 5, 1, 2, 6, 4, 9, 4])
    #     ys = np.array([1, 5, 8, 6, 4, 7, 7, 1, 3, 5])

    #     xlims = (4.5, 7.5)
    #     xmin, xmax = xlims
    #     ylims = (3.5, 8.5)
    #     ymin, ymax = ylims

    #     # test_reject_inside_ids
    #     returned = Peaks._reject_inside_ids(xs, xmin, xmax, ys, ymin, ymax)
    #     expected = np.array([3, 6])
    #     self.assertArrayEqual(expected, returned)

    #     # Method -> Reject on frequency and velocity
    #     other = np.arange(10)
    #     peaks = Peaks(xs, ys, other=other)
    #     peaks.reject(xtype="frequency", xlims=xlims,
    #                  ytype="velocity", ylims=ylims)

    #     keep_ids = [0, 1, 2, 4, 5, 7, 8, 9]
    #     attrs = dict(frequency=xs, velocity=ys, other=other)
    #     for attr, value in attrs.items():
    #         self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

    #     # Method -> Reject on frequency and other
    #     other = np.arange(10)
    #     peaks = Peaks(xs, ys, other=other)
    #     peaks.reject(xtype="frequency", xlims=xlims,
    #                  ytype="other", ylims=ylims)

    #     keep_ids = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    #     attrs = dict(frequency=xs, velocity=ys, other=other)
    #     for attr, value in attrs.items():
    #         self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

    #     # Method -> Reject on slowness and velocity
    #     xs = np.array([1, 5, 8, 6, 4, 7, 7, 1, 3, 5])
    #     ys = 1/np.array([1, 5, 9, 4, 5, 6, 7, 8, 1, 7])

    #     peaks = Peaks(xs, ys)
    #     peaks.reject(xtype="frequency", xlims=(0, 10),
    #                  ytype="slowness", ylims=(4.5, 7.5))

    #     keep_ids = [0, 2, 3, 7, 8]
    #     attrs = dict(frequency=xs, velocity=ys)
    #     for attr, value in attrs.items():
    #         self.assertArrayEqual(getattr(peaks, attr), value[keep_ids])

    # def test_from_dict(self):
    #     # Basic Case: No keyword arguments
    #     data = {"frequency": self.frq, "velocity": self.vel}
    #     peaks = Peaks.from_dict(data, identifier=self._id)
    #     self.assertArrayEqual(np.array(self.frq), peaks.frequency)
    #     self.assertArrayEqual(np.array(self.vel), peaks.velocity)
    #     self.assertEqual(self._id, peaks.identifier)

    #     # Advanced Case: Four keyword arguments
    #     data = {"frequency": self.frq, "velocity": self.vel,
    #             "azi": self.azi, "ell": self.ell, "noi": self.noi,
    #             "pwr": self.pwr}
    #     peaks = Peaks.from_dict(data, identifier=self._id)
    #     self.assertArrayEqual(np.array(self.azi), peaks.azi)
    #     self.assertArrayEqual(np.array(self.ell), peaks.ell)
    #     self.assertArrayEqual(np.array(self.noi), peaks.noi)
    #     self.assertArrayEqual(np.array(self.pwr), peaks.pwr)

    #     # Advanced Case: Two keyword arguments
    #     data = {"frequency": self.frq, "velocity": self.vel,
    #             "azi": self.azi, "pwr": self.pwr}
    #     peaks = Peaks.from_dict(data, identifier=self._id)
    #     self.assertArrayEqual(np.array(self.azi), peaks.azi)
    #     self.assertArrayEqual(np.array(self.pwr), peaks.pwr)

    #     # Bad: missing frequency or velocity
    #     del data["frequency"]
    #     self.assertRaises(TypeError, Peaks.from_dict, data)

    # def test_to_and_from_json(self):
    #     # Standard to_json and from_json
    #     fname = "test.json"
    #     expected = Peaks(self.frq, self.vel, self._id,
    #                      azi=self.azi, pwr=self.pwr)
    #     expected.to_json(fname)
    #     returned = Peaks.from_json(fname)
    #     self.assertEqual(expected, returned)
    #     os.remove(fname)

    #     # Deprecated write_peak_json
    #     expected = Peaks(self.frq, self.vel, self._id,
    #                      azi=self.azi, pwr=self.pwr)
    #     fname = "test_1.json"
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         expected.write_peak_json(fname)
    #     returned = Peaks.from_json(fname)
    #     self.assertEqual(expected, returned)
    #     os.remove(fname)

    #     # to_json append data does not exist
    #     fname = "test_2.json"
    #     peaks = Peaks(self.frq, self.vel, "org")
    #     peaks.to_json(fname)
    #     peaks.identifier = "app"
    #     peaks.to_json(fname, append=True)
    #     suite = swprocess.PeaksSuite.from_json(fname)
    #     for _peaks, _id in zip(suite, suite.ids):
    #         peaks.identifier = _id
    #         self.assertEqual(peaks, _peaks)

    #     # to_json append data already exists
    #     self.assertRaises(KeyError, peaks.to_json, fname, append=True)
    #     os.remove(fname)

    #     # to_json overwrite
    #     fname = "test_3.json"
    #     peaks = Peaks(self.frq, self.vel, "org")
    #     peaks.to_json(fname)
    #     peaks.identifier = "app"
    #     peaks.to_json(fname, append=False)
    #     returned = Peaks.from_json(fname)
    #     self.assertEqual(peaks, returned)
    #     os.remove(fname)

    #     # from_json ignore multiple data
    #     fname = self.full_path + "data/peak/peaks_c2.json"
    #     peak_suite = swprocess.PeaksSuite.from_json(fname)
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         returned = Peaks.from_json(fname)
    #     expected = peak_suite[peak_suite.ids.index(returned.identifier)]
    #     self.assertEqual(expected, returned)

    # def test_from_max(self):
    #     # rayleigh, nmaxima=1, nblocksets=1, samples=30
    #     fname_max = self.full_path + "data/rtbf/rtbf_nblockset=1_nmaxima=1.max"
    #     fname_csv = self.full_path + "data/rtbf/rtbf_nblockset=1_nmaxima=1_r_parsed.csv"

    #     peaks = Peaks.from_max(fname=fname_max, wavetype="rayleigh")
    #     df = pd.read_csv(fname_csv)
    #     for attr in ["frequency", "slowness", "azimuth", "ellipticity", "noise", "power"]:
    #         self.assertArrayAlmostEqual(getattr(df, attr).to_numpy(),
    #                                     getattr(peaks, attr))

    #     # love, nmaxima=1, nblocksets=1, samples=30
    #     fname_max = self.full_path + "data/rtbf/rtbf_nblockset=1_nmaxima=1.max"
    #     fname_csv = self.full_path + "data/rtbf/rtbf_nblockset=1_nmaxima=1_l_parsed.csv"

    #     peaks = Peaks.from_max(fname=fname_max, wavetype="love")
    #     df = pd.read_csv(fname_csv)
    #     for attr in ["frequency", "slowness", "azimuth", "ellipticity", "noise", "power"]:
    #         self.assertArrayAlmostEqual(getattr(df, attr).to_numpy(),
    #                                     getattr(peaks, attr))

    #     # rayleigh, nmaxima=5, nblocksets=1, samples=3
    #     fname_max = self.full_path + "data/rtbf/rtbf_nblockset=1_nmaxima=5_trim.max"
    #     peaks = Peaks.from_max(fname=fname_max, wavetype="rayleigh")

    #     frequency = np.array([[26.78452072, 23.91368501, 21.35055305],
    #                           [26.78452072, 23.91368501, 21.35055305],
    #                           [26.78452072, 23.91368501, np.nan],
    #                           [26.78452072, 23.91368501, np.nan],
    #                           [26.78452072, 23.91368501, np.nan]])
    #     self.assertArrayAlmostEqual(frequency, peaks.frequency, equal_nan=True)

    #     slowness = np.array([[0.000440641, 0.001356121, 0.001527776],
    #                         [0.001276366, 0.001434077, 0.002798866],
    #                         [0.001855475, 0.00150539, np.nan],
    #                         [0.001911923, 0.001839504, np.nan],
    #                         [0.002008938, 0.002440085, np.nan]])
    #     self.assertArrayAlmostEqual(slowness, peaks.slowness, equal_nan=True)

    #     # TODO (jpv): Extend check to other attrs.

    #     # love, nmaxima=5, nblocksets=1, samples=3
    #     fname_max = self.full_path + "data/rtbf/rtbf_nblockset=1_nmaxima=5_trim.max"
    #     peaks = Peaks.from_max(fname=fname_max, wavetype="love")

    #     frequency = np.array([[26.78452072, 23.91368501, 21.35055305],
    #                           [26.78452072, 23.91368501, 21.35055305],
    #                           [26.78452072, 23.91368501, 21.35055305],
    #                           [26.78452072, 23.91368501, 21.35055305],
    #                           [np.nan, 23.91368501, 21.35055305]])
    #     self.assertArrayAlmostEqual(frequency, peaks.frequency, equal_nan=True)

    #     slowness = np.array([[0.000291155, 0.000410159, 0.001344777],
    #                         [0.000432074, 0.002521279, 0.001778382],
    #                         [0.001733716, 0.00257382, 0.002784536],
    #                         [0.001773228, 0.002602644, 0.003074123],
    #                         [np.nan, 0.002692776, 0.003118143]])
    #     self.assertArrayAlmostEqual(slowness, peaks.slowness, equal_nan=True)

    #     # TODO (jpv): Extend check to other attrs.

    #     # Bad wavetype
    #     self.assertRaises(ValueError, Peaks.from_max,
    #                       fname_max, wavetype="incorrect")

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
