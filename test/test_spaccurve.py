"""Tests for SpacCurve class."""

import numpy as np
import matplotlib.pyplot as plt

import swprocess
from testtools import unittest, TestCase, get_full_path


class Test_SpacCurve(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_init(self):
        frequencies = np.array([1., 2, 3])
        ratios = np.array([1., 0.5, 0.2])
        time = "7200"
        component = 0
        ring = 0
        dmin = 100
        dmax = 110

        spaccurve = swprocess.SpacCurve(frequencies, ratios, time,
                                        component, ring, dmin, dmax)

        self.assertArrayEqual(frequencies, spaccurve.frequencies)
        self.assertIsNot(frequencies, spaccurve.frequencies)

        self.assertArrayEqual(ratios, spaccurve.ratios)
        self.assertIsNot(ratios, spaccurve.ratios)

        self.assertEqual(time, spaccurve.time)

        self.assertEqual(component, spaccurve.component)

        self.assertEqual(ring, spaccurve.ring)

        self.assertEqual(dmin, spaccurve.dmin)
        self.assertIsNot(dmin, spaccurve.dmin)

        self.assertEqual(dmax, spaccurve.dmax)
        self.assertIsNot(dmax, spaccurve.dmax)


if __name__ == "__main__":
    unittest.main()
