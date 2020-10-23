"""Tests for PeaksSuite class."""

import logging

from swprocess.regex import get_all, get_peak_from_max
from testtools import unittest, TestCase, get_full_path

logger = logging.getLogger("swprocess")
logger.setLevel(logging.ERROR)


class Test_PeaksSuite(TestCase):

    # TODO (jpv): Add this to the test cases.
    def test_get_peak_from_max(self):
        pass
        # txt = "20201021184000.000000 0.88707185499315710508 Rayleigh 0.0052274209500743342924 146.16012705150347983 -9.7624928074594592431 4.1036102507755423119 2.4140442228149637278e-05 1"