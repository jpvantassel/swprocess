"""Tests for SpacCurveSuite class."""

import logging

import numpy as np
import matplotlib.pyplot as plt

import swprocess
from testtools import unittest, TestCase, get_full_path

logger = logging.getLogger("swprocess")
logger.setLevel(logging.ERROR)


class Test_SpacCurveSuite(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_from_max(self):
        fname = self.full_path + "data/mspac/mspac_c0.max"
        scsuite = swprocess.SpacCurveSuite.from_max(fname)



if __name__ == "__main__":
    unittest.main()
