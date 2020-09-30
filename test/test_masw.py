"""Tests for class Masw."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

import unittest
import swprocess
from testtools import TestCase, unittest, get_full_path


class Test_Masw(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)
        cls.wghs_path = cls.full_path + "../examples/sample_data/wghs/"

    def test_run(self):
        @swprocess.register.MaswWorkflowRegistry.register("dummy")
        class DummyMaswWorkflow(swprocess.maswworkflows.AbstractMaswWorkflow):
            def run(self):
                return 0

            def __str__(self):
                return "DummyMaswWorkflow"

        swprocess.Masw.create_settings_file(fname="dummy_settings",
                                            workflow="dummy")
        self.assertEqual(0, swprocess.Masw.run(fnames="dummy_file",
                                               settings_fname="dummy_settings"))
        os.remove("dummy_settings")

    def test_create_settings_file(self):
        efname = self.full_path + "data/settings/settings_test.json"
        with open(efname, "r") as f:
            expected = json.load(f)

        rfname = "tmp.json"
        swprocess.Masw.create_settings_file(fname=rfname)
        with open(rfname, "r") as f:
            returned = json.load(f)
        self.assertDictEqual(expected, returned)
        os.remove(rfname)


if __name__ == "__main__":
    unittest.main()
