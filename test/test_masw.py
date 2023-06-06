# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Tests for class Masw."""

import unittest
import swprocess
from testtools import TestCase, unittest, get_path


class Test_Masw(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = get_path(__file__)
        cls.wghs_path = cls.path / "../examples/masw/data/wghs/"

    def test_run(self):
        @swprocess.register.MaswWorkflowRegistry.register("dummy")
        class DummyMaswWorkflow(swprocess.maswworkflows.AbstractMaswWorkflow):
            def run(self):
                return 0

            def __str__(self):
                return "DummyMaswWorkflow"

        settings = swprocess.Masw.create_settings_dict(workflow="dummy")
        self.assertEqual(0, swprocess.Masw.run(fnames="dummy_file",
                                               settings=settings))
        # os.remove("dummy_settings")

    # TODO (jpv): Need to review test coverage here.
    # def test_create_settings_file(self):
    #     efname = self.path / "data/settings/settings_test.json"
    #     with open(efname, "r") as f:
    #         expected = json.load(f)

    #     rfname = "tmp.json"
    #     swprocess.Masw.create_settings_file(fname=rfname)
    #     with open(rfname, "r") as f:
    #         returned = json.load(f)
    #     self.assertDictEqual(expected, returned)
    #     os.remove(rfname)


if __name__ == "__main__":
    unittest.main()
