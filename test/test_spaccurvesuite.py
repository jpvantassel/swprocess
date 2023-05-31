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

# """Tests for SpacCurveSuite class."""

# import logging

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# import swprocess
# from testtools import unittest, TestCase, get_path

# logger = logging.getLogger("swprocess")
# logger.setLevel(logging.ERROR)


# class Test_SpacCurveSuite(TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.full_path = get_path(__file__)

#     def test_from_max(self):
#         fname = self.path / "data/mspac/mspac_c0.max"
#         spaccurvesuite = swprocess.SpacCurveSuite.from_max(fname)

#         # 59 time windows * 1 component * 4 rings = 236
#         self.assertEqual(59*1*4, len(spaccurvesuite))

#         def compare(time, component, ring, df):
#             for index, curve in enumerate(spaccurvesuite):
#                 if curve.time == time and curve.component==component and curve.ring==ring:
#                     break
#             else:
#                 raise ValueError("invalid entry to compare(), no break detected.")
#             spaccurve = spaccurvesuite[index]
#             self.assertArrayAlmostEqual(df.frequencies.to_numpy(), spaccurve.frequencies, places=3)
#             self.assertArrayAlmostEqual(df.ratios.to_numpy(), spaccurve.ratios, places=3)
#             self.assertEqual(df.time[0], spaccurve.time)
#             self.assertEqual(df.component[0], spaccurve.component)
#             self.assertEqual(df.ring[0], spaccurve.ring)

#         # time=7201, component=0, ring=0
#         df = pd.read_csv(self.full_path+"data/mspac/t=7201_c=0_r=0.csv",
#                          dtype={"time":str})
#         compare(time="7201", component=0, ring=0, df=df)

#         # time=7441.01, component=0, ring=1
#         df = pd.read_csv(self.full_path+"data/mspac/t=7441.01_c=0_r=1.csv",
#                          dtype={"time":str})
#         compare(time="7441.01", component=0, ring=1, df=df)

#         # time=7561.02, component=0, ring=2
#         df = pd.read_csv(self.full_path+"data/mspac/t=7561.02_c=0_r=2.csv",
#                          dtype={"time":str})
#         compare(time="7561.02", component=0, ring=2, df=df)

#         # bad time reference
#         fname = self.path / "data/mspac/mspac_c0_abstimereference.max"
#         self.assertRaises(ValueError, swprocess.SpacCurveSuite.from_max, fname)

# if __name__ == "__main__":
#     unittest.main()
