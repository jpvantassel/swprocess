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

# """Tests for SpacCurve class."""

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# import swprocess
# from testtools import unittest, TestCase, get_path


# class Test_SpacCurve(TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.path = get_path(__file__)

#     def test_init(self):
#         frequencies = np.array([1., 2, 3])
#         ratios = np.array([1., 0.5, 0.2])
#         time = "7200"
#         component = 0
#         ring = 0
#         dmin = 100
#         dmax = 110

#         spaccurve = swprocess.SpacCurve(frequencies, ratios, time,
#                                         component, ring, dmin, dmax)

#         self.assertArrayEqual(frequencies, spaccurve.frequencies)
#         # self.assertIsNot(frequencies, spaccurve.frequencies)

#         self.assertArrayEqual(ratios, spaccurve.ratios)
#         self.assertIsNot(ratios, spaccurve.ratios)

#         self.assertEqual(time, spaccurve.time)

#         self.assertEqual(component, spaccurve.component)

#         self.assertEqual(ring, spaccurve.ring)

#         self.assertEqual(dmin, spaccurve.dmin)
#         self.assertIsNot(dmin, spaccurve.dmin)

#         self.assertEqual(dmax, spaccurve.dmax)
#         self.assertIsNot(dmax, spaccurve.dmax)

#     def test_theoretical_spac_ratio_function_custom(self):
#         # component=0; vertical.
#         fname = self.path / "data/mspac/bessel_vr=400_bettig2001.csv"
#         df = pd.read_csv(fname)

#         spaccurve = swprocess.SpacCurve([0.1, 0.2], [0.1, 0.2],
#                                         time="0", component=0, ring=0,
#                                         dmin=352, dmax=383)
#         func = spaccurve.theoretical_spac_ratio_function_custom()
#         expected = df.ratios.to_numpy()
#         returned = func(df.frequencies.to_numpy(), 400)
#         self.assertArrayAlmostEqual(expected, returned)

#         # component=1; radial.
#         spaccurve = swprocess.SpacCurve([0.1, 0.2], [0.1, 0.2],
#                                         time="0", component=1, ring=0,
#                                         dmin=352, dmax=383)
#         self.assertRaises(NotImplementedError,
#                           spaccurve.theoretical_spac_ratio_function_custom)

#         # component=2; transverse.
#         spaccurve = swprocess.SpacCurve([0.1, 0.2], [0.1, 0.2],
#                                         time="0", component=2, ring=0,
#                                         dmin=352, dmax=383)
#         self.assertRaises(NotImplementedError,
#                           spaccurve.theoretical_spac_ratio_function_custom)

#     # def test_fit_to_theoretical(self):
#     #     # known solution with vr = 400.
#     #     fname = self.path / "data/mspac/bessel_vr=400_bettig2001.csv"
#     #     df = pd.read_csv(fname)

#     #     spaccurve = swprocess.SpacCurve(df.frequencies, df.ratios,
#     #                                     time="0", component=0, ring=0,
#     #                                     dmin=352, dmax=383)
#     #     frequencies, vrs = spaccurve.fit_to_theoretical(vrange=(100, 1000))
#     #     self.assertArrayEqual(df.frequencies.to_numpy(), frequencies)
#     #     self.assertArrayAlmostEqual(np.ones_like(frequencies)*400, vrs)



# if __name__ == "__main__":
#     unittest.main()
