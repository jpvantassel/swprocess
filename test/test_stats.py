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

"""Tests for PeaksSuite class."""

import logging
from unittest.mock import MagicMock

import numpy as np
from numpy.random import default_rng, PCG64

from swprocess.stats import Statistics

from testtools import unittest, TestCase

logger = logging.getLogger("swprocess")
logger.setLevel(logging.ERROR)


class Test_Statistics(TestCase):

    def test_sort(self):
        n = np.nan

        # Well-sorted matrix, requires no alteration.
        expected = np.array([[1, 2, 3, n, n],
                             [1, 2, n, n, n],
                             [n, 1, 2, 3, n],
                             [n, 1, 2, 3, 4],
                             [n, n, n, 1, 2]])
        returned = Statistics._sort(expected)
        self.assertArrayAlmostEqual(expected, returned, equal_nan=True)

        # Unsorted array requires rearranging.
        data = np.array([[n, 1, 2, 3, n],
                         [n, n, n, 1, 2],
                         [1, 2, 3, n, n],
                         [n, 1, 2, 3, 4],
                         [1, 2, n, n, n]])
        returned = Statistics._sort(data)
        self.assertArrayAlmostEqual(expected, returned, equal_nan=True)

        # Another example of an unsorted array, that require rearranging.
        data = np.array([[n, n, n, 1, 2, 3, 4, 5, n],
                         [n, 1, 2, 3, 4, n, n, n, n],
                         [1, 2, 3, 4, n, n, n, n, n],
                         [1, 2, 3, 4, 5, n, n, n, n],
                         [n, n, n, n, 1, 2, 3, 4, 5],
                         [n, n, n, n, n, n, 1, 2, 3],
                         ])

        expected = np.array([[1, 2, 3, 4, n, n, n, n, n],
                             [1, 2, 3, 4, 5, n, n, n, n],
                             [n, 1, 2, 3, 4, n, n, n, n],
                             [n, n, n, 1, 2, 3, 4, 5, n],
                             [n, n, n, n, 1, 2, 3, 4, 5],
                             [n, n, n, n, n, n, 1, 2, 3],
                             ])
        returned = Statistics._sort(data)
        self.assertArrayAlmostEqual(expected, returned, equal_nan=True)

    def test_identify_regions(self):
        n = np.nan

        # Full matrix, will succeed regardless of threshold.
        data = np.array([[1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5]])

        for threshold in [0, 0.5, 1]:
            [(tl, br)] = Statistics._identify_regions(data,
                                                      density_threshold=threshold)
            self.assertTupleEqual((0, 0), tl)
            self.assertTupleEqual((3, 5), br)

        # P-full matrix, result dependant on threshold.
        data = np.array([[1, 2, 3, n, n],
                         [1, 2, 3, 4, 5],
                         [n, n, 3, 4, 5]])

        # If threshold below min(7/9=0.78, 9/12=0.75, 11/15=0.73) -> single region
        for threshold in [0, 4/6, 11/15]:
            [(tl, br)] = Statistics._identify_regions(data,
                                                      density_threshold=threshold)
            self.assertTupleEqual((0, 0), tl)
            self.assertTupleEqual((3, 5), br)

        # If threshold above 7/9=0.77 -> separate regions
        threshold = 0.77
        r_regions = Statistics._identify_regions(data,
                                                 density_threshold=threshold)
        e_regions = [((0, 0), (3, 3)), ((1, 3), (3, 5))]
        for expected, returned in zip(e_regions, r_regions):
            self.assertTupleEqual(expected, returned)

    def test_fill(self):
        n = np.nan

        # Full matrix, no fill required.
        data = np.array([[1, 2, 3],
                         [1, 2, 3],
                         [1, 2, 3]])

        returned = Statistics._fill_data(data)
        self.assertArrayEqual(data, returned)

        # P. full matrix, with locked seed rng.
        pdata = np.array([[1., n, 3, 4, 5],
                          [3., 2, 8, n, 5],
                          [1., n, 3, 4, n],
                          [2., 1, n, 1, 8],
                          [1., 2, 3, 4, 5]])

        #  Expected result
        means = np.nanmean(pdata, axis=0)
        stddevs = np.nanstd(pdata, axis=0, ddof=1)
        means[0] = means[1]
        stddevs[0] = stddevs[1]
        rng = default_rng(PCG64(seed=1994))
        expected = np.array(pdata)
        rows = [0, 2, 3, 1, 2]
        cols = [1, 1, 2, 3, 4]
        for row, col, mean, stddev in zip(rows, cols, means, stddevs):
            expected[row, col] = rng.normal(mean, stddev)

        #  Returned result
        rng = default_rng(PCG64(seed=1994))
        returned = Statistics._fill_data(pdata, rng=rng)
        self.assertArrayEqual(expected, returned)

        # P. full matrix, with no uncertainty.
        pdata = np.array([[1., n, 3, 4, 5],
                          [1., 2, 3, n, 5],
                          [1., n, 3, 4, n],
                          [1., 2, n, 4, 5],
                          [1., 2, 3, 4, 5]])
        returned = Statistics._fill_data(pdata)
        expected = np.array([[1, 2, 3, 4, 5] for _ in range(5)], dtype=float)
        self.assertArrayEqual(expected, returned)

        # P. full matrix, with mocked rng.
        pdata = np.array([[1, 2, 3],
                          [1, 2, 3],
                          [1, n, 3]])
        mock_rng = MagicMock()
        mock_rng.normal.return_value = 2
        returned = Statistics._fill_data(pdata, rng=mock_rng)
        self.assertArrayEqual(data, returned)
        mock_rng.normal.assert_called_once()

    def test_calc_density(self):
        n = np.nan

        # Full matrix.
        data = np.array([[1, 2, 3],
                         [1, 2, 3],
                         [1, 2, 3]])
        #   Full matrix
        expected = 1.
        returned = Statistics._calc_density(data,
                                            tl_corner=(0, 0),
                                            br_corner=(2, 2))
        self.assertEqual(expected, returned)
        #   Single cell
        expected = 1.
        returned = Statistics._calc_density(data,
                                            tl_corner=(0, 0),
                                            br_corner=(0, 0))
        self.assertEqual(expected, returned)

        # Semi-full matrix
        data = np.array([[1, 2, n],
                         [1, n, 3],
                         [1, 2, n]])
        #  Full matrix
        expected = 6/9
        returned = Statistics._calc_density(data,
                                            tl_corner=(0, 0),
                                            br_corner=(2, 2))
        self.assertEqual(expected, returned)
        #  Subset of matrix
        expected = 3/4
        returned = Statistics._calc_density(data,
                                            tl_corner=(0, 0),
                                            br_corner=(1, 1))
        self.assertEqual(expected, returned)


if __name__ == "__main__":
    unittest.main()
