"""Tests for PeaksSuite class."""

import logging
import warnings

import numpy as np

from swprocess.stats import Statistics

from testtools import unittest, TestCase, get_full_path

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
        e_regions = [((0,0), (3,3)), ((1,3), (3,5))]
        for expected, returned in zip(e_regions, r_regions):
            self.assertTupleEqual(expected, returned)

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
