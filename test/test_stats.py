"""Tests for PeaksSuite class."""

import logging
import warnings

import numpy as np

from swprocess.stats import Statistics

from testtools import unittest, TestCase, get_full_path

logger = logging.getLogger("swprocess")
logger.setLevel(logging.ERROR)


class Test_Statistics(TestCase):

    def test_sort_data(self):
        n = np.nan

        # Well-sorted matrix, requires no alteration.
        expected = np.array([[n, n, 1, 3, 5],
                             [n, n, n, 2, 3],
                             [n, 1, 2, 3, n],
                             [1, 2, 3, 4, n],
                             [1, 2, n, n, n]])
        returned = Statistics._sort_data(expected)
        self.assertArrayAlmostEqual(expected, returned, equal_nan=True)

        # Unsorted array requires rearranging.
        data = np.array([[n, 1, 2, 3, n],
                         [1, 2, n, n, n],
                         [n, n, 1, 3, 5],
                         [1, 2, 3, 4, n],
                         [n, n, n, 2, 3]])
        returned = Statistics._sort_data(data)
        self.assertArrayAlmostEqual(expected, returned, equal_nan=True)

        # Another example of an unsorted array, that require rearranging.
        data = np.array([[n, 1, 2, 3, 4, 5, n, n, n],
                         [n, n, n, n, 1, 2, 3, 4, n],
                         [n, n, n, n, n, 1, 2, 3, 4],
                         [n, n, n, n, 1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5, n, n, n, n],
                         [1, 2, 3, n, n, n, n, n, n],
                         ])

        expected = np.array([[n, n, n, n, n, 1, 2, 3, 4],
                             [n, n, n, n, 1, 2, 3, 4, 5],
                             [n, n, n, n, 1, 2, 3, 4, n],
                             [n, 1, 2, 3, 4, 5, n, n, n],
                             [1, 2, 3, 4, 5, n, n, n, n],
                             [1, 2, 3, n, n, n, n, n, n],
                             ])
        returned = Statistics._sort_data(data)
        self.assertArrayAlmostEqual(expected, returned, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
