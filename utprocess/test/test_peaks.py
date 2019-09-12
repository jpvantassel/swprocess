"""Tests for Peaks abstract class."""
import unittest
import utprocess


class TestPeaks(unittest.TestCase):
    def test_init(self):
        frequency = [100, 50, 30, 10, 5, 3]
        velocity = [100, 120, 130, 140, 145, 150]
        identifier = '-5m'
        my_peaks = utprocess.Peaks(frequency=frequency,
                                   velocity=velocity,
                                   identifier=identifier)
        

if __name__ == "__main__":
    unittest.main()
