import unittest

class TestCase(unittest.TestCase):
    def assertArrayEqual(self, array1, array2):
        self.assertListEqual(array1.tolist(), array2.tolist())

    def assertArrayAlmostEqual(self, array1, array2, **kwargs):
        assert(array1.size == array2.size)
        array1 = array1.flatten()
        array2 = array2.flatten()
        for v1, v2 in zip(array1, array2):
            self.assertAlmostEqual(v1, v2, **kwargs)
