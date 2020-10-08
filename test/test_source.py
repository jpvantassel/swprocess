"""Test for Source class."""

from swprocess import Source
from testtools import get_full_path, unittest, TestCase


class Test_Utils(TestCase):

    def test_init(self):
        location = dict(x=0., y=5., z=6.)
        source = Source(**location)

        for key, expected in location.items():
            returned = getattr(source, key)
            self.assertEqual(expected, returned)

    def test_from_source(self):
        location = dict(x=7., y=2., z=0.)
        source_a = Source(**location)
        source_b = Source.from_source(source_a)
        self.assertEqual(source_a, source_b)
        self.assertNotEqual(id(source_a), id(source_b))

    def test_repr(self):
        location = dict(x=7., y=0., z=9.)
        source = Source(**location)

        expected = f"Source(x={location['x']}, y={location['y']}, z={location['z']})"
        returned = source.__repr__()
        self.assertEqual(expected, returned)

    def test_eq(self):
        location = dict(x=1., y=2., z=2.)
        expected = Source(**location)

        a = "I am not a Source object."
        b = Source(**{**location, **dict(x=0.)})
        c = Source(**{**location, **dict(y=3.)})
        d = Source(**{**location, **dict(z=4.)})
        e = Source(**location)
        f = Source.from_source(expected)

        self.assertNotEqual(expected, a)
        self.assertNotEqual(expected, b)
        self.assertNotEqual(expected, c)
        self.assertNotEqual(expected, d)

        self.assertEqual(expected, e)
        self.assertEqual(expected, f)


if __name__ == "__main__":
    unittest.main()
