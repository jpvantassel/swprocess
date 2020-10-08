"""Test for register module."""

import warnings

from testtools import unittest, TestCase
from swprocess.register import AbstractRegistry


class Test_Register(TestCase):

    def test_register(self):
        # Create a MockRegistry, inheriting from AbstractRegistry.
        class MockRegistry(AbstractRegistry):

            _register = {}

        # Register a MockItem, into the MockRegistry.
        @MockRegistry.register("mock")
        class MockItem():
            def __init__(self):
                self.name = "I am a MockItem"

        # Create class -> MockItem1
        self.assertEqual(MockItem, MockRegistry.create_class("mock"))

        # Create instance -> MockItem1
        mock = MockRegistry.create_instance("mock")
        self.assertEqual("I am a MockItem", mock.name)

        # Replace instance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            @MockRegistry.register("mock")
            class MockItem2():
                def __init__(self):
                    self.name = "I am a MockItem2"

        # Create class -> MockItem2
        self.assertEqual(MockItem2, MockRegistry.create_class("mock"))

        # Create instance -> MockItem2
        mock = MockRegistry.create_instance("mock")
        self.assertEqual("I am a MockItem2", mock.name)


if __name__ == "__main__":
    unittest.main()
