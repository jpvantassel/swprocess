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
