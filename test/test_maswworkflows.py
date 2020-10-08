"""Test for maswworkflows module."""

from unittest.mock import patch, Mock

import swprocess
from testtools import unittest, TestCase


class MockArray1D():

    def __init__(self, *args, **kwargs):
        self.sensors = [Mock()]

    @classmethod
    def from_files(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_array1d(cls, *args, **kwargs):
        return Mock()

    @property
    def _source_inside(self):
        return False

    def __getitem__(self, index):
        return self.sensors[index]


class SubMockArray1D(MockArray1D):
    @property
    def _source_inside(self):
        return True


class Test_MaswWorkflows(TestCase):

    @patch("swprocess.maswworkflows.Array1D", new=MockArray1D)
    def test_time_domain(self):
        Workflow = swprocess.register.MaswWorkflowRegistry.create_class(
            "time-domain")
        settings = swprocess.Masw.create_settings_dict(snr=True,
                                                       noise_begin=-0.4,
                                                       noise_end=-0.1)
        workflow = Workflow(settings=settings)
        transform = workflow.run()

        # Call detrend on sensor(s).
        for sensor in workflow.array.sensors:
            sensor.detrend.assert_called_once_with()

        # Select noise.
        workflow.noise.trim.assert_called_once_with(-0.4, -0.1)


    @patch("swprocess.maswworkflows.Array1D", new=SubMockArray1D)
    def test_fail_if_source_inside(self):
        workflow = swprocess.register.MaswWorkflowRegistry.create_class(
            "time-domain")()
        self.assertRaises(ValueError, workflow.run)


if __name__ == "__main__":
    unittest.main()
