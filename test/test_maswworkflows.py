"""Test for maswworkflows module."""

import warnings
from unittest.mock import patch, MagicMock, call

import swprocess
from testtools import unittest, TestCase


# class MockArray1D():

#     def __init__(self, *args, **kwargs):
#         self.sensors = [Mock()]

#     @classmethod
#     def from_files(cls, *args, **kwargs):
#         return cls(*args, **kwargs)

#     @classmethod
#     def from_array1d(cls, *args, **kwargs):
#         return Mock()

#     def trim

#     @property
#     def _source_inside(self):
#         return False

#     def __getitem__(self, index):
#         return self.sensors[index]


# class SubMockArray1D(MockArray1D):

#     @property
#     def _source_inside(self):
#         return True


class Test_MaswWorkflows(TestCase):

    @patch("swprocess.maswworkflows.WavefieldTransformRegistry", new=MagicMock(spec=swprocess.register.WavefieldTransformRegistry))
    @patch("swprocess.maswworkflows.SignaltoNoiseRatio", new=MagicMock(spec=swprocess.snr.SignaltoNoiseRatio))
    def test_time_domain(self):
        with patch("swprocess.maswworkflows.Array1D", spec=swprocess.Array1D) as MockArray1D:
            MockFromFiles = MagicMock(autospec=swprocess.Array1D)
            MockArray1D.from_files.return_value = MockFromFiles
            mock_from_files = MockArray1D.from_files()
            mock_from_files._source_inside = False
            mock_from_files.sensors = [MagicMock()]
            mock_from_files.interactive_mute.return_value = ("signal_start",
                                                             "signal_end")

            Registry = swprocess.register.MaswWorkflowRegistry
            Workflow = Registry.create_class("time-domain")
            settings = swprocess.Masw.create_settings_dict(
                snr=True, noise_begin=-0.4, noise_end=-0.1, signal_begin=0.,
                signal_end=0.3, pad_snr=True, df_snr=0.2, trim=True,
                start_time=0., end_time=1.0, mute=True, pad=True, df=0.1)
            workflow = Workflow(settings=settings)
            transform = workflow.run()

            # Call detrend on sensor(s).
            for sensor in workflow.array.sensors:
                sensor.detrend.assert_called_once_with()

            # Select noise.
            trim_calls = [call(-0.4, -0.1)]

            # Trim time record.
            workflow.array.trim.assert_called_once_with(0.0, 1.0)

            # Mute time record.
            workflow.array.interactive_mute.assert_called_once_with()
            self.assertEqual("signal_start", workflow.signal_start)
            self.assertEqual("signal_end", workflow.signal_end)

            # Select signal.
            trim_calls += [call(0., 0.3)]
            self.assertEqual(workflow.signal, workflow.noise)
            workflow.noise.trim.assert_has_calls(trim_calls)

            # Calculate snr.
            self.assertTrue(isinstance(workflow.snr, MagicMock))

            # Pad.
            workflow.array.zero_pad.assert_called_once_with(0.1)

            # __str__
            expected = "\n"
            expected += "MaswWorkflow: time-domain\n"
            expected += "  - Create Array1D from files.\n"
            expected += "  - Check array is acceptable.\n"
            expected += "  - Perform linear detrend on each trace.\n"
            expected += "  - Perform trim (if desired).\n"
            expected += "  - Perform mute (if desired).\n"
            expected += "  - Perform pad  (if desired).\n"
            expected += "  - Perform transform."
            self.assertEqual(expected, workflow.__str__())

    def test_fail_with_source_inside(self):
        with patch("swprocess.maswworkflows.Array1D", spec=swprocess.Array1D) as MockArray1D:
            MockFromFiles = MagicMock(autospec=swprocess.Array1D)
            MockArray1D.from_files.return_values = MockFromFiles
            mock_from_files = MockArray1D.from_files()
            mock_from_files._source_inside = True

            Registry = swprocess.register.MaswWorkflowRegistry
            Workflow = Registry.create_class("time-domain")
            settings = swprocess.Masw.create_settings_dict()
            workflow = Workflow(settings=settings)
            self.assertRaises(ValueError, workflow.run)

    def test_fail_with_mute_method(self):
        with patch("swprocess.maswworkflows.Array1D", spec=swprocess.Array1D) as MockArray1D:
            MockFromFiles = MagicMock(autospec=swprocess.Array1D)
            MockArray1D.from_files.return_values = MockFromFiles
            mock_from_files = MockArray1D.from_files()
            mock_from_files._source_inside = False

            Registry = swprocess.register.MaswWorkflowRegistry
            Workflow = Registry.create_class("time-domain")
            settings = swprocess.Masw.create_settings_dict(snr=False,
                                                           trim=False,
                                                           mute=True,
                                                           method="bad_method"
                                                           )
            workflow = Workflow(settings=settings)
            self.assertRaises(KeyError, workflow.run)

    @patch("swprocess.maswworkflows.WavefieldTransformRegistry", new=MagicMock(spec=swprocess.register.WavefieldTransformRegistry))
    @patch("swprocess.maswworkflows.SignaltoNoiseRatio", new=MagicMock(spec=swprocess.snr.SignaltoNoiseRatio))
    def test_single(self):
        with patch("swprocess.maswworkflows.Array1D", spec=swprocess.Array1D) as MockArray1D:
            MockFromFiles = MagicMock(autospec=swprocess.Array1D)
            MockArray1D.from_files.return_value = MockFromFiles
            mock_from_files = MockArray1D.from_files()
            mock_from_files._source_inside = False
            mock_from_files.sensors = [MagicMock()]
            mock_from_files.interactive_mute.return_value = ("signal_start",
                                                             "signal_end")

            Registry = swprocess.register.MaswWorkflowRegistry
            Workflow = Registry.create_class("single")
            settings = swprocess.Masw.create_settings_dict(
                snr=False, trim=True, start_time=0., end_time=1.0,
                mute=True, pad=False, df=0.1)
            fnames = ["name1", "name2"]
            workflow = Workflow(fnames=fnames, settings=settings)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transform = workflow.run()

            # Call detrend on sensor(s).
            for sensor in workflow.array.sensors:
                sensor.detrend.assert_called_once_with()

            # Trim time record.
            workflow.array.trim.assert_called_once_with(0.0, 1.0)

            # Mute time record.
            workflow.array.interactive_mute.assert_called_once_with()
            self.assertEqual("signal_start", workflow.signal_start)
            self.assertEqual("signal_end", workflow.signal_end)

            # Pad.
            workflow.array.zero_pad.assert_not_called()

            # __str__
            expected = "\n"
            expected += "MaswWorkflow: single\n"
            expected += "  - Create Array1D from file (ignore if multiple).\n"
            expected += "  - Check array is acceptable.\n"
            expected += "  - Perform linear detrend on each trace.\n"
            expected += "  - Perform trim (if desired).\n"
            expected += "  - Perform mute (if desired).\n"
            expected += "  - Perform pad  (if desired).\n"
            expected += "  - Perform transform."
            self.assertEqual(expected, workflow.__str__())



if __name__ == "__main__":
    unittest.main()
