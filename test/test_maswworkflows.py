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

"""Test for maswworkflows module."""

import warnings
from unittest.mock import patch, MagicMock, call

import swprocess
from testtools import unittest, TestCase


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
                trim_begin=0., trim_end=1.0, mute=True, pad=True, df=0.1)
            workflow = Workflow(settings=settings)
            _ = workflow.run()

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
            expected += "  - Perform trim in space (if desired).\n"
            expected += "  - Perform linear detrend on each trace.\n"
            expected += "  - Perform trim in time (if desired).\n"
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
                snr=False, trim=True, trim_begin=0., trim_end=1.0,
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
            expected += "  - Perform trim in space (if desired).\n"
            expected += "  - Perform linear detrend on each trace.\n"
            expected += "  - Perform trim in time (if desired).\n"
            expected += "  - Perform mute (if desired).\n"
            expected += "  - Perform pad  (if desired).\n"
            expected += "  - Perform transform."
            self.assertEqual(expected, workflow.__str__())

    @patch("swprocess.maswworkflows.WavefieldTransformRegistry", new=MagicMock(spec=swprocess.register.WavefieldTransformRegistry))
    @patch("swprocess.maswworkflows.SignaltoNoiseRatio", new=MagicMock(spec=swprocess.snr.SignaltoNoiseRatio))
    def test_frequency_domain(self):
        with patch("swprocess.maswworkflows.Array1D", spec=swprocess.Array1D) as MockArray1D:
            MockFromFiles = MagicMock(autospec=swprocess.Array1D)
            MockArray1D.from_files.return_value = MockFromFiles
            mock_from_files = MockArray1D.from_files()
            mock_from_files._source_inside = False
            mock_from_files.sensors = [MagicMock(autospec=swprocess.Sensor1C)]
            mock_from_files.interactive_mute.return_value = ("signal_start",
                                                             "signal_end")

            Registry = swprocess.register.MaswWorkflowRegistry
            Workflow = Registry.create_class("frequency-domain")
            settings = swprocess.Masw.create_settings_dict(
                snr=True, noise_begin=-0.2, noise_end=-0.1, signal_begin=0.1,
                signal_end=0.2, trim=True, trim_begin=0., trim_end=1.0,
                mute=True, pad=True, df=0.1)
            fnames = ["name1", "name2", "name3"]
            workflow = Workflow(fnames=fnames, settings=settings)

            transform = workflow.run()

            # Call detrend on sensor(s).
            for sensor in workflow.array.sensors:
                self.assertEqual(4, sensor.detrend.call_count)

            # Select noise & signal
            trim_calls = []
            # TODO (jpv): SNR of frequency-domain returns SNR of last trace only.
            # for _ in range(len(fnames)+1):
            for _ in range(1):
                trim_calls.append(call(-0.2, -0.1))
                trim_calls.append(call(0.1, 0.2))
            self.assertEqual(workflow.signal, workflow.noise)
            workflow.noise.trim.assert_has_calls(trim_calls)

            # Trim time record.
            workflow.array.trim.assert_has_calls([call(0.0, 1.0)]*4)
            self.assertEqual(4, workflow.array.trim.call_count)

            # Mute time record.
            workflow.array.interactive_mute.assert_called_once_with()
            self.assertEqual("signal_start", workflow.signal_start)
            self.assertEqual("signal_end", workflow.signal_end)

            # Calculate snr.
            self.assertTrue(isinstance(workflow.snr, MagicMock))

            # Pad.
            workflow.array.zero_pad.assert_has_calls([call(0.1)]*4)

            # __str__
            expected = "\n"
            expected += "MaswWorkflow: frequency-domain\n"
            expected += "  - Create Array1D from file.\n"
            expected += "  - Check array is acceptable.\n"
            expected += "  - Perform trim in space (if desired).\n"
            expected += "  - Perform linear detrend on each trace.\n"
            expected += "  - Perform trim in time (if desired).\n"
            expected += "  - Perform mute (if desired).\n"
            expected += "  - Perform pad  (if desired).\n"
            expected += "  - Perform transform.\n"
            expected += "  - Repeat steps for remaining files, stacking in frequency-domain.\n"
            self.assertEqual(expected, workflow.__str__())

    @patch("swprocess.maswworkflows.WavefieldTransformRegistry", new=MagicMock(spec=swprocess.register.WavefieldTransformRegistry))
    def test_fail_is_similar(self):

        # Create a response generator.
        def response_generator(responses):
            index = 0
            while index < len(responses):
                yield responses[index]
                index += 1

        # Use a closure to wrap generator.
        def wrap_generator(generator):
            def wrapper(*args, **kwargs):
                return next(generator)
            return wrapper

        with patch("swprocess.maswworkflows.Array1D", spec=swprocess.Array1D) as MockArray1D:
            MockFromFiles = MagicMock(autospec=swprocess.Array1D)
            MockArray1D.from_files.return_value = MockFromFiles
            mock_from_files = MockArray1D.from_files()
            mock_from_files._source_inside = False
            mygen = response_generator([True, False])
            is_similar = wrap_generator(generator=mygen)
            mock_from_files.is_similar = is_similar

            Registry = swprocess.register.MaswWorkflowRegistry
            Workflow = Registry.create_class("frequency-domain")
            settings = swprocess.Masw.create_settings_dict(
                snr=False, noise_begin=-0.2, noise_end=-0.1, signal_begin=0.1,
                signal_end=0.2, trim=False, trim_begin=0., trim_end=1.0,
                mute=False, pad=False, df=0.1)
            fnames = ["name1", "name2", "name3"]
            workflow = Workflow(fnames=fnames, settings=settings)
            self.assertRaises(ValueError, workflow.run)


if __name__ == "__main__":
    unittest.main()
