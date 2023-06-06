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

"""Masw workflow class definitions."""

from abc import ABC, abstractmethod
import warnings
import logging

from .register import MaswWorkflowRegistry
from .wavefieldtransforms import WavefieldTransformRegistry
from .array1d import Array1D, Array1DwSource
from .snr import SignaltoNoiseRatio

logger = logging.getLogger("swprocess.maswworkflows")


class AbstractMaswWorkflow(ABC):
    """Abstract base class (ABC) defining an MASW workflow."""

    def __init__(self, fnames=None, settings=None, map_x=None, map_y=None):
        """Perform initialization common to all MaswWorkflows.

        """
        # Set objects state
        self.fnames = fnames
        self.settings = settings
        self.map_x = map_x
        self.map_y = map_y

        # Pre-define state variables for ease of reading.
        self.array = None

        # Pre-define optional state variables too.
        self.signal_start = None
        self.signal_end = None
        self.noise = None
        self.signal = None
        self.snr = None

    def check(self):
        """Check array is acceptable for WavefieldTransform."""
        if self.array._source_inside:
            raise ValueError("Source must be located outside of the array.")

    def trim_offsets(self):
        """Remove receivers outside of the offset range."""
        offsets = self.settings["pre-processing"]["offsets"]
        self.array.trim_offsets(offsets["min"], offsets["max"])

    def detrend(self):
        """Perform linear detrend operation."""
        for sensor in self.array.sensors:
            sensor.detrend()

    def select_noise(self):
        """Select a portion of the record as noise."""
        snr = self.settings["signal-to-noise"]
        if snr["perform"] and self.noise is None:
            # Copy array and trim noise.
            self.noise = Array1D.from_array1d(self.array)
            self.noise.trim(snr["noise"]["begin"], snr["noise"]["end"])

    def trim_time(self):
        """Trim record in the time domain."""
        trim = self.settings["pre-processing"]["trim"]
        if trim["apply"]:
            self.array.trim(trim["begin"], trim["end"])

    def mute(self):
        """Mute record in the time domain."""
        mute = self.settings["pre-processing"]["mute"]
        if mute["apply"]:
            if self.signal_start is None and self.signal_end is None:
                if mute["method"] == "interactive":
                    self.signal_start, self.signal_end = self.array.interactive_mute()
                # TODO (jpv): Implement predefined times for muting.
                else:
                    msg = f"mute type {mute['method']} is unknown, use 'interactive'."
                    raise KeyError(msg)
            else:
                self.array.mute(signal_start=self.signal_start,
                                signal_end=self.signal_end,
                                window_kwargs=mute.get("window kwargs"),
                                )

    def select_signal(self):
        """Select a portion of the record as signal."""
        snr = self.settings["signal-to-noise"]
        if snr["perform"] and self.signal is None:
            # Copy array and trim noise.
            self.signal = Array1D.from_array1d(self.array)
            self.signal.trim(snr["signal"]["begin"], snr["signal"]["end"])

    def calculate_snr(self):
        snr = self.settings["signal-to-noise"]
        process = self.settings["processing"]
        if snr["perform"]:
            self.snr = SignaltoNoiseRatio.from_array1ds(
                self.signal, self.noise,
                fmin=process["fmin"], fmax=process["fmax"],
                pad_snr=snr["pad"]["apply"], df_snr=snr["pad"]["df"])

    def pad(self):
        """Pad record in the time domain."""
        pad = self.settings["pre-processing"]["pad"]
        if pad["apply"]:
            self.array.zero_pad(pad["df"])

    @abstractmethod
    def run(self):  # pragma: no cover
        pass

    @abstractmethod
    def __str__(self):  # pragma: no cover
        """Human-readable representaton of the workflow."""
        pass


class TimeDomainWorkflow(AbstractMaswWorkflow):

    def run(self):
        self.array = Array1D.from_files(self.fnames, map_x=self.map_x,
                                        map_y=self.map_y)
        self.check()
        self.trim_offsets()
        self.detrend()
        self.select_noise()
        self.trim_time()
        self.mute()
        self.select_signal()
        self.calculate_snr()
        self.pad()
        proc = self.settings["processing"]
        Transform = WavefieldTransformRegistry.create_class(proc["transform"])
        transform = Transform.from_array(array=self.array, settings=proc)
        transform.array = self.array
        if self.settings["signal-to-noise"]["perform"]:
            transform.snr = self.snr.snr
            transform.snr_frequencies = self.snr.frequencies
        return transform


@MaswWorkflowRegistry.register("single")
class SingleMaswWorkflow(TimeDomainWorkflow):
    """Perform transform on a single time-domain record."""

    def run(self):
        if not isinstance(self.fnames, (str,)) and len(self.fnames) != 1:
            self.fnames = self.fnames[0]
            msg = f"fnames may only include a single file for the selected workflow, only processing {self.fnames}."
            warnings.warn(msg)
        return super().run()

    def __str__(self):
        msg = "\n"
        msg += "MaswWorkflow: single\n"
        msg += "  - Create Array1D from file (ignore if multiple).\n"
        msg += "  - Check array is acceptable.\n"
        msg += "  - Perform trim in space (if desired).\n"
        msg += "  - Perform linear detrend on each trace.\n"
        msg += "  - Perform trim in time (if desired).\n"
        msg += "  - Perform mute (if desired).\n"
        msg += "  - Perform pad  (if desired).\n"
        msg += "  - Perform transform."
        return msg


@MaswWorkflowRegistry.register("time-domain")
class TimeDomainMaswWorkflow(TimeDomainWorkflow):
    """Stack in the time-domain."""

    def __str__(self):
        msg = "\n"
        msg += "MaswWorkflow: time-domain\n"
        msg += "  - Create Array1D from files.\n"
        msg += "  - Check array is acceptable.\n"
        msg += "  - Perform trim in space (if desired).\n"
        msg += "  - Perform linear detrend on each trace.\n"
        msg += "  - Perform trim in time (if desired).\n"
        msg += "  - Perform mute (if desired).\n"
        msg += "  - Perform pad  (if desired).\n"
        msg += "  - Perform transform."
        return msg


@MaswWorkflowRegistry.register("time-domain-xcorr")
class TimeDomainXcorrMaswWorkflow(AbstractMaswWorkflow):
    """Stack in the time-domain and xcorr."""

    def __init__(self, fnames_rec=None, fnames_src=None, src_channel=None,
                 settings=None, map_x=None, map_y=None):
        """Perform initialization common to all MaswWorkflows.

        """
        super().__init__(fnames=fnames_rec, settings=settings,
                         map_x=map_x, map_y=map_y)
        self.fnames_src = fnames_src
        self.src_channel = src_channel

    def run(self):
        self.array = Array1DwSource.from_files(self.fnames,
                                               self.fnames_src,
                                               self.src_channel,
                                               map_x=self.map_x,
                                               map_y=self.map_y)
        self.array.trim(0, max(self.array.sensors[0].time))
        self.array.source.trim(0, max(self.array.source.time))
        self.check()
        self.trim_offsets()
        self.detrend()
        self.select_noise()
        self.trim_time()
        self.mute()
        self.select_signal()
        self.calculate_snr()
        self.array.xcorrelate(self.settings["processing"]["vmin"],
                              self.settings["processing"]["vmax"])
        self.pad()
        proc = self.settings["processing"]
        Transform = WavefieldTransformRegistry.create_class(proc["transform"])
        transform = Transform.from_array(array=self.array, settings=proc)
        transform.array = self.array
        if self.settings["signal-to-noise"]["perform"]:
            transform.snr = self.snr.snr
            transform.snr_frequencies = self.snr.frequencies
        return transform

    def __str__(self):
        msg = "\n"
        msg += "MaswWorkflow: time-domain\n"
        msg += "  - Create Array1D from files.\n"
        msg += "  - Check array is acceptable.\n"
        msg += "  - Perform trim in space (if desired).\n"
        msg += "  - Perform linear detrend on each trace.\n"
        msg += "  - Cross-correlate traces with source.\n"
        msg += "  - Perform trim in time (if desired).\n"
        msg += "  - Perform mute (if desired).\n"
        msg += "  - Perform pad  (if desired).\n"
        msg += "  - Perform transform."
        return msg


@MaswWorkflowRegistry.register("frequency-domain")
class FrequencyDomainMaswWorkflow(AbstractMaswWorkflow):
    """Stack in the frequency-domain."""

    def run(self):
        example_array = Array1D.from_files(self.fnames[0],
                                           map_x=self.map_x,
                                           map_y=self.map_y)
        for sensor in example_array.sensors:
            sensor.detrend()
        preprocess = self.settings["pre-processing"]
        if preprocess["trim"]["apply"]:
            example_array.trim(preprocess["trim"]["begin"],
                               preprocess["trim"]["end"])
        if preprocess["pad"]["apply"]:
            example_array.zero_pad(preprocess["pad"]["df"])

        Transform = WavefieldTransformRegistry.create_class("empty")
        proc = self.settings["processing"]
        running_stack = Transform.from_array(array=example_array,
                                             settings=proc)
        Transform = WavefieldTransformRegistry.create_class(proc["transform"])
        for fname in self.fnames:
            self.array = Array1D.from_files(fname,
                                            map_x=self.map_x,
                                            map_y=self.map_y)
            if not self.array.is_similar(example_array):
                msg = "Can only stack arrays which are similar, "
                msg += f"first dissimilar file is {fname}."
                raise ValueError(msg)
            self.check()
            self.trim_offsets()
            self.detrend()
            # TODO (jpv): Calling select_noise n times.
            self.select_noise()
            self.trim_time()
            self.mute()
            # TODO (jpv): Calling select_singal n times.
            self.select_signal()
            self.pad()
            transform = Transform.from_array(array=self.array,
                                             settings=proc)
            running_stack.stack(transform)
        running_stack.array = self.array
        if self.settings["signal-to-noise"]["perform"]:
            self.calculate_snr()
            running_stack.snr = self.snr.snr
            running_stack.snr_frequencies = self.snr.frequencies
        return running_stack

    def __str__(self):
        msg = "\n"
        msg += "MaswWorkflow: frequency-domain\n"
        msg += "  - Create Array1D from file.\n"
        msg += "  - Check array is acceptable.\n"
        msg += "  - Perform trim in space (if desired).\n"
        msg += "  - Perform linear detrend on each trace.\n"
        msg += "  - Perform trim in time (if desired).\n"
        msg += "  - Perform mute (if desired).\n"
        msg += "  - Perform pad  (if desired).\n"
        msg += "  - Perform transform.\n"
        msg += "  - Repeat steps for remaining files, stacking in frequency-domain.\n"
        return msg
