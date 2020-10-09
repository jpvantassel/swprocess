"""Masw workflow class definitions."""

from abc import ABC, abstractmethod
import warnings
import logging

import numpy as np

from .register import MaswWorkflowRegistry
from .wavefieldtransforms import WavefieldTransformRegistry
from .array1d import Array1D

logger = logging.getLogger("swprocess.maswworkflows")


class AbstractMaswWorkflow(ABC):
    """Abstract base class (ABC) defining an MASW workflow."""

    def __init__(self, fnames=None, settings=None, map_x=None, map_y=None):
        """Perform initialization common to all `MaswWorkflow`s."""
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
        self.snr_frequencies = None

    def check(self):
        """Check array is acceptable for WavefieldTransform."""
        if self.array._source_inside:
            raise ValueError("Source must be located outside of the array.")

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

    def trim(self):
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
            # Copy array
            self.signal = Array1D.from_array1d(self.array)

            # Trim out noise.
            self.signal.trim(snr["signal"]["begin"], snr["signal"]["end"])

            # Pad
            if snr["pad"]["apply"]:
                self.noise.zero_pad(snr["pad"]["df"])
                self.signal.zero_pad(snr["pad"]["df"])

            # Check signal and noise windows are indeed the same length.
            if self.noise[0].nsamples != self.signal[0].nsamples:
                msg = f"Signal and noise windows must be of equal length, or set 'pad_snr' to 'True'."
                raise IndexError(msg)

            # Frequency vector
            sensor = self.noise[0]
            frqs = np.arange(sensor.nsamples) * sensor.df
            Empty = WavefieldTransformRegistry.create_class("empty")
            keep_ids = Empty._frequency_keep_ids(frqs,
                                                 self.settings["processing"]["fmin"],
                                                 self.settings["processing"]["fmax"],
                                                 sensor.multiple)
            self.snr_frequencies = frqs[keep_ids]

            # Compute SNR
            self.snr = np.mean(np.abs(np.fft.fft(
                self.signal.timeseriesmatrix())[:, keep_ids]), axis=0)
            self.snr /= np.mean(np.abs(np.fft.fft(
                self.noise.timeseriesmatrix())[:, keep_ids]), axis=0)

            # Clean-up
            self.noise = False
            self.signal = False

    def pad(self):
        """Pad record in the time domain."""
        pad = self.settings["pre-processing"]["pad"]
        if pad["apply"]:
            self.array.zero_pad(pad["df"])

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def __str__(self):
        """Human-readable representaton of the workflow."""
        pass


class TimeDomainWorkflow(AbstractMaswWorkflow):

    def run(self):
        self.array = Array1D.from_files(self.fnames, map_x=self.map_x,
                                        map_y=self.map_y)
        self.check()
        self.detrend()
        self.select_noise()
        self.trim()
        # self.mute()
        # self.select_signal()
        # self.pad()
        # Transform = WavefieldTransformRegistry.create_class(
        #     self.settings["processing"]["transform"])
        # transform = Transform.from_array(array=self.array,
        #                                  settings=self.settings["processing"])
        # transform.snr = self.snr
        # transform.snr_frequencies = self.snr_frequencies
        # transform.array = self.array
        # return transform

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
        msg += "  - Perform linear detrend on each trace.\n"
        msg += "  - Perform trim (if desired).\n"
        msg += "  - Perform mute (if desired).\n"
        msg += "  - Perform pad  (if desired).\n"
        msg += "  - Perform transform."
        return msg


@MaswWorkflowRegistry.register("time-domain")
class TimeDomainMaswWorkflow(TimeDomainWorkflow):
    """Stack in the frequency-domain."""

    def __str__(self):
        msg = "\n"
        msg += "MaswWorkflow: time-domain\n"
        msg += "  - Create Array1D from files.\n"
        msg += "  - Check array is acceptable.\n"
        msg += "  - Perform linear detrend on each trace.\n"
        msg += "  - Perform trim (if desired).\n"
        msg += "  - Perform mute (if desired).\n"
        msg += "  - Perform pad  (if desired).\n"
        msg += "  - Perform transform."
        return msg


@MaswWorkflowRegistry.register("frequency-domain")
class FrequencyDomainMaswWorkflow(AbstractMaswWorkflow):
    """Stack in the frequency-domain."""

    def run(self):
        ex_array = Array1D.from_files(self.fnames[0],
                                      map_x=self.map_x,
                                      map_y=self.map_y)
        for sensor in ex_array:
            sensor.detrend()
        preprocess = self.settings["pre-processing"]
        if preprocess["trim"]["apply"]:
            ex_array.trim(preprocess["trim"]["start_time"],
                          preprocess["trim"]["end_time"])
        if preprocess["pad"]["apply"]:
            ex_array.zero_pad(preprocess["pad"]["df"])

        Transform = WavefieldTransformRegistry.create_class("empty")
        processing = self.settings["processing"]
        running_stack = Transform.from_array(array=ex_array,
                                             settings=processing)
        Transform = WavefieldTransformRegistry.create_class(
            processing["transform"])
        for fname in self.fnames:
            self.array = Array1D.from_files(fname, map_x=self.map_x,
                                            map_y=self.map_y)
            if not self.array.is_similar(ex_array):
                msg = f"Can only stack arrays which are similar, first dissimilar file is {fname}."
                raise ValueError(msg)
            self.check()
            self.detrend()
            self.select_noise()
            self.trim()
            self.mute()
            self.select_signal()
            self.pad()
            transform = Transform.from_array(array=self.array,
                                             settings=processing)
            running_stack.stack(transform)

        running_stack.snr = self.snr
        running_stack.snr_frequencies = self.snr_frequencies
        running_stack.array = self.array

        return running_stack

    def __str__(self):
        msg = "\n"
        msg += "MaswWorkflow: frequency-domain\n"
        msg += "  - Create Array1D from file.\n"
        msg += "  - Check array is acceptable.\n"
        msg += "  - Perform trim (if desired).\n"
        msg += "  - Perform mute (if desired).\n"
        msg += "  - Perform pad  (if desired).\n"
        msg += "  - Perform transform.\n"
        msg += "  - Repeat steps for remaining files, stacking in frequency-domain.\n"
        return msg
