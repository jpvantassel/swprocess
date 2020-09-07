"""Workflows class definitions."""

from abc import ABC, abstractmethod
import warnings

from .register import MaswWorkflowRegistry
from .wavefieldtransforms import WavefieldTransformRegistry
from .array1d import Array1D


class AbstractMaswWorkflow(ABC):
    """Abstract base class defining an MASW workflow."""

    def __init__(self, fnames=None, settings=None, map_x=None, map_y=None):
        """Perform initialization common to all `MaswWorkflow`s."""
        # Set objects state
        self.fnames = fnames
        self.settings = settings
        self.map_x = map_x
        self.map_y = map_y

        # Pre-define optional state variables to None for ease of reading
        self.signal_start = None
        self.signal_end = None

    def check(self):
        """Check array is acceptable for WavefieldTransform."""
        if self.array._source_inside:
            raise ValueError("Source must be located outside of the array.")

    def trim(self):
        """Trim record in the time domain."""
        trim = self.settings["pre-processing"]["trim"]
        if trim["apply"]:
            self.array.trim(trim["start_time"], trim["end_time"])

    def mute(self):
        """Mute record in the time domain."""
        mute = self.settings["pre-processing"]["mute"]
        if mute["apply"]:
            if self.signal_start is None and self.signal_end is None:
                if mute["type"] == "interactive":
                    self.signal_start, self.signal_end = self.array.interactive_mute()
                # elif muting["type"] == "predefined":
                #     # TODO (jpv): Implement predefined type for time-domain muting.
                #     raise NotImplementedError
                else:
                    msg = f"mute type {mute['type']} is unknown, use 'interactive'."
                    raise KeyError(msg)
            else:
                self.array.mute(signal_start=self.signal_start,
                                signal_end=self.signal_end,
                                window_kwargs=mute.get("window kwargs"),
                                )

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        self.array = Array1D.from_files(self.fnames, map_x=self.map_x,
                                        map_y=self.map_y)
        self.check()
        self.trim()
        self.mute()
        self.pad()
        Transform = WavefieldTransformRegistry.create_class(
            self.settings["processing"]["type"])
        return Transform.from_array(array=self.array,
                                    settings=self.settings["processing"])

    def __str__(self):
        pass


@MaswWorkflowRegistry.register("single")
class SingleMaswWorkflow(TimeDomainWorkflow):
    """Perform transform on a single time-domain record."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        if not isinstance(self.fnames, (str,)) and len(self.fnames) != 1:
            self.fnames = self.fnames[0]
            msg = "fnames may only include a single file for the selected workflow, only processing the first."
            warnings.warn(msg)
        return super().run()

    def __str__(self):
        msg = "\n"
        msg += "MaswWorkflow: single\n"
        msg += "  - Create Array1D from file (ignore if multiple).\n"
        msg += "  - Check array is acceptable.\n"
        msg += "  - Perform trim (if desired).\n"
        msg += "  - Perform mute (if desired).\n"
        msg += "  - Perform pad  (if desired).\n"
        msg += "  - Perform transform."
        return msg


@MaswWorkflowRegistry.register("time-domain")
class TimeDomainMaswWorkflow(TimeDomainWorkflow):
    """Stack in the frequency-domain."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        msg = "\n"
        msg += "MaswWorkflow: time-domain\n"
        msg += "  - Create Array1D from files.\n"
        msg += "  - Check array is acceptable.\n"
        msg += "  - Perform trim (if desired).\n"
        msg += "  - Perform mute (if desired).\n"
        msg += "  - Perform pad  (if desired).\n"
        msg += "  - Perform transform."
        return msg


@MaswWorkflowRegistry.register("frequency-domain")
class FrequencyDomainMaswWorkflow(AbstractMaswWorkflow):
    """Stack in the frequency-domain."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        ex_array = Array1D.from_files(self.fnames[0], map_x=self.map_x,
                                      map_y=self.map_y)
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
        Transform = WavefieldTransformRegistry.create_class(processing["type"])
        for fname in self.fnames:
            self.array = Array1D.from_files(fname, map_x=self.map_x, map_y=self.map_y)
            if not self.array.is_similar(ex_array):
                msg = f"Can only stack arrays which are similar, first dissimilar file is {fname}."
                raise ValueError(msg)
            self.check()
            self.trim()
            self.mute()
            self.pad()
            transform = Transform.from_array(array=self.array,
                                             settings=processing)
            running_stack.stack(transform)

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
