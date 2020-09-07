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
        self.settings = settings

        # Pre-define optional state variables to None for ease of reading
        self.signal_start = None
        self.signal_end = None

        # Perform transform
        self.run(fnames=fnames, map_x=map_x, map_y=map_y)

    def check(self):
        """Check array is acceptable for WavefieldTransform."""
        if self.array._source_inside:
            raise ValueError("Source must be located outside of the array.")

    def trim(self):
        """Trim record in the time domain."""
        trim = self.settings["pre-processing"]["trim"]
        if not trim["apply"]:
            return
        else:
            self.array.trim(trim["start_time"], trim["end_time"])

    def mute(self):
        """Mute record in the time domain."""
        mute = self.settings["pre-processing"]["mute"]
        if not mute["apply"]:
            return
        else:
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
        if not pad["apply"]:
            return
        else:
            self.array.zero_pad(pad["df"])

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def __str__(self):
        """Human-readable representaton of the workflow."""
        pass


@MaswWorkflowRegistry.register("single")
class SingleMaswWorkflow(AbstractMaswWorkflow):
    """Perform transform on a single time-domain record."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, fnames=None, map_x=None, map_y=None):
        if not isinstance(fnames, (str,)) and len(fnames) != 1:
            fnames = fnames[0]
            msg = "fnames may only include a single file for the selected workflow, only processing the first."
            raise warnings.warn(msg)

        self.array = Array1D.from_files(fnames, map_x=map_x, map_y=map_y)
        self.check()
        self.trim()
        self.mute()
        self.pad()
        transform = WavefieldTransformRegistry.create_instance(self.settings["processing"]["type"],
                                                               array=self.array,
                                                               settings=self.settings["processing"])
        self.transform = transform.transform()

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


@MaswWorkflowRegistry.register("frequency-domain")
class FrequencyDomainMaswWorkflow(AbstractMaswWorkflow):
    """Stack in the frequency-domain."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, fnames=None, map_x=None, map_y=None):

        ex_array = Array1D.from_files(fnames[0], map_x=map_x, map_y=map_y)
        preprocess = self.settings["pre-processing"]
        if preprocess["trim"]["apply"]:
            ex_array.trim(preprocess["trim"]["start_time"], preprocess["trim"]["end_time"])
        if preprocess["pad"]["apply"]:
            ex_array.zero_pad(preprocess["pad"]["df"])
        
        running_stack = WavefieldTransformRegistry.create_instance("empty",
                                                                   array=ex_array,
                                                                   settings=self.settings["processing"])
        for fname in fnames:
            self.array = Array1D.from_files(fname, map_x=map_x, map_y=map_y)
            if not self.array.is_similar(ex_array):
                msg = f"Can only stack arrays which are similar, first dissimilar file is {fname}."
                raise ValueError(msg)
            self.check()
            self.trim()
            self.mute()
            self.pad()
            transform = WavefieldTransformRegistry.create_instance(self.settings["processing"]["type"],
                                                                   array=self.array,
                                                                   settings=self.settings["processing"])
            trans = transform.transform()
            running_stack.stack(trans)

        self.transform = running_stack

    def __str__(self):
        msg = ""
        return msg
