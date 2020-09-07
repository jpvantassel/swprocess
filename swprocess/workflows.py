"""Workflows class definitions."""

from abc import ABC, abstractmethod

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
        self.pre_mute = None
        self.post_mute = None

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
            if self.pre_mute is None and self.post_mute is None:
                if mute["type"] == "interactive":
                    self.pre_mute, self.post_mute = self.array.interactive_mute()
                # elif muting["type"] == "predefined":
                #     # TODO (jpv): Implement predefined type for time-domain muting.
                #     raise NotImplementedError
                else:
                    msg = f"mute type {mute['type']} is unknown, use 'interactive'."
                    raise KeyError(msg)
            else:
                self.array.mute(pre_mute=self.pre_mute,
                                post_mute=self.post_mute,
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
        print(args)
        print(kwargs)
        super().__init__(*args, **kwargs)

    def run(self, fnames=None, map_x=None, map_y=None):
        if not isinstance(fnames, (str,)) and len(fnames) != 1:
            msg = f"fnames must only include a single file for the selected workflow."
            raise IndexError(msg)

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
        msg = ""
        msg += "MaswWorkflow: single\n"
        msg += "  - Create Array1D fromm file."
        msg += "  - Check array is acceptable."
        msg += "  - Perform trim (if desired)."
        msg += "  - Perform mute (if desired)."
        msg += "  - Perform pad  (if desired)."
        msg += "  - Perform transform."

# @MaswWorkflowRegistry.register("frequency-domain")
# class FrequencyDomainMaswWorkflow(MaswWorkflow):
#     """Stack in the frequency-domain."""

#     def __init__(self, fnames, settings, map_x=lambda x: x, map_y=lambda y: y):
#         super.__init__(settings)
#         transform = WavefieldTransformRegistry.create_instance()

#         for fname in fnames:
#             self.array = Array1D.from_files(fnames, map_x=map_x, map_y=map_y)
#             self.mute()
