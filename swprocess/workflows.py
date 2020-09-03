"""Workflows class definitions."""

from abc import ABCMeta, abstractmethod

from register import WorkflowRegistry, WavefieldTransformRegistry
from array1d import Array1D


class MaswWorkflow(ABCMeta):
    """Abstract base class defining an MASW workflow."""

    def __init__(self, settings):
        self.settings = settings

        # Pre-define variables to None for ease of reading
        self.pre_mute = None
        self.post_mute = None

    def mute(self):
        """Mute noise in the time-domain."""
        muting = self.settings["workflow"]["time-domain muting"]
        if not muting["apply"]:
            return
        else:
            if self.pre_mute is None and self.post_mute is None:
                if muting["type"] == "interactive":
                    self.pre_mute, self.post_mute = self.array.interactive_mute()
                # elif muting["type"] == "predefined":
                #     # TODO (jpv): Implement predefined type for time-domain muting.
                #     raise NotImplementedError
                else:
                    msg = f"muting type {muting["type"]} is unknown, use 'interactive'."
                    raise KeyError(msg)
            else:
                self.array.mute(pre_mute=self.pre_mute,
                                post_mute=self.post_mute,
                                window_kwargs=muting.get("window kwargs"),
                                )

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def __str__(self):
        """Human-readable representaton of the workflow."""
        pass



@MaswWorkflowRegistry.register("single")
class TimeDomainMaswWorkflow(MaswWorkflow):
    """Stack wavefield in the time domain, prior to transform."""

    def __init__(self, fnames, settings, map_x=lambda x: x, map_y=lambda y: y):
        super.__init__(settings)

    def run(self):
        self.array = Array1D.from_files(fnames, map_x=map_x, map_y=map_y)
        self.mute()
        transform = WavefieldTransformRegistry.create_instance()

    def __str__(self):
        msg = ""
        msg += "MaswWorkflow: single\n"
        msg += "  - Create Array1D."
        msg += "  - Perform mute (if desired)."
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
