"""Workflows class definitions."""

from abc import ABCMeta, abstractproperty

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
        muting = self.settings["workflow"]["time-domain muting"]
        if not muting["apply"]:
            return
        else:
            if self.pre_mute is None and self.post_mute is None:
                if muting["type"] == "interactive":
                    self.pre_mute, self.post_mute = self.array._get_mute_values()
                elif muting["type"] == "predefined":
                    # TODO (jpv): Implement predefined type for time-domain muting.
                    raise NotImplementedError
                else:
                    msg = f"muting type {muting["type"]} is unknown, use 'interactive'."
                    raise KeyError(msg)

        self.array.mute(pre_mute=self.pre_mute,
                        post_mute=self.post_mute,
                        window_kwargs=muting.get("muting window kwargs"),
                        )
            


@MaswWorkflowRegistry.register("time-domain")
class TimeDomainMaswWorkflow(MaswWorkflow):
    """Stack wavefield in the time domain, prior to transform."""

    def __init__(self, fnames, settings, map_x=lambda x: x, map_y=lambda y: y):
        super.__init__(settings)
        self.array = Array1D.from_files(fnames, map_x=map_x, map_y=map_y)
        self.mute()
        transform = WavefieldTransformRegistry.create_instance()
