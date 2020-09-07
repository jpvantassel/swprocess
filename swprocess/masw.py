"""Masw class definition."""

import logging
import json

from .workflows import MaswWorkflowRegistry

logger = logging.getLogger(__name__)


class Masw():
    """Customizable Multichannel Analysis of Surface Waves workflow.

    Convenient customer-facing interface for implementing different
    and extensible MASW processing workflows.

    """

    @staticmethod
    def run(fnames, settings, map_x=lambda x: x, map_y=lambda y: y):
        """Run an MASW workflow from SU or SEGY files.

        Create an instance of an `Masw` object for a specific
        `Masw` workflow. Note that each file should contain
        multiple traces where each trace corresponds to a
        single receiver. The header information for these files must
        be correct and readable. Currently supported file types are
        SEGY and SU.

        Parameters
        ----------
        fnames : str or iterable of str
            File name or iterable of file names.
        settings : str
            JSON settings file detailing how MASW should be performed.
            See `meth: Masw.example_settings_file()` for more
            information.
        map_x, map_y : function, optional
            Functions to convert the x and y coordinates of source and
            receiver information, default is no transformation. Useful
            for converting between coordinate systems.

        Returns
        -------
        AbstractTransform-like
            Initialized subclass (i.e., child) of `AbstractTransform`.

        Raises
        ------
        TypeError
            If `fnames` is not of type `str` or `iterable`.

        """
        # Load settings
        with open(settings, "r") as f:
            settings = json.load(f)

        Workflow = MaswWorkflowRegistry.create_class(settings["workflow"])
        workflow = Workflow(fnames=fnames, settings=settings, map_x=map_x,
                            map_y=map_y)
        return workflow.run()

    # TODO (jpv): Generate an example settings file on the fly.
    @classmethod
    def example_settings_file(cls, fname):
        pass
