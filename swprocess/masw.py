"""Masw class definition."""

import logging
import json

from register import MaswWorkflowRegistry
from array1d import Array1D

logger = logging.getLogger(__name__)


class Masw():
    """Customizable Multichannel Analysis of Surface Waves workflow.

    Convenient customer-facing interface for implementing different
    and extensible MASW processing workflows.

    """
    def __init__(cls, fnames, settings, map_x=lambda x: x, map_y=lambda y: y):
        """Initialize an `MaswWorkflow` from SU or SEGY files.

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
            For an example file see
            `meth: MaswOffset.example_settings_file()`.
        map_x, map_y : function, optional
            Functions to convert the x and y coordinates of source and
            receiver information, default is no transformation. Useful
            for converting between coordinate systems.

        Returns
        -------
        MaswOffset
            Initialized `MaswOffset` object.

        Raises
        ------
        TypeError
            If `fnames` is not of type `str` or `iterable`.

        """
        # Load settings
        with open(settings, "r") as f:
            settings = json.load(f)

        transform = MaswWorkflowRegistry.create_instance(settings["workflow"],
                                                         fnames, settings)
        
        def run(self):
            return self.transform.run()

    # TODO (jpv): Generate a default settings file on the fly.
    # @classmethod
    # def default_settings_file(fname):
    #     pass



        # if array._source_inside:
        #     raise ValueError("Source must be located outside of the array.")

        # with open(settings, "r") as f:
        #     logger.info("loading settings ... ")
        #     self.settings = json.load(f)

        # if self.settings["trim"]:
        #     logger.info("trimming ... ")
        #     array.trim(start_time=self.settings["start_time"],
        #                end_time=self.settings["end_time"])

        # if self.settings["zero_pad"]:
        #     logger.info("padding ... ")
        #     array.zero_pad(df=self.settings["df"])

        # self.kres = array.kres
        # self.transform = WavefieldTransformRegistry.create_instance()
