"""MaswOffset class definition."""

import logging
import json

from register import MaswWorkflowRegistry
from array1d import Array1D

logger = logging.getLogger(__name__)


class MaswExperiment():
    """Multi-channel Analysis of Surface Waves (MASW) Offset.

    Controls the MASW processing workflow for a single array and source
    offset. `MaswOffset` can account for multiple shot gathers for
    a given array and source setup.

    Attributes
    ----------
    arrays : iterable of Array1D objects
        May contain a single or multiple `Array1D` objects depending
        upon whether one or many files are used during instantiation
        and if multiple files are provided whether the stacking should
        occur in the time or frequency domain.

    """
    # def __init__(self, array, settings):
    #     """Initialize from an `Array1D` and a settings file.

    #     Parameters
    #     ----------
    #     array : Array1D
    #         Instantiated `Array1D` object.
    #     settings : str
    #         JSON settings file detailing how MASW should be performed.
    #         For an example file see
    #         `meth: MaswOffset.example_settings_file()`.

    #     Returns
    #     -------
    #     MaswOffset
    #         Instantiated `MaswOffset`.

    #     """
    #     # Append array
    #     if isinstance(array, (Array1D,)):
    #         self.arrays = [array]

    #     # Load settings file
    #     with open(settings, "r") as f:
    #         self.settings = json.load(f)

    #     # Define stacking procedure
    #     stacking_options = {"time-domain": self._stack_time_domain,
    #                         "frequency-domain": self._stack_frequency_domain}
    #     try:
    #         stacking_choice = self.settings["workflow"]["stacking"]
    #         self.append = stacking_options[stacking_choice]
    #     except KeyError as e:
    #         msg = f"Option {stacking_choice} not found in {stacking_options}."
    #         raise e(msg)

    # def _stack_time_domain(self, array):
    #     if self.arrays[-1].is_similar(array):
    #         self.arrays.append(array)
    #     else:
    #         msg = "Appended `array` must be similar to current `arrays`."
    #         raise ValueError(msg)

    @classmethod
    def from_files(cls, fnames, settings, map_x=lambda x: x, map_y=lambda y: y):
        """Initialize an `MaswOffset` from SU or SEGY files.

        This classmethod creates an `MaswOffset` object by reading
        the the files provided. Note that each file should contain
        multiple traces where each trace corresponds to a
        single receiver. The header information for these files must
        be correct and readable. Currently supported file types are
        SEGY and SU.

        Parameters
        ----------
        fnames : str or iterable
            File name or iterable of file names. If multiple files 
            are provided the traces are stacked according to the method
            described in the settings file.
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
        if isinstance(fnames, (str,)):

        # Load settings
        with open(settings, "r") as f:
            settings = json.load(f)

        MaswWorkflowRegistry.create_instance(settings["workflow"]["type"],
                                                 fnames,)





    # TODO (jpv): Generate a default settings file on the fly.
    @classmethod
    def default_settings_file(fname):
        pass



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
