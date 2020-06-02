# """PeaksPassive class definition."""

# import logging

# import numpy as np

# from swprocess import Peaks
# from swprocess.regex import *

# logger = logging.getLogger(__name__)


# class PeaksPassive(Peaks):
#     """Spectral peaks of passive-wavefield dispersion."""

#     def __init__(self, frequency, velocity, identifier, **kwargs):
#         """Initialize an instance of Peaks from a list of frequency
#         and velocity values.

#         Parameters
#         ----------
#         frequency, velocity : list
#             Frequency and velocity (one per peak), respectively.
#         identifiers : str
#             String to uniquely identify the provided frequency-velocity
#             pair.
#         **kwargs : kwargs
#             Optional keyword argument(s) these may include
#             additional details about the dispersion peaks such as:
#             azimuth (azi), ellipticity (ell), power (pwr), and noise
#             (noi).

#         Returns
#         -------
#         PeaksPassive
#             Initialized `PeaksPassive` object.

#         """
#         super().__init__(frequency, velocity, identifier, **kwargs)

