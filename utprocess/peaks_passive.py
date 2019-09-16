# """File for derived class PeaksPassive for handling peaks from passive
# data."""

# from utprocess import Peaks
# import numpy as np
# import re
# import logging
# logger = logging.getLogger(__name__)


# class PeaksPassive(Peaks):
#     """Class for handling spectral peaks from passive-wavefield
#     dispersion measurments."""

#     def __init__(self, peak_data_dict):
#         """Initialize PeaksPassive object from either rayleigh, love,
#         or both.

#         TODO (jpv): Take care of the other details later.

#         Args:
#             array_name: String to identifty array.

#             rayleigh: Dictonary with the following key, value pairs:
#                 key         value
#                 'freq'      float denoting frequency value
#                 'vel'       float or list of velocity values
#                 'az'        float or list denoting azimuth (deg? TODO: (jpv))
#                 'ell'       float or list denoting ellipticity
#                 'nosie'     float or list dentoting (TODO: (jpv))
#                 'power'     float or list dentoting power
#                 'valid'     bool or list dentoting if point is valid

#         Returns:
#             Initialized PeaksPassive object.

#         Raises:
#             This method raises no exceptions.
#         """
#         super().__init__(peak_data_dict)