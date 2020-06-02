# """PeaksActive class definition."""

# import json
# import logging

# import numpy as np

# from swprocess import Peaks

# logger = logging.getLogger(__name__)


# class PeaksActive(Peaks):
#     """Spectral peaks of active-source dispersion."""

#     def __init__(self, frequency, velocity, identifier=0):
#         super().__init__(frequency, velocity, identifier)

#     @classmethod
#     def from_json(cls, fname):
#         with open(fname, "r") as f:
#             data = json.load(f)
#         return cls(data["frequency"], data["velocity"])
