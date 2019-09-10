"""File for derived class PeaksActive for handling peaks from active
data."""

from utprocess import Peaks

import numpy as np
import json


class PeaksActive(Peaks):
    """Class for storing and manipulating spectral peaks from
    active-source dispersion measurments."""

    def __init__(self, frequency, velocity, offset):
        super().__init__(frequency, velocity, offset)
        # self.offset = offset

    @classmethod
    def from_json(cls, fname):
        with open(fname, "r") as f:
            data = json.load(f)

        frequency, velocity, offset = [], [], []
        for key in data:
            frequency += [np.array(data[key]["frequency"])]
            velocity += [np.array(data[key]["velocity"])]
            offset += [key]
        return cls(frequency, velocity, offset)
