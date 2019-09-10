"""File for derived class PeaksPassive for handling peaks from passive
data."""

from utprocess import Peaks

class PeaksPassive(Peaks):
    """Class for handling spectral peaks from passive data."""

    def __init__(self):
        pass

    @classmethod
    def from_hfk_historical(cls, fname):
        pass

    @classmethod
    def from_hfk(cls, fname):
        pass