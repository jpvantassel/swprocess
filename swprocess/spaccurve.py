"""SpacCurve class definition."""

import numpy as np

class SpacCurve():
    """Container for SPAC ratios."""

    def __init__(frequencies, ratios):
        """

        Paramters
        ---------
        frequencies : array-like
            Frequencies associated with each SPAC ratio.
        ratios : array-like
            SPAC ratio values, one per frequency.

        """
        self._frq = np.array(frequencies, dytpe=float)
        self.ratios = np.array(ratios, dtype=float)

    @classmethod
    def from_max(cls, fname):
        