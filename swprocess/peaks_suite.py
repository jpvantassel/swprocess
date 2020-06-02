"""PeaksSuite class definition."""

from swprocess import Peaks

class PeaksSuite():

    @staticmethod
    def _check_input(peaks):
        if isinstance(peaks, Peaks):
            msg = f"peaks must be of type `Peaks`, not {type(Peaks)}."
            raise TypeError(msg)

    def __init__(self, peaks):
        """Instantiate a `PeaksSuite` object from a `Peaks` object.

        Parameters
        ----------
        peaks : Peaks
            A `Peaks` object to include in the suite.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        self._check_input(peaks)
        self.peaks = [peaks]

    def append(self, peaks):
        """Append a `Peaks` object.

        Parameters
        ----------
        peaks : Peaks
            A `Peaks` object to include in the suite.

        Returns
        -------
        None
            Appends `Peaks` to `PeaksSuite`.

        """
        self._check_input(peaks)
        self.peaks.append(peaks)



            # ValueError
            # If `fnames` and `identifiers` are not the same length.

    

