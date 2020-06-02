"""PeaksSuite class definition."""

import json 

from swprocess import Peaks

class PeaksSuite():

    @staticmethod
    def _check_input(peaks):
        if not isinstance(peaks, Peaks):
            msg = f"peaks must be an instance of `Peaks`, not {type(peaks)}."
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
        self.ids = [peaks.ids]

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
        if peaks.ids in self.ids:
            msg = f"There already exists a member object with ids = {peaks.ids}."
            raise ValueError(msg)
        self.peaks.append(peaks)
        self.ids.append(peaks.ids)

    @classmethod
    def from_dicts(cls, dicts):
        """Instantiate `PeaksSuite` from `list` of `dict`s.

        Parameters
        ----------
        dicts : list of dict or dict
            List of `dict` or a single `dict` containing dispersion
            data. 

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        if isinstance(dicts, dict):
            dicts = [dicts]

        iterable = []
        for _dict in dicts:
            for identifier, data in _dict.items():
                iterable.append(Peaks.from_dict(data, identifier=identifier))
        
        return cls.from_iter(iterable)

    @classmethod
    def from_jsons(cls, fnames):
        """Instantiate `PeaksSuite` from json file(s).

        Parameters
        ----------
        fnames : list of str or str
            List of or a single file name containing dispersion data.
            Names may contain a relative or the full path.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        if isinstance(fnames, str):
            fnames = [fnames]

        dicts = []
        for fname in fnames:
            with open(fname, "r") as f:
                dicts.append(json.load(f))
        return cls.from_dicts(dicts)

    @classmethod
    def from_maxs(cls, fnames, identifiers, rayleigh=True, love=False):

        if len(fnames) != len(identifiers):
            msg = f"len(fnames) must equal len(identifiers), {len(fnames)} != {len(identifiers)}"
            ValueError(msg)
        
        iterable = []
        for fname, identifier in zip(fnames, identifiers):
            peaks = Peaks.from_max(fname, identifier=identifier, rayleigh=rayleigh, love=love)
            iterable.append(peaks)

        return cls.from_iter(iterable)

    @classmethod
    def from_iter(cls, iterable):
        """Instantiate `PeaksSuite` from iterable object.

        Parameters
        ----------
        iterable : iterable
            Iterable containing `Peaks` objects.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        obj = cls(iterable[0])
        
        if len(iterable) >= 1:
            for _iter in iterable[1:]:
                obj.append(_iter)

        return obj

    def __eq__(self, other):
        if not isinstance(other, PeaksSuite):
            return False

        for mypeaks, urpeaks in zip(self.peaks, other.peaks):
            if mypeaks != urpeaks:
                return False
            
        return True