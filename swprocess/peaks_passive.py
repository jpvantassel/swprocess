"""PeaksPassive class definition."""

import re
import logging

import numpy as np

from swprocess import Peaks

logger = logging.getLogger(__name__)


class PeaksPassive(Peaks):
    """Spectral peaks of passive-wavefield dispersion."""

    def __init__(self, frequency, velocity, identifier, **kwargs):
        """Initialize an instance of Peaks from a list of frequency
        and velocity values.

        Parameters
        ----------
        frequency, velocity : list
            Frequency and velocity (one per peak), respectively.
        identifiers : str
            String to uniquely identify the provided frequency-velocity
            pair.
        **kwargs : kwargs
            Optional keyword argument(s) these may include
            additional details about the dispersion peaks such as:
            azimuth (azi), ellipticity (ell), power (pwr), and noise
            (noi). Will generally not be used directly.

        Returns
        -------
        PeaksPassive
            Initialized `PeaksPassive` object.

        """
        super().__init__(frequency, velocity, identifier, **kwargs)

    @classmethod
    def from_max(cls, fname, identifier="0", rayleigh=True, love=False):
        """Initialize `Peaks` from `.max` file(s).

        Parameters
        ----------
        fnames : str
            Denotes the filename(s) for the .max file, may include a
            relative or the full path.
        identifier : str
            Uniquely identifying the dispersion data from each file.
        rayleigh : bool, optional
            Denote if Rayleigh data should be extracted, default is
            `True`.
        love : bool, optional
            Denote if Love data should be extracted, default is
            `False`.

        Returns
        -------
        PeaksPassive
            Initialized `PeaksPassive` object.

        Raises
        ------
        ValueError
            If neither or both `rayleigh` and `love` are equal to
            `True`.

        """
        if not isinstance(rayleigh, bool) and not isinstance(love, bool):
            msg = f"`rayleigh` and `love` must both be of type `bool`, not {type(rayleigh)} and {type(love)}."
            raise TypeError(msg)
        if rayleigh == True and love == True:
            raise ValueError("`rayleigh` and `love` cannot both be `True`.")
        if rayleigh == False and love == False:
            raise ValueError("`rayleigh` and `love` cannot both be `False`.")

        if isinstance(fnames, str):
            fnames = [fnames]
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        if len(fnames) != len(identifiers):
            raise ValueError("`len(fnames)` must equal `len(identifiers)`.")

        disp_type = "Rayleigh" if rayleigh else "Love"
        number = "-?\d+.?\d*[eE]?[+-]?\d*"
        pattern = f"^\d+\.?\d* (\d+\.?\d*) (Rayleigh|Love) ({number}) ({number}) ({number}) (\d+\.?\d*|-?inf|nan) (\d+\.?\d*) (0|1)$"
        # TODO (jpv): Change re here to be compiled, change from findall to finditer.
        # TODO (jpv): Add time block here.

        for fnum, (fname, identifier) in enumerate(zip(fnames, identifiers)):
            logging.debug(f"Attempting to Open File: {fname}")
            with open(fname, "r") as f:
                lines = f.read().splitlines()

            for line_number, line in enumerate(lines):
                if line.startswith("# BEGIN DATA"):
                    start = line_number + 3
                    break

            frqs, vels, azis, ells, nois, pwrs = [], [], [], [], [], []
            for line_number, line in enumerate(lines[start:]):
                try:
                    fr, pol, sl, az, el, noi, pw, ok = re.findall(pattern, line)[
                        0]
                except IndexError as e:
                    print(line)
                    raise e
                if pol == disp_type and ok == "1":
                    frqs.append(float(fr))
                    vels.append(1/float(sl))
                    azis.append(float(az))
                    ells.append(float(el))
                    nois.append(float(noi))
                    pwrs.append(float(pw))
                elif pol != disp_type:
                    continue
                elif ok == "0":
                    logging.warn(f"Invalid point! Line #{line_number+start+1}")
                else:
                    logging.debug(pol)
                    logging.debug(ok)
                    raise ValueError("Check line")

            args = (frqs, vels, identifier)
            kwargs = dict(azi=azis, ell=ells, noi=nois, pwr=pwrs)
            if fnum == 0:
                obj = cls(*args, **kwargs)
            else:
                obj.append(*args, **kwargs)

        return obj
