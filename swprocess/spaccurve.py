"""SpacCurve class definition."""

import numpy as np
from scipy.special import j1
from scipy.optimize import curve_fit

from .regex import get_spac_ratio


class SpacCurve():
    """Definition of SPAC ratio curve."""

    def __init__(self, frequencies, ratios, time, component, ring, dmin, dmax):
        """

        Paramters
        ---------
        frequencies : array-like
            Frequencies associated with each SPAC ratio.
        ratios : array-like
            SPAC ratio values, one per frequency.

        """
        self.frequencies = np.array(frequencies, dtype=float)
        self.ratios = np.array(ratios, dtype=float)
        self.time = str(time)
        self.component = int(component)
        self.ring = int(ring)
        self.dmin, self.dmax = float(dmin), float(dmax)

    @classmethod
    def _from_data(cls, data, time, component, ring, dmin, dmax):
        regex = get_spac_ratio(time=time, component=component, ring=ring)
        frequencies, ratios = [], []
        for found in regex.finditer(data):
            frequency, ratio = found.groups()
            frequencies.append(frequency)
            ratios.append(ratios)
        return cls(frequencies, ratios, time, component, ring, dmin, dmax)

    def fit_to_bessel(self, intial_guess=400,
                      lower_bounds=0, upper_bounds=4000):
        """Fit SPAC curve to associated Bessel functional form.

        Paramters
        ---------
        initial_guess : {float, array-like}, optional
            Initial guess for data's approximate wave velocity, default
            is 400 m/s.
        lower_bounds, upper_bounds : {float, array-like}, optional
            Upper and lower boundaries on surface wave phase velocity
            for each frequency, defaults are `0` and `4000`
            respectively.

        Returns
        -------
        tuple
            Of the form `(frequencies, vrs)` where `frequencies` are the
            frequencies where the SPAC curve is defined and `vrs` are
            the fit Rayleigh wave phase velocities.

        TODO (jpv): Extend this to radial and transverse.

        """
        if self.component == 0:
            def func(frequencies, vrs):
                w = 2*np.pi*frequencies
                ratios = 2*vrs
                ratios /= (w*(self.dmax*self.dmax - self.dmin*self.dmin))
                ratios *= (self.dmax*j1(w*self.dmax/vrs) -
                           self.dmin*j1(w*self.dmin/vrs))
                return ratios
        else:
            msg = f"component={self.component} is not allowed; only vertical component=0 is implemented."
            raise NotImplementedError(msg)

        guess = np.ones_like(self.frequencies)*intial_guess
        lower_bounds = np.ones_like(self.frequencies)*lower_bounds
        upper_bounds = np.ones_like(self.frequencies)*upper_bounds
        bounds = (lower_bounds, upper_bounds)

        vrs, _ = curve_fit(func, self.frequencies,
                           self.ratios, p0=guess, bounds=bounds)

        return (self.frequencies, vrs)
