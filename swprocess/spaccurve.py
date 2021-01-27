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
        sort_ids = np.argsort(frequencies)
        self.frequencies = self.frequencies[sort_ids]
        self.ratios = np.array(ratios, dtype=float)
        self.ratios = self.ratios[sort_ids]
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
            frequencies.append(float(frequency))
            ratios.append(float(ratio))
        return cls(frequencies, ratios, time, component, ring, dmin, dmax)

    def theoretical_spac_ratio_function_custom(self):
        return self.theoretical_spac_ratio_function_general(component=self.component,
                                                            dmin=self.dmin,
                                                            dmax=self.dmax)

    @staticmethod
    def theoretical_spac_ratio_function_general(component, dmin, dmax):
        if component == 0:
            def func(frequencies, vrs, dmin=dmin, dmax=dmax):
                w = 2*np.pi*frequencies
                ratios = 2*vrs
                ratios /= (w*(dmax*dmax - dmin*dmin))
                ratios *= (dmax*j1(w*dmax/vrs) -
                           dmin*j1(w*dmin/vrs))
                return ratios
        else:
            msg = f"component={component} is not allowed; only vertical component=0 is implemented."
            raise NotImplementedError(msg)

        return func

    def fit_to_theoretical(self):
        """Fit SPAC ratio curve to theoretical functional form.

        Paramters
        ---------
        # initial_guesses : {float, array-like}, optional
        #     Initial guesses for data's approximate wave velocity,
        #     default is 400 m/s.
        # lower_bounds, upper_bounds : {float, array-like}, optional
        #     Upper and lower boundaries on surface wave phase velocity
        #     for each frequency, defaults are `0` and `4000`
        #     respectively.

        Returns
        -------
        tuple
            Of the form `(frequencies, vrs)` where `frequencies` are the
            frequencies where the SPAC curve is defined and `vrs` are
            the fit Rayleigh wave phase velocities.

        TODO (jpv): Extend this to radial and transverse.

        """
        func = self.theoretical_spac_ratio_function_custom()
        
        vtrial = np.linspace(50, 4000, 4000)
        vrs = np.empty_like(self.frequencies)
        errors = np.empty_like(self.frequencies)
        for index, (frequency, exp_ratio) in enumerate(zip(self.frequencies, self.ratios)):
            theo_ratios = func(frequency, vtrial)
            
            error = theo_ratios - exp_ratio
            
            v_index = np.argmin(error*error)
            
            errors[index] = error[v_index]
            vrs[index] = vtrial[v_index]


        # def wrapper_func(frequencies, *vrs):
        #     vrs = np.array(vrs)
        #     return func(frequencies, vrs)


        # lower_bounds = np.ones_like(self.frequencies)*lower_bounds
        # upper_bounds = np.ones_like(self.frequencies)*upper_bounds

        # vrs, _ = curve_fit(wrapper_func, self.frequencies, self.ratios,
        #                    p0=guesses, bounds=(lower_bounds, upper_bounds))

        # vrs = np.empty_like(self.frequencies)
        # for index, (guess, lower_bound, upper_bound) in enumerate(zip(guesses, lower_bounds, upper_bounds)):
        #     vrs, _ = curve_fit(func, self.frequencies, self.ratios,
        #                       p0=guess, bounds=(lower_bound, upper_bound))
        #     vrs[index] = vr

        return (self.frequencies, vrs, errors)
