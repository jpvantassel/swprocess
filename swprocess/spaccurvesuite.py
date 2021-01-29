"""SpacCurveSuite class definition."""

import numpy as np
from scipy.special import jv

from .regex import get_spac_ratio, get_spac_ring
from .spaccurve import SpacCurve
from .peakssuite import PeaksSuite
from .inversion import leastsquare_iterativealgorithm, leastsquare_posterioricovmatrix


class SpacCurveSuite():

    def __init__(self, spaccurve):

        self.spaccurves = [spaccurve]

    def append(self, spaccurve):
        self.spaccurves.append(spaccurve)

    @property
    def rings(self):
        """List of rings associated with the particular suite."""
        rings = []
        for spaccurve in self.spaccurves:
            ring = spaccurve.ring
            if ring not in rings:
                rings.append(ring)
        return rings

    def _data_matrix(self):
        """Create data matrix.

        Returns
        -------
        ndarray
            Of shape `(frequencies, ncurves)` such that each column
            corresponds to one `SpacCurve`.

        Notes
        -----
        Assumes all `SpacCurve`s share same frequency sampling.

        """
        data_matrix = np.empty((len(self[0].frequencies), len(self)))
        for col, spaccurve in enumerate(self.spaccurves):
            data_matrix[:, col] = spaccurve.ratios
        return data_matrix

    def _calc_spac_ratio_stats(self, data_matrix=None):
        """Calculate statistics on experimental SPAC ratios.

        Returns
        -------
        tuple
            Of the form `(frequencies, means, stddevs, cov)`.

        """
        if data_matrix is None:
            data_matrix = self._data_matrix()

        mean = np.mean(data_matrix, axis=1)
        std = np.std(data_matrix, axis=1, ddof=1)
        cov = np.cov(data_matrix)

        return (self[0].frequencies, mean, std, cov)

    def _to_phase_stat(self, p0=1500, p0_std=500, covp0p0=None,
                       omega=5, iterations=20, tol=0.01):
        """Use non-linear least squares to compute phase velocity stats.

        Parameters
        ----------
        p0 : {float, ndarray}, optional
            Prior estimate of the phase velocity means.
        p0_std : {float, ndarray}, optional
            Prior estimate of the phase velocity standard deviations,
            ignored if `p0_cov` is provided.
        covp0p0 : ndarray, optional
            Prior estimate of the phase velocity covariance matrix,
            default is `None` so covariance matrix will be assumed.
        omega : float, optional
            Correlation length in the frequency domain, ignored if
            `covp0p0` is not `None`. Larger values promote a smoother
            solution in the phase velocity domain and consequently a
            potentially worse fit in the SPAC ratio domain.
        iterations : int, optional
            Default number of iterations to attempt.
        tol : float, optional
            Accpetable misfit tolerance, default is `0.01`.

        Returns
        -------
        tuple
            Of the form `(frequencies, means, stds, cov)` in the phase
            velocity domain.

        """
        # Prepare the data (i.e., the SPAC ratios).
        frq, d0, _, covd0d0 = self._calc_spac_ratio_stats()
        k = len(frq)
        d0 = np.reshape(d0, (k, 1))

        # Prepare the prior parameter information (i.e., on phase velocity).
        j = len(frq)
        p0 = np.ones((j, 1)) * p0
        if covp0p0 is None:
            p0_std = np.ones(j) * p0_std
            covp0p0 = np.empty((j, j))
            omega2 = omega*omega
            for row, (f0, std) in enumerate(zip(frq, p0_std)):
                df = frq - f0
                covp0p0[row] = std*std*np.exp(-0.5*(df*df)/omega2)

        # TODO (jpv): Make SpacCurveSuite only for a single ring & component.
        def calc_partial_derivative_matrix(fs, pm,
                                           dmin=self[0].dmin,
                                           dmax=self[0].dmax):
            pm = pm.flatten()
            ws = 2*np.pi*fs
            dgdp = np.empty((len(fs), len(pm)))
            pm2 = pm*pm
            dmax3 = dmax**3
            dmin3 = dmin**3
            for row, w in enumerate(ws):
                a = w*dmax3/pm2 * jv(2, w*dmax/pm)
                b = w*dmin3/pm2 * jv(2, w*dmin/pm)
                dgdp[row] = a - b
            return dgdp

        # Prepare iterative fit.
        forward = self[0].theoretical_spac_ratio_function_custom()
        pm = p0
        dm = forward(frq, p0.flatten())
        dm = np.reshape(dm, (k,1))
        dgdp = calc_partial_derivative_matrix(frq, p0)

        # Iterate
        for iteration in range(iterations):
            pm1 = leastsquare_iterativealgorithm(p0, pm, covp0p0, d0, dm, covd0d0, dgdp)

            # Calculate posteriori covariance matrix.
            dgdp = calc_partial_derivative_matrix(frq, pm1)
            covpm1pm1 = leastsquare_posterioricovmatrix(covp0p0, covd0d0, dgdp)
            
            # Error calculation (only done for the mean currently).
            error = forward(frq, pm1.flatten()) - d0.flatten()
            rms = np.sqrt(np.mean(error*error))
            if rms < tol:
                break
            
            # Update in preparation for next iteration.
            pm = pm1
            dm = forward(frq, pm.flatten())
            dm = np.reshape(dm, (k,1))

        print(iteration)
        return (frq, pm1.flatten(), np.sqrt(np.abs(np.diag(covpm1pm1))), covpm1pm1)

    # def to_peaksuite(self, rings="all"):
    #     """Transform `SpacCurveSuite` to `PeaksSuite`.

    #     Parameters
    #     ----------
    #     rings : {int, list, "all"}, optional
    #         Desired ring(s) to be transformed, default is "all".
    #     vrange : tuple, optional
    #         See parameter description in
    #         :meth:`~swprocess.spaccurve.SpacCurve.fit_to_theoretical`

    #     Returns
    #     -------
    #     list of PeaksSuite
    #         `list` of `PeaksSuite` objects, one per ring.

    #     """
    #     if rings == "all":
    #         rings = self.rings
    #     elif isinstance(ring, int):
    #         rings = [rings]
    #     else:
    #         rings = list(rings)

    #     peakssuites_one_per_ring = []
    #     for ring in rings:
    #         peaks_for_suite = []
    #         for spaccurve in self.spaccurves:
    #             if spaccurve.ring == ring:
    #                 peaks = spaccurve.to_peaks(vrange=vrange)
    #                 peaks_for_suite.append(peaks)
    #         peakssuite = PeaksSuite.from_peaks(peaks_for_suite)
    #         peakssuites_one_per_ring.append(peakssuite)
    #     return peakssuites_one_per_ring

    @classmethod
    def from_list(cls, spaccurves):
        obj = cls(spaccurves[0])

        try:
            for spaccurve in spaccurves[1:]:
                obj.append(spaccurve)
        except IndexError:
            pass

        return obj

    @classmethod
    def from_max(cls, fname, fname_log=None, component="(0)", ring="(\d+)"):
        if fname_log is None:
            fname_log = fname[:-4]+".log"

        # Read .log file for ring information.
        with open(fname_log, "r") as f:
            data = f.read()

        regex = get_spac_ring()
        rings = {}
        for iring, found in enumerate(regex.finditer(data)):
            dmin, dmax = found.groups()
            rings[str(iring)] = dict(dmin=dmin, dmax=dmax)

        # Read .max file for auto-correlation ratios.
        with open(fname, "r") as f:
            data = f.read()

        regex = get_spac_ratio(component=component, ring=ring)
        spaccurves, found_curves = [], []
        for found in regex.finditer(data):
            time, _, component, ring, _ = found.groups()

            curve_id = (time, component, ring)
            if curve_id in found_curves:
                continue
            found_curves.append(curve_id)

            spaccurve = SpacCurve._from_data(data, time, component,
                                             ring, **rings[ring])
            spaccurves.append(spaccurve)

        # # Check if all values are accounted for.
        # meas_npeaks = data.count("\n") - data.count("#")
        # # TODO (jpv): Implement radial and transverse spaccurve, remove * 3.
        # found_npeaks = len(found_curves) * 3 * len(spaccurve.frequencies)

        # if meas_npeaks != found_npeaks:
        #     msg = f"Number of measured peaks {meas_npeaks} does not equal the number of found peaks {found_npeaks}."
        #     raise ValueError(msg)

        return cls.from_list(spaccurves)

    def __getitem__(self, index):
        return self.spaccurves[index]

    def __len__(self):
        return len(self.spaccurves)
