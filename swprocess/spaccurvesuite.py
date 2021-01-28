"""SpacCurveSuite class definition."""

import numpy as np

from .regex import get_spac_ratio, get_spac_ring
from .spaccurve import SpacCurve
from .peakssuite import PeaksSuite


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
        
        Notes
        -----
        Assumes all `SpacCurve`s share same frequency sampling.

        """
        data_matrix = np.empty((len(self), len(self[0].frequency)))
        for row, spaccurve in enumerate(self.spaccurves):
            data_matrix[row] = spaccurve
        return data_matrix

    def _calc_spac_ratio_uncertainty(self, data_matrix=None):
        if data_matrix is None:
            data_matrix = self._data_matrix()
        
        mean = np.mean(data_matrix, axis=1)
        std = np.std(data_matrix, axis=1, ddof=1)
        cov = np.cov(data_matrix.T).T

        return (mean, std, cov)
        
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

        # Check if all values are accounted for.
        meas_npeaks = data.count("\n") - data.count("#")
        # TODO (jpv): Implement radial and transverse spaccurve, remove * 3.
        found_npeaks = len(found_curves) * 3 * len(spaccurve.frequencies)
        
        if meas_npeaks != found_npeaks:
            msg = f"Number of measured peaks {meas_npeaks} does not equal the number of found peaks {found_npeaks}."
            raise ValueError(msg)

        return cls.from_list(spaccurves)

    def __getitem__(self, index):
        return self.spaccurves[index]

    def __len__(self):
        return len(self.spaccurves)
