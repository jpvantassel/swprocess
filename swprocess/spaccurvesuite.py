"""SpacCurveSuite class definition."""

import numpy as np

from .regex import get_spac_ratio, get_spac_ring
from .spaccurve import SpacCurve


class SpacCurveSuite():

    def __init__(self, spaccurve):

        self.spaccurves = [spaccurve]

    def append(self, spaccurve):
        self.spaccurves.append(spacurve)

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
        if meas_npeaks != pred_npeaks:
            msg = f"Number of measured peaks {meas_npeaks} does not equal the number of found peaks {found_npeaks}."
            raise ValueError(msg)

        return cls.from_list(spaccurves)

    def __getitem__(self, index):
        return self.spaccurves[index]

    def __len__(self):
        return len(self.spaccurves)
