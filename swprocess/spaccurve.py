# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

# """SpacCurve class definition."""

# import numpy as np
# from scipy.special import j1
# from scipy.optimize import curve_fit

# from .regex import get_spac_ratio
# from .peaks import Peaks


# class SpacCurve():
#     """Definition of SPAC ratio curve."""

#     def __init__(self, frequencies, ratios, time, component, ring, dmin, dmax):
#         """

#         Parameters
#         ---------
#         frequencies : array-like
#             Frequencies associated with each SPAC ratio.
#         ratios : array-like
#             SPAC ratio values, one per frequency.

#         """
#         self.frequencies = np.array(frequencies, dtype=float)
#         sort_ids = np.argsort(frequencies)
#         self.frequencies = self.frequencies[sort_ids]
#         self.ratios = np.array(ratios, dtype=float)
#         self.ratios = self.ratios[sort_ids]
#         self.time = str(time)
#         self.component = int(component)
#         self.ring = int(ring)
#         self.dmin, self.dmax = float(dmin), float(dmax)

#     @classmethod
#     def _from_data(cls, data, time, component, ring, dmin, dmax):
#         regex = get_spac_ratio(time=time, component=component, ring=ring)
#         frequencies, ratios = [], []
#         for found in regex.finditer(data):
#             frequency, ratio = found.groups()
#             frequencies.append(float(frequency))
#             ratios.append(float(ratio))
#         return cls(frequencies, ratios, time, component, ring, dmin, dmax)

#     def theoretical_spac_ratio_function_custom(self):
#         return self.theoretical_spac_ratio_function_general(component=self.component,
#                                                             dmin=self.dmin,
#                                                             dmax=self.dmax)

#     @staticmethod
#     def theoretical_spac_ratio_function_general(component, dmin, dmax):
#         # TODO (jpv): Extend this to radial and transverse.
#         if component == 0:
#             def func(frequencies, vrs, dmin=dmin, dmax=dmax):
#                 w = 2*np.pi*frequencies
#                 ratios = 2*vrs
#                 ratios /= (w*(dmax*dmax - dmin*dmin))
#                 ratios *= (dmax*j1(w*dmax/vrs) -
#                            dmin*j1(w*dmin/vrs))
#                 return ratios
#         else:
#             msg = f"component={component} is not allowed; only vertical component=0 is implemented."
#             raise NotImplementedError(msg)

#         return func

#     # def fit_to_theoretical(self, vrange=(50, 4000)):
#     #     """Fit SPAC ratio curve to theoretical functional form.

#     #     Parameters
#     #     ---------
#     #     vrange : tuple, optional
#     #         Contains the upper and lower limit of the search range for
#     #         the Rayleigh phase velocity, default is `(50, 4000)`. It is
#     #         strongly recommended that this range be narrowed according
#     #         to the anticipated site conditions and/or other experimental
#     #         measurements (e.g., FK-type processing).

#     #     Returns
#     #     -------
#     #     tuple
#     #         Of the form `(frequencies, vrs)` where `frequencies` are the
#     #         frequencies where the SPAC ratios are defined and `vrs` are
#     #         the Rayleigh wave phase velocities which best explain the
#     #         experimental SPAC ratios.

#     #     """
#     #     func = self.theoretical_spac_ratio_function_custom()

#     #     # Define trial velocities with 0.9 m/s spacing.
#     #     vmin, vmax = min(vrange), max(vrange)
#     #     n = int((vmax - vmin)/0.9)
#     #     vtrial = np.linspace(vmin, vmax, n)

#     #     vrs = np.empty_like(self.frequencies)
#     #     for index, (frequency, exp_ratio) in enumerate(zip(self.frequencies, self.ratios)):
#     #         theo_ratios = func(frequency, vtrial)

#     #         diff = theo_ratios - exp_ratio
#     #         error = np.abs(diff*diff)

#     #         v_index = np.argmin(error)
#     #         vrs[index] = vtrial[v_index]

#     #     return (self.frequencies, vrs)

#     # def to_peaks(self, vrange=(50, 4000)):
#     #     """Transform `SpacCurve` to `Peaks` object.

#     #     Parameters
#     #     ----------
#     #     vrange : tuple, optional
#     #         See parameter description in
#     #         :meth:`~swprocess.spaccurve.SpacCurve.fit_to_theoretical`

#     #     Returns
#     #     -------
#     #     Peaks
#     #         Containing the frequency-velocity peaks fit to the SPAC
#     #         ratios.

#     #     """
#     #     frequency, velocities = self.fit_to_theoretical(vrange=vrange)
#     #     return Peaks(frequency, velocities,
#     #                  identifier=f"time={self.time}; ring={self.ring}")
