# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (jvantassel@utexas.edu)
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

"""A speed test on WavefieldTransform1D."""

import cProfile
import pstats

from swprocess.masw import Masw
from testtools import get_full_path

full_path = get_full_path(__file__)


def main():
    fname = full_path+"../examples/masw/data/wghs/6.dat"
    settings = Masw.create_settings_dict(workflow="single", trim=False,
                                         mute=False, pad=True, df=1.0,
                                         transform="fk", fmin=5, fmax=100,
                                         vmin=100, vmax=400, nvel=200,
                                         vspace="linear", snr=False,
                                         weighting="sqrt",
                                         steering="cylindrical")
    Masw.run(fnames=fname, settings=settings)

fname = full_path+".tmp_profiler_run"
data = cProfile.run('main()', filename=fname)
stat = pstats.Stats(fname)
stat.sort_stats('tottime')
stat.print_stats(0.01)

# fk
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 27 : 0.132 s -> Baseline.
# 2020 - 10 - 14 : 0.479 s -> New baseline after workflow rewrite.
# 2021 - 03 - 05 : 0.479 s -> Update for release.

# phaseshift
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 27 : 0.510 s -> Baseline.
# 2020 - 10 - 14 : 0.551 s -> New baseline after workflow rewrite.
# 2021 - 03 - 05 : 0.580 s -> Update for release.

# slantstack
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 27 : 7.238 s -> Baseline
# 2020 - 08 - 27 : 0.318 s -> Complex indexing implementation
# 2020 - 08 - 27 : 8.063 s -> Functional implementation sans jit
# 2020 - 08 - 27 : 9.096 s -> Functional implementation with jit
# 2020 - 08 - 28 : 0.325 s -> Revert to earlier slant-stack implementation
# 2020 - 10 - 14 : 0.363 s -> New baseline after workflow rewrite.
# 2021 - 03 - 05 : 0.315 s -> Update for release.

# fdbf w/ weight=sqrt and steer=cylindrical
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 30 : 0.816 s -> Baseline
# 2020 - 09 - 01 : 0.733 s -> Remove excessive transposes in scm
# 2020 - 09 - 01 : 0.883 s -> Change problem to sqrt and cylindrical
# 2020 - 10 - 14 : 0.872 s -> New baseline after workflow rewrite.
# 2021 - 03 - 05 : 0.791 s -> Update for release.
