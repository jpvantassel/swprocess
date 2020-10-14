"""A speed test on WavefieldTransform1D."""

import cProfile
import pstats
import json

import numpy as np

from swprocess.masw import Masw, MaswWorkflowRegistry
from testtools import get_full_path

full_path = get_full_path(__file__)


def main():
    fname = full_path+"../examples/masw/data/wghs/6.dat"
    settings_fname = full_path+"data/settings/settings_performance.json"
    Masw.create_settings_file(fname=settings_fname, workflow="single", 
                              trim=False, mute=False, pad=True, df=1.0,
                              transform="fk", fmin=5, fmax=100, vmin=100,
                              vmax=400, nvel=200, vspace="linear", snr=False,
                              weighting="sqrt", steering="cylindrical",
                              )
    transform = Masw.run(fnames=fname, settings_fname=settings_fname)

fname = full_path+".tmp_profiler_run"
data = cProfile.run('main()', filename=fname)
stat = pstats.Stats(fname)
stat.sort_stats('tottime')
stat.print_stats(0.01)

# fk
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 27 : 0.132 s -> Baseline
# 2020 - 10 - 14 : 0.479 s -> New baseline after workflow rewrite.

# phase-shift
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 27 : 0.510 s -> Baseline
# 2020 - 10 - 14 : 0.551 s -> New baseline after workflow rewrite.

# slantstack
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 27 : 7.238 s -> Baseline
# 2020 - 08 - 27 : 0.318 s -> Complex indexing implementation
# 2020 - 08 - 27 : 8.063 s -> Functional implementation sans jit
# 2020 - 08 - 27 : 9.096 s -> Functional implementation with jit
# 2020 - 08 - 28 : 0.325 s -> Revert to earlier slant-stack implementation
# 2020 - 10 - 14 : 0.363 s -> New baseline after workflow rewrite.

# fdbf w/ weight=sqrt and steer=cylindrical
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 30 : 0.816 s -> Baseline
# 2020 - 09 - 01 : 0.733 s -> Remove excessive transposes in scm
# 2020 - 09 - 01 : 0.883 s -> Change problem to sqrt and cylindrical
# 2020 - 10 - 14 : 0.872 s -> New baseline after workflow rewrite.