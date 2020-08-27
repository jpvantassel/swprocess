"""A speed test on WavefieldTransform1D."""

import cProfile
import pstats
import json

import numpy as np

import swprocess
from testtools import get_full_path

full_path = get_full_path(__file__)

def main():
    method = "slant-stack"
    settings=full_path+"settings/settings_performance.json"
    with open(settings, "r") as f:
        data = json.load(f)
    data["method"] = method
    with open(settings, "w") as f:    
        json.dump(data, f)
    fname = full_path+"../examples/sample_data/wghs/6.dat"
    array = swprocess.Array1D.from_files(fnames=fname)
    transform = swprocess.WavefieldTransform1D(array=array, settings=settings)

fname = full_path+".tmp_profiler_run"
data = cProfile.run('main()', filename=fname)
stat = pstats.Stats(fname)
stat.sort_stats('tottime')
stat.print_stats(0.01)

# fk
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 27 : 0.132 s -> Baseline

# phase-shift
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 27 : 0.510 s -> Baseline

# slant-stack
# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 08 - 27 : 0.759 s -> Baseline
