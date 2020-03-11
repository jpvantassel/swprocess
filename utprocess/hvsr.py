"""This file contains the class Hvsr for organizing data 
related to the horizontal-to-vertical spectral ratio method."""

import os
import glob
import re
import logging
logger = logging.getLogger(__name__)


class Hvsr():
    def __init__(self, frequency, amplitude, identifier):
        self.frq = [frequency]
        self.amp = [amplitude]
        self.idn = identifier

    def append(self, frequency, amplitude):
        for cfreq, nfreq in zip(self.frq[0], frequency):
            if cfreq!=nfreq:
                raise ValueError(f"appended f {cfreq} != existing f{nfreq}")
        self.frq.append(frequency)
        self.amp.append(amplitude)

    @classmethod
    def from_geopsy_folder(cls, dirname, identifier):
        logging.info(f"Reading .hv files from {dirname}")
        fnames = glob.glob(dirname+"/*.hv")
        logging.debug(f"File names to load are {fnames}")
        logging.info(f"Starting file {fnames[0]}")
        obj = cls.from_geopsy_file(fnames[0], identifier)
        for fname in fnames[1:]:
            logging.info(f"Starting file {fname}")
            tmp_obj = cls.from_geopsy_file(fname, "temp")
            obj.append(tmp_obj.frq[0], tmp_obj.amp[0])
        return obj

    @classmethod
    def from_geopsy_file(cls, fname, identifier):
        with open(fname, "r") as f:
            lines = f.read().splitlines()

        for num, line in enumerate(lines):
            if line.startswith("# Frequency"):
                start_line = num + 1 
                break

        frq, amp = [], []
        for line in lines[start_line:]:
            fr, am = re.findall(r"^(\d+.?\d*)\t(\d+.?\d*)\t\d+.?\d*\t\d+.?\d*$", line)[0]
            frq.append(float(fr))
            amp.append(float(am))

        return cls(frq, amp, identifier)