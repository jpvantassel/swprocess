"""Regular expression definitions."""

import re

__all__ = ["getpeak_rayleigh", "getpeak_love", "getall_rayleigh", "getall_love"]

time = "\d+\.?\d*"
number = "-?\d+.?\d*[eE]?[+-]?\d*"
rpat = f"({time}) (\d+\.?\d*) Rayleigh ({number}) ({number}) ({number}) (\d+\.?\d*|-?inf|nan) (\d+\.?\d*) 1"
lpat = f"({time}) (\d+\.?\d*) Love ({number}) ({number}) ({number}) (\d+\.?\d*|-?inf|nan) (\d+\.?\d*) 1"

rcnt = f".* Rayleigh .* 1"
lcnt = f".* Love .* 1"

getpeak_rayleigh = re.compile(rpat)
getpeak_love = re.compile(lpat)

getall_rayleigh = re.compile(rcnt)
getall_love = re.compile(lcnt)
