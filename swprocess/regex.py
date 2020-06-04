"""Regular expression definitions."""

import re

__all__ = ["getpeak_rayleigh", "getpeak_love", "getall_rayleigh", "getall_love"]

# rpat = 
# lpat = f"({time}) (\d+\.?\d*) Love ({number}) ({number}) ({number}) (\d+\.?\d*|-?inf|nan) (\d+\.?\d*) 1"

# lcnt = f".* Love .* 1"

# getpeak_rayleigh = re.compile(rpat)
# getpeak_love = re.compile(lpat)

# getall_rayleigh = re.compile(rcnt)
# getall_love = re.compile(lcnt)

def get_peak_from_max(time=None, wavetype=None):
    """Returns a compiled regex for details specified.

    Parameters
    ----------
    time : str, optional

    wavetype : {'rayleigh', 'love'}, optional

    """
    if time is None:
        time = "(\d+\.?\d*)"

    if wavetype is None:
        wavetype = "(Rayleigh|Love)"
    else:
        wavetype = wavetype.capitalize()

    number = "-?\d+.?\d*[eE]?[+-]?\d*"
    pattern = f"{time} (\d+\.?\d*) {wavetype} ({number}) ({number}) ({number}) (\d+\.?\d*|-?inf|nan) (\d+\.?\d*) 1"
    return re.compile(pattern)

def get_all(time, wavetype):
    pattern = f"{time} .* {wavetype.capitalize()} .* 1"
    return re.compile(pattern)
