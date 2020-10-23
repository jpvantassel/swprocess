"""Regular expression definitions."""

import re

__all__ = ["get_peak_from_max", "get_all"]


def get_peak_from_max(wavetype="rayleigh", time="(\d+\.?\d*)"):
    """Compile regular expression to extract peaks from a `.max` file.

    Parameters
    ----------
    wavetype : {'rayleigh', 'love', 'vertical', 'radial', 'transverse'}, optional
        Define a specific wavetype to extract, default is `'rayleigh'`.
    time : str, optional
        Define a specific time of interest, default is `"(\d+\.?\d*)")`,
        a generic regular expression which will match all time.

    Return
    ------
    Compiled Regular Expression
        To extract peaks from a `.max` file.

    """
    wavetype = validate_wavetypes(wavetype)
    number = "-?\d+.?\d*[eE]?[+-]?\d*"
    pattern = f"{time} ({number}) {wavetype} ({number}) ({number}) ({number}) ({number}|-?inf|nan) ({number}) 1"
    return re.compile(pattern)


def get_all(wavetype="rayleigh", time="(\d+\.?\d*)"):
    """Compile regular expression to identify peaks from a `.max` file.

    Parameters
    ----------
    wavetype : {'rayleigh', 'love', 'vertical', 'radial', 'transverse'}, optional
        Define a specific wavetype to extract, default is `'rayleigh'`.
    time : str, optional
        Define a specific time of interest, default is `"(\d+\.?\d*)")`,
        a generic regular expression which will match all time.

    Return
    ------
    Compiled Regular Expression
        To identify peaks from a `.max` file.

    """
    wavetype = validate_wavetypes(wavetype)
    pattern = f"{time} .* {wavetype} .* 1"
    return re.compile(pattern)


def validate_wavetypes(wavetype):
    if wavetype in ("rayleigh", "love", "vertical", "radial", "transverse"):
        return wavetype.capitalize()
    else:
        raise ValueError(f"wavetype={wavetype}, not recognized.")
