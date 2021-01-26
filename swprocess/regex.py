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

def get_spac_ratio(component="0", ring="\d*"):
    """TODO (jpv): Finish docstring.

    Paramters
    ---------
    component : {"0", "1", "2"}, optional
        Component vertical="0", radial="1", and transverse="2" to be
        read, default is "0".
    ring : str
        Desired ring, default is "\d*" so all rings will be
        exported.

    Returns
    -------
    Compiled regular expression
        To read lines from SPAC-style `.max` file.

    """
    if component != "0":
        msg = f"component={component} is not allowed; only vertical component is implemented."
        raise NotImplementedError(msg)

    number =  "-?\d+.?\d*[eE]?[+-]?\d*"

    pattern = f"({number}) ({number}) ({component}) ({ring}) ({number})"
    return re.compile(pattern)
