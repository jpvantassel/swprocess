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

"""Regular expression definitions."""

import re

__all__ = ["get_peak_from_max", "get_nmaxima",  "get_all", "get_spac_ratio", "get_spac_ring"]

NUMBER = "-?\d+.?\d*[eE]?[+-]?\d*"


def get_peak_from_max(time="\d+\.?\d*", wavetype="rayleigh",
                      frequency="-?\d+.?\d*[eE]?[+-]?\d*"):
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
    pattern = f"({time}) ({frequency}) {wavetype} ({NUMBER}) ({NUMBER}) ({NUMBER}) ({NUMBER}|-?inf|nan) ({NUMBER}) 1"
    return re.compile(pattern)


def get_nmaxima():
    return re.compile("N_MAXIMA=(\d+)")


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


def get_spac_ratio(time="(-?\d+.?\d*[eE]?[+-]?\d*)", component="(0)",
                   ring="(\d+)"):
    """
    TODO (jpv): Finish docstring.

    Parameters
    ---------
    component : {"0", "1", "2"}, optional
        Component vertical="0", radial="1", and transverse="2" to be
        read, default is "0".
    ring : str
        Desired ring, default is "\d+" so all rings will be
        exported.

    Returns
    -------
    Compiled regular expression
        To read lines from SPAC-style `.max` file.

    """
    if component not in ["(0)", "0"]:
        msg = f"component={component} is not allowed; only vertical component=0 is implemented."
        raise NotImplementedError(msg)

    number = "(-?\d+.?\d*[eE]?[+-]?\d*)"

    pattern = f"{time} {number} {component} {ring} {number}"
    return re.compile(pattern)


def get_spac_ring():
    """Find all rings in MSPAC .log file.
    TODO (jpv): Finish docstring.

    """
    number = "(-?\d+.?\d*[eE]?[+-]?\d*)"
    pattern = f" --- Ring \({number} m, {number} m\)"
    return re.compile(pattern)
