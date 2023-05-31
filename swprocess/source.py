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

"""This file contains the Source class for storing information on the
type and location of an active-source."""

from sigpropy import TimeSeries


class Source():
    """A Source class for storing information about an active-source."""

    def __init__(self, x, y, z):
        """Initialize a Source class object.

        Parameters
        ----------
        x, y, z : float
            Source position in terms of x, y, and z all in meters.

        Returns
        -------
        Source
            Initialized `Source` object.

        """
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @classmethod
    def from_source(cls, other):
        args = [getattr(other, attr) for attr in ["x", "y", "z"]]
        return cls(*args)

    def __repr__(self):
        return f"Source(x={self._x}, y={self._y}, z={self._z})"

    def __eq__(self, other):
        if not isinstance(other, Source):
            return False

        for attr in ["x", "y", "z"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True


class SourceWithSignal(Source, TimeSeries):
    """Contains source position and signal information."""

    def __init__(self, x, y, z, amp, dt):
        """Create from spatial and signal information.

        Parameters
        ----------
        x, y, z : float
            Source position in terms of x, y, and z all in meters.
        amp : interable of floats
            Amplitude of source signal.
        dt : float
            Time step in seconds.

        Returns
        -------
        Source
            Initialized `Source` object.

        """
        Source.__init__(self, x, y, z)
        TimeSeries.__init__(self, amp, dt)
