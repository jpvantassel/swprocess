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

"""Sensor1C class definition."""

import logging

from swprocess import ActiveTimeSeries

logger = logging.getLogger(name=__name__)


class Sensor1C(ActiveTimeSeries):
    """Class for single component sensor objects."""

    def _set_position(self, x, y, z):
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    def __init__(self, amplitude, dt, x, y, z, nstacks=1, delay=0):
        """Initialize `Sensor1C`."""
        super().__init__(amplitude, dt, nstacks=nstacks, delay=delay)
        self._set_position(x, y, z)

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
    def from_sensor1c(cls, sensor1c):
        """Create deep copy of an existing `Sensor1C` object."""
        arg_map = {"amplitude": lambda x: x, "dt": lambda x: x,
                   "x": float, "y": float, "z": float}
        args = [opr(getattr(sensor1c, arg)) for arg, opr in arg_map.items()]

        kwarg_map = {"nstacks": int, "delay": float}
        kwargs = {key: opr(getattr(sensor1c, key))
                  for key, opr in kwarg_map.items()}
        return cls(*args, **kwargs)

    @classmethod
    def from_activetimeseries(cls, activetimeseries, x, y, z):
        return cls(activetimeseries.amplitude, activetimeseries.dt, x, y, z,
                   nstacks=activetimeseries.nstacks,
                   delay=activetimeseries.delay)

    @classmethod
    def from_trace(cls, trace,
                   read_header=True, map_x=lambda x: x, map_y=lambda y: y,
                   nstacks=1, delay=0, x=0, y=0, z=0):
        """Create a `Sensor1C` object from a `Trace` object.

        Parameters
        ----------
        trace : Trace
            `Trace` object with attributes `data` and `stats.delta`.
        read_header : bool
            Flag to indicate whether the data in the header of the
            file should be parsed, default is `True` indicating that
            the header data will be read.
        map_x, map_y : function, optional
            Convert x and y coordinates using some function, default
            is not transformation. Can be useful for converting between
            coordinate systems.
        nstacks : int, optional
            Number of stacks included in the present trace, default
            is 1 (i.e., no stacking). Ignored if `read_header=True`.
        delay : float, optional
            Pre-trigger delay in seconds, default is 0 seconds.
            Ignored if `read_header=True`.
        x, y, z : float, optional
            Receiver's relative position in x, y, and z, default is
            zero for all components (i.e., the origin). Ignored if
            `read_header=True`.

        Returns
        -------
        Sensor1C
            An initialized `Sensor1C` object.

        Raises
        ------
        ValueError
            If trace type cannot be identified.

        """
        try:
            _format = trace.stats._format.upper()
        except:
            raise ValueError("Trace type could not be identified.")

        if read_header:
            if _format == "SEG2":
                return cls._from_trace_seg2(trace, map_x=map_x, map_y=map_y)
            elif _format == "SU":
                return cls._from_trace_su(trace, map_x=map_x, map_y=map_y)
            else:
                raise NotImplementedError
        else:
            return cls(amplitude=trace.data, dt=trace.stats.delta,
                       x=x, y=y, z=z, nstacks=nstacks, delay=delay)

    @classmethod
    def _from_trace_seg2(cls, trace, map_x=lambda x: x, map_y=lambda y: y):
        """Create a `Sensor1C` object form a SEG2-style `Trace` object.

        Parameters
        ----------
        trace : Trace
            SEG2-style Trace with header information entered correctly.
        map_x, map_y : function, optional
            Convert x and y coordinates using some function, default
            is not transformation. Can be useful for converting between
            coordinate systems.

        Returns
        -------
        Sensor1C
            An initialized `Sensor1C` object.

        """
        header = trace.stats.seg2
        return cls.from_trace(trace,
                              read_header=False,
                              nstacks=int(header.STACK),
                              delay=float(header.DELAY),
                              x=map_x(float(header.RECEIVER_LOCATION)),
                              y=map_y(0),
                              z=0)

    @classmethod
    def _from_trace_su(cls, trace, map_x=lambda x: x, map_y=lambda y: y):
        """Create a `Sensor1C` object form a SU-style `Trace` object.

        Parameters
        ----------
        trace : Trace
            SU-style trace with header information entered correctly.
        map_x, map_y : function, optional
            Convert x and y coordinates using some function, default
            is not transformation. Can be useful for converting between
            coordinate systems.

        Returns
        -------
        Sensor1C
            An initialized `Sensor1C` object.

        """
        header = trace.stats.su.trace_header
        nstack_key = "number_of_horizontally_stacked_traces_yielding_this_trace"
        scaleco = int(header["scalar_to_be_applied_to_all_coordinates"])

        int_x = int(header["group_coordinate_x"])
        x =  int_x / abs(scaleco) if scaleco < 0 else int_x * scaleco

        int_y = int(header["group_coordinate_y"])
        y = int_y / abs(scaleco) if scaleco < 0 else int_x * scaleco

        return cls.from_trace(trace,
                              read_header=False,
                              nstacks=int(header[nstack_key])+1,
                              delay=int(header["delay_recording_time"])/1000,
                              x=map_x(x),
                              y=map_y(y),
                              z=0)

    def _is_similar(self, other, exclude=None):
        """Check if `other` is similar to `self` though not equal."""
        if exclude is None:
            exclude = []

        if not isinstance(other, Sensor1C):
            return False

        for attr in ["x", "y", "z"]:
            if attr in exclude:
                continue
            if getattr(self, attr) != getattr(other, attr):
                return False

        if not super()._is_similar(other, exclude=exclude):
            return False

        return True

    def __eq__(self, other):
        """Check if `other` is equal to the `Sensor1C`."""
        if not super().__eq__(other):
            return False

        return True
