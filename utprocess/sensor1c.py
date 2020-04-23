"""Sensor1C class definition."""

import logging

from utprocess import ActiveTimeSeries

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
    def from_activetimeseries(cls, activetimeseries, x, y, z):
        return cls(activetimeseries.amp, activetimeseries.dt, x, y, z,
                   nstacks=activetimeseries.nstacks,
                   delay=activetimeseries.delay)

    def _is_similar(self, other, exclude=[]):
        """Check if `other` is similar to `self` though not equal."""
        if not isinstance(other, Sensor1C):
            return False

        if not super()._is_similar(other, exclude=exclude):
            return False

        return True

    def __eq__(self, other):
        """Check if `other` is equal to the `Sensor1C`."""
        if not super().__eq__(other):
            return False

        for attr in ["x", "y", "z"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

