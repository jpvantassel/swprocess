"""This file contains the Source class for storing information on the
type and location of an active-source."""

class Source():
    """A Source class for storing information about an active-source.
    
    Attributes:
    """
    def __init__(self, x, y, z):
        """Initialize a Source class object.
        
        Args:
            position: Dictionary showing the relative position of the
                source from the first receiver of the form:
                {'x': xval, 'y':yval, 'z':zval}
                TODO (jpv): allow for non-relative positions.

            TODO (jpv): Add type of source, timeseries etc
         
        Returns:
            This method returns no value.
        
        Raises:
            This method raises no exceptions.
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


