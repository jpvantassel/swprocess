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
