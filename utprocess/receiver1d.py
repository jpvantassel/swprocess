"""This file contains the derived class `Receiver1D` for receiver with 
a single component."""

from utprocess import Receiver, TimeSeries
import logging
logger = logging.getLogger(__name__)


class Receiver1D(Receiver):
    """Derived class for receivers with a single component.

    Attributes:
        timseeries : TimeSeries
            `TimeSeries` of the receiver.
        position : dict
            Relative receiever positions of the form
            {'x': xval, 'y':yval, 'z':zval}.
    """

    def __init__(self, timeseries, position):
        """Initialize a one-dimensional (i.e., single component)
        receiver.

        Args:
            timeseries : TimeSeries
                Initialized `TimeSeries` object, containing the
                information recorded on the receiver.
            position : dict
                With relative receiever positions of the form
                {'x': xval, 'y':yval, 'z':zval}.

        Returns:
            An initialized `Receiver1D` object.
        """
        self.timeseries = timeseries
        self.position = position

    @classmethod
    def from_seg2_trace(cls, trace):
        """Create a `Receiver1D` object form a SEG2-style `Trace` object.

        Args:
            trace : Trace
                SEG2 style trace with header information entered
                correctly.

        Returns:
            An initialized `Receiver1D` object.
        """
        header = trace.stats.seg2
        return cls.from_trace(trace,
                              ftype="custom",
                              read_header=False,
                              nstacks=int(header.STACK),
                              delay=float(header.DELAY),
                              x=float(header.RECEIVER_LOCATION),
                              y=0,
                              z=0)

    @classmethod
    def from_su_trace(cls, trace):
        """Create a `Receiver1D` object form a SU-style `Trace` object.

        Args:
            trace : Trace
                SU style trace with header information entered
                correctly.

        Returns:
            An initialized `Receiver1D` object.
        """
        header = trace.stats.su.trace_header
        nstack_key = "number_of_horizontally_stacked_traces_yielding_this_trace"
        return cls.from_trace(trace,
                              ftype="custom",
                              read_header=False,
                              nstacks=int(header[nstack_key])+1,
                              delay=float(header["delay_recording_time"]),
                              x=float(header["group_coordinate_x"]/1000),
                              y=float(header["group_coordinate_y"]/1000),
                              z=0)

    @classmethod
    def from_trace(cls, trace, ftype="unkown", read_header=True, nstacks=1, delay=0, x=0, y=0, z=0):
        """Create a `Receiver1D` object from a `Trace` object.

        Args:
            trace : Trace
                `Trace` object with attributes `data` and `stats.delta`.
            ftype : {'unknown', 'seg2', 'su', 'custom'}, optional
                File type, default is 'unkown' indicating that the file
                type is unknown and should be checked against the
                available options.
            read_header : bool
                Flag to indicate whether the data in the header of the
                file should be parsed, default is `True` indicating that
                the header data will be read.
            nstacks : int, optional
                Number of stacks included in the present trace, default
                is 1 (i.e., no stacking).
            delay : float, optional
                Pre-trigger delay in seconds, default is 0 seconds.
            x, y, z : float, optional
                Receiver's relative position in x, y, and z, default is
                zero for all components (i.e., the origin).

        Returns:
            An initialized `Receiver1D` object.

        Raises:
            ValueError:
                If 'ftype' is 'unkown' and no matches can be found.
                If 'ftype' does not match the options listed.

        Examples:
            ftype="unkown", read_header=True -> If ftype is known then
                header will be read otherwise ValueError.
            ftype="unkown", read_header=False -> Header will not be
                read, even if file is a known type.
            ftype="custom" -> Header will not be read.
            ftype="seg2", read_header=True -> If file is seg2 then
                header will be read otherwise ValueError.
        """

        if ftype == "unkown":
            try:
                _format = trace.stats._format
            except:
                raise ValueError("ftype of trace could not be identified.")
        else:
            _format = ftype.upper()

        if read_header and _format == "SEG2":
            return cls.from_seg2_trace(trace)
        elif read_header and _format == "SU":
            return cls.from_su_trace(trace)
        elif _format == "CUSTOM":
            return cls(TimeSeries(amplitude=trace.data,
                                  dt=trace.stats.delta,
                                  nstacks=nstacks,
                                  delay=delay),
                       position={"x": x, "y": y, "z": z})
        else:
            raise ValueError(f"ftype={_format} not recognized.")

    def stack_trace(self, trace):
        """Append `Trace` object to an existing `Receiver1D` object.

        Args:
            trace : Trace
                `Trace` object with attributes `data` and `stats.delta`,
                assumed to come from a single stack.
        
        Returns:
            `None`, instead updates the attribute timeseries.
        """
        # TODO (jpv): Add checks in here to ensure series are complient.
        self.timeseries.stack_append(amplitude=trace.data,
                                     dt=trace.stats.delta,
                                     nstacks=1)
