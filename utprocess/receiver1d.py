"""This file contains the derived class receiver1D for organizing data 
for a receiver with only one component."""

from utprocess import Receiver, TimeSeries
import logging

# logging.basicConfig()


class Receiver1D(Receiver):
    """Derived receiver class for sensors with only one component.

    Attributes:
        timeeries: TimeSeries corresponding to this reciever.
        amp:
        dt: 
        position: 

    """

    def __init__(self, recordings, position):
        """Initialize a one-dimensional (i.e., single components) 
        receiver.

        Args:
            recordings: Recordings performed on this receiver.
                Each receiver should be of the form 
                {'amp':ampvals, 'dt':dtval, 'nstacks':nstackval}.
                TODO (jpv): add type info here.

            position: Relative position of this receiver of the form
                {'x': xval, 'y':yval, 'z':zval}
                TODO (jpv): allow for non-relative positions.

        Returns:
            An intialized one-dimensional receiver object.

        Raises:

        """
        if "nstacks" in recordings:
            self.timeseries = TimeSeries(recordings["amp"],
                                         recordings["dt"],
                                         recordings["nstacks"])
        else:
            self.timeseries = TimeSeries(recordings["amp"], recordings["dt"])
        self.amp = self.timeseries.amp
        self.dt = self.timeseries.dt
        self.nsamples = len(self.amp)
        self.position = position

    @classmethod
    def from_trace(cls, trace):
        """Create a Receiver1D object from a trace object.

        Args:

        Returns:
            An intialized one-dimension receiver object.

        Raises:

        """
        # TODO: Much more information to extract here if desired

        # logging.info("Receiver1D.from_trace")
        # logging.info("data = {}".format(trace.data))
        # print(trace.data)
        # print(type(trace.data))
        # print(trace.stats.delta)
        # logging.info("delta = {}".format(trace.stats.delta))
        # logging.info("stack = {}".format(trace.stats.seg2.STACK))

        return cls(recordings={"amp": trace.data,
                               "dt": trace.stats.delta,
                               "nstacks": int(trace.stats.seg2.STACK)},
                   position={"x": float(trace.stats.seg2.RECEIVER_LOCATION),
                             "y": 0.,
                             "z": 0.})

    # @classmethod
    # def from_traces(cls, traces):
    #     """Create a Receiver1D object from multiple trace objects that 
    #     will be stacked.

    #     Args:

    #     Returns:
    #         An intialized one-dimension receiver object.

    #     Raises:

    #     """

    #     return cls(recordings={"amp": trace.data,
    #                            "dt": trace.stats.delta,
    #                            "nstacks": int(trace.stats.seg2.STACK)},
    #                position={"x": float(trace.stats.seg2.RECEIVER_LOCATION),
    #                          "y": 0.,
    #                          "z": 0.})
