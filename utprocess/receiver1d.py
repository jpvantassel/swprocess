"""This file contains the derived class receiver1D for organizing data 
for a receiver with only one component."""

from utprocess import Receiver, TimeSeries
import logging
logger = logging.getLogger(__name__)


class Receiver1D(Receiver):
    """Derived receiver class for sensors with only one component.

    Attributes:
        timseeries: TimeSeries corresponding to this reciever.
        amp:
        dt: 
        position: 

    """

    def __init__(self, timeseries, position):
        """Initialize a one-dimensional (i.e., single components) 
        receiver.

        Args:
            timeseries: Initialized TimeSeries object.

            position: Relative position of this receiver of the form
                {'x': xval, 'y':yval, 'z':zval}
                TODO (jpv): allow for non-relative positions.

        Returns:
            An intialized one-dimensional receiver object.

        Raises:
            This method raises no exceptions.
        """
        self.timeseries = timeseries
        # self.amp = self.timeseries.amp
        # self.dt = self.timeseries.dt
        # self.delay = self.timeseries.delay
        # self.nsamples = len(self.amp)
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

        return cls(TimeSeries(amplitude=trace.data,
                              dt=trace.stats.delta,
                              nstacks=int(trace.stats.seg2.STACK),
                              delay=float(trace.stats.seg2.DELAY)),
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
