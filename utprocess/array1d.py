"""This file contains the derived class Array1D for organizing data for
a one-dimensional array."""

from utprocess import TimeSeries, ReceiverArray, Receiver1D
import numpy as np
import obspy
import matplotlib.pyplot as plt
from scipy import signal


class Array1D(ReceiverArray):
    """A derived class to organize the information for a 1D (i.e.,
    linear) array.

    Attributes:
        self.nsensor: Number of sensors in 1D array.
    """

    def __init__(self, receivers):
        """Initialize an Array1D object from a list of receivers.

        Args:
            receivers: List of Receiver1D arrays.

        Returns:
            Initiaize Array1D object.

        Raises:

        """
        self.nchannels = len(receivers)
        self.nsamples = receivers[0].nsamples
        self.dt = receivers[0].dt
        self.fs = 1/self.dt
        self.df = self.fs/self.nsamples
        self.fnyq = 0.5*self.fs
        self.timeseriesmatrix = np.zeros((self.nsamples, self.nchannels))
        self.position = []
        for current_receiver, receiver in enumerate(receivers):
            assert(self.nsamples == receiver.nsamples)
            assert(self.dt == receiver.dt)
            self.timeseriesmatrix[:, current_receiver] = receiver.amp
            self.position.append(receiver.position["x"])
        self.kres = 2*np.pi / min(np.diff(self.position))
        assert(self.kres > 0)

    def plot_waterfall(self, scale_factor=1.0, timelength=1, plot_ax='x'):
        """

        Args:

        Returns:
            This method returns no value.

        Raises:
            This method raises no exceptions.
        """
        # TODO (jpv): Include delay attribute in receiver class.
        self.delay = 0

        time = np.arange(self.delay, (self.nsamples *
                                      self.dt + self.delay), self.dt)

        # Normalize and detrend
        norm_traces = np.zeros(np.shape(self.timeseriesmatrix))
        for k in range(self.nchannels):
            current_trace = self.timeseriesmatrix[:, k]
            current_trace = signal.detrend(current_trace)
            current_trace = current_trace / np.amax(current_trace)
            current_trace = current_trace*scale_factor + self.position[k]
            norm_traces[:, k] = current_trace

        # Plotting
        if str.lower(plot_ax) == 'y':
            fig = plt.figure(figsize=(2.75, 6))
            ax = fig.add_axes([0.14, 0.20, 0.8, 0.8])
            for m in range(self.nchannels):
                ax.plot(time, norm_traces[:, m], 'b-', linewidth=0.5)
            ax.set_xlim((min(time), max(time)))
            ax.set_ylim(
                (-self.position[1], self.position[1]+self.position[len(self.position)-1]))
            ax.set_xticklabels(ax.get_xticks(), fontsize=11, fontname='arial')
            ax.set_yticklabels(ax.get_yticks(), fontsize=11, fontname='arial')
            ax.grid(axis='x', linestyle='--')
            ax.set_xlabel('Time (s)', fontsize=11, fontname="arial")
            ax.set_ylabel('Normalized Amplitude',
                          fontsize=11, fontname="arial")
            ax.tick_params(labelsize=11)
            ax.tick_params('x', length=4, width=1, which='major')
            ax.tick_params('y', length=4, width=1, which='major')
        elif str.lower(plot_ax) == 'x':
            fig = plt.figure(figsize=(6, 2.75))
            ax = fig.add_axes([0.14, 0.20, 0.8, 0.75])
            for m in range(self.nchannels):
                ax.plot(norm_traces[:, m], time, 'b-', linewidth=0.5)
            ax.set_ylim((max(time), min(time)))
            ax.set_xlim(
                (-self.position[1], self.position[1]+self.position[len(self.position)-1]))
            ax.set_yticklabels(ax.get_yticks(), fontsize=11, fontname='arial')
            ax.set_xticklabels(ax.get_xticks(), fontsize=11, fontname='arial')
            ax.grid(axis='y', linestyle='--')
            ax.set_ylabel('Time (s)', fontsize=11, fontname="arial")
            ax.set_xlabel('Normalized Amplitude',
                          fontsize=11, fontname="arial")
            ax.tick_params(labelsize=11)
            ax.tick_params('y', length=4, width=1, which='major')
            ax.tick_params('x', length=4, width=1, which='major')
        return (fig, ax)

    @property
    def spacing(self):
        min_spacing = min(np.diff(self.position))
        max_spacing = max(np.diff(self.position))
        if min_spacing == max_spacing:
            return min_spacing
        else:
            raise ValueError(
                "spacing is not defined for non-equally spaced arrays")

    @classmethod
    def from_seg2(cls, fname):
        """Initialize an Array1D object from seg2 file with one stream
        and multiple traces.

        Args:
            fname: Name of input file (should be of type seg2).

        Returns:
            Initialized Array1D object.

        Raises:
            This method raises no exceptions.
        """
        stream = obspy.read(fname)
        receivers = []
        for trace in stream.traces:
            receivers.append(Receiver1D.from_trace(trace))
        return cls(receivers)
