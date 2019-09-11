"""This file contains the derived class Array1D for organizing data for
a one-dimensional array."""

from utprocess import TimeSeries, ReceiverArray, Receiver1D, Source
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

    def __init__(self, receivers, source):
        """Initialize an Array1D object from a list of receivers.

        Args:
            receivers: List of initialized Receiver1D objects.

            source: Initialized Source object.

        Returns:
            Initialize an Array1D object.

        Raises:

        """
        self.receivers = receivers
        self.nchannels = len(receivers)
        self.nsamples = receivers[0].nsamples
        self.delay = receivers[0].delay
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
        self.source = source

    def plot_waterfall(self, scale_factor=1.0, timelength=1, plot_ax='x'):
        """Create waterfall plot for the shot or stack of shots for this
        array setup.

        Creates a waterfall plot from the timehistories belonging to
        this array. The waterfall includes normalized timeseries plotted
        vertically with distance. The abscissa (cartesian x-axis) is the
        relative receiver location in meters, and the ordinate 
        (cartesian y-axis) is time in seconds.

        Args: 
            scale_factor: Float denoting the scale of the nomalized
                timeseries height (peak-to-trough). Half the receiver
                spacing is generally a good value.

            timelength: Float denoting the length of the time series 
                to plot in seconds.

            plot_ax: {'x' or 'y'} denoting on which axis the waterfall
                plot should be plotted on.

        Returns:
            This method returns a tuple of the form (figure, axes) where
            figure is the figure object and axes is the axes object on
            which the schematic is plotted.

        Raises:
            This method raises no exceptions.
        """
        # TODO (jpv): Include delay attribute in receiver class.

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

    def plot_array(self):
        """Plot a schematic of the Array1D object.

        The schematic shows the relative position of the receivers and
        the source and lists the total number of receivers. The figure
        and axes are returned to the user for use in further editing if
        desired.

        Example:
            >>import matplotlib.pyplot as plt
            >>import utprocess
            >>
            >># 1.dat is a seg2 file from an MASW survey
            >>my_array = utprocess.Array1D.from_seg2s("1.dat")
            >>fig, ax = my_array.plot_array()
            >>plt.show()

        Args:
            This method takes no arguements.

        Returns:
            This method returns a tuple of the form (figure, axes) where
            figure is the figure object and axes is the axes object on
            which the schematic is plotted.

        Raises:
            This method raises no exceptions.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 2))
        
        for n_rec, receiver in enumerate(self.receivers):
            label = "Receiver" if n_rec == 1 else None
            ax.plot(receiver.position["x"],
                    receiver.position["y"],
                    marker="^",
                    color="k",
                    linestyle="None",
                    label=label)

        try:
            spacing_txt = f"Receiver spacing is {self.spacing}m."
        except ValueError:
            spacing_txt = f"Receiver spacings are not equal."
        ax.text(min(self.receivers[0].position["x"], self.source.position["x"]),
                3,
                f"Number of Receivers: {self.nchannels}\n{spacing_txt}")

        ax.plot(self.source.position["x"],
                self.source.position["y"],
                marker="D",
                color="b",
                linestyle="None",
                label=f"Source at {self.source.position['x']}m")

        ax.legend()
        ax.set_ylim([-2, 5])
        ax.set_xlabel("Distance Along Array (m)")
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
    def from_seg2s(cls, fnames):
        """Initialize an Array1D object from one or more seg2 files.

        The seg2 file(s) should be provided as a list of file names,
        the full path may be provided if desired. Each file should
        contain multiple traces where each trace corresponds to a single
        receiver.

        Args:
            fnames: A single str or a list of str where each str is an 
                input file name. These files should be of type seg2.

        Returns:
            Initialized Array1D object.

        Raises:
            TypeError: if fnames is not of type list or string.
        """
        if type(fnames) in [list, str]:
            if type(fnames) in [str]:
                fnames = [fnames]
        else:
            raise TypeError(
                f"fnames should be of type list or str, not {type(fnames)}.")

        stream = obspy.read(fnames[0])
        receivers = []
        for trace in stream.traces:
            receivers.append(Receiver1D.from_trace(trace))
        source = Source({"x": float(trace.stats.seg2.SOURCE_LOCATION),
                         "y": 0,
                         "z": 0})
        arr = cls(receivers, source)

        if len(fnames) > 0:
            for fname in fnames[1:]:
                stream = obspy.read(fname)
                for rid, trace in enumerate(stream.traces):
                    arr.receivers[rid].timeseries.stack_append(amplitude=trace.data,
                                                               dt=trace.stats.delta,
                                                               nstacks=int(trace.stats.seg2.STACK))
                    # TODO (jpv): Alllow you to index an array will return receiver
                    assert(arr.source.position["x"] ==
                           float(trace.stats.seg2.SOURCE_LOCATION))
                    assert(arr.delay ==
                           float(trace.stats.seg2.DELAY))
                    # if rid == 1:
                    #     print(np.mean(arr.receivers[rid].timeseries.amp))
        return arr
