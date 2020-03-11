"""This file contains the derived class Array1d for organizing data for
a one-dimensional array."""

from utprocess import ActiveTimeSeries, Source, Sensor1c
import numpy as np
import obspy
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Cursor
import logging
import warnings
logger = logging.getLogger(__name__)


class Array1d():
    """A class to organize the information for a 1D (linear) array.

    Attributes
    ----------
    nsensor: Number of sensors in the array.
    """

    def __make_timeseries_matrix(self):
        """Pull information from each Sensor1c into a 2D matrix.

        Returns
        -------
        ndarray
            Where each row is a different timehistory.

        """
        self.ex_rec = self.receivers[0]
        self.n_samples = self.ex_rec.timeseries.n_samples
        self.timeseriesmatrix = np.empty((self.nchannels, self.n_samples))
        self.position = []
        for current_receiver, receiver in enumerate(self.receivers):
            assert(self.ex_rec.timeseries.n_samples ==
                   receiver.timeseries.n_samples)
            assert(self.ex_rec.timeseries.dt == receiver.timeseries.dt)
            self.timeseriesmatrix[current_receiver,
                                  :] = receiver.timeseries.amp
            self.position.append(receiver.position["x"])
        logger.info("\tAll dt and n_samples are equal.")

        self.flipped = False if self.source.position['x'] < 0 else True
        if self.flipped:
            self.timeseriesmatrix = np.flipud(self.timeseriesmatrix)
            logger.info("\ttimeseriesmatrix is flipped.")

    def __init__(self, receivers, source):
        """Initialize an Array1d object from a list of receivers.

        Parameters
        ----------
        receivers : iterable
            Iterable of initialized Sensor1c objects.
        source : Source
            Initialized `Source` object.

        Returns
        -------
        Array1d
            Initialize an `Array1d` object.
        """
        logger.info("Initalize an Array1d object.")
        self.receivers = receivers
        self.nchannels = len(self.receivers)
        self.source = source
        self.__make_timeseries_matrix()
        self.kres = 2*np.pi / min(np.diff(self.position))
        self.absolute_minus_relative = 0
        assert(self.kres > 0)
        logger.info("\tkres > 0")

    def trim_record(self, start_time, end_time):
        """Trim timeseries from each Receiver1c.

        Parameters
        ----------
        start_time, end_time: float
            Desired start time and end time in seconds measured from the
            point the acquisition system was triggered.

        Returns
        -------
        None
            May update the attributes `n_samples`, `delay`, and `df`.

        Raises
        ------
        IndexError
            If the `start_time` and `end_time` is illogical.
        """
        for rec_num, receiver in enumerate(self.receivers):
            logging.debug(f"Starting to trim receiver {rec_num}")
            receiver.timeseries.trim(start_time, end_time)
        self.__make_timeseries_matrix()

    def _clean_matrix(self, scale_factor):
        norm_traces = np.empty_like(self.timeseriesmatrix)
        for k, current_trace in enumerate(self.timeseriesmatrix):
            current_trace = signal.detrend(current_trace)
            current_trace /= np.amax(current_trace)
            current_trace *= scale_factor
            current_trace += self.position[k]
            norm_traces[k, :] = current_trace
        return norm_traces

    def plot_waterfall(self, scale_factor=1.0, plot_ax='x'):
        """Create waterfall plot for this array setup.

        Creates a waterfall plot from the time series belonging to
        this array. The waterfall includes normalized timeseries plotted
        vertically with distance. The abscissa (cartesian x-axis) is the
        relative receiver location in meters, and the ordinate
        (cartesian y-axis) is time in seconds.

        Parameters
        ----------
        scale_factor : float, optional
            Denotes the scale of the nomalized timeseries height
            (peak-to-trough), default is 1. Half the receiver spacing is
            generally a good value.
        plot_ax : {'x', 'y'}, optional
            Denotes on which axis the waterfall should reside, 'x' is
            the default.

        Returns
        -------
        Tuple
            Of the form (fig, ax) where `fig` is the figure object and
            `ax` the axes object on which the schematic is plotted.
        """
        time = self.ex_rec.timeseries.time
        norm_traces = self._clean_matrix(scale_factor=scale_factor)

        # Plotting
        if str.lower(plot_ax) == 'y':
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6))
            for trace in norm_traces:
                ax.plot(time, trace, 'b-', linewidth=0.5)
            ax.set_xlim((min(time), max(time)))
            ax.set_ylim(
                (-self.position[1], self.position[1]+self.position[len(self.position)-1]))
            ax.grid(axis='x', linestyle=':')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Distance (m)')
            ax.tick_params('x', length=4, width=1, which='major')
            ax.tick_params('y', length=4, width=1, which='major')
        elif str.lower(plot_ax) == 'x':
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
            for trace in norm_traces:
                ax.plot(trace, time, 'b-', linewidth=0.5)
            ax.set_ylim((max(time), min(time)))
            ax.set_xlim(
                (-self.position[1], self.position[1]+self.position[len(self.position)-1]))
            ax.grid(axis='y', linestyle=':')
            ax.set_ylabel('Time (s)')
            ax.set_xlabel("Distance (m)")
            ax.tick_params('y', length=4, width=1, which='major')
            ax.tick_params('x', length=4, width=1, which='major')

        if self.flipped:
            ax.text(0,
                    1.1,
                    "This is a far-offset shot. Traces have been flipped.",
                    transform=ax.transAxes)
        return (fig, ax)

    def plot_array(self):
        """Plot a schematic of the `Array1d` object.

        The schematic shows the relative position of the receivers and
        the source and lists the total number of receivers. The figure
        and axes are returned to the user for use in further editing if
        desired.

        Returns
        -------
            This method returns a tuple of the form (figure, axes) where
            figure is the figure object and axes is the axes object on
            which the schematic is plotted.

        Examples
        --------
            >>import matplotlib.pyplot as plt
            >>import utprocess
            >>
            >># 1.dat is a seg2 file from an MASW survey
            >>my_array = utprocess.Array1d.from_seg2s("1.dat")
            >>fig, ax = my_array.plot_array()
            >>plt.show()
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
            raise ValueError("spacing undefined for non-equally spaced arrays")

    def _pick_threshold(self, threshold=0.05):
        """

        Parameters
        ----------
        threshold : {0.-1.}, optional
            Picking threashold as a percent.

        """
        norm_traces = self._clean_matrix(scale_factor=1)
        _time = self.ex_rec.timeseries.time

        times = []
        for trace in norm_traces:
            pindices = np.argwhere(abs(trace-np.mean(trace[:10])) > threshold)
            index = int(pindices[0])
            times.append(_time[index])

        return (self.position, times)

    def auto_pick_first_arrivals(self, algorithm='threshold', **kwargs):
        if algorithm == "threshold":
            picks = self._pick_threshold(**kwargs)
        else:
            raise NotImplementedError
        return picks

    def pick_first_arrivals(self, waterfall_kwargs=None):
        """Allow for interactive picking of first arrivals.

        Parameters
        ----------
        waterfall_kwargs : dict, optional
            Dictionary of keyword arguements for
            meth: `<plot_waterfall>`, default is `None` indicating
            default keyword arugements.

        Returns
        -------
        Tuple
            Of the form (distance, picked_time) 
        """
        if waterfall_kwargs is None:
            waterfall_kwargs = {}

        fig, ax = self.plot_waterfall(**waterfall_kwargs)

        xs, ys = [], []

        cursor = Cursor(ax, useblit=True, color='k', linewidth=1)

        print("Make adjustments: (Press spacebar when ready)")
        zoom_ok = False
        while not zoom_ok:
            zoom_ok = plt.waitforbuttonpress(timeout=-1)

        while True:
            print(
                "Pick arrival: (Left Click to Add, Right Click to Remove, Enter to Finish")
            vals = plt.ginput(n=-1, timeout=0)

            x, y = vals[-1]
            ax.plot(x, y, "r", marker="+", linestyle="")
            xs.append(x)
            ys.append(y)

            print(
                "Continue? (Make adjustments then press spacebar once to contine, twice to exit)")
            zoom_ok = False
            while not zoom_ok:
                zoom_ok = plt.waitforbuttonpress(timeout=-1)

            if plt.waitforbuttonpress(timeout=0.5):
                print("Exiting ... ")
                break

            print("Continuing ... ")

        print("Close figure when ready.")

        if waterfall_kwargs.get("plot_ax") is None:
            waterfall_kwargs["plot_ax"] = "x"

        if waterfall_kwargs["plot_ax"] == "x":
            distance, time = vals
        else:
            time, distance = vals

        return (distance, time)

    @classmethod
    def from_files(cls, fnames):
        """Initialize an `Array1d` object from one or more data files.

        This classmethod creates an `Array1d` object by reading the
        header information in the provided file(s). Each file
        should contain multiple traces where each trace corresponds to a
        single receiver. Currently supported file types include: SEGY
        and SU.

        Parameters
        ----------
        fnames : str, iterable
            File name or iterable of file names. If multiple files 
            are provided the traces are stacked.

        Returns
        -------
        Array1d
            Initialized `Array1d` object.

        Raises
        ------
        TypeError
            If `fnames` is not of type `str` or `iterable`.
        """
        # Check that fnames has the correct attributes.
        if type(fnames) == str:
            fnames = [fnames]
        elif hasattr(fnames, "__iter__"):
            pass
        else:
            msg = f"fnames must be a str or an iterable, not {type(fnames)}."
            raise TypeError(msg)

        # Read file for traces
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stream = obspy.read(fnames[0])

        # Create array of receivers
        receivers = []
        minimum_x = 1E9
        for trace in stream.traces:
            sensor = Sensor1c.from_trace(trace)
            minimum_x = min(minimum_x, sensor.position["x"])
            receivers.append(sensor)

        for receiver in receivers:
            receiver.position["x"] -= minimum_x

        # Define source
        _format = trace.stats._format
        if _format == "SEG2":
            source = Source({"x": float(trace.stats.seg2.SOURCE_LOCATION),
                             "y": 0,
                             "z": 0})
        elif _format == "SU":
            source = Source({"x": (float(trace.stats.su.trace_header["source_coordinate_x"])/1000)-minimum_x,
                             "y": float(trace.stats.su.trace_header["source_coordinate_y"])/1000,
                             "z": 0})
        else:
            raise ValueError(f"_format={_format} not recognized.")
        obj = cls(receivers, source)

        # Stack additional traces, if necessary
        if len(fnames) > 0:
            for fname in fnames[1:]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stream = obspy.read(fname)
                for receiver, trace in zip(obj, stream.traces):
                    receiver.stack_trace(trace)

        obj.absolute_minus_relative = minimum_x
        return obj

    def __getitem__(self, slices):
        return self.receivers[slices]
