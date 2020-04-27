"""Array1d class definition."""

import logging
import warnings

import numpy as np
import obspy
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Cursor

from utprocess import ActiveTimeSeries, Source, Sensor1C

logger = logging.getLogger(__name__)


class Array1D():
    """A class to organize the information for a 1D (linear) array.

    Attributes
    ----------
    sensors : list of Sensor1C
        Sensors which compose the 1D array.
    source : Source
        Source for active shot gather.

    """

    @staticmethod
    def _check_array(sensors, source):
        example = sensors[0]
        positions = [example.x]

        # Ensure sensors are "similar"
        for sensor in sensors[1:]:
            if not example.is_similar(sensor):
                raise ValueError("All sensors must be similar.")
            if sensor.x in positions:
                raise ValueError("Sensors must have unique x positions.")

        # Sort sensors in terms of x from small to large
        indices = np.argsort(positions)
        sensors = [sensors[i] for i in indices]

        return (sensors, source)

    def _make_timeseries_matrix(self):
        """Pull information from each `Sensor1C` into a 2D matrix."""
        matrix = np.empty((self.nchannels, example.nsamples))
        for i, sensor in enumerate(self.sensors):
            matrix[i, :] = sensor.amp
        return matrix

    def __init__(self, sensors, source):
        """Initialize from an iterable of `Receiver`s and a `Source`.

        Parameters
        ----------
        sensors : iterable
            Iterable of initialized `Sensor1C` objects.
        source : Source
            Initialized `Source` object.

        Returns
        -------
        Array1D
            Initialized `Array1D` object.

        """
        logger.info("Howdy from Array1D!")
        sensors, source = self._check_array(sensors, source)
        self.sensors = sensors
        self.source = source
        self.absolute_minus_relative = 0
        self._regen_matrix = True
        self._regen_position = True

        if self.kres < 0:
            msg = "Invalid receiver position, kres must be greater than 0."
            raise ValueError(msg)

    @property
    def timeseriesmatrix(self):
        if self._regen_matrix:
            self._regen_matrix = True
            self._matrix = self._make_timeseries_matrix()
        return self._matrix

    @property
    def position(self):
        if self._regen_position:
            self._regen_position = False
            self._position = [sensor.x for sensor in self.sensors]
        return self._position

    @property
    def kres(self):
        return 2*np.pi / min(np.diff(self.position))

    @property
    def nchannels(self):
        return len(self.sensors)

    @property
    def spacing(self):
        min_spacing = min(np.diff(self.position))
        max_spacing = max(np.diff(self.position))
        if min_spacing == max_spacing:
            return min_spacing
        else:
            raise ValueError("spacing undefined for non-equally spaced arrays")

    def trim_record(self, start_time, end_time):
        """Trim time series from each Sensor1C.

        Parameters
        ----------
        start_time, end_time: float
            Desired start time and end time in seconds measured from the
            point the acquisition system was triggered.

        Returns
        -------
        None
            Updates internal attributes.

        """
        self._regen_matrix = True
        for sensor in enumerate(self.sensors):
            sensor.trim(start_time, end_time)

    def _norm_traces(self, scale_factor):
        norm_traces = np.empty_like(self.timeseriesmatrix)
        for k, current_trace in enumerate(self.timeseriesmatrix):
            current_trace = signal.detrend(current_trace)
            current_trace /= np.amax(current_trace)
            current_trace *= scale_factor
            current_trace += self.position[k]
            norm_traces[k, :] = current_trace
        return norm_traces

    def waterfall(self, ax=None, scale_factor=1.0, time_along='x',
                  waterfall_kwargs=None):
        """Create waterfall plot for this array setup.

        Parameters
        ----------
        ax : Axis, optional
            Axes on which to plot, default is `None` indicating a 
            `Figure` and `Axis` will be generated on-the-fly.
        scale_factor : float, optional
            Denotes the scale of the nomalized timeseries height
            (peak-to-trough), default is 1. Half the receiver spacing is
            generally a good value.
        time_along : {'x', 'y'}, optional
            Denotes on which axis the waterfall should reside, 'x' is
            the default.
        waterfall_kwargs : None, dict, optional
            Plot kwargs for plotting the normalized time histories as
            a dictionary, default is `None`.

        Returns
        -------
        Tuple
            Of the form (fig, ax) where `fig` is the figure object and
            `ax` the axes object on which the schematic is plotted.

        """
        ax_was_none = False
        if ax is None:
            ax_was_none = True
            if time_along == "y":
                size = (4, 6)
            else:
                size = (6, 4)
            fig, ax = plt.subplots(figsize=size)

        time = self.sensors[0].time
        norm_traces = self._norm_traces(scale_factor=scale_factor)

        if waterfall_kwargs is None:
            waterfall_kwargs = {}
        default_kwargs = dict("b-", linewidth=0.5)
        kwargs = {**waterfall_kwargs, **default_kwargs}
        if time_along == "y":
            for trace in norm_traces:
                ax.plot(time, trace, **kwargs)
        elif time_along == "x":
            for trace in norm_traces:
                ax.plot(trace, time, **kwargs)
        else:
            msg = f"time_along = {time_along} not recognized, use 'x' or 'y'."
            raise NotImplementedError(msg)

        time_ax = time_along
        dist_ax = "x" if time_ax == "y" else "y"

        setattr(ax, f"set_{time_ax}lim", (min(time), max(time)))
        setattr(ax, f"set_{dist_ax}lim",
                -self.position[1], self.position[1]+self.position[-1])
        setattr(ax, "grid", axis=time_ax, linestyle=":")
        setattr(ax, f"set_{time_ax}label", "Time (s)")
        setattr(ax, f"set_{dist_ax}label", "Distance (m)")

        for label in list("xy"):
            ax.tick_params(label, length=4, width=1, which='major')

        if ax_was_none:
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
        """Initialize an `Array1D` object from one or more data files.

        This classmethod creates an `Array1D` object by reading the
        header information in the provided file(s). Each file
        should contain multiple traces where each trace corresponds to a
        single receiver. Currently supported file types are SEGY
        and SU.

        Parameters
        ----------
        fnames : str or iterable
            File name or iterable of file names. If multiple files 
            are provided the traces are stacked.

        Returns
        -------
        Array1D
            Initialized `Array1d` object.

        Raises
        ------
        TypeError
            If `fnames` is not of type `str` or `iterable`.

        """
        if type(fnames) == str:
            fnames = [fnames]

        # Read file for traces
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stream = obspy.read(fnames[0])

        # Create array of receivers
        receivers = []
        minimum_x = 1E9
        for trace in stream.traces:
            sensor = Sensor1C.from_trace(trace)
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

    def __getitem__(self, index):
        return self.receivers[index]
