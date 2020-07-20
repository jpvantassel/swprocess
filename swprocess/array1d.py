"""Array1d class definition."""

import logging
import warnings

import numpy as np
import obspy
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Cursor

from swprocess import ActiveTimeSeries, Source, Sensor1C

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
            if not example._is_similar(sensor, exclude=["x", "y", "z"]):
                raise ValueError("All sensors must be similar.")
            if sensor.x in positions:
                raise ValueError("Sensors must have unique x positions.")
            positions.append(sensor.x)

        # Sort sensors in terms of x from small to large
        indices = np.argsort(positions)
        sensors = [sensors[i] for i in indices]

        return (sensors, source)

    def _normalize_positions(self):
        self.absolute_minus_relative = min(self.position)
        self._regen_position = True
        for sensor in self.sensors:
            sensor._x -= self.absolute_minus_relative
        self.source._x -= self.absolute_minus_relative

    def _make_timeseries_matrix(self):
        """Pull information from each `Sensor1C` into a 2D matrix."""
        matrix = np.empty((self.nchannels, self.sensors[0].nsamples))
        for i, sensor in enumerate(self.sensors):
            matrix[i, :] = sensor.amp
        return matrix

    def __init__(self, sensors, source, normalize_positions=False):
        """Initialize from an iterable of `Receiver`s and a `Source`.

        Parameters
        ----------
        sensors : iterable
            Iterable of initialized `Sensor1C` objects.
        source : Source
            Initialized `Source` object.
        normalize_positions : bool, optional
            Normalize the relative locations of the sensors and source
            such that the smallest postion sensor is located at (0,0).

        Returns
        -------
        Array1D
            Initialized `Array1D` object.

        """
        logger.info("Howdy from Array1D!")
        sensors, source = self._check_array(sensors, source)
        self.sensors = sensors
        self.source = source
        self._regen_matrix = True
        self._regen_position = True

        if normalize_positions:
            self._normalize_positions()
        else:
            self.absolute_minus_relative = 0

    @property
    def timeseriesmatrix(self):
        """Sensors amplitudes as 2D `np.ndarray`."""
        if self._regen_matrix:
            self._regen_matrix = True
            self._matrix = self._make_timeseries_matrix()
        return self._matrix

    @property
    def position(self):
        """Relative sensor positions as `list`."""
        if self._regen_position:
            self._regen_position = False
            self._position = [sensor.x for sensor in self.sensors]
        return self._position

    @property
    def kres(self):
        """The array's resolution wavenumber."""
        return np.pi / min(np.diff(self.position))

    @property
    def nchannels(self):
        """Number of `Sensors` in the array."""
        return len(self.sensors)

    @property
    def spacing(self):
        min_spacing = min(np.diff(self.position))
        max_spacing = max(np.diff(self.position))
        if min_spacing == max_spacing:
            return min_spacing
        else:
            raise ValueError("spacing undefined for non-equally spaced arrays")

    @property
    def _source_inside(self):
        sx = self.source.x
        position = self.position
        return ((sx > position[0]) and (sx < position[-1]))

    @property
    def _safe_spacing(self):
        try:
            spacing = self.spacing
        except ValueError:
            logger.warning("Array1D does not have equal spacing.")
            spacing = self.position[1] - self.position[0]
        return spacing

    def trim(self, start_time, end_time):
        """Trim time series belonging to each Sensor1C.

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
        for sensor in self.sensors:
            sensor.trim(start_time, end_time)

    def zero_pad(self, df):
        """Append zero to sensors to achieve a desired frequency step.

        Parameters
        ----------
        df : float
            Desired linear frequency step in Hertz.

        Returns
        -------
        None
            Instead modifies `sensors`.

        """
        for sensor in self.sensors:
            sensor.zero_pad(df=df)

    @property
    def _flip_required(self):
        if self.source.x > self.position[-1]:
            return True
        else:
            return False

    def _norm_traces(self, scale_factor):
        norm_traces = np.empty_like(self.timeseriesmatrix)
        position = self.position
        for k, current_trace in enumerate(self.timeseriesmatrix):
            current_trace = signal.detrend(current_trace)
            current_trace /= current_trace[abs(current_trace) == np.max(np.abs(current_trace))]
            current_trace *= scale_factor
            current_trace += position[k]
            norm_traces[k, :] = current_trace
        return norm_traces

    def waterfall(self, ax=None, scale_factor=1.0, time_along="y",
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
            Denotes on which axis time should reside, 'x' is the
            default.
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
            if time_along == "x":
                size = (6, 4)
            elif time_along == "y":
                size = (4, 6)
            else:
                msg = f"time_along = {time_along} not recognized, use 'x' or 'y'."
                raise ValueError(msg)
            fig, ax = plt.subplots(figsize=size)

        time = self.sensors[0].time
        norm_traces = self._norm_traces(scale_factor=scale_factor)

        if waterfall_kwargs is None:
            waterfall_kwargs = {}
        default_kwargs = dict(color="b", linewidth=0.5)
        kwargs = {**default_kwargs, **waterfall_kwargs}
        if time_along == "x":
            for trace in norm_traces:
                ax.plot(time, trace, **kwargs)
                kwargs["label"] = None
        else:
            for trace in norm_traces:
                ax.plot(trace, time, **kwargs)
                kwargs["label"] = None

        time_ax = time_along
        dist_ax = "x" if time_ax == "y" else "y"

        time_tuple = (min(time), max(time)) if time_ax == "x" else (
            max(time), min(time))
        getattr(ax, f"set_{time_ax}lim")(time_tuple)

        spacing = self._safe_spacing
        getattr(ax, f"set_{dist_ax}lim")(self.position[0]-spacing,
                                         self.position[-1]+spacing)
        getattr(ax, "grid")(axis=time_ax, linestyle=":")
        getattr(ax, f"set_{time_ax}label")("Time (s)")
        getattr(ax, f"set_{dist_ax}label")("Distance (m)")

        for label in list("xy"):
            ax.tick_params(label, length=4, width=1, which='major')

        if ax_was_none:
            fig.tight_layout()
            return (fig, ax)

    def plot(self):
        """Plot a schematic of the `Array1D` object.

        The schematic shows the relative position of the receivers and
        the source and lists the total number of receivers. The figure
        and axes are returned to the user for use in further editing if
        desired.

        Returns
        -------
        Tuple
            Of the form `(fig, ax)` where `fig` is the Figure object
            and `ax` is the Axes object on which the schematic is
            plotted.

        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 2))

        for n_rec, sensor in enumerate(self.sensors):
            label = "Sensor" if n_rec == 1 else None
            ax.plot(sensor.x, sensor.y, marker="^", color="k", linestyle="",
                    label=label)

        try:
            spacing_txt = f"Receiver spacing is {self.spacing}m."
        except ValueError:
            spacing_txt = f"Receiver spacings are not equal."

        ax.text(0.03, 0.95,
                f"Number of Receivers: {self.nchannels}\n{spacing_txt}",
                transform=ax.transAxes, va="top", ha="left")

        ax.plot(self.source.x, self.source.y, marker="D", color="b",
                linestyle="", label=f"Source at {self.source.x}m")

        ax.legend()
        ax.set_ylim([-2, 5])
        ax.set_xlabel("Distance Along Array (m)")
        return (fig, ax)

    def auto_pick_first_arrivals(self, algorithm='threshold', **kwargs):
        if algorithm == "threshold":
            picks = self._pick_on_threshold(**kwargs)
        else:
            raise NotImplementedError
        return picks

    def _pick_on_threshold(self, threshold=0.05):
        """

        Parameters
        ----------
        threshold : {0.-1.}, optional
            Picking threashold as a percent.

        """
        norm_traces = self._norm_traces(scale_factor=1)
        _time = self.sensors[0].time

        times = []
        for trace in norm_traces:
            pindices = np.argwhere(abs(trace-np.mean(trace[:10])) > threshold)
            index = int(pindices[0])
            times.append(_time[index])

        return (self.position, times)

    def manual_pick_first_arrivals(self, waterfall_kwargs=None): # pragma: no cover
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

        fig, ax = self.waterfall(**waterfall_kwargs)

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
    def from_files(cls, fnames, map_x=lambda x:x, map_y=lambda y:y):
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
        map_x, map_y : function, optional
            Convert x and y coordinates using some function, default
            is not transformation. Can be useful for converting between
            coordinate systems.

        Returns
        -------
        Array1D
            Initialized `Array1d` object.

        Raises
        ------
        TypeError
            If `fnames` is not of type `str` or `iterable`.

        """
        if isinstance(fnames, (str)):
            fnames = [fnames]

        # Read traces from first file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stream = obspy.read(fnames[0])

        # Create array of sensors
        sensors = []
        for trace in stream.traces:
            sensor = Sensor1C.from_trace(trace, map_x=map_x, map_y=map_y)
            sensors.append(sensor)

        # Define source
        _format = trace.stats._format
        if _format == "SEG2":
            def parse_source(stats):
                x = map_x(float(stats.seg2.SOURCE_LOCATION))
                return Source(x=x, y=0, z=0)
        elif _format == "SU":
            def parse_source(stats):
                x = map_x(float(stats.su.trace_header["source_coordinate_x"]))
                y = map_y(float(stats.su.trace_header["source_coordinate_y"]))
                return Source(x=x, y=y, z=0)
        else:
            # Here for "belt and suspenders".s
            raise NotImplementedError(f"_format={_format} not recognized.")
        source = parse_source(trace.stats)
        obj = cls(sensors, source)

        # Stack additional traces, if necessary
        if len(fnames) > 0:
            for fname in fnames[1:]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stream = obspy.read(fname)
                if source != parse_source(stream[0].stats):
                    msg = f"fname = {fname} has incompatable source."
                    raise ValueError(msg)
                for sensor, trace in zip(obj.sensors, stream.traces):
                    new_sensor = Sensor1C.from_trace(trace)
                    sensor.stack_append(new_sensor)

        return obj

    @classmethod
    def from_array1d(cls, array1d):
        obj = cls(array1d.sensors, array1d.source)

        if array1d.absolute_minus_relative != 0:
            obj.absolute_minus_relative = float(array1d.absolute_minus_relative)
        
        return obj

    def __eq__(self, other):
        if not isinstance(other, Array1D):
            return False

        for attr in ["nchannels", "source", "absolute_minus_relative"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        for my, ur in zip(self.sensors, other.sensors):
            if my != ur:
                return False
        
        return True

    def __getitem__(self, index):
        return self.sensors[index]
