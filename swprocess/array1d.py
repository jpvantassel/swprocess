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
        """Shift array so that the left-most sensor is at x=0."""
        self.absolute_minus_relative = float(min(self.position))
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
        """Initialize from an iterable of `Sensor1C`s and a `Source`.

        Parameters
        ----------
        sensors : iterable of Sensor1c
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
        logger.info("Initializing Array1D")
        sensors, source = self._check_array(sensors, source)
        self.sensors = sensors
        self.source = source
        self._regen_matrix = True
        self._regen_position = True

        if normalize_positions:
            self._normalize_positions()
        else:
            self.absolute_minus_relative = 0.

    @property
    def timeseriesmatrix(self):
        """Sensor amplitudes as 2D `ndarray`."""
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
    def offsets(self):
        """Receiver offsets relative to source position as `list`."""
        positions = self.position
        offsets = []
        for position in positions:
            offsets.append(abs(self.source._x - position))
        return offsets

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

    # @property
    # def _safe_spacing(self):
    #     try:
    #         spacing = self.spacing
    #     except ValueError:
    #         logger.warning("Array1D does not have equal spacing.")
    #         spacing = self.position[1] - self.position[0]
    #     return spacing

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
        return True if self.source.x > self.position[-1] else False

    def _norm_traces(self, scale_factor):
        norm_traces = np.empty_like(self.timeseriesmatrix)
        position = self.position
        for k, current_trace in enumerate(self.timeseriesmatrix):
            current_trace = signal.detrend(current_trace)
            current_trace /= np.max(np.abs(current_trace))
            current_trace *= scale_factor
            current_trace += position[k]
            norm_traces[k, :] = current_trace
        return norm_traces

    def waterfall(self, ax=None, scale_factor=1.0, time_along="y",
                  plot_kwargs=None):
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
            Denotes on which axis time should reside, 'y' is the
            default.
        plot_kwargs : None, dict, optional
            Kwargs for `matplotlib.pyplot.plot <https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.plot.html>`_
            to control the style of each trace, default is `None`.

        Returns
        -------
        Tuple
            Of the form (fig, ax) where `fig` is the figure object and
            `ax` the axes object on which the schematic is plotted, if
            `ax=None`.

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

        time = self[0].time
        norm_traces = self._norm_traces(scale_factor=scale_factor)

        if plot_kwargs is None:
            plot_kwargs = {}
        default_kwargs = dict(color="b", linewidth=0.5)
        kwargs = {**default_kwargs, **plot_kwargs}
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

        if time_ax == "x":
            time_tuple = (min(time), max(time))
        else:
            time_tuple = (max(time), min(time))
        getattr(ax, f"set_{time_ax}lim")(time_tuple)

        spacing = self.position[1] - self.position[0]
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

    def plot(self, ax=None, sensor_kwargs=None, source_kwargs=None):
        """Plot a schematic of the `Array1D` object.

        The schematic shows the position of the receivers and source
        and lists the total number of receivers and their spacing.

        Parameters
        ----------
        ax : Axis, optional
            Axes on which to plot, default is `None` indicating a
            `Figure` and `Axis` will be generated on-the-fly.
        sensor_kwargs, source_kwargs : None, dict, optional
            Kwargs for `matplotlib.pyplot.plot <https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.plot.html>`_
            to control the plotting of the sensors and source,
            respectively. Default is `None`, indicating the predefined
            default values will be used.

        Returns
        -------
        Tuple
            Of the form (fig, ax) where `fig` is the figure object and
            `ax` the axes object on which the schematic is plotted, if
            `ax=None`.

        """
        # Create axis on the fly.
        ax_was_none = False
        if ax is None:
            ax_was_none = True
            fig, ax = plt.subplots(figsize=(6, 2))

        # Plot sensors.
        xs = [sensor.x for sensor in self.sensors]
        ys = [sensor.y for sensor in self.sensors]
        default_sensor_kwargs = dict(marker="^", color="k", linestyle="",
                                     label="Sensor")
        if sensor_kwargs is None:
            sensor_kwargs = {}
        sensor_kwargs = {**default_sensor_kwargs, **sensor_kwargs}
        ax.plot(xs, ys, **sensor_kwargs)

        # List number of sensors and their spacing.
        number_txt = f"Number of sensors: {self.nchannels}"
        try:
            spacing_txt = f"Sensor spacing is {self.spacing}m."
        except ValueError:
            spacing_txt = "Sensor spacing is not constant."
        ax.text(0.03, 0.95, f"{number_txt}\n{spacing_txt}",
                transform=ax.transAxes, va="top", ha="left")

        # Plot source.
        default_source_kwargs = dict(marker="D", color="b", linestyle="",
                                     label=f"Source")
        if source_kwargs is None:
            source_kwargs = {}
        source_kwargs = {**default_source_kwargs, **source_kwargs}
        ax.plot(self.source.x, self.source.y, **source_kwargs)

        # General figure settings.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([min(ymin, -2), max(ymax, 5)])
        ax.set_xlabel("Distance (m)")
        ax.legend()

        # Return figure and axes objects if generated on the fly.
        if ax_was_none:
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

    def manual_pick_first_arrivals(self, waterfall_kwargs=None):  # pragma: no cover
        """Allow for interactive picking of first arrivals.

        Parameters
        ----------
        waterfall_kwargs : dict, optional
            Dictionary of keyword arguments for
            meth: `<plot_waterfall>`, default is `None` indicating
            default keyword arguments.

        Returns
        -------
        Tuple
            Of the form (distance, picked_time)

        """
        if waterfall_kwargs is None:
            waterfall_kwargs = {}

        fig, ax = self.waterfall(**waterfall_kwargs)

        pairs = self._ginput_session(ax)

        if ax.get_xlabel() == "Distance (m)":
            distance, time = pairs
        else:
            time, distance = pairs

        return (distance, time)

    def interactive_mute(self, mute_location="both", window_kwargs=None,
                         waterfall_kwargs=None):
        """Interactively select source window boundary.

        Parameters
        ----------
        mute_location : {"before", "after", "both"}, optional
            Select which part of the record to mute, default is `"both"`
            indicating two lines defining the source window boundary
            will be required.
        window_kwargs : dict, optional
            Dictionary of keyword arguments defining the signal window,
            see `scipy.singal.windows.tukey <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
            for available options.
        waterfall_kwargs : dict, optional
            Dictionary of keyword arguments defining how the waterfall
            should be created, see `:meth Array1D.waterfall` for
            the available options.

        Returns
        -------
        tuple
            Of the form `(signal_start, signal_end)`.

        """
        # Create waterfall plot.
        if waterfall_kwargs is None:
            waterfall_kwargs = dict(time_along="y")
        fig, ax = self.waterfall(**waterfall_kwargs)

        # Parse x, y to distance, time
        def parse(xs, ys, time_along=waterfall_kwargs["time_along"]):
            if time_along == "y":
                return ((xs[0], ys[0]), (xs[1], ys[1]))
            else:
                return ((ys[0], xs[0]), (ys[1], xs[1]))

        # Define the start of the signal.
        if mute_location == "before" or mute_location == "both":
            xs, ys = self._ginput_session(ax, npts=2,
                                          initial_adjustment=False,
                                          ask_to_continue=False)
            plt.plot(xs, ys, color="r", linewidth=0.5)
            signal_start = parse(xs, ys)
        else:
            signal_start = None

        # Define the end of the signal.
        if mute_location == "after" or mute_location == "both":
            xs, ys = self._ginput_session(ax, npts=2,
                                          initial_adjustment=False,
                                          ask_to_continue=False)
            plt.plot(xs, ys, color="r", linewidth=0.5)
            signal_end = parse(xs, ys)
        else:
            signal_end = None

        # Perform mute.
        self.mute(signal_start=signal_start,
                  signal_end=signal_end,
                  window_kwargs=window_kwargs)

        return (signal_start, signal_end)

    def mute(self, signal_start=None, signal_end=None, window_kwargs=None):
        """Mute traces outside of a narrow signal window.

        Parameters
        ----------
        signal_start, signal_end : iterable of floats, optional
            Two points to define start and stop of the narrow signal
            window of the form
            `((pt1_dist, pt1_time), (pt2_dist, pt2_time))`, default is
            `None` .
        window_kwargs : dict, optional
            Dictionary of keyword arguments defining the signal window,
            see `scipy.singal.windows.tukey <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
            for available options.

        Returns
        -------
        None
            Modifies the object internal state.

        """
        # General variables common to intercept calculations.
        position = np.array(self.position)
        dt = self[0].dt
        delay = self[0].delay
        ndelay = abs(delay/dt)
        nsamples = self[0].nsamples

        # Define starting index for each trace.
        if signal_start is not None:
            ((x1, t1), (x2, t2)) = signal_start
            slope = (t2 - t1) / (x2 - x1)
            times = t1 + slope*(position - x1)
            start_indices = np.array((times / dt) + ndelay, dtype=int)
        else:
            start_indices = np.zeros_like(position, dtype=int)

        # Define stopping index for each trace.
        if signal_end is not None:
            ((x1, t1), (x2, t2)) = signal_end
            slope = (t2 - t1) / (x2 - x1)
            times = t1 + slope*(position - x1)
            stop_indices = np.array((times / dt) + ndelay, dtype=int)
        else:
            stop_indices = np.ones_like(position, dtype=int) * (nsamples-1)

        # Define default windowing to be Tukey w/ 20% taper (10% on each side).
        if window_kwargs is None:
            window_kwargs = dict(alpha=0.2)

        # TODO (jpv): Windowing can be pushed down into Sensor object.
        # Perform windowing.
        window = np.zeros(nsamples)
        for i, (start, stop) in enumerate(zip(start_indices, stop_indices)):
            window[start:stop] = signal.windows.tukey(stop-start,
                                                      **window_kwargs)
            self.sensors[i].amp *= window
            window *= 0

    # TODO (jpv): This can be factored out of Array1D.
    # TODO (jpv): Replace hard-coded messaged with user-defined inputs. Generator?
    @staticmethod
    def _ginput_session(ax, initial_adjustment=True, ask_to_continue=True,
                        npts=None):
        """Start ginput session using the provided axes object.

        Parameters
        ----------
        ax : Axes
            Axes on which points are to be selected.
        initial_adjustment : bool, optional
            Allow user to pan and zoom prior to the selection of the
            first point, default is `True`.
        ask_to_continue : bool, optional
            Pause the selection process after each point. This allows
            the user to pan and zoom the figure as well as select when
            to continue, default is `True`.
        npts : int, optional
            Predefine the number of points the user is allowed to
            select, the default is `None` which allows the selection of
            an infinite number of points.

        Returns
        -------
        tuple
            Of the form `(xs, ys)` where `xs` is a `list` of x
            coordinates and `ys` is a `list` of y coordinates in the
            order in which they were picked.

        """
        # Set npts to infinity if npts is None
        if npts is None:
            npts = np.inf

        # Enable cursor to make precise selection easier.
        cursor = Cursor(ax, useblit=True, color='k', linewidth=1)

        # Permit initial adjustment with blocking call to figure.
        if initial_adjustment:
            print("Adjust view, spacebar when ready.")
            while True:
                if plt.waitforbuttonpress(timeout=-1):
                    break

        # Begin selection of npts.
        npt, xs, ys = 0, [], []
        while npt < npts:
            print("Left click to add, right click to remove, enter to accept.")
            vals = plt.ginput(n=-1, timeout=0)
            x, y = vals[-1]
            ax.plot(x, y, "r", marker="+", linestyle="")
            xs.append(x)
            ys.append(y)
            npt += 1

            if ask_to_continue:
                print("Press spacebar once to contine, twice to exit)")
                while True:
                    if plt.waitforbuttonpress(timeout=-1):
                        break

            if plt.waitforbuttonpress(timeout=0.5):
                print("Exiting ... ")
                break
        print("Close figure when ready.")

        return (xs, ys)

    @classmethod
    def from_files(cls, fnames, map_x=lambda x: x, map_y=lambda y: y):
        """Initialize an `Array1D` object from one or more data files.

        This classmethod creates an `Array1D` object by reading the
        header information in the provided file(s). Each file
        should contain multiple traces where each trace corresponds to a
        single receiver. Currently supported file types are SEGY and SU.

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
            raise NotImplementedError(f"_format={_format} not recognized.")
        source = parse_source(trace.stats)
        obj = cls(sensors, source)

        # Stack additional traces, if necessary
        if len(fnames) > 0:
            for fname in fnames[1:]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stream = obspy.read(fname)
                nsource = parse_source(stream[0].stats)
                if nsource != source:
                    msg = f"fname = {fname} has an incompatable source."
                    raise ValueError(msg)
                for sensor, trace in zip(obj.sensors, stream.traces):
                    new_sensor = Sensor1C.from_trace(trace)
                    sensor.stack_append(new_sensor)

        return obj

    @classmethod
    def from_array1d(cls, array1d):
        obj = cls(array1d.sensors, array1d.source)
        obj.absolute_minus_relative = float(array1d.absolute_minus_relative)
        return obj

    def is_similar(self, other):
        """See if `other` is similar to `self`, though not strictly equal."""
        if not isinstance(other, (Array1D,)):
            return False

        for attr in ["nchannels", "source", "absolute_minus_relative"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def __eq__(self, other):
        if not self.is_similar(other):
            return False

        for my, ur in zip(self.sensors, other.sensors):
            if my != ur:
                return False

        return True

    def __getitem__(self, index):
        return self.sensors[index]
