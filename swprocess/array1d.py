# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Array1D class definition."""

import logging
import warnings

import numpy as np
import obspy
from obspy.io.segy.segy import SEGYTraceHeader
import matplotlib.pyplot as plt
from scipy import signal

from swprocess.source import SourceWithSignal

from .interact import ginput_session
from swprocess import Source, Sensor1C

logger = logging.getLogger("swprocess.array1d")


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

    def __init__(self, sensors, source):
        """Initialize from an iterable of `Sensor1C`s and a `Source`.

        Parameters
        ----------
        sensors : iterable of Sensor1c
            Iterable of initialized `Sensor1C` objects.
        source : Source
            Initialized `Source` object.

        Returns
        -------
        Array1D
            Initialized `Array1D` object.

        """
        logger.info("Initializing Array1D")
        self.sensors, self.source = self._check_array(sensors, source)

    def timeseriesmatrix(self, detrend=False, normalize="none"):
        """Sensor amplitudes as 2D `ndarray`.

        Parameters
        ----------
        detrend : bool, optional
            Boolean to control whether a linear detrending operation is
            performed, default is `False` so no detrending is performed.
        normalize : {"none", "each", "all"}, optional
            Enable different normalizations to be performed. `"each"`
            normalizes each traces by its maximum. `"all"` normalizes
            all traces by the same maximum. Default is `"none"` so no
            normalization is performed.

        Returns
        -------
        ndarray
            Of shape `(nchannels, nsamples)` where each row is the
            amplitude of a given sensor.

        """
        matrix = np.empty((self.nchannels, self[0].nsamples))
        for i, sensor in enumerate(self.sensors):
            _amp = sensor.amplitude
            amp = signal.detrend(_amp) if detrend else _amp
            amp = amp/np.max(np.abs(amp)) if normalize == "each" else amp
            matrix[i, :] = amp

        if normalize == "all":
            matrix /= np.max(np.abs(matrix))

        return matrix

    def position(self, normalize=False):
        """Array's sensor positions as `list`.

        Parameters
        ----------
        normalize : bool, optional
            Determines whether the array positions are shifted such
            that the first sensor is located at x=0.

        """
        if normalize:
            position_0 = self.sensors[0].x
            return [sensor.x - position_0 for sensor in self.sensors]
        else:
            return [sensor.x for sensor in self.sensors]

    @property
    def offsets(self):
        """Receiver offsets relative to source position as `list`."""
        positions = self.position(normalize=False)
        offsets = []
        for position in positions:
            offsets.append(abs(self.source._x - position))
        return offsets

    @property
    def kres(self):
        """The array's resolution wavenumber."""
        return np.pi / min(np.diff(self.position()))

    @property
    def nchannels(self):
        """Number of `Sensors` in the array."""
        return len(self.sensors)

    @property
    def array_center_distance(self):
        return np.mean(self.offsets)

    @property
    def spacing(self):
        position = self.position()
        min_spacing = min(np.diff(position))
        max_spacing = max(np.diff(position))
        if (max_spacing - min_spacing) < 1E-2:
            return round(min_spacing, 2)
        else:
            raise ValueError("spacing undefined for non-equally spaced arrays")

    @property
    def _source_inside(self):
        sx = self.source.x
        position = self.position(normalize=False)
        return ((sx > position[0]) and (sx < position[-1]))

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
        for sensor in self.sensors:
            sensor.trim(start_time, end_time)

    def trim_offsets(self, min_offset, max_offset):
        """Remove sensors outside of the offsets specified.

        Parameters
        ----------
        min_offset, max_offset : float
            Specify the minimum and maximum allowable offset in meters.

        Return
        ------
        None
            Updates internal attributes.

        """
        sensors = []
        for offset, sensor in zip(self.offsets, self.sensors):
            if (offset > min_offset) and (offset < max_offset):
                sensors.append(sensor)

            if (offset > max_offset):
                break

        if len(sensors) == 0:
            msg = "Removing all sensors at offsets between "
            msg += f"{min_offset} and {max_offset}, results in no sensors."
            raise ValueError(msg)

        self.sensors = sensors

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
        return True if self.source.x > self.position(normalize=False)[-1] else False

    def waterfall(self, ax=None, time_ax="y", amplitude_detrend=True,
                  amplitude_normalization="each", amplitude_scale=None,
                  position_normalization=False, plot_kwargs=None):
        """Create waterfall plot for this array setup.

        Parameters
        ----------
        ax : Axes, optional
            Axes on which to plot, default is `None` indicating a
            `Figure` and `Axes` will be generated on-the-fly.
        time_ax : {'x', 'y'}, optional
            Denotes the time axis, 'y' is the default.
        amplitude_detrend : bool, optional
            Boolean to control whether a linear detrending operation is
            performed, default is `False` so no detrending is performed.
        amplitude_normalization : {"none", "each", "all"}, optional
            Enable different normalizations including:
            `"each"` which normalizes each traces by its maximum,
            `"all"` which normalizes all traces by the same maximum, and
            `"none"` which perform no normalization, default is
            `"each"`.
        amplitude_scale : float, optional
            Factor by which each trace is multiplied, default is `None`
            which uses a factor equal to half the average receiver
            receiver spacing.
        position_normalization : bool, optional
            Determines whether the array positions are shifted such
            that the first sensor is located at x=0.
        plot_kwargs : None, dict, optional
            Kwargs for `matplotlib.pyplot.plot <https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.plot.html>`_
            to control the style of each trace, default is `None`.

        Returns
        -------
        Tuple
            Of the form `(fig, ax)` where `fig` is the figure object
            and `ax` the axes object on which the schematic is plotted,
            if `ax=None`.

        """
        # Create axes (if required)
        ax_was_none = False
        if ax is None:
            ax_was_none = True
            if time_ax == "x":
                size = (6, 4)
            elif time_ax == "y":
                size = (4, 6)
            else:
                msg = f"time_ax = {time_ax} not recognized, use 'x' or 'y'."
                raise ValueError(msg)
            fig, ax = plt.subplots(figsize=size)

        # Prepare waterfall data.
        time = self[0].time
        traces = self.timeseriesmatrix(detrend=amplitude_detrend,
                                       normalize=amplitude_normalization)
        positions = self.position(normalize=position_normalization)
        if amplitude_scale is None:
            amplitude_scale = np.mean(np.diff(positions))/2
        for i, position in enumerate(positions):
            traces[i, :] *= amplitude_scale
            traces[i, :] += position

        # Allow custom plotting kwargs.
        if plot_kwargs is None:
            plot_kwargs = {}
        default_kwargs = dict(color="b", linewidth=0.5)
        kwargs = {**default_kwargs, **plot_kwargs}

        # Plot waveforms.
        if time_ax == "x":
            for trace in traces:
                ax.plot(time, trace, **kwargs)
                kwargs["label"] = None
        else:
            for trace in traces:
                ax.plot(trace, time, **kwargs)
                kwargs["label"] = None

        dist_ax = "x" if time_ax == "y" else "y"
        if time_ax == "x":
            time_tuple = (min(time), max(time))
        else:
            time_tuple = (max(time), min(time))
        getattr(ax, f"set_{time_ax}lim")(time_tuple)

        spacing = positions[1] - positions[0]
        getattr(ax, f"set_{dist_ax}lim")(positions[0]-spacing,
                                         positions[-1]+spacing)
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
        default_sensor_kwargs = dict(marker="^", color="k",
                                     linestyle="", label="Sensor")

        if sensor_kwargs is None:
            sensor_kwargs = {}
        sensor_kwargs = {**default_sensor_kwargs, **sensor_kwargs}
        ax.plot(xs, ys, **sensor_kwargs)

        # List number of sensors and their spacing.
        number_txt = f"Number of sensors: {self.nchannels}"
        try:
            spacing_txt = f"Sensor spacing: {self.spacing} m."
        except ValueError:
            spacing_txt = "Sensor spacing is not constant."
        source_txt = f"Source @ {round(self.source.x, 2)} m."
        ax.text(0.03, 0.95, f"{number_txt}\n{spacing_txt}\n{source_txt}",
                transform=ax.transAxes, va="top", ha="left")

        # Plot source.
        default_source_kwargs = dict(marker="D", color="b",
                                     linestyle="", label="Source")
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

    def auto_pick_first_arrivals(self, algorithm="threshold",
                                 **algorithm_kwargs):
        if algorithm == "threshold":
            picks = self._pick_on_threshold(**algorithm_kwargs)
        else:
            raise NotImplementedError
        return picks

    def _pick_on_threshold(self, threshold=0.05):
        """

        Parameters
        ----------
        threshold : {0.-1.}, optional
            Picking threshold as a percent.

        """
        traces = self.timeseriesmatrix(detrend=True, normalize="each")
        time = self.sensors[0].time

        times = []
        for trace in traces:
            pindices = np.argwhere(abs(trace-np.mean(trace[:10])) > threshold)
            index = int(pindices.flatten()[0])
            times.append(time[index])

        return (self.position(), times)

    def manual_pick_first_arrivals(self, waterfall_kwargs=None):
        """Allow for interactive picking of first arrivals.

        Parameters
        ----------
        waterfall_kwargs : dict, optional
            Dictionary of keyword arguments for
            meth: `<Array1D.waterfall>`, default is `None` indicating
            default keyword arguments.

        Returns
        -------
        Tuple
            Of the form (distance, picked_time)

        """
        if waterfall_kwargs is None:
            waterfall_kwargs = {}

        _, ax = self.waterfall(**waterfall_kwargs)

        pairs = self._ginput_session(ax, npts=np.inf)

        if ax.get_xlabel() == "Distance (m)":
            distance, time = pairs
        else:
            time, distance = pairs

        return (distance, time)

    @staticmethod
    def _ginput_session(*args, **kwargs):
        return ginput_session(*args, **kwargs)

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
        default_waterfall_kwargs = dict(time_ax="y")
        if waterfall_kwargs is None:
            waterfall_kwargs = {}
        waterfall_kwargs = {**default_waterfall_kwargs, **waterfall_kwargs}
        _, ax = self.waterfall(**waterfall_kwargs)

        # Parse x, y to distance, time
        def parse(xs, ys, time_ax=waterfall_kwargs["time_ax"]):
            if time_ax == "y":
                return ((xs[0], ys[0]), (xs[1], ys[1]))
            else:
                return ((ys[0], xs[0]), (ys[1], xs[1]))

        # Define the start of the signal.
        if (mute_location == "before") or (mute_location == "both"):
            xs, ys = self._ginput_session(ax, npts=2,
                                          initial_adjustment=False,
                                          ask_to_continue=False)
            ax.plot(xs, ys, color="r", linewidth=0.5)
            signal_start = parse(xs, ys)
        else:
            signal_start = None

        # Define the end of the signal.
        if (mute_location == "after") or (mute_location == "both"):
            xs, ys = self._ginput_session(ax, npts=2,
                                          initial_adjustment=False,
                                          ask_to_continue=False)
            ax.plot(xs, ys, color="r", linewidth=0.5)
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
        position = np.array(self.position())
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
            self.sensors[i]._amp *= window
            window *= 0

    def _flipped_tseries_and_offsets(self):
        """Timeseriesmatrix and offsets, flipped from near to far."""
        if self._flip_required:
            offsets = self.offsets[::-1]
            tmatrix = np.flipud(self.timeseriesmatrix())
        else:
            offsets = self.offsets
            tmatrix = self.timeseriesmatrix()
        offsets = np.array(offsets)
        return (tmatrix, offsets)

    @classmethod
    def from_files(cls, fnames, map_x=lambda x: x, map_y=lambda y: y, obspy_read_kwargs=None):
        """Initialize an `Array1D` object from one or more data files.

        This classmethod creates an `Array1D` object by reading the
        header information in the provided file(s). Each file
        should contain multiple traces where each trace corresponds to a
        single receiver. Currently supported file types are SEG2 and SU.

        Parameters
        ----------
        fnames : str or iterable
            File name or iterable of file names. If multiple files
            are provided the traces are stacked.
        map_x, map_y : function, optional
            Convert x and y coordinates using some function, default
            is not transformation. Can be useful for converting between
            coordinate systems.
        obspy_read_kwargs : dict, optional
            Keyword arguments to be passed to the obspy.read().

        Returns
        -------
        Array1D
            Initialized `Array1d` object.

        Raises
        ------
        TypeError
            If `fnames` is not of type `str` or `iterable`.

        """
        if isinstance(fnames, str):
            fnames = [fnames]
        
        try:
            iter(fnames)
        except TypeError:
            fnames = [str(fnames)]

        if obspy_read_kwargs is None:
            obspy_read_kwargs = {}

        # Read traces from first file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stream = obspy.read(str(fnames[0]), **obspy_read_kwargs)

        # Create array of sensors
        sensors = []
        for trace in stream.traces:
            sensor = Sensor1C.from_trace(trace, map_x=map_x, map_y=map_y)
            sensors.append(sensor)

        # Define source
        _format = trace.stats._format
        if _format == "SEG2":
            def parse_source(stats):
                x = float(stats.seg2.SOURCE_LOCATION)
                return Source(x=map_x(x), y=0, z=0)

        elif _format == "SU":
            def parse_source(stats):
                if stats.su.trace_header["coordinate_units"] == 0:
                    msg = "Coordinate units is unset, assuming length in meters and not degrees minutes seconds."
                    warnings.warn(msg)
                elif stats.su.trace_header["coordinate_units"] != 1:
                    msg = "Coordinate units must be in units of length, not degrees minutes seconds."
                    raise ValueError(msg)

                scaleco = int(stats.su.trace_header["scalar_to_be_applied_to_all_coordinates"])
                if scaleco == 0:
                    msg = "Resetting scale to be applied to all coordinates from zero to one."
                    warnings.warn(msg)
                    scaleco = 1

                int_x = int(stats.su.trace_header["source_coordinate_x"])
                x = int_x / abs(scaleco) if scaleco < 0 else int_x * scaleco
                x = round(x, -np.sign(scaleco) * int(np.log10(abs(scaleco))))

                int_y = int(stats.su.trace_header["source_coordinate_y"])
                y = int_y / abs(scaleco) if scaleco < 0 else int_y * scaleco
                y = round(y, -np.sign(scaleco) * int(np.log10(abs(scaleco))))

                return Source(x=map_x(x), y=map_y(y), z=0)

        elif _format == "SEGY":
            def parse_source(stats):
                if stats.segy.trace_header["coordinate_units"] == 0:
                    msg = "Coordinate units is unset, assuming length in meters and not degrees minutes seconds."
                    warnings.warn(msg)
                elif stats.segy.trace_header["coordinate_units"] != 1:
                    msg = "Coordinate units must be in units of length, not degrees minutes seconds."
                    raise ValueError(msg)

                scaleco = int(
                    stats.segy.trace_header["scalar_to_be_applied_to_all_coordinates"])

                int_x = int(stats.segy.trace_header["source_coordinate_x"])
                x = int_x / abs(scaleco) if scaleco < 0 else int_x * scaleco
                x = round(x, -np.sign(scaleco) * int(np.log10(abs(scaleco))))

                int_y = int(stats.segy.trace_header["source_coordinate_y"])
                y = int_y / abs(scaleco) if scaleco < 0 else int_y * scaleco
                y = round(y, -np.sign(scaleco) * int(np.log10(abs(scaleco))))

                return Source(x=map_x(x), y=map_y(y), z=0)

        source = parse_source(trace.stats)
        obj = cls(sensors, source)

        # Stack additional traces, if necessary
        if len(fnames) > 0:
            for fname in fnames[1:]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stream = obspy.read(str(fname))
                nsource = parse_source(stream[0].stats)
                if nsource != source:
                    msg = f"fname = {fname} has an incompatible source."
                    raise ValueError(msg)
                for sensor, trace in zip(obj.sensors, stream.traces):
                    new_sensor = Sensor1C.from_trace(trace)
                    sensor.stack_append(new_sensor)

        return obj

    @classmethod
    def from_array1d(cls, array1d):
        """Create a deep copy of an existing `Array1D` object."""
        sensors = []
        for _sensor in array1d.sensors:
            sensors.append(Sensor1C.from_sensor1c(_sensor))
        source = Source.from_source(array1d.source)
        obj = cls(sensors, source)
        return obj

    # su
    # https://pubs.usgs.gov/of/2001/of01-326/HTML/FILEFORM.HTM
    # https://wiki.seismic-unix.org/sudoc:su_data_format
    # http://web.mit.edu/cwpsu_v44r1/sumanual_600dpi_letter.pdf

    # segy
    # https://sioseis.ucsd.edu/segy.header.html
    # https://library.seg.org/pb-assets/technical-standards/seg_y_rev2_0-mar2017-1686080998003.pdf
    def to_file(self, fname, ftype="su"):
        if ftype != "su":
            raise ValueError(f"ftype = {ftype} not recognized.")

        stream = obspy.Stream()

        def rint(x): return int(round(x))

        for sensor in self.sensors:
            trace = obspy.Trace(np.array(sensor.amplitude, dtype=np.float32))
            trace.stats.delta = sensor.dt
            trace.stats.starttime = obspy.UTCDateTime(2020, 12, 18, 10, 0, 0)

            if not hasattr(trace.stats, 'su'):
                trace.stats.su = {}
            trace.stats.su.trace_header = SEGYTraceHeader()
            trace.stats.su.trace_header.scalar_to_be_applied_to_all_coordinates = -1000
            trace.stats.su.trace_header.source_coordinate_x = rint(
                self.source._x*1000)
            trace.stats.su.trace_header.source_coordinate_y = rint(
                self.source._y*1000)
            trace.stats.su.trace_header.number_of_horizontally_stacked_traces_yielding_this_trace = rint(
                sensor.nstacks-1)
            trace.stats.su.trace_header.delay_recording_time = rint(
                sensor.delay*1000)
            trace.stats.su.trace_header.group_coordinate_x = rint(
                sensor.x*1000)
            trace.stats.su.trace_header.group_coordinate_y = rint(
                sensor.y*1000)
            trace.stats.su.trace_header.coordinate_units = 1

            stream.append(trace)

        stream.write(filename=fname, format="SU")

    def is_similar(self, other):
        """Check if `other` is similar to `self`."""
        if not isinstance(other, (Array1D,)):
            return False

        for attr in ["nchannels", "source"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def __eq__(self, other):
        """Define how `other` can be equal to `self`."""
        if not self.is_similar(other):
            return False

        for my, ur in zip(self.sensors, other.sensors):
            if my != ur:
                return False

        return True

    def __getitem__(self, index):
        """Select sensor by index."""
        return self.sensors[index]


class Array1DwSource(Array1D):

    def __init__(self, sensors, source):
        super().__init__(sensors, source)

    @classmethod
    def from_files(cls, fnames_rec, fnames_src, src_channel,
                   map_x=lambda x: x, map_y=lambda y: y):
        _cls = Array1D.from_files(fnames=fnames_rec, map_x=map_x, map_y=map_y)

        # TODO (jpv): Badly assume fnames_src is a properly formatted list.
        trace = obspy.read(str(fnames_src[0]))[src_channel]
        dt = trace.meta.delta
        amp = trace.data
        for fname in fnames_src[1:]:
            trace = obspy.read(str(fname))[src_channel]
            amp += trace.data
        amp /= len(fnames_src)

        source = SourceWithSignal(_cls.source.x,
                                  _cls.source.y,
                                  _cls.source.z,
                                  amp,
                                  dt)

        return cls(_cls.sensors, source)

    def xcorrelate(self, vmin=None, vmax=None):

        # TODO (jpv): Check dt for source and signal
        # TODO (jpv): Size of source and receiver signal

        if vmax is None:
            vmax = 3000

        if vmin is None:
            vmin = 50

        min_offset, max_offset = min(self.offsets), max(self.offsets)

        tmin = min_offset/vmax
        # TODO (jpv): Revisit minimum time add fudge factor for now.
        idx_dx_tmin = int((tmin-0.5)/self.source.dt)

        tmax = max_offset/vmin
        idx_dx_tmax = int(tmax/self.source.dt)

        # start correlation at/near zero lag position.
        src_amp = self.source.amplitude
        idx_zero = self.source.nsamples
        idx_min, idx_max = idx_zero+idx_dx_tmin, idx_zero+idx_dx_tmax

        sensors = []
        for sensor in self.sensors:
            corr = signal.correlate(sensor.amplitude, src_amp)
            corr = corr[idx_min:idx_max]
            sensor = Sensor1C(corr, dt=sensor.dt,
                              x=sensor.x, y=sensor.y, z=sensor.z,
                              nstacks=sensor.nstacks, delay=0)
            sensors.append(sensor)

        self.sensors = sensors
