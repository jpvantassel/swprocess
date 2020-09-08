"""Wavefield transform class definitions."""

from abc import ABC, abstractclassmethod
import json
import logging
from math import ceil
import warnings

from numpy import linspace, geomspace
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

from .peaks import Peaks
from .register import WavefieldTransformRegistry

logger = logging.getLogger(__name__)


class AbstractWavefieldTransform(ABC):
    """Wavefield transformation of an `Array1D`.

    Attributes
    ----------

    """

    def __init__(self, frequencies, velocities, power):
        self.frequencies = frequencies
        self.velocities = velocities
        self.power = power

        # Pre-define optional attributes
        self.snr = None
        self.snr_frequencies = None
        self.array = None

    @staticmethod
    def _create_velocities(settings):
        samplers = {"linear": linspace, "lin": linspace,
                    "log": geomspace, "logarithmic": geomspace}
        sampler = samplers[settings["vspace"]]
        return sampler(settings["vmin"], settings["vmax"], settings["nvel"])

    @staticmethod
    def _flip_if_required(array):
        """Flip array and offsets if required by source location."""
        if array._flip_required:
            offsets = array.offsets[::-1]
            tmatrix = np.flipud(array.timeseriesmatrix)
        else:
            offsets = array.offsets
            tmatrix = array.timeseriesmatrix
        offsets = np.array(offsets)
        return (tmatrix, offsets)

    @classmethod
    def from_array(cls, array, settings):
        # Create velocity vector.
        vels = cls._create_velocities(settings)

        # Perform transform.
        frqs, powr = cls.transform(array, vels, settings)

        # Create WavefieldTransform object.
        return cls(frqs, vels, powr)

    @classmethod
    @abstractclassmethod
    def transform(cls, array, velocities, settings):
        pass

    @staticmethod
    def _frequency_keep_ids(frequencies, fmin, fmax, multiple):
        fmin_ids = np.argmin(np.abs(frequencies-fmin))
        fmax_ids = np.argmin(np.abs(frequencies-fmax))
        keep_ids = range(fmin_ids, (fmax_ids+1), multiple)
        return keep_ids

    def normalize(self, by="frequency-maximum"):
        """Normalize `WavefieldTransform` power.

        Parameters
        ----------
        by : {"none", "absolute-maximum", "frequency-maximum"}, optional
            Determines how the surface wave dispersion power is
            normalized, default is 'frequency-maximum'.

        Returns
        -------
        None
            Update the internal state of power.

        """
        register = {"none": lambda x: np.abs(x),
                    "absolute-maximum": lambda x: np.abs(x)/np.max(np.abs(x)),
                    "frequency-maximum": lambda x: np.abs(x)/np.max(np.abs(x), axis=0),
                    }
        self.power = register[by](self.power)

    def find_peak_power(self, by="frequency-maximum"):
        """Pick maximum `WavefieldTransform` power.

        Parameters
        ----------
        by : {"frequency-maximum"}, optional
            Determines how the maximum surface wave dispersion power is
            selected, default is 'frequency-maximum'.

        Returns
        -------
        # Peaks
        #     An instantiated `Peaks` object.
        """
        # TODO(jpv): Decide if this should return a peaks object or update state.
        self.peaks = self.velocities[np.argmax(self.power, axis=0)]
        # return Peaks(self.frequencies, self.peaks,)

    def write_peaks_to_file(self, fname, identifier, append=False, ftype="json"):
        pass
        # """Write peak disperison values to file.

        # Parameters
        # ----------
        # fname : str
        #     Name of the output file, may be a relative or the full path.
        # identifier :  str
        #     A unique identifier for the peaks. The source offset is
        #     typically sufficient.
        # append : bool, optional
        #     Flag to denote whether `fname` should be appended to or
        #     overwritten, default is `False` indicating the file will
        #     be overwritten.
        # ftype : {'json'}, optional
        #     Denotes the desired filetype.
        #     TODO (jpv): Add also a csv option.

        # Returns
        # -------
        # None
        #     Instead writes/appends dispersion peaks to file `fname`.

        # """
        # if self.domain == "wavenumber":
        #     v_peak = 2*np.pi / self.peaks*self.frqs
        # elif self.domain == "velocity":
        #     v_peak = self.peaks
        # else:
        #     raise NotImplementedError()

        # if ftype != "json":
        #     raise ValueError()

        # if fname.endswith(".json"):
        #     fname = fname[:-5]

        # data = {}
        # if append:
        #     try:
        #         f = open(f"{fname}.json", "r")
        #     except FileNotFoundError:
        #         pass
        #     else:
        #         data = json.load(f)
        #         f.close()

        # with open(f"{fname}.json", "w") as fp:
        #     if identifier in data:
        #         raise KeyError(f"identifier {identifier} is repeated.")
        #     else:
        #         # keep_ids = np.where(v_peak < self.settings["vmax"])
        #         # ftrim = self.frqs[keep_ids].tolist()
        #         # vtrim = v_peak[keep_ids].tolist()
        #         # data.update({identifier: {"frequency": ftrim,
        #         #                           "velocity": vtrim}})

        #         data.update({identifier: {"frequency": self.frqs.tolist(),
        #                                   "velocity": v_peak.tolist()}})

        #     json.dump(data, fp)

    # def plot_spectra(self, stype="fv", plot_peak=True, plot_limit=None):  # pragma: no cover

    def plot_waterfall(self, *args, **kwargs):
        # Only proceed if array is not None:
        if self.array is None:
            warnings.warn("array is not defined, therefore cannot be plotted.")
            return

        return self.array.waterfall(*args, **kwargs)

    def plot_array(self, *args, **kwargs):
        # Only proceed if array is not None:
        if self.array is None:
            warnings.warn("array is not defined, therefore cannot be plotted.")
            return

        return self.array.plot(*args, **kwargs)

    def plot_snr(self, ax=None, plot_kwargs=None):

        # Only proceed if snr is not None:
        if self.snr is None:
            warnings.warn("snr is not defined, therefore cannot be plotted.")
            return

        # Construct fig and ax (if necessary).
        ax_was_none = False
        if ax is None:
            ax_was_none = True
            fig, ax = plt.subplots(figsize=(4, 3), dpi=150)

        # Allow for user customization of plot.
        if plot_kwargs is None:
            plot_kwargs = {}

        # Plot signal-to-noise ratio.
        ax.plot(self.snr_frequencies, self.snr, **plot_kwargs)

        # Labels
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Signal-to-Noise Ratio")

        # Return fig and ax (if necessary).
        if ax_was_none:
            fig.tight_layout()
            return (fig, ax)

    def plot(self, ax=None, normalization="frequency-maximum",
             peaks="frequency-maximum", cmap="jet", peak_kwargs=None):
        """Plot the `WavefieldTransform`'s dispersion image.

        Parameters
        ----------
        ax : Axes, optional
            Axes object on which to plot the dispersion image, default
            is `None` so an `Axes` will be created on-the-fly.
        normalization : {"none", "absolute-maximum", "frequency-maximum"}, optional
            Determines how the surface wave dispersion power is
            normalized, default is 'frequency-maximum'.
        peaks : {"none", "frequency-maximum"}, optional
            Determines if the spectral peaks are shown and if so how
            they will be determined, default is 'frequency-maximum'.
        peak_kwargs : dict, optional
            Keyword arguments to control the appearance of the spectral
            peaks, default is `None` so the default settings will be
            used.

        Returns
        -------
        tuple or None
            `tuple` of the form `(fig, ax)` if `ax=None`, `None`
            otherwise.

        """
        # Construct fig and ax (if necessary).
        ax_was_none = False
        if ax is None:
            ax_was_none = True
            fig, ax = plt.subplots(figsize=(4, 3), dpi=150)

        # Perform normalization.
        self.normalize(by=normalization)

        # Select peaks.
        self.find_peak_power(by=peaks)

        # Plot dispersion image.
        contour = ax.contourf(self.frequencies,
                              self.velocities,
                              self.power,
                              np.linspace(0, np.max(self.power), 20),
                              cmap=plt.cm.get_cmap(cmap))
        fig.colorbar(contour, ax=ax, ticks=np.round(
            np.linspace(0, np.max(self.power), 11), 1))

        # Plot peaks (if necessary).
        if peaks != ["none"]:
            default_kwargs = dict(marker="o", markersize=1, markeredgecolor="w",
                                  markerfacecolor='none', linestyle="none")
            peak_kwargs = {} if peak_kwargs is None else peak_kwargs
            peak_kwargs = {**default_kwargs, **peak_kwargs}
            ax.plot(self.frequencies, self.peaks, **peak_kwargs)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase Velocity (m/s)")

        # Return fig and ax (if necessary).
        if ax_was_none:
            fig.tight_layout()
            return (fig, ax)

    def plot_slices(self, frequencies, axs=None, plot_kwargs=None):
        """Plot frequency-velocity slices of the `WavefieldTransform`.

        Parameters
        ----------
        frequencies : iterable of floats
            Select frequencies at which the slices are to be plotted.
            Note the plotted frequencies may not match these exactly
            depending upon the frequency discretization used during
            processing. To ensure the two match exactly first reprocess
            the data using frequency domain padding to ensure a known
            `df` then select only slice frequencies which are multiples
            of `df`.
        axs : ndarray of Axes, optional
            `ndarray` of `Axes` objects on which to plot the
            frequency-velocity slices, default is `None` indicating
            the appropriate `Axes` will be generated on-the-fly.
        plot_kwargs : dict, optional
            Keyword arguments to the plot command, default is `None`
            so the default settings will be used.


        Returns
        -------
        tuple or None
            `tuple` of the form `(fig, axs)` if `axs=None`, and `None`
            otherwise.

        """
        # Construct fig and axs (if necessary).
        axs_was_none = False
        if axs is None:
            axs_was_none = True
            npanels = len(frequencies)
            cols = 4
            rows = ceil(npanels/cols)
            blanks = cols*rows - npanels
            fig, axs = plt.subplots(nrows=rows, ncols=cols,
                                    figsize=(1.5*cols, 1.5*rows), dpi=150)
            axs[-1, -blanks:] = None

        # Allow user to customize the slice's appearance.
        plot_kwarags = {} if plot_kwargs is None else plot_kwargs
        default_kwargs = dict(linewidth=0.75, color="#000000")
        plot_kwargs = {**default_kwargs, **plot_kwarags}

        # Plot the slices.
        for ax, requested in zip(axs.flatten(), frequencies):
            fid = np.argmin(np.abs(self.frequencies - requested))
            ax.plot(self.velocities, self.power[:, fid], **plot_kwargs)
            ax.text(0.95, 0.95, f"@{np.round(self.frequencies[fid])}Hz",
                    ha="right", va="top", transform=ax.transAxes)
            # ax.set_xlabel("Velocity (m/s)")
            # ax.set_ylabel("Power")

        # Return fig and ax (if necessary).
        if axs_was_none:
            fig.tight_layout()
            return (fig, axs)


@WavefieldTransformRegistry.register('empty')
class EmptyWavefieldTransform(AbstractWavefieldTransform):

    def __init__(self, frequencies, velocities, power):
        self.n = 0
        super().__init__(frequencies, velocities, power)

    @classmethod
    def from_array(cls, array, settings):
        sensor = array[0]
        frqs = np.arange(sensor.nsamples)*sensor.df
        keep_ids = cls._frequency_keep_ids(frqs,
                                           settings["fmin"],
                                           settings["fmax"],
                                           sensor.multiple)
        nvel, nfrq = settings["nvel"], len(keep_ids)
        frequencies = frqs[keep_ids]
        velocities = cls._create_velocities(settings)
        power = np.empty((nvel, nfrq), dtype=complex)
        return cls(frequencies, velocities, power)

    def stack(self, other):
        try:
            self.power = (self.power*self.n + other.power*1)/(self.n+1)
        except AttributeError as e:
            msg = "Can only append objects if decendent of AbstractWavefieldTransform"
            raise AttributeError(msg) from e

        self.n += 1

    @classmethod
    def transform(cls, array, velocities, settings):
        pass


class FK(AbstractWavefieldTransform):

    def __init__(self, frequencies, velocities, power):
        """Perform Frequency-Wavenumber (fk) transform.

        The FK approach utilizes a 2D Fourier Transform to transform
        data from the time-space domain to the frequency-wavenumber
        domain. The FK approach was adapted by Gabriels et al. (1987)
        for linear arrays from the FK approach developed by Nolet and
        Panza (1976) for 2D arrays.

        Parameters
        ----------
        array : Array1d
            One-dimensional array object.
        nwave : int
            Number of wavenumbers to consider.
        fmin, fmax : float
            Minimum and maximum frequency of interest in the
            transformation.

        Returns
        -------
        Tuple
            Of the form `(frqs, domain, ks, pnorm, kpeaks)`.

        """
        # Frequency vector
        sensor = array.sensors[0]
        frqs = np.arange(sensor.nsamples) * sensor.df

        # Perform 2D FFT
        if array._flip_required:
            tseries = np.flipud(array.timeseriesmatrix)
        else:
            tseries = array.timeseriesmatrix
        fk = np.fft.fft2(tseries, s=(nwave, sensor.nsamples))
        fk = np.abs(fk[-2::-1, 0:len(frqs)])

        # Trim frequencies and downsample (if required by zero padding)
        fmin_ids = np.argmin(np.abs(frqs-fmin))
        fmax_ids = np.argmin(np.abs(frqs-fmax))
        freq_ids = range(fmin_ids, (fmax_ids+1), sensor.multiple)
        frqs = frqs[freq_ids]
        fk = fk[:, freq_ids]

        # Wavenumber vector
        kres = array.kres
        dk = 2*kres / nwave
        ks = np.arange(dk, 2*kres, dk)

        # Normalize power and find peaks
        pnorm = np.empty_like(fk)
        kpeaks = np.empty_like(frqs)
        for k, _fk in enumerate(fk.T):
            normed_fk = np.abs(_fk/np.max(_fk))
            pnorm[:, k] = normed_fk
            kpeaks[k] = ks[np.argmax(normed_fk)]

        return (frqs, "wavenumber", ks, pnorm, kpeaks)

    def plot(self, *args, **kwargs):
        raise NotImplementedError


@WavefieldTransformRegistry.register('slantstack')
class SlantStack(AbstractWavefieldTransform):

    @classmethod
    def slant_stack(cls, array, velocities):
        """Perform a slant-stack on the given wavefield data.

        Parameters
        ----------
        array : Array1d
            One-dimensional array object.
        velocities : ndarray
            One-dimensional array of trial velocities.

        Returns
        -------
        tuple
            Of the form `(tau, slant_stack)` where `tau` is an ndarray
            of the attempted intercept times and `slant_stack` are the
            slant-stacked waveforms.

        """
        tmatrix, position = cls._flip_if_required(array)

        position = np.array(position)
        position -= np.min(position)
        nchannels = array.nchannels
        diff = position[1:] - position[:-1]
        diff = diff.reshape((len(diff), 1))
        dt = array.sensors[0].dt
        npts = tmatrix.shape[1]
        ntaus = npts - int(np.max(position)*np.max(1/velocities)/dt) - 1
        slant_stack = np.empty((len(velocities), ntaus))
        rows = np.tile(np.arange(nchannels).reshape(nchannels, 1), (1, ntaus))
        cols = np.tile(np.arange(ntaus).reshape(1, ntaus), (nchannels, 1))

        pre_float_indices = position.reshape(nchannels, 1)/dt
        previous_lower_indices = np.zeros((nchannels, 1), dtype=int)
        for i, velocity in enumerate(velocities):
            float_indices = pre_float_indices/velocity
            lower_indices = np.array(float_indices, dtype=int)
            delta = float_indices - lower_indices
            cols += lower_indices - previous_lower_indices
            amplitudes = tmatrix[rows, cols] * \
                (1-delta) + tmatrix[rows, cols+1]*delta
            integral = 0.5*diff*(amplitudes[1:, :] + amplitudes[:-1, :])
            summation = np.sum(integral, axis=0)
            slant_stack[i, :] = summation

            previous_lower_indices[:] = lower_indices
        # taus = np.arange(ntaus)*dt
        # return (taus, slant_stack)
        return slant_stack

    @classmethod
    def transform(cls, array, velocities, settings):
        """Perform the Slant-Stack transform.

        Parameters
        ----------
        array : Array1D
            Instance of `Array1D`.
        velocities : ndarray
            Vector of trial velocities.
        settings : dict
            `dict` with processing settings.

        Returns
        -------
        tuple
            Of the form `(frequencies, power)`.

        """
        # Perform slant-stack
        nsamples = array[0].nsamples
        slant_stack = cls.slant_stack(array, velocities)

        # Frequency vector
        sensor = array[0]
        df = sensor.df
        frequencies = np.arange(nsamples) * df

        # Fourier Transform of the slant-stack
        power = np.fft.fft(slant_stack, n=nsamples)

        # Trim and downsample frequencies.
        keep_ids = cls._frequency_keep_ids(frequencies,
                                           settings["fmin"],
                                           settings["fmax"],
                                           sensor.multiple)
        return (frequencies[keep_ids], power[:, keep_ids])


@WavefieldTransformRegistry.register('phaseshift')
class PhaseShift(AbstractWavefieldTransform):

    @classmethod
    def transform(cls, array, velocities, settings):
        """Perform the Phase-Shift Transform.

        Parameters
        ----------
        array : Array1D
            Instance of `Array1D`.
        velocities : ndarray
            Vector of trial velocities.
        settings : dict
            `dict` with processing settings.

        Returns
        -------
        tuple
            Of the form `(frequencies, power)`.

        """
        # Flip reference frame (if required).
        tmatrix, offsets = cls._flip_if_required(array)

        # u(x,t) -> FFT -> U(x,f).
        fft = np.fft.fft(tmatrix)

        # Frequency vector.
        sensor = array.sensors[0]
        frqs = np.arange(sensor.nsamples) * sensor.df

        # Trim and downsample frequencies.
        keep_ids = cls._frequency_keep_ids(frqs,
                                           settings["fmin"],
                                           settings["fmax"],
                                           sensor.multiple)
        frequencies = frqs[keep_ids]

        # Integrate across the array offsets.
        power = np.empty((len(velocities), len(frequencies)))
        dx = offsets[1:] - offsets[:-1]
        for row, vel in enumerate(velocities):
            exponent = 1j * 2*np.pi/vel * offsets
            for col, (f_index, frq) in enumerate(zip(keep_ids, frequencies)):
                shift = np.exp(exponent*frq)
                inner = shift*fft[:, f_index]/np.abs(fft[:, f_index])
                integral = np.abs(np.sum(0.5*dx*(inner[:-1] + inner[1:])))
                power[row, col] = integral

        return (frequencies, power)


@WavefieldTransformRegistry.register('fdbf')
class FDBF(AbstractWavefieldTransform):

    @classmethod
    def transform(cls, array, velocities, settings):
        """Perform Frequency-Domain Beamforming.

        Parameters
        ----------
        array : Array1D
            Instance of `Array1D`.
        velocities : ndarray
            Vector of trial velocities.
        settings : dict
            `dict` with processing settings.

        Returns
        -------
        tuple
            Of the form `(frequencies, power)`.

        """
        # Flip reference frame (if required).
        tmatrix, offsets = cls._flip_if_required(array)

        # Reshape to 3D array, for calculating sscm.
        sensor = array.sensors[0]
        tmatrix = tmatrix.reshape(array.nchannels, sensor.nsamples, 1)

        # Frequency vector
        frqs = np.arange(sensor.nsamples)*sensor.df

        # Trim and downsample frequencies.
        keep_ids = cls._frequency_keep_ids(frqs,
                                           settings["fmin"],
                                           settings["fmax"],
                                           sensor.multiple)
        frequencies = frqs[keep_ids]

        # Calculate the spatiospectral correlation matrix
        fdbf_specific = settings.get("fdbf-specific", {})
        weighting = fdbf_specific.get("weighting")
        sscm = cls._spatiospectral_correlation_matrix(tmatrix,
                                                      frq_ids=keep_ids,
                                                      weighting=weighting)

        # Weighting
        if weighting == "sqrt":
            offsets_n = offsets.reshape(array.nchannels, 1)
            offsets_h = np.transpose(np.conjugate(offsets_n))
            w = np.dot(offsets_n, offsets_h)
        else:
            w = np.ones((array.nchannels, array.nchannels))

        # Steering
        steering = fdbf_specific.get("steering")
        if steering == "cylindrical":
            def create_steering(kx):
                return np.exp(-1j * np.angle(special.j0(kx) + 1j*special.y0(kx)))
        else:
            def create_steering(kx):
                return np.exp(-1j * kx)

        steer = np.empty((array.nchannels, 1), dtype=complex)
        power = np.empty((len(velocities), len(frequencies)), dtype=complex)
        kx = np.empty_like(offsets)
        for i, f in enumerate(frequencies):
            weighted_sscm = sscm[:, :, i]*w
            for j, v in enumerate(velocities):
                kx[:] = 2*np.pi*f/v * offsets[:]
                steer[:, 0] = create_steering(kx)[:]
                _power = np.dot(np.dot(np.transpose(np.conjugate(steer)),
                                       weighted_sscm), steer)
                power[j, i] = _power

        return (frequencies, power)

    @staticmethod
    def _spatiospectral_correlation_matrix(tmatrix, frq_ids=None, weighting=None):
        """Compute the spatiospectral correlation matrix.

        Parameters
        ----------
        tmatrix : ndarray
            Three-dimensional matrix of shape
            `(samples_per_block, nblocks, nchannels)`. 
        fmin, fmax : float, optional
            Minimum and maximum frequency of interest.

        Returns
        -------
        ndarray
            Of size `(nchannels, nchannels, nfrqs)` containing the
            spatiospectral correlation matrix.

        """
        nchannels, samples_per_block, nblocks = tmatrix.shape

        # Perform FFT
        transform = np.fft.fft(tmatrix, axis=1)

        # Trim FFT
        if frq_ids is not None:
            transform = transform[:, frq_ids, :]

        # Define weighting matrix
        if weighting == "invamp":
            _, nfrqs, _ = transform.shape
            weighting = 1/np.abs(np.mean(transform, axis=-1))

            for i in range(nfrqs):
                w = weighting[:, i]
                for b in range(nblocks):
                    transform[:, i, b] *= w

        # Calculate spatiospectral correlation matrix
        nchannels, nfrqs, nblocks = transform.shape
        spatiospectral = np.empty((nchannels, nchannels, nfrqs), dtype=complex)
        scm = np.zeros((nchannels, nchannels), dtype=complex)
        tslice = np.zeros((nchannels, 1), dtype=complex)
        tslice_h = np.zeros((1, nchannels), dtype=complex)
        for i in range(nfrqs):
            scm[:, :] = 0
            for j in range(nblocks):
                tslice[:, 0] = transform[:, i, j]
                tslice_h[0, :] = np.conjugate(tslice)[:, 0]
                scm += np.dot(tslice, tslice_h)
            scm /= nblocks
            spatiospectral[:, :, i] = scm[:]

        return spatiospectral
