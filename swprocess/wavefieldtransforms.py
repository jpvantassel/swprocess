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

"""Wavefield transform class definitions."""

from abc import ABC, abstractmethod
import logging
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import special

from .register import WavefieldTransformRegistry

logger = logging.getLogger("swprocess.wavefieldtransforms")


class AbstractWavefieldTransform(ABC):
    """Wavefield transformation of an `Array1D`."""

    def __init__(self, frequencies, velocities, power):
        """Define AbstractWavefieldTransform."""
        self.n = 1
        self.frequencies = frequencies
        self.velocities = velocities
        self.power = power

        # Pre-define internal state attributes
        self._to_abs = 1.

        # Pre-define optional attributes
        self.snr = None
        self.snr_frequencies = None
        self.array = None

    @staticmethod
    def _create_vector(pmin, pmax, pn, pspace):
        logger.info("Howdy from _create_vector")
        samplers = {"linear": np.linspace, "lin": np.linspace,
                    "log": np.geomspace, "logarithmic": np.geomspace}
        sampler = samplers[pspace]
        return sampler(pmin, pmax, pn)

    @classmethod
    def from_array(cls, array, settings):
        # Create velocity vector.
        vels = cls._create_vector(settings["vmin"], settings["vmax"],
                                  settings["nvel"], settings["vspace"])

        # Perform transform.
        frqs, powr = cls.transform(array, vels, settings)

        # Create WavefieldTransform object.
        return cls(frqs, vels, powr)

    @classmethod
    @abstractmethod
    def transform(cls, array, velocities, settings):  # pragma: no cover
        """Abstract transform method."""

    @staticmethod
    def _frequency_keep_ids(frequencies, fmin, fmax, multiple):
        """Ids to keep between [fmin, fmax] (inclusive) by multiple."""
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
        # Return power to unnormalized state.
        self.power *= self._to_abs

        # Normalize power
        register = {"none": (np.abs,
                             lambda x: 1.),
                    "absolute-maximum": (lambda x: np.abs(x)/np.max(np.abs(x)),
                                         lambda x: np.max(np.abs(x))),
                    "frequency-maximum": (lambda x: np.abs(x)/np.max(np.abs(x), axis=0),
                                          lambda x: np.max(np.abs(x), axis=0))
                    }
        to_norm, to_abs = register[by]
        self.power, self._to_abs = (to_norm(self.power), to_abs(self.power))

    def find_peak_power(self, by="frequency-maximum", **kwargs):
        """Find maximum `WavefieldTransform` power.

        Parameters
        ----------
        by : {"frequency-maximum", "find_peaks"}, optional
            Determines how the maximum surface wave dispersion power is
            selected, default is 'frequency-maximum'.
            `frequency-maximum` as the name indicates simply returns the
            single maximum power point's velocity at each frequency.
            `find_peaks` uses the function by the same name from the
            scipy package, keyword arguments can be entered as kwargs.
        kwargs : kwargs, optional
            Keyword arguments, different for each search method.

        Returns
        -------
        ndarray
            Containing the peak velocity at each frequency.

        """
        if by == "frequency-maximum":
            return self.velocities[np.argmax(self.power, axis=0)]
        else:
            msg = f"find_peak_power by {by} not recognized, see docs for options."
            raise NotImplementedError(msg)


    def plot_snr(self, ax=None, plot_kwargs=None):
        # Only proceed if snr is not None.
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

    def plot(self, fig=None, ax=None, cax=None,
             normalization="frequency-maximum", peaks="frequency-maximum",
             nearfield=None, cmap="jet", peak_kwargs=None, colorbar_kwargs=None, rasterize=False):
        """Plot the `WavefieldTransform`'s dispersion image.

        Parameters
        ----------
        ax : Axes, optional
            Axes object on which to plot the dispersion image, default
            is `None` so an `Axes` will be created on-the-fly.
        cax : Axes, optional
            Axes object on which to plot the colorbar for the disperison
            image, default is `None` so an `Axes` will be created from
            `ax`.
        normalization : {"none", "absolute-maximum", "frequency-maximum"}, optional
            Determines how the surface wave dispersion power is
            normalized, default is 'frequency-maximum'.
        peaks : {"none", "frequency-maximum"}, optional
            Determines if the spectral peaks are shown and if so how
            they will be determined, default is 'frequency-maximum'.
        nearfield : int, optional
            Number of array center distances per wavelength following
            Yoon and Rix (2009), default is `None` so nearfield
            criteria will not be plotted. A value of 1 corresponds
            to ~15% error and 2 ~5% error.
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
        # TODO (jpv): Rewrite docstring.

        # Construct fig and ax (if necessary).
        if ax is None:
            ax_was_none = True
            fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        else:
            ax_was_none = False
            if fig is None:
                raise ValueError("Both `fig` and `ax` must be defined.")

        # Perform normalization.
        self.normalize(by=normalization)

        # Plot dispersion image.
        img = ax.contourf(self.frequencies,
                          self.velocities,
                          self.power,
                          np.linspace(0, np.max(self.power), 21),
                          cmap=mpl.colormaps[cmap])
        if rasterize:
            for pathcoll in img.collections:
                pathcoll.set_rasterized(True)

        # Plot colorbar for image.
        if cax is None:
            ax_kwargs = dict(ax=ax, pad=0.01)
        else:
            ax_kwargs = dict(cax=cax)

        default_colorbar_kwargs = dict(
            **ax_kwargs, ticks=np.round(np.linspace(0, np.max(self.power), 6), 1))
        if colorbar_kwargs is None:
            colorbar_kwargs = default_colorbar_kwargs
        else:
            colorbar_kwargs = {**default_colorbar_kwargs, **colorbar_kwargs}

        fig.colorbar(img, **colorbar_kwargs)

        # Plot peaks (if necessary).
        if peaks != "none":
            selected_peaks = self.find_peak_power(by=peaks)
            default_kwargs = dict(marker="o", markersize=1, markeredgecolor="w",
                                  markerfacecolor='none', linestyle="none")
            peak_kwargs = {} if peak_kwargs is None else peak_kwargs
            peak_kwargs = {**default_kwargs, **peak_kwargs}
            ax.plot(self.frequencies, selected_peaks, **peak_kwargs)

        # Plot Yoon and Rix (2009) (if necessary).
        if nearfield is not None:
            ylims = ax.get_ylim()
            fmin, fmax = ax.get_xlim()
            fs = np.linspace(fmin, fmax, 30)
            wv = self.array.array_center_distance / nearfield
            vs = fs*wv
            ax.plot(fs, vs, color="black", linestyle="-.",
                    label=r"$\frac{\overline{x}}{\lambda}=$"+f"{nearfield:.2f}")
            ax.set_ylim(ylims)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase Velocity (m/s)")

        # Return fig and ax (if necessary).
        if ax_was_none:
            fig.tight_layout()
            return (fig, ax)

    # def plot_slices(self, frequencies, axs=None, plot_kwargs=None):
    #     """Plot frequency-velocity slices of the `WavefieldTransform`.

    #     Parameters
    #     ----------
    #     frequencies : iterable of floats
    #         Select frequencies at which the slices are to be plotted.
    #         Note the plotted frequencies may not match these exactly
    #         depending upon the frequency discretization used during
    #         processing. To ensure the two match exactly first reprocess
    #         the data using frequency domain padding to ensure a known
    #         `df` then select only slice frequencies which are multiples
    #         of `df`.
    #     axs : ndarray of Axes, optional
    #         `ndarray` of `Axes` objects on which to plot the
    #         frequency-velocity slices, default is `None` indicating
    #         the appropriate `Axes` will be generated on-the-fly.
    #     plot_kwargs : dict, optional
    #         Keyword arguments to the plot command, default is `None`
    #         so the default settings will be used.

    #     Returns
    #     -------
    #     tuple or None
    #         `tuple` of the form `(fig, axs)` if `axs=None`, and `None`
    #         otherwise.

    #     """
    #     # Construct fig and axs (if necessary).
    #     axs_was_none = False
    #     if axs is None:
    #         axs_was_none = True
    #         npanels = len(frequencies)
    #         cols = 4
    #         rows = ceil(npanels/cols)
    #         blanks = cols*rows - npanels
    #         fig, axs = plt.subplots(nrows=rows, ncols=cols,
    #                                 figsize=(1.5*cols, 1.5*rows), dpi=150)
    #         axs[-1, -blanks:] = None

    #     # Allow user to customize the slice's appearance.
    #     plot_kwarags = {} if plot_kwargs is None else plot_kwargs
    #     default_kwargs = dict(linewidth=0.75, color="#000000")
    #     plot_kwargs = {**default_kwargs, **plot_kwarags}

    #     # Plot the slices.
    #     for ax, requested in zip(axs.flatten(), frequencies):
    #         fid = np.argmin(np.abs(self.frequencies - requested))
    #         ax.plot(self.velocities, self.power[:, fid], **plot_kwargs)
    #         ax.text(0.95, 0.95, f"@{np.round(self.frequencies[fid])}Hz",
    #                 ha="right", va="top", transform=ax.transAxes)
    #         # ax.set_xlabel("Velocity (m/s)")
    #         # ax.set_ylabel("Power")

    #     # Return fig and ax (if necessary).
    #     if axs_was_none:
    #         fig.tight_layout()
    #         return (fig, axs)


@WavefieldTransformRegistry.register('empty')
class EmptyWavefieldTransform(AbstractWavefieldTransform):

    def __init__(self, frequencies, velocities, power):
        super().__init__(frequencies, velocities, power)
        self.n = 0

    @classmethod
    def from_array(cls, array, settings):
        # Create frequency vector.
        sensor = array[0]
        frqs = np.arange(sensor.nsamples)*sensor._df
        keep_ids = cls._frequency_keep_ids(frqs, settings["fmin"],
                                           settings["fmax"], sensor.multiple)
        frequencies = frqs[keep_ids]

        # Create velocity vector.
        velocities = cls._create_vector(settings["vmin"], settings["vmax"],
                                        settings["nvel"], settings["vspace"])

        # Create empty power tensor.
        nvel, nfrq = settings["nvel"], len(keep_ids)
        power = np.zeros((nvel, nfrq), dtype=complex)

        return cls(frequencies, velocities, power)

    def stack(self, other):
        try:
            self.power = (self.power*self.n + other.power*other.n)
            self.power /= (self.n+other.n)
        except AttributeError as e:
            msg = "Can only append objects if descendant of `AbstractWavefieldTransform`."
            raise AttributeError(msg) from e
        self.n += other.n

    @classmethod
    def transform(cls, array, velocities, settings):  # pragma: no cover
        """Empty transform method."""


# class FK(AbstractWavefieldTransform):

#     def __init__(self, frequencies, velocities, power):
#         """Perform Frequency-Wavenumber (fk) transform.

#         The FK approach utilizes a 2D Fourier Transform to transform
#         data from the time-space domain to the frequency-wavenumber
#         domain. The FK approach was adapted by Gabriels et al. (1987)
#         for linear arrays from the FK approach developed by Nolet and
#         Panza (1976) for 2D arrays.

#         Parameters
#         ----------
#         array : Array1d
#             One-dimensional array object.
#         nwave : int
#             Number of wavenumbers to consider.
#         fmin, fmax : float
#             Minimum and maximum frequency of interest in the
#             transformation.

#         Returns
#         -------
#         Tuple
#             Of the form `(frqs, domain, ks, pnorm, kpeaks)`.

#         """
#         # Frequency vector
#         sensor = array.sensors[0]
#         frqs = np.arange(sensor.nsamples) * sensor._df

#         # Perform 2D FFT
#         if array._flip_required:
#             tseries = np.flipud(array.timeseriesmatrix)
#         else:
#             tseries = array.timeseriesmatrix
#         fk = np.fft.fft2(tseries, s=(nwave, sensor.nsamples))
#         fk = np.abs(fk[-2::-1, 0:len(frqs)])

#         # Trim frequencies and downsample (if required by zero padding)
#         fmin_ids = np.argmin(np.abs(frqs-fmin))
#         fmax_ids = np.argmin(np.abs(frqs-fmax))
#         freq_ids = range(fmin_ids, (fmax_ids+1), sensor.multiple)
#         frqs = frqs[freq_ids]
#         fk = fk[:, freq_ids]

#         # Wavenumber vector
#         kres = array.kres
#         dk = 2*kres / nwave
#         ks = np.arange(dk, 2*kres, dk)

#         # Normalize power and find peaks
#         pnorm = np.empty_like(fk)
#         kpeaks = np.empty_like(frqs)
#         for k, _fk in enumerate(fk.T):
#             normed_fk = np.abs(_fk/np.max(_fk))
#             pnorm[:, k] = normed_fk
#             kpeaks[k] = ks[np.argmax(normed_fk)]

#         return (frqs, "wavenumber", ks, pnorm, kpeaks)

#     def plot(self, *args, **kwargs):
#         raise NotImplementedError


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
        tmatrix, position = array._flipped_tseries_and_offsets()

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
        frequencies = np.arange(nsamples) * sensor._df

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
        tmatrix, offsets = array._flipped_tseries_and_offsets()

        # u(x,t) -> FFT -> U(x,f).
        fft = np.fft.fft(tmatrix)

        # Frequency vector.
        sensor = array.sensors[0]
        frqs = np.arange(sensor.nsamples)*sensor._df

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
        tmatrix, offsets = array._flipped_tseries_and_offsets()

        # Reshape to 3D array, for calculating sscm.
        sensor = array.sensors[0]
        tmatrix = tmatrix.reshape(array.nchannels, sensor.nsamples, 1)

        # Frequency vector
        frqs = np.arange(sensor.nsamples)*sensor._df

        # Trim and downsample frequencies.
        keep_ids = cls._frequency_keep_ids(frqs, settings["fmin"],
                                           settings["fmax"], sensor.multiple)
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
        # TODO (jpv): Rewrite docstring.
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


@WavefieldTransformRegistry.register('fk')
class FK(FDBF):

    @classmethod
    def from_array(cls, array, settings):
        settings["fdbf-specific"]["weighting"] = "none"
        settings["fdbf-specific"]["steering"] = "plane"

        return super().from_array(array, settings)
