"""Wavefield transform class definitions."""

import json
import logging
from abc import ABC, abstractclassmethod

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
        fig.colorbar(contour, ax=ax, ticks=np.round(np.linspace(0, np.max(self.power), 11), 1))

        # Plot peaks (if necessary).
        if peaks != ["none"]:
            default_kwargs = dict(marker="o", markersize=5, markeredgecolor="w",
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


        # if self.domain == "wavenumber":
        #     fgrid, kgrid = np.meshgrid(self.frqs, self.vals)
        #     vgrid = 2*np.pi*fgrid / kgrid
        #     wgrid = 2*np.pi/kgrid
        #     kpeak = self.peaks
        #     wpeak = 2*np.pi / kpeak
        #     vpeak = wpeak*self.frqs
        # elif self.domain == "velocity":
        #     fgrid, vgrid = np.meshgrid(self.frqs, self.vals)
        #     wgrid = vgrid / fgrid
        #     wgrid = 2*np.pi / wgrid
        #     vpeak = self.peaks
        #     kpeak = 2*np.pi*self.frqs / vpeak
        #     wpeak = 2*np.pi / kpeak
        # else:
        #     raise NotImplementedError()

        # if stype == "fk":
        #     xgrid = fgrid
        #     ygrid = kgrid
        #     xpeak = self.frqs
        #     ypeak = kpeak
        #     if plot_limit == None:
        #         plot_limit = [0, np.max(self.frqs), 0, 2*self.kres]
        #     xscale = "linear"
        #     yscale = "linear"
        #     xLabText = "Frequency (Hz)"
        #     yLabText = "Wavenumber (rad/m)"
        # elif stype == "fw":
        #     xgrid = fgrid
        #     ygrid = wgrid
        #     xpeak = self.frqs
        #     ypeak = wpeak
        #     if plot_limit == None:
        #         plot_limit = [0, np.max(self.frqs), 1, 200]
        #     xscale = "linear"
        #     yscale = "log"
        #     xLabText = "Frequency (Hz)"
        #     yLabText = "Wavelength (m)"
        # elif stype == "fv":
        #     xgrid = fgrid
        #     ygrid = vgrid
        #     xpeak = self.frqs
        #     ypeak = vpeak
        #     if plot_limit == None:
        #         plot_limit = [0, np.max(self.frqs),
        #                       self.settings["vmin"], self.settings["vmax"]]
        #     xscale = "linear"
        #     yscale = "linear"
        #     xLabText = "Frequency (Hz)"
        #     yLabText = "Velocity (m/s)"
        # elif stype == "fp":
        #     xgrid = fgrid
        #     ygrid = 1.0 / vgrid
        #     xpeak = self.frqs
        #     ypeak = 1.0 / vpeak
        #     if plot_limit == None:
        #         plot_limit = [0, np.max(self.frqs), 1.0/1000, 1.0/100]
        #     xscale = "linear"
        #     yscale = "linear"
        #     xLabText = "Frequency (Hz)"
        #     yLabText = "Slowness (s/m)"
        # elif stype == "wv":
        #     xgrid = wgrid
        #     ygrid = vgrid
        #     xpeak = wpeak
        #     ypeak = vpeak
        #     if plot_limit == None:
        #         plot_limit = [1, 200, 0, 1000]
        #     xscale = "log"
        #     yscale = "linear"
        #     xLabText = "Wavelength (m)"
        #     yLabText = "Velocity (m/s)"
        # else:
        #     msg = f"`stype`= {stype} not recognized, use 'fk', 'fw', 'fv', 'fp', or 'wv'."
        #     raise ValueError(msg)

        # zgrid = self.pnorm
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        # contour = ax.contourf(xgrid,
        #                       ygrid,
        #                       zgrid,
        #                       np.linspace(0, np.max(zgrid), 20),
        #                       cmap=plt.cm.get_cmap("jet"))
        # fig.colorbar(contour, ax=ax, ticks=np.arange(0, 1.1, 0.1))

        # # TODO (jpv): Looking into using imshow for speedup!
        # # plt.imshow(max_z,
        # #           extent=plot_limit,
        # #           origin='lower',
        # #           cmap="jet")

        # if plot_peak:
        #     ax.plot(xpeak,
        #             ypeak,
        #             marker="o",
        #             markersize=5,
        #             markeredgecolor="w",
        #             markerfacecolor='none',
        #             linestyle="none")

        # # TODO (jpv): Look into making a plotting function to set these defaults
        # ax.set_xlim(plot_limit[:2])
        # ax.set_ylim(plot_limit[2:])
        # ax.set_xlabel(xLabText, fontsize=12, fontname="arial")
        # ax.set_ylabel(yLabText, fontsize=12, fontname="arial")
        # ax.set_xscale(xscale)
        # ax.set_yscale(yscale)
        # return fig, ax

    # # TODO (jpv) Pythonize this
    def plotSlices(self, plotType="fv", freqPlotValues=np.arange(6, 22, 1), xlims=[]):
        pass
        # # Determine appropriate number of panels and their arrangement
        # n_slices = len(freqPlotValues)
        # xFigDim = int(math.ceil(math.sqrt(n_slices)))
        # if (math.ceil(math.sqrt(n_slices))*math.floor(math.sqrt(n_slices)) < n_slices):
        #     yFigDim = int(math.ceil(math.sqrt(n_slices)))
        # else:
        #     yFigDim = int(math.floor(math.sqrt(n_slices)))

        # # Create an array containing the row and column for each panel
        # panel = 0
        # panel_ids = np.zeros((n_slices, 2))
        # for r in range(yFigDim):
        #     for c in range(xFigDim):
        #         if (panel+1) <= n_slices:
        #             panel_ids[panel, 0] = r+1
        #             panel_ids[panel, 1] = c+1
        #             panel += 1

        # fig = plt.figure(figsize=(mwdth, mhght))

        # # Loop through freqPlotValues
        # for k in range(n_slices-1, -1, -1):

        #     # Find frequency closest to freqPlotValues(k)
        #     c_id = np.argmin(np.absolute(self.freq-freqPlotValues[k]))
        #     cfreq = self.freq[c_id]

        #     # Plotting parameters
        #     if str.lower(self.val_type) == "wavenumber":
        #         k_vals = self.trial_vals
        #         k_peak = self.peak_vals
        #         v_vals = 2*np.pi*cfreq / k_vals
        #         v_peak = 2*np.pi*cfreq / k_peak
        #     elif str.lower(self.val_type) == "velocity":
        #         v_vals = self.trial_vals
        #         v_peak = self.peak_vals
        #         k_vals = 2*np.pi*cfreq / v_vals
        #         k_peak = 2*np.pi*cfreq / v_peak
        #     else:
        #         raise ValueError(
        #             "Invalid value type, should be \"wavenumber\" or \"velocity\"")

        #     # Compute maximum power
        #     maxY = np.nanmax(np.abs(self.pnorm[:, c_id]))

        #     # Determine x-axis and corresponding limits based on chosen graph type
        #     if str.lower(plotType) == "fk":
        #         x = k_vals
        #         xp = k_peak[c_id]
        #         xLabText = "Wavenumber (rad/m)"
        #         if not xlims:
        #             xlims = (0, self.kres)
        #         xscale = "linear"
        #         text_xloc = 0.66*(xlims[1] - xlims[0]) + xlims[0]
        #     elif str.lower(plotType) == "fw":
        #         x = 2*np.pi / k_vals
        #         xp = 2*np.pi / k_peak[c_id]
        #         xLabText = 'Wavelength (m)'
        #         if not xlims:
        #             xlims = (1, 200)
        #         xscale = "log"
        #         text_xloc = math.pow(
        #             10, (0.66*(math.log10(xlims[1]) - math.log10(xlims[0])) + math.log10(xlims[0])))
        #     elif str.lower(plotType) == "fv":
        #         x = v_vals
        #         xp = v_peak[c_id]
        #         xLabText = "Velocity (m/s)"
        #         if not xlims:
        #             xlims = (0, 1000)
        #         xscale = "linear"
        #         text_xloc = 0.66*(xlims[1] - xlims[0]) + xlims[0]
        #     elif str.lower(plotType) == "fp":
        #         x = 1.0 / v_vals
        #         xp = 1.0 / v_peak[c_id]
        #         xLabText = "Slowness (s/m)"
        #         if (k+1) == n_slices:
        #             minX = 1.0 / np.max(v_vals)
        #         if not xlims:
        #             xlims = (minX, 1.0/100)
        #         xscale = "linear"
        #         text_xloc = 0.33*(xlims[1] - xlims[0]) + xlims[0]
        #     else:
        #         raise ValueError(
        #             "Invalid plot type, should be \"fk\", \"fw\", \"fv\" or \"fp\"")

        #     # Plot power at current frequency
        #     ax = fig.add_subplot(yFigDim, xFigDim, k+1)
        #     ax.set_xlim(xlims)
        #     ax.set_ylim((0, maxY))
        #     ax.set_xscale(xscale)
        #     ax.plot(x, np.abs(self.pnorm[:, c_id]))
        #     ax.plot(xp, np.max(
        #         np.abs(self.pnorm[:, c_id])), marker="*", color="r", markersize=10)
        #     ax.set_xticklabels(ax.get_xticks(), fontsize=9, fontname='arial')
        #     ax.set_yticklabels(ax.get_yticks(), fontsize=9, fontname='arial')
        #     ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        #     ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        #     prfreq = '%.2f' % cfreq
        #     plt.text(text_xloc, 0.75*maxY, prfreq +
        #             " Hz", fontsize=9, fontname="arial")
        #     if panel_ids[k, 0] == yFigDim:
        #         ax.set_xlabel(xLabText, fontsize=9, fontname="arial")
        #     if panel_ids[k, 1] == 1:
        #         ax.set_ylabel("Normalized Amplitude",
        #                     fontsize=9, fontname="arial")


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

# @WavefieldTransformRegistry.register('fk')


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
