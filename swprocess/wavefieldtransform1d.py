"""WavefieldTransform1D class definition."""

import json
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

logger = logging.getLogger(__name__)


class WavefieldTransform1D():
    """Wavefield transformation of a `Array1D`.

    Attributes
    ----------

    """

    def __init__(self, array, settings):
        """Initialize from `Array1D` and settings file.

        Parameters
        ----------
        array : Array1D
            Instantiated `Array1D` object.
        settings : str
            JSON settings file detailing the settings for the wavefield
            transformation.

        Returns
        -------
        WavefieldTransform1D
            Instantiated `WavefieldTransform1D`.

        """
        logger.info("Howdy!")

        if array._source_inside:
            raise ValueError("Source must be located outside of the array.")

        with open(settings, "r") as f:
            logger.info("loading settings ... ")
            self.settings = json.load(f)

        if self.settings["trim"]:
            logger.info("trimming ... ")
            array.trim(start_time=self.settings["start_time"],
                       end_time=self.settings["end_time"])

        if self.settings["zero_pad"]:
            logger.info("padding ... ")
            array.zero_pad(df=self.settings["df"])

        self.kres = array.kres
        self._perform_transform(array=array)

    def _perform_transform(self, array):
        if self.settings["method"] == "fk":
            results = self._fk_transform(array=array,
                                         nwave=self.settings["fk-specific"]["nwavenumbers"],
                                         fmin=self.settings["fmin"],
                                         fmax=self.settings["fmax"],
                                         )
        elif self.settings["method"] == "phase-shift":
            results = self._phase_shift_transform(array=array,
                                                  fmin=self.settings["fmin"],
                                                  fmax=self.settings["fmax"],
                                                  vmin=self.settings["vmin"],
                                                  vmax=self.settings["vmax"],
                                                  nvel=self.settings["phaseshift-specific"]["nvelocities"]
                                                  )
        elif self.settings["method"] == "slant-stack":
            vels = np.linspace(self.settings["vmin"], self.settings["vmax"],
                               self.settings["slantstack-specific"]["nvelocities"])
            results = self._slant_stack_transform(array=array,
                                                  fmin=self.settings["fmin"],
                                                  fmax=self.settings["fmax"],
                                                  velocities=vels
                                                  )
        elif self.settings["method"] == "fdbf":
            vels = np.linspace(self.settings["vmin"], self.settings["vmax"],
                               self.settings["fdbf-specific"]["nvelocities"])
            results = self._frequency_domain_beamformer(array=array,
                                                        fmin=self.settings["fmin"],
                                                        fmax=self.settings["fmax"],
                                                        velocities=vels
                                                        )
        else:
            raise NotImplementedError

        self.frqs, self.domain, self.vals, self.pnorm, self.peaks = results

    @staticmethod
    def _fk_transform(array, nwave, fmin, fmax):
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

    @staticmethod
    def _phase_shift_transform(array, fmin, fmax, vmin, vmax, nvel):
        """Perform the Phase-Shift transform.

        Parameters
        ----------
        array : Array1d
            One-dimensional array object.
        fmin, fmax : float
            Minimum and maximum frequency of interest in the
            transformation.
        vmin, vmax : float
            Minimum and maximum velocity of interest in the
            transformation.
        nvel : int
            Number of trial velocities to attempt between vmin and vmax.

        Returns
        -------
        Tuple
            Of the form `(frqs, domain, vels, pnorm, vpeaks)`.

        """
        # Frequency vector
        sensor = array.sensors[0]
        frqs = np.arange(sensor.nsamples) * sensor.df

        # u(x,t) -> FFT -> U(x,f)
        if array._flip_required:
            offsets = array.offsets[::-1]
            tmatrix = np.flipud(array.timeseriesmatrix)
        else:
            offsets = array.offsets
            tmatrix = array.timeseriesmatrix
        offsets = np.array(offsets)
        trans = np.fft.fft(tmatrix)

        # Trim frequencies and downsample (if required by zero padding)
        fmin_ids = np.argmin(np.abs(frqs-fmin))
        fmax_ids = np.argmin(np.abs(frqs-fmax))
        freq_ids = range(fmin_ids, (fmax_ids+1), sensor.multiple)
        frqs = frqs[freq_ids]
        trans = trans[:, freq_ids]

        # Integrate across the array offsets
        power = np.empty((len(frqs), nvel))
        vels = np.linspace(vmin, vmax, nvel)
        dx = offsets[1:] - offsets[:-1]
        for f_index, frq in enumerate(frqs):
            for v_index, vel in enumerate(vels):
                shift = np.exp(1j * 2*np.pi*frq/vel * offsets)
                inner = shift*trans[:, f_index]/np.abs(trans[:, f_index])
                power[f_index, v_index] = np.abs(
                    np.sum(0.5*dx*(inner[:-1] + inner[1:])))

        # Normalize power and find peaks
        pnorm = np.empty_like(power)
        vpeaks = np.empty_like(frqs)
        pnorm = power/np.max(power)
        for k, _power in enumerate(pnorm):
            # normed_power = np.abs(_power/np.max(_power))
            # pnorm[k, :] = normed_power
            vpeaks[k] = vels[np.argmax(_power)]

        return (frqs, "velocity", vels, pnorm.T, vpeaks)

    @staticmethod
    def _slant_stack(array, velocities):
        """Perform a slant-stack on the given wavefield data.

        Parameters
        ----------
        array : Array1d
            One-dimensional array object.

        Returns
        -------
        tuple
            Of the form `(tau, slant_stack)` where `tau` is an ndarray
            of the attempted intercept times and `slant_stack` are the
            slant-stacked waveforms.

        """
        if array._flip_required:
            tmatrix = np.flipud(array.timeseriesmatrix)
        else:
            tmatrix = array.timeseriesmatrix

        position = np.array(array.position)
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
            amplitudes = WavefieldTransform1D.calc_amp(
                tmatrix, rows, cols, delta)
            integral = WavefieldTransform1D.integral(diff, amplitudes)
            summation = WavefieldTransform1D.summer(integral)
            slant_stack[i, :] = summation

            previous_lower_indices[:] = lower_indices
        taus = np.arange(ntaus)*dt
        return (taus, slant_stack)

    @staticmethod
    def summer(integral):
        return np.sum(integral, axis=0)

    @staticmethod
    def calc_amp(tmatrix, rows, cols, delta):
        return tmatrix[rows, cols]*(1-delta) + tmatrix[rows, cols+1]*delta

    @staticmethod
    def integral(diff, amplitudes):
        return 0.5*diff*(amplitudes[1:, :] + amplitudes[:-1, :])

    @staticmethod
    def _slant_stack_transform(array, fmin, fmax, velocities):
        """Perform the Slant-Stack transform.

        Parameters
        ----------
        array : Array1d
            One-dimensional array object.


        Returns
        -------

        """
        _, tau_p = WavefieldTransform1D._slant_stack(array, velocities)

        # Frequency vector
        sensor = array.sensors[0]
        ntaus = tau_p.shape[1]
        df = 1/(ntaus*sensor.dt)
        frqs = np.arange(ntaus) * df

        # TODO (jpv): Adjust zero padding for the slant-stack. Need to
        # be padding in the tau-p domain rather than in the x-t domain.

        # Fourier Transform of Slant Stack
        fp = np.fft.fft(tau_p)

        # Trim frequencies and downsample (if required by zero padding)
        fmin_ids = np.argmin(np.abs(frqs-fmin))
        fmax_ids = np.argmin(np.abs(frqs-fmax))
        freq_ids = range(fmin_ids, (fmax_ids+1), sensor.multiple)
        frqs = frqs[freq_ids]
        fp = fp[:, freq_ids]

        # Normalize power and find peaks
        pnorm = np.empty(fp.shape)
        vpeaks = np.empty_like(frqs)
        # fp = np.abs(fp/np.max(fp))
        abs_fp = np.abs(fp)
        for k, _fp in enumerate(abs_fp.T):
            normed_fp = _fp/np.max(_fp)
            pnorm[:, k] = normed_fp
            vpeaks[k] = velocities[np.argmax(normed_fp)]

        return (frqs, "velocity", velocities, pnorm, vpeaks)

    @staticmethod
    def _spatiospectral_correlation_matrix(tmatrix, dt, fmin, fmax, multiple):
        """Compute the spatiospectral correlation matrix.

        Parameters
        ----------
        tmatrix : ndarray
            Three dimensional matrix of shape `(samples_per_block,
            nblocks, nchannels)`. 
        fmin, fmax : float, optional
            Minimum and maximum frequency of interest.

        Returns
        -------
        ndarray
            Of size `(nchannels, nchannels, nfrqs)` containing the
            spatiospectral correlation matrix.

        """
        nchannels, samples_per_block, nblocks = tmatrix.shape
        # print(f"tmatrix.shape = {tmatrix.shape}")

        # Perform FFT
        transform = np.fft.fft(tmatrix, axis=1)

        # print(transform[0, :10, 0])
        # plt.figure()
        # plt.plot(tmatrix[0, :, 0])
        # print(np.fft.fft(tmatrix[0, :, 0])[:10])
        # print(f"transform.shape = {transform.shape}")

        # Frequency vector
        frqs = np.arange(samples_per_block) * 1/(samples_per_block*dt)
        fmin_ids = np.argmin(np.abs(frqs-fmin))
        fmax_ids = np.argmin(np.abs(frqs-fmax))
        freq_ids = range(fmin_ids, (fmax_ids+1), multiple)
        frqs = frqs[freq_ids]
        transform = transform[:, freq_ids, :]

        # print(f"transform.shape = {transform.shape}")

        spatiospectral = np.empty((nchannels, nchannels, len(frqs)), dtype=complex)
        for i in range(len(frqs)):
            scm = np.zeros((nchannels, nchannels), dtype=complex)
            for block in range(nblocks):
                tslice = transform[:, i, block].reshape(nchannels, 1)
                _scm = np.dot(tslice, np.transpose(np.conjugate(tslice)))
                scm += _scm
            scm /= nblocks
            spatiospectral[:, :, i] = scm[:]

        return (frqs, spatiospectral)

# p130
# p334 - Spatiospectral Correlaton Matrix

    @staticmethod
    def _frequency_domain_beamformer(array, fmin, fmax, velocities, weight=None):
        if array._flip_required:
            offsets = array.offsets[::-1]
            tmatrix = np.flipud(array.timeseriesmatrix)
        else:
            offsets = array.offsets
            tmatrix = array.timeseriesmatrix

        # plt.figure()
        # plt.plot(tmatrix[0, :])
        # plt.plot(tmatrix[5, :])

        sensor = array.sensors[0]
        # print(f"sensor.dt = {sensor.dt}")
        # print(f"sensor.nsamples = {sensor.nsamples}")
        # print(f"sensor.df = {sensor.df}")
        # print(f"tmatrix.shape = {tmatrix.shape}")

        # plt.plot(tmatrix[0, :])
        tmatrix = tmatrix.reshape(array.nchannels, sensor.nsamples, 1)
        # print(f"tmatrix.shape = {tmatrix.shape}")
        # plt.plot(tmatrix[0, :, 0], linestyle="--")    

        frqs, spatiospectral = WavefieldTransform1D._spatiospectral_correlation_matrix(tmatrix, sensor.dt, fmin, fmax, sensor.multiple)

        # Weighting
        offsets = np.array(offsets).reshape(array.nchannels, 1)
        offsets_h = np.transpose(np.conjugate(offsets))
        w = np.dot(offsets, offsets_h)
        # w = np.ones((array.nchannels, array.nchannels) )

        # steering = np.empty((1, array.nchannels), dtype=complex)
        power = np.empty((len(velocities), len(frqs)), dtype=complex)
        for i, f in enumerate(frqs):
            weighted_spatiospectral = spatiospectral[:, :, i]*w
            for j, v in enumerate(velocities):
                kx = 2*np.pi*f/v * offsets
                steering = np.exp(-1j*np.angle(special.j0(kx) + 1j*special.y0(kx)))
                # print(f"steering.shape = {steering.shape}")
                _power = np.dot(np.dot(np.transpose(np.conjugate(steering)), weighted_spatiospectral), steering)
                power[j, i] = _power
                # print(f"_power.shape = {_power.shape}")
                # raise ValueError

        # Normalize power and find peaks
        pnorm = np.empty(power.shape)
        vpeaks = np.empty_like(frqs)
        for k, _fp in enumerate(power.T):
            normed_p = np.abs(_fp)/np.max(np.abs(_fp))
            pnorm[:, k] = normed_p
            vpeaks[k] = velocities[np.argmax(normed_p)]

        return (frqs, "velocity", velocities, pnorm, vpeaks)

    # TODO (jpv): Generate a default settings file on the fly.
    @staticmethod
    def default_settings_file(fname):
        pass

    def write_peaks_to_file(self, fname, identifier, append=False, ftype="json"):
        """Write peak disperison values to file.

        Parameters
        ----------
        fname : str
            Name of the output file, may be a relative or the full path.
        identifier :  str
            A unique identifier for the peaks. The source offset is
            typically sufficient.
        append : bool, optional
            Flag to denote whether `fname` should be appended to or
            overwritten, default is `False` indicating the file will
            be overwritten.
        ftype : {'json'}, optional
            Denotes the desired filetype.
            TODO (jpv): Add also a csv option.

        Returns
        -------
        None
            Instead writes/appends dispersion peaks to file `fname`.

        """
        if self.domain == "wavenumber":
            v_peak = 2*np.pi / self.peaks*self.frqs
        elif self.domain == "velocity":
            v_peak = self.peaks
        else:
            raise NotImplementedError()

        if ftype != "json":
            raise ValueError()

        if fname.endswith(".json"):
            fname = fname[:-5]

        data = {}
        if append:
            try:
                f = open(f"{fname}.json", "r")
            except FileNotFoundError:
                pass
            else:
                data = json.load(f)
                f.close()

        with open(f"{fname}.json", "w") as fp:
            if identifier in data:
                raise KeyError(f"identifier {identifier} is repeated.")
            else:
                # keep_ids = np.where(v_peak < self.settings["vmax"])
                # ftrim = self.frqs[keep_ids].tolist()
                # vtrim = v_peak[keep_ids].tolist()
                # data.update({identifier: {"frequency": ftrim,
                #                           "velocity": vtrim}})

                data.update({identifier: {"frequency": self.frqs.tolist(),
                                          "velocity": v_peak.tolist()}})

            json.dump(data, fp)

    def plot_spectra(self, stype="fv", plot_peak=True, plot_limit=None):  # pragma: no cover
        if plot_limit is not None and len(plot_limit) != 4:
            raise ValueError("plot_limit should be a four element list")

        if self.domain == "wavenumber":
            fgrid, kgrid = np.meshgrid(self.frqs, self.vals)
            vgrid = 2*np.pi*fgrid / kgrid
            wgrid = 2*np.pi/kgrid
            kpeak = self.peaks
            wpeak = 2*np.pi / kpeak
            vpeak = wpeak*self.frqs
        elif self.domain == "velocity":
            fgrid, vgrid = np.meshgrid(self.frqs, self.vals)
            wgrid = vgrid / fgrid
            wgrid = 2*np.pi / wgrid
            vpeak = self.peaks
            kpeak = 2*np.pi*self.frqs / vpeak
            wpeak = 2*np.pi / kpeak
        else:
            raise NotImplementedError()

        if stype == "fk":
            xgrid = fgrid
            ygrid = kgrid
            xpeak = self.frqs
            ypeak = kpeak
            if plot_limit == None:
                plot_limit = [0, np.max(self.frqs), 0, 2*self.kres]
            xscale = "linear"
            yscale = "linear"
            xLabText = "Frequency (Hz)"
            yLabText = "Wavenumber (rad/m)"
        elif stype == "fw":
            xgrid = fgrid
            ygrid = wgrid
            xpeak = self.frqs
            ypeak = wpeak
            if plot_limit == None:
                plot_limit = [0, np.max(self.frqs), 1, 200]
            xscale = "linear"
            yscale = "log"
            xLabText = "Frequency (Hz)"
            yLabText = "Wavelength (m)"
        elif stype == "fv":
            xgrid = fgrid
            ygrid = vgrid
            xpeak = self.frqs
            ypeak = vpeak
            if plot_limit == None:
                plot_limit = [0, np.max(self.frqs),
                              self.settings["vmin"], self.settings["vmax"]]
            xscale = "linear"
            yscale = "linear"
            xLabText = "Frequency (Hz)"
            yLabText = "Velocity (m/s)"
        elif stype == "fp":
            xgrid = fgrid
            ygrid = 1.0 / vgrid
            xpeak = self.frqs
            ypeak = 1.0 / vpeak
            if plot_limit == None:
                plot_limit = [0, np.max(self.frqs), 1.0/1000, 1.0/100]
            xscale = "linear"
            yscale = "linear"
            xLabText = "Frequency (Hz)"
            yLabText = "Slowness (s/m)"
        elif stype == "wv":
            xgrid = wgrid
            ygrid = vgrid
            xpeak = wpeak
            ypeak = vpeak
            if plot_limit == None:
                plot_limit = [1, 200, 0, 1000]
            xscale = "log"
            yscale = "linear"
            xLabText = "Wavelength (m)"
            yLabText = "Velocity (m/s)"
        else:
            msg = f"`stype`= {stype} not recognized, use 'fk', 'fw', 'fv', 'fp', or 'wv'."
            raise ValueError(msg)

        zgrid = self.pnorm
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        contour = ax.contourf(xgrid,
                              ygrid,
                              zgrid,
                              np.linspace(0, np.max(zgrid), 20),
                              cmap=plt.cm.get_cmap("jet"))
        fig.colorbar(contour, ax=ax, ticks=np.arange(0, 1.1, 0.1))

        # TODO (jpv): Looking into using imshow for speedup!
        # plt.imshow(max_z,
        #           extent=plot_limit,
        #           origin='lower',
        #           cmap="jet")

        if plot_peak:
            ax.plot(xpeak,
                    ypeak,
                    marker="o",
                    markersize=5,
                    markeredgecolor="w",
                    markerfacecolor='none',
                    linestyle="none")

        # TODO (jpv): Look into making a plotting function to set these defaults
        ax.set_xlim(plot_limit[:2])
        ax.set_ylim(plot_limit[2:])
        ax.set_xlabel(xLabText, fontsize=12, fontname="arial")
        ax.set_ylabel(yLabText, fontsize=12, fontname="arial")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        return fig, ax

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
