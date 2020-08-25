"""WavefieldTransform1D class definition."""

import json
import logging

import numpy as np
import matplotlib.pyplot as plt

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

        # TODO (jpv): Somethings not quite right on zero pad.
        # If df is large (assumed greater than current df) padding
        # results in incorrect phase velocity.
        # See vs_uncertainty/0_intro/masw
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

        else:
            raise NotImplementedError

        self.frqs, self.domain, self.vals, self.pnorm, self.peaks = results

    @staticmethod
    def _fk_transform(array, nwave, fmin, fmax):
        """Perform fk transform on the provided array."""
        # Frequency vector
        sensor = array.sensors[0]
        frqs = np.arange(0, sensor.fnyq+sensor.df, sensor.df)

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
        # Frequency vector
        sensor = array.sensors[0]
        frqs = np.arange(0, sensor.fnyq+sensor.df, sensor.df)

        # u(x,t) -> FFT -> U(x,f)
        if array._flip_required:
            offsets = array.offsets[::-1]
            tseries = np.flipud(array.timeseriesmatrix)
        else:
            offsets = array.offsets
            tseries = array.timeseriesmatrix
        offsets = np.array(offsets)
        transformed = np.fft.fft(tseries)

        # Trim frequencies and downsample (if required by zero padding)
        fmin_ids = np.argmin(np.abs(frqs-fmin))
        fmax_ids = np.argmin(np.abs(frqs-fmax))
        freq_ids = range(fmin_ids, (fmax_ids+1), sensor.multiple)
        frqs = frqs[freq_ids]
        transformed = transformed[:, freq_ids]

        # Integrate across the array offsets
        phase_shift = np.empty((len(frqs), nvel))
        vels = np.linspace(vmin, vmax, nvel)
        dx = offsets[1:] - offsets[:-1]
        for f_index, frq in enumerate(frqs):
            for v_index, vel in enumerate(vels):
                exponent = 1j * 2*np.pi*frq/vel * offsets
                inner = np.exp(exponent) * transformed[:, f_index]/np.abs(transformed[:, f_index])
                phase_shift[f_index, v_index] = np.abs(np.sum(0.5*dx*(inner[:-1] + inner[1:])))

        # Normalize power and find peaks
        pnorm = np.empty_like(phase_shift)
        vpeaks = np.empty_like(frqs)
        for k, _phase_shift in enumerate(phase_shift):
            normed_phase_shift = np.abs(_phase_shift/np.max(_phase_shift))
            pnorm[k, :] = normed_phase_shift
            vpeaks[k] = vels[np.argmax(normed_phase_shift)]

        return (frqs, "velocity", vels, pnorm.T, vpeaks)

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
