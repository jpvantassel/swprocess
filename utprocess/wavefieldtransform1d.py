"""This file contains a class for performing and manipulating a
wavefield transformation for a 1D array."""

import matplotlib.pyplot as plt
import json
import numpy as np
import logging
logger = logging.getLogger(__name__)


class WavefieldTransform1D():
    """Class for performing and manipulating wavefield 
    transformations for a 1D array.

    Attributues:
        timelength: Length of time record used in seconds.

        method: Type of wavefield transformation.

        f_trial: Dictionary of the form {'min': ,'max': , 'npts': } 
            which defines the minimum, maximum, and number of frequency
            points.

        weighting: Type of weighting vector, can be ['none', ...]

        steering_vector: Type of steering vector, can be ['plane', ...]

    """

    def __init__(self, array, settings_file):
        """Initialize an instance of the WavefieldTransformation1D class 
        from an instance of the Array1D class.

        Args:
            array: Instance of an Array1D class.

            settings: Name of a .json file that descibes the settings to
                be used for the 1D wavefield transformation.

        Returns:


        Raises:

        """
        self.array = array
        inside_array = (array.source.position["x"] > 0 and
                        array.source.position["x"] < (array.nchannels-1)*array.spacing)
        if inside_array:
            raise ValueError("Source is inside the array.")

        with open(settings_file, "r") as f:
            settings = json.load(f)
            self.general = settings["general"]

        if self.general["start_time"] != None and self.general["end_time"] != None:
            array.trim_timeseries(start_time=self.general["start_time"],
                                  end_time=self.general["end_time"])

        if settings["type"] == "fk":
            numk = self.general["n_trial"]
            if numk % 2 != 0:
                numk += 1
            # TODO (jpv) self.generalize numk to be n_trial, so move above outside of if statement

            freq = np.arange(0,
                             self.array.ex_rec.timeseries.fnyq+self.array.ex_rec.timeseries.df,
                             self.array.ex_rec.timeseries.df)

            kres = self.array.kres
            dk = 2*np.pi / (numk*self.array.spacing)
            k_vals = np.arange(dk, kres, dk)

            fk = np.fft.fft2(self.array.timeseriesmatrix,
                             s=(self.array.ex_rec.timeseries.nsamples, numk))
            fk = np.fliplr(np.abs(fk))
            fk = fk[0:len(freq), 1::]

            # Remove frequencies above/below specificied max/min frequencies and downsample (if required by zero padding)
            fminID = np.argmin(np.absolute(freq-self.general["min_freq"]))
            fmaxID = np.argmin(np.absolute(freq-self.general["max_freq"]))
            freq_id = range(fminID,
                            (fmaxID+1),
                            array.ex_rec.timeseries.multiple)
            freq = freq[freq_id]
            fk = fk[freq_id, :]

            # Identify wavenumber associated with maximum in fk domain..................
            pnorm = np.zeros(np.shape(fk))
            k_peak = np.zeros(np.shape(freq))
            for k in range(np.shape(fk)[0]):
                pnorm[k, :] = fk[k, :] / np.max(fk[k, :])
                k_peak[k] = k_vals[np.argmax(pnorm[k, :])]

            self.freq = freq
            self.peak_vals = k_peak
            self.trial_vals = k_vals
            self.val_type = 'wavenumber'
            self.kres = kres
            self.pnorm = np.transpose(pnorm)

    def save_peaks(self, fname, identifier, append=False):
        """Save the peak disperison values frequency and velocity to
        json file.

        Args: 
            fname: String denoting the name of the output file, may be
                a full path if desired. Do not include the '.json'
                extension.

            identifier: Any immutable/hashable type to denote the object
                (i.e., array) to which this data belongs. For MASW this
                may be the offset, for MAM this may be the array name.

            append: Boolean to denote whether a file with the same name
                as fname (if it exists) should be appended to or that
                the file should be overwritten.

        Returns:
            Do not return a value, instead write dispersion to json file
            fname.

        Raises:
            This method raises no exceptions.
        """
        logging.info("DispersionPower.save_peaks")

        if self.val_type == "wavenumber":
            v_peak = 2*np.pi / self.peak_vals*self.freq
        elif self.val_type == "velocity":
            v_peak = self.peak_vals
        else:
            raise NotImplementedError()

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
                data.update({identifier: {"frequency": self.freq.tolist(),
                                          "velocity": v_peak.tolist()}})
            json.dump(data, fp)

    def plot_spec(self, plot_type="fv", plot_limit=None):
        if plot_limit != None and len(plot_limit) != 4:
            raise ValueError("plotLim should be a four element list")

        if self.val_type == "wavenumber":
            k_vals = self.trial_vals
            freq_grid, wavenum_grid = np.meshgrid(self.freq, k_vals)
            vel_grid = 2*np.pi*freq_grid / wavenum_grid
            wavel_grid = 2*np.pi/wavenum_grid
            k_peak = self.peak_vals
            wavel_peak = 2*np.pi / k_peak
            v_peak = wavel_peak*self.freq
        elif self.val_type == "velocity":
            v_vals = self.trial_vals
            freq_grid, vel_grid = np.meshgrid(self.freq, v_vals)
            wavel_grid = vel_grid / freq_grid
            wavenum_grid = 2*np.pi / wavel_grid
            v_peak = self.peak_vals
            k_peak = 2*np.pi*self.freq / v_peak
            wavel_peak = 2*np.pi / k_peak
        else:
            raise NotImplementedError()

        if str.lower(plot_type) == "fk":
            xgrid = freq_grid
            ygrid = wavenum_grid
            xpeak = self.freq
            ypeak = k_peak
            if plot_limit == None:
                plot_limit = [0, np.max(self.freq), 0, self.kres]
            xscale = "linear"
            yscale = "linear"
            xLabText = "Frequency (Hz)"
            yLabText = "Wavenumber (rad/m)"
        elif str.lower(plot_type) == "fw":
            xgrid = freq_grid
            ygrid = wavel_grid
            xpeak = self.freq
            ypeak = wavel_peak
            if plot_limit == None:
                plot_limit = [0, np.max(self.freq), 1, 200]
            xscale = "linear"
            yscale = "log"
            xLabText = "Frequency (Hz)"
            yLabText = "Wavelength (m)"
        elif str.lower(plot_type) == "fv":
            xgrid = freq_grid
            ygrid = vel_grid
            xpeak = self.freq
            ypeak = v_peak
            if plot_limit == None:
                plot_limit = [
                    0, np.max(self.freq), self.general["min_vel"], self.general["max_vel"]]
            xscale = "linear"
            yscale = "linear"
            xLabText = "Frequency (Hz)"
            yLabText = "Velocity (m/s)"
        elif str.lower(plot_type) == "fp":
            xgrid = freq_grid
            ygrid = 1.0 / vel_grid
            xpeak = self.freq
            ypeak = 1.0 / v_peak
            if plot_limit == None:
                plot_limit = [0, np.max(self.freq), 1.0/1000, 1.0/100]
            xscale = "linear"
            yscale = "linear"
            xLabText = "Frequency (Hz)"
            yLabText = "Slowness (s/m)"
        elif str.lower(plot_type) == "wv":
            xgrid = wavel_grid
            ygrid = vel_grid
            xpeak = wavel_peak
            ypeak = v_peak
            if plot_limit == None:
                plot_limit = [1, 200, 0, 1000]
            xscale = "log"
            yscale = "linear"
            xLabText = "Wavelength (m)"
            yLabText = "Velocity (m/s)"
        else:
            raise ValueError(
                "Invalid plot_type. Should be \"fk\", \"fw\", \"fv\", \"fp\", or \"wv\".")

        mwdth = 6
        mhght = 4
        zgrid = np.abs(self.pnorm)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(mwdth, mhght))
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
        # ax.set_xticklabels(ax.get_xticks(), fontsize=12, fontname="arial")
        # ax.set_yticklabels(ax.get_yticks(), fontsize=12, fontname="arial")
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
