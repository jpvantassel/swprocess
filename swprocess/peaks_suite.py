"""PeaksSuite class definition."""

import json
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from swprocess import Peaks

_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5

class PeaksSuite():

    @staticmethod
    def _check_input(peaks):
        if not isinstance(peaks, Peaks):
            msg = f"peaks must be an instance of `Peaks`, not {type(peaks)}."
            raise TypeError(msg)

    def __init__(self, peaks):
        """Instantiate a `PeaksSuite` object from a `Peaks` object.

        Parameters
        ----------
        peaks : Peaks
            A `Peaks` object to include in the suite.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        self._check_input(peaks)
        self.peaks = [peaks]
        self.ids = [peaks.ids]

    def append(self, peaks):
        """Append a `Peaks` object.

        Parameters
        ----------
        peaks : Peaks
            A `Peaks` object to include in the suite.

        Returns
        -------
        None
            Appends `Peaks` to `PeaksSuite`.

        """
        self._check_input(peaks)
        if peaks.ids in self.ids:
            msg = f"There already exists a member object with ids = {peaks.ids}."
            raise ValueError(msg)
        self.peaks.append(peaks)
        self.ids.append(peaks.ids)

    @classmethod
    def from_dicts(cls, dicts):
        """Instantiate `PeaksSuite` from `list` of `dict`s.

        Parameters
        ----------
        dicts : list of dict or dict
            List of `dict` or a single `dict` containing dispersion
            data. 

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        if isinstance(dicts, dict):
            dicts = [dicts]

        iterable = []
        for _dict in dicts:
            for identifier, data in _dict.items():
                iterable.append(Peaks.from_dict(data, identifier=identifier))

        return cls.from_iter(iterable)

    @classmethod
    def from_jsons(cls, fnames):
        """Instantiate `PeaksSuite` from json file(s).

        Parameters
        ----------
        fnames : list of str or str
            List of or a single file name containing dispersion data.
            Names may contain a relative or the full path.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        if isinstance(fnames, str):
            fnames = [fnames]

        dicts = []
        for fname in fnames:
            with open(fname, "r") as f:
                dicts.append(json.load(f))
        return cls.from_dicts(dicts)

    @classmethod
    def from_maxs(cls, fnames, identifiers, rayleigh=True, love=False):

        if len(fnames) != len(identifiers):
            msg = f"len(fnames) must equal len(identifiers), {len(fnames)} != {len(identifiers)}"
            ValueError(msg)

        iterable = []
        for fname, identifier in zip(fnames, identifiers):
            peaks = Peaks.from_max(
                fname, identifier=identifier, rayleigh=rayleigh, love=love)
            iterable.append(peaks)

        return cls.from_iter(iterable)

    @classmethod
    def from_iter(cls, iterable):
        """Instantiate `PeaksSuite` from iterable object.

        Parameters
        ----------
        iterable : iterable
            Iterable containing `Peaks` objects.

        Returns
        -------
        PeaksSuite
            Instantiated `PeaksSuite` object.

        """
        obj = cls(iterable[0])

        if len(iterable) >= 1:
            for _iter in iterable[1:]:
                obj.append(_iter)

        return obj

    def blitz(self, attr, limits):
        """Reject peaks outside the stated boundary.

        TODO (jpv): Refence Peaks.blitz for more information.

        """
        for peak in self.peaks:
            peak.blitz(attr, limits)

    def reject(self, xtype, xlims, ytype, ylims):
        """Reject peaks inside the stated boundary.

        TODO (jpv): Refence Peaks.reject for more information.

        """
        for peak in self.peaks:
            peak.reject(xtype, xlims, ytype, ylims)
    
    def plot(self, xtype="frequency", ytype="velocity", ax=None,
             plot_kwargs=None, ax_kwargs=None):
        """Create plot of dispersion data.

        TODO (jpv): Refence Peaks.plot for more information.

        plot_kwargs = {"key":value}
        plot_kwargs = {"key":[value1, value2, value3 ... ]}

        """
        if plot_kwargs is None:
            plot_kwargs = {}

        if ax_kwargs is None:
            ax_kwargs = {}

        if "color" not in plot_kwargs:
            plot_kwargs["color"] = _colors

        _plot_kwargs = self._prepare_kwargs(plot_kwargs, 0)
        _ax_kwargs = self._prepare_kwargs(ax_kwargs, 1)
        result = self.peaks[0].plot(xtype, ytype, ax, _plot_kwargs, _ax_kwargs)

        if ax is None:
            ax_was_none = True
            fig, ax = result
        else:
            ax_was_none = False
            
        if len(self.peaks) > 1:
            for index, peak in enumerate(self.peaks[1:], 1):
                _plot_kwargs = self._prepare_kwargs(plot_kwargs, index)
                _ax_kwargs = self._prepare_kwargs(ax_kwargs, index)
                peak.plot(xtype, ytype, ax, _plot_kwargs, _ax_kwargs)

        if ax_was_none:
            return (fig, ax)

    @staticmethod
    def _prepare_kwargs(kwargs, index):
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float)):
                new_kwargs[key] = value
            # elif len(value) == 1:
            #     new_kwargs[key] = value[0]
            else:
                new_kwargs[key] = value[index]
        return new_kwargs        

    def interactive_trimming(self, settings_file):
        with open(settings_file, "r") as f:
            settings = json.load(f)

        for key, value in settings.get("limits", []):
            self.blitz(key, value)

        fig = 0
        while True:
            if fig:
                plt.close(fig)
            # self.mean_disp = self.compute_dc_stats(self.frq,
            #                                        self.vel,
            #                                        minp=settings["minval"],
            #                                        maxp=settings["maxval"],
            #                                        numbins=settings["nbins"],
            #                                        binscale=settings["binscale"],
            #                                        bintype=settings["bintype"])
            fig = self.plot_dc_for_rmv(self.frq,
                                       self.vel,
                                       self.mean_disp,
                                       self.ids,
                                       klimits=klimits)

            ((xmin, xmax), (ymin, ymax)) = self._draw_box(fig)

            # self.rmv_dc_points(self.frq,
            #                    self.vel,
            #                    self.wav,
            #                    self.ids,
            #                    cfig,
            #                    extras=self.ext)

        # If all data is removed for a given offset, delete corresponding entries
        # (Only delete entries for one offset at a time because indices change after
        # deletion. Continue while loop as long as emty entries are encountered).
            prs = True
            while prs:
                n_empty = 0
                for k in range(len(self.frq)):
                    if len(self.frq[k]) == 0:
                        del self.frq[k]
                        del self.vel[k]
                        del self.ids[k]
                        n_empty += 1
                        break
                if n_empty == 0:
                    prs = False

            while True:
                cont = input("Enter 1 to continue, 0 to quit: ")
                if cont == "":
                    continue
                else:
                    cont = int(cont)
                    break

    # Reference for weighted mean and standard deviation
    # https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

    # @staticmethod
    # def compute_dc_stats(frequency, velocity, minp=5, maxp=100, numbins=96, binscale="linear", bintype="frequency", arrayweights=None):
    #     """Compute statistics on a set of frequency, velocity data"""

    #     if arrayweights is None:
    #         arrayweights = [1]*len(frequency)
    #     elif len(arrayweights) != len(frequency):
    #         raise ValueError(
    #             "len frequency velocity arrayweights must be equal.")

    #     frq, vel, wgt = np.array([]), np.array([]), np.array([])
    #     for ftmp, vtmp, wtmp in zip(frequency, velocity, arrayweights):
    #         if wtmp <= 0:
    #             raise ValueError("arrayweights must be > 0.")
    #         frq = np.concatenate((frq, ftmp))
    #         vel = np.concatenate((vel, vtmp))
    #         wgt = np.concatenate((wgt, wtmp*np.ones(frq.shape)))
    #     wav = vel/frq

    #     if binscale.lower() == "linear":
    #         binedges = np.linspace(minp, maxp, numbins+1)
    #     elif binscale.lower() == "log":
    #         binedges = np.logspace(np.log10(minp), np.log10(maxp), numbins+1)
    #     else:
    #         raise ValueError(f"Unknown binscale `{binscale}`.")
    #     logging.debug(f"binedges = {binedges}")

    #     if bintype.lower() == "frequency":
    #         bin_indices = np.digitize(frq, binedges)
    #     elif bintype.lower() == "wavelength":
    #         bin_indices = np.digitize(wav, binedges)
    #     else:
    #         raise ValueError(f"Unknown bintype `{bintype}")
    #     logging.debug(f"bin_indices = {bin_indices}")

    #     # bin_cnt = np.zeros(numbins)
    #     bin_wgt = np.zeros(numbins)
    #     bin_vel_mean = np.zeros(numbins)
    #     bin_vel_std = np.zeros(numbins)
    #     bin_slo_mean = np.zeros(numbins)
    #     bin_slo_std = np.zeros(numbins)
    #     bin_frq_mean = np.zeros(numbins)
    #     bin_wav_mean = np.zeros(numbins)

    #     for cbin in range(0, numbins):
    #         bin_id = np.where(bin_indices == (cbin+1))[0]
    #         cfrq = frq[bin_id]
    #         cvel = vel[bin_id]
    #         cslo = 1/cvel
    #         cwav = wav[bin_id]
    #         cwgt = wgt[bin_id]
    #         ccnt = len(bin_id)

    #         if ccnt != 0:

    #             logging.debug(f"ccnt = {ccnt}")
    #             logging.debug(f"bin_id = {bin_id}")
    #             logging.debug(f"cfrq = {cfrq}")
    #             logging.debug(f"cvel = {cvel}")
    #             logging.debug(f"cslo = {cslo}")
    #             logging.debug(f"cwav = {cwav}")
    #             logging.debug(f"cwgt = {cwgt}")

    #             bin_wgt[cbin] = sum(cwgt)

    #             bin_frq_mean[cbin] = sum(cfrq*cwgt) / sum(cwgt)
    #             bin_vel_mean[cbin] = sum(cvel*cwgt) / sum(cwgt)
    #             bin_slo_mean[cbin] = sum(cslo*cwgt) / sum(cwgt)
    #             bin_wav_mean[cbin] = sum(cwav*cwgt) / sum(cwgt)

    #             if ccnt > 1:
    #                 vres = sum(cwgt * ((cvel-bin_vel_mean[cbin])**2))
    #                 bin_vel_std[cbin] = np.sqrt(
    #                     (ccnt*vres)/((ccnt-1)*sum(cwgt)))
    #                 sres = sum(cwgt * ((cslo-bin_slo_mean[cbin])**2))
    #                 bin_slo_std[cbin] = np.sqrt(
    #                     (ccnt*sres)/((ccnt-1)*sum(cwgt)))
    #             else:
    #                 bin_vel_std[cbin] = 0
    #                 bin_slo_std[cbin] = 0

    #     keep_ids = np.where(bin_wgt > 0)[0]
    #     bin_frq_mean = bin_frq_mean[keep_ids]
    #     bin_vel_mean = bin_vel_mean[keep_ids]
    #     bin_vel_std = bin_vel_std[keep_ids]
    #     bin_slo_mean = bin_slo_mean[keep_ids]
    #     bin_slo_std = bin_slo_std[keep_ids]
    #     bin_wav_mean = bin_wav_mean[keep_ids]
    #     bin_wgt = bin_wgt[keep_ids]
    #     # cov = bin_vel_mean / bin_vel_std

    #     # mean_disp = np.vstack(
    #     #     (freqMean, velMean, velStd, slowMean, slowStd, waveMean, binWeight, cov))
    #     # mean_disp = mean_disp.transpose()
    #     # # Remove rows corresponding to empty bins (meanFreq==0)
    #     # z_ids = np.where(mean_disp[:, 0] == 0)[0]
    #     # mean_disp = np.delete(mean_disp, z_ids, 0)
    #     # return mean_disp

    #     meandisp = {"mean": {"frq": bin_frq_mean,
    #                          "vel": bin_vel_mean,
    #                          "slo": bin_slo_mean,
    #                          "wav": bin_wav_mean},
    #                 "std": {"vel": bin_vel_std,
    #                         "slo": bin_slo_std,
    #                         }
    #                 }
    #     return meandisp

    # @staticmethod
    # def plot_dc_for_rmv(frequency, velocity, mean_disp, legend, marker_type=None, color_spec=None, xscaletype="log", klimits=None):
    #     """Function to plot dispersion data along with averages and standard deviations.
    #     (Note that the min(klimits) and max(klimits) curves are used for passive-source FK processing,
    #     thus, min(klimits) and max(klimits) are set equal to NaN for MASW testing to avoid plotting.)
    #     """

    #     n_off = len(velocity)

    #     if marker_type is None:
    #         marker_type = ['o']*n_off
    #     if color_spec is None:
    #         color_spec = plot_tools.makecolormap(n_off)

    #     minf = np.min(mean_disp["mean"]["frq"])
    #     maxf = np.max(mean_disp["mean"]["frq"])
    #     maxv, maxw = 0, 0
    #     for vl, fr in zip(velocity, frequency):
    #         if max(vl) > maxv:
    #             maxv = max(vl)
    #         if max(vl/fr) > maxw:
    #             maxw = max(vl/fr)

    #     if klimits:
    #         freq_klim = np.logspace(np.log10(minf), np.log10(maxf), 100)
    #         vel_klimf = np.vstack((2*np.pi*freq_klim/max(klimits), 2*np.pi*freq_klim /
    #                                (max(klimits)/2), 2*np.pi*freq_klim/min(klimits), 2*np.pi*freq_klim/(min(klimits)/2)))
    #         vel_klimf = vel_klimf.transpose()
    #         if not(np.isnan(max(klimits))):
    #             for j in range(np.shape(vel_klimf)[1]):
    #                 rmvID = np.where(vel_klimf[:, j] > maxv)[0]
    #                 vel_klimf[rmvID, j] = float('nan')
    #         wave_lim = np.hstack((2*np.pi/max(klimits)*np.array([[1], [1]]), 2*np.pi/(max(klimits)/2)*np.array(
    #             [[1], [1]]), 2*np.pi/min(klimits)*np.array([[1], [1]]), 2*np.pi/(min(klimits)/2)*np.array([[1], [1]])))
    #         vel_klimW = np.array([0, maxv])

    #     mwdth = 10
    #     mhght = 6
    #     fsize = 11
    #     cfig, (axf, axw) = plt.subplots(
    #         nrows=1, ncols=2, figsize=(mwdth, mhght))

    #     for fr, vl, mk, co in zip(frequency, velocity, marker_type, color_spec):
    #         axf.plot(fr,
    #                  vl,
    #                  marker=mk,
    #                  markersize=5,
    #                  markeredgecolor=co,
    #                  markerfacecolor="none",
    #                  linestyle="none")
    #     axf.errorbar(mean_disp["mean"]["frq"],
    #                  mean_disp["mean"]["vel"],
    #                  mean_disp["std"]["vel"],
    #                  marker="o",
    #                  markersize=5,
    #                  color="k",
    #                  linestyle="none")

    #     if klimits:
    #         axf.plot(freq_klim, vel_klimf[:, 0],
    #                  linestyle=":", color='#000000')
    #         axf.plot(freq_klim, vel_klimf[:, 1],
    #                  linestyle="-", color='#000000')
    #         axf.plot(freq_klim, vel_klimf[:, 2],
    #                  linestyle="--", color='#000000')
    #         axf.plot(freq_klim, vel_klimf[:, 3],
    #                  linestyle="-.", color='#000000')
    #     axf.set_xlabel("Frequency (Hz)", fontsize=fsize, fontname="arial")
    #     axf.set_ylabel("Velocity (m/s)", fontsize=fsize, fontname="arial")
    #     axf.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    #     axf.set_xscale(xscaletype)

    #     for fr, vl, mk, co, le in zip(frequency, velocity, marker_type, color_spec, legend):
    #         axw.plot(vl/fr,
    #                  vl,
    #                  marker=mk,
    #                  markersize=5,
    #                  markeredgecolor=co,
    #                  markerfacecolor="none",
    #                  linestyle="none",
    #                  label=le)
    #     axw.errorbar(mean_disp["mean"]["wav"],
    #                  mean_disp["mean"]["vel"],
    #                  mean_disp["std"]["vel"],
    #                  marker="o",
    #                  markersize=5,
    #                  color="k",
    #                  linestyle="none")

    #     if klimits:
    #         axw.plot(wave_lim[:, 0], vel_klimW, linestyle=":",
    #                  color='#000000', label='kmax')
    #         axw.plot(wave_lim[:, 1], vel_klimW, linestyle="-",
    #                  color='#000000', label='kmax/2')
    #         axw.plot(wave_lim[:, 2], vel_klimW,
    #                  linestyle="--", color='#000000', label='kmin')
    #         axw.plot(wave_lim[:, 3], vel_klimW, linestyle="-.",
    #                  color='#000000', label='kmin/2')

    #     # handles, labels = axw.get_legend_handles_labels()
    #     # axw.legend(handles, labels, loc='upper left')
    #     axw.legend(loc="upper left")
    #     axw.set_xlabel("Wavelength (m)", fontsize=fsize, fontname="arial")
    #     axw.set_xscale(xscaletype)
    #     # axw.set_xticklabels(axw.get_xticks(), fontsize=fsize, fontname="arial")
    #     # axw.set_yticklabels(axw.get_yticks(), fontsize=fsize, fontname="arial")
    #     axw.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    #     cfig.show()
    #     axf.set_autoscale_on(False)
    #     axw.set_autoscale_on(False)
    #     return cfig

    # @staticmethod
    # def rmv_dc_points(frequency, velocity, wavelength, offset, cfig, extras=None):
    #     """Function to prompt user to draw a box on a dispersion curve
    #     figure. Data points inside of the box are removed and data
    #     points outside of the box are kept.
    #     """
    #     while True:
    #         axclick = []

    #         def on_click(event):
    #             if event.inaxes is not None:
    #                 if len(axclick) < 2:
    #                     axclick.append(event.inaxes)
    #         cid = cfig.canvas.mpl_connect('button_press_event', on_click)

    #         axf = cfig.axes[0]
    #         axw = cfig.axes[1]

    #         fcursor = Cursor(axf, useblit=True, color='k', linewidth=1)
    #         wcursor = Cursor(axw, useblit=True, color='k', linewidth=1)

    #         rawBounds = np.asarray(cfig.ginput(2, timeout=0))
    #         cfig.canvas.mpl_disconnect(cid)
    #         xmin = np.min(rawBounds[:, 0])
    #         xmax = np.max(rawBounds[:, 0])
    #         ymin = np.min(rawBounds[:, 1])
    #         ymax = np.max(rawBounds[:, 1])

    #         n_rmv = 0
    #         # for g in range(len(frequency)):
    #         for bin_id, (f, v, w) in enumerate(zip(frequency, velocity, wavelength)):
    #             # Create arrays to store indices of data that will be kept and removed
    #             rmv_id = np.zeros(len(f), int)
    #             keep_id = np.zeros(len(f), int)

    #             # If user clicked on two different axes, warn user and return
    #             if axclick[0] != axclick[1]:
    #                 logging.warning("BOTH CLICKS MUST BE ON SAME AXIS")
    #                 return

    #             for i in range(len(f)):
    #                 condition1 = (axclick[0] == axf) and (
    #                     xmin < f[i] and f[i] < xmax and ymin < v[i] and v[i] < ymax)
    #                 condition2 = (axclick[0] == axw) and (
    #                     xmin < w[i] and w[i] < xmax and ymin < v[i] and v[i] < ymax)
    #                 # Points inside of selectd box are removed
    #                 if condition1 or condition2:
    #                     rmv_id[i] = i+1
    #                 # Points outside of selected box are kept
    #                 else:
    #                     keep_id[i] = i+1

    #             # Remove zeros from rmv_id and keep_id
    #             zid = np.where(rmv_id == 0)[0]
    #             rmv_id = np.delete(rmv_id, zid, 0)
    #             zid = np.where(keep_id == 0)[0]
    #             keep_id = np.delete(keep_id, zid, 0)

    #             frmv = f[(rmv_id-1)]
    #             vrmv = v[(rmv_id-1)]
    #             wrmv = w[(rmv_id-1)]
    #             n_rmv += len(vrmv)

    #             axf.plot(frmv, vrmv,
    #                      marker="x", color="k", markersize=5, linestyle="none")
    #             axw.plot(wrmv, vrmv,
    #                      marker="x", color="k", markersize=5, linestyle="none")
    #             cfig.canvas.draw_idle()

    #             logging.debug(f"f = {f}")
    #             logging.debug(f"v = {v}")
    #             logging.debug(f"w = {w}")

    #             frequency[bin_id] = f[(keep_id-1)]
    #             velocity[bin_id] = v[(keep_id-1)]
    #             wavelength[bin_id] = w[(keep_id-1)]

    #             logging.debug(f"f = {f}")
    #             logging.debug(f"v = {v}")
    #             logging.debug(f"w = {w}")
    #             logging.debug(f"extras = {extras}")

    #             for key, value in extras.items():
    #                 logging.debug(f"key = {key}")
    #                 logging.debug(f"value = {value}")
    #                 logging.debug(f"bin_id = {bin_id}")
    #                 logging.debug(f"keep_id-1 = {keep_id-1}")

    #                 extras[key][bin_id] = value[bin_id][(keep_id-1)]

    #         if n_rmv == 0:
    #             break

    #         del cid

    def _draw_box(self, fig):
        """Prompt user to define a rectangular box on a two-panel plot.

        Parameters
        ----------
        fig : Figure
            Figure object, on which the user is to draw the box.

        Returns
        -------
        tuple
            Of the form `((xmin, xmax), (ymin,ymax))` where `xmin` and
            `xmax` are the minimum and maximum abscissa and `ymin` and
            `ymax` are the minimum and maximum ordinate of the
            user-defined box.

        """
        cursors = []
        for ax in fig.axes:
            cursors.append(Cursor(ax, useblit=True, color='k', linewidth=1))

        def on_click(event):
            if event.inaxes is not None:
                axclick.append(event.inaxes)

        while True:
            axclick = []
            session = fig.canvas.mpl_connect('button_press_event', on_click)
            xs, ys = fig.ginput(2, timeout=0)
            fig.canvas.mpl_disconnect(session)

            if len(axclick) == 2 and axclick[0] == axclick[1]:
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                return ((xmin, xmax), (ymin, ymax))
            else:
                warnings.warn("Both clicks must be inside the same axis.")

    # def _indicate_rejections(self, fig, plot_kwargs):
    #     axf.plot(frmv, vrmv,
    #                 marker="x", color="k", markersize=5, linestyle="none")
    #     axw.plot(wrmv, vrmv,
    #                 marker="x", color="k", markersize=5, linestyle="none")
    #     fig.canvas.draw_idle()

    # TODO (jpv) : Broken b/c mean_disp
    # def write_stat_swprepost(self, fname):
    #     """Write statistics (mean and standard deviation) to csv file
    #     of the form accepted by swprepost.

    #     Args:
    #         fname = String for file name. Can be a relative or a full
    #             path.

    #     Returns:
    #         Method returns None, but saves file to disk.

    #     """
    #     if fname.endswith(".csv"):
    #         fname = fname[:-4]
    #     with open(fname+".csv", "w") as f:
    #         f.write("Frequency (Hz), Velocity (m/s), VelStd (m/s)\n")
    #         for fr, ve, st in zip(self.mean_disp["mean"]["frq"],
    #                               self.mean_disp["mean"]["vel"],
    #                               self.mean_disp["std"]["vel"]):
    #             f.write(f"{fr},{ve},{st}\n")

    def __eq__(self, other):
        if not isinstance(other, PeaksSuite):
            return False

        for mypeaks, urpeaks in zip(self.peaks, other.peaks):
            if mypeaks != urpeaks:
                return False

        return True
