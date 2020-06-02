"""Peaks class definition."""

import json
import re
import warnings
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from swprocess import plot_tools

_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5
logger = logging.getLogger(__name__)


class Peaks():
    """Class for handling peak dispersion data.

    Attributes
    ----------
        TODO (jpv): Finish documentation.

    """

    def __init__(self, frequency, velocity, identifier, **kwargs):
        """Initialize an instance of Peaks from a list of frequency
        and velocity values.

        Parameters
        ----------
        frequency, velocity : list
            Frequency and velocity (one per peak), respectively.
        identifiers : str
            String to uniquely identify the provided frequency-velocity
            pair.
        **kwargs : kwargs
            Optional keyword argument(s) these may include
            additional details about the dispersion peaks such as:
            azimuth (azi), ellipticity (ell), power (pwr), and noise
            (pwr). Will generally not be used directly.

        Returns
        -------
        Peaks
            Instantiated `Peaks` object.

        """
        self.frq = [np.array(frequency)]
        self.vel = [np.array(velocity)]
        self.ids = [str(identifier)]
        logging.debug(f"**kwargs {kwargs}")
        self.extra = list(kwargs.keys())
        for key, val in kwargs.items():
            setattr(self, key, [np.array(val)])
    
    @property
    def frequency(self):
        return self.frq

    @property
    def velocity(self):
        return self.velocity
    
    @property
    def identifiers(self):
        return self.ids

    @property
    def wav(self)
        msg = "Use of wav is deprecated use wavelength instead."
        warnings.warn(msg, DeprecationWarning)
        return self.wavelength

    @property
    def wavelength(self):
        wav = []
        for frq, vel in zip(self.frq, self.vel):
            wav.append(vel/frq)
        return wave
        
    @property
    def slowness(self)
        slo = []
        for vel in self.vel:
            slo.append(1/vel)
        return slo

    @property
    def mean_disp(self, **kwargs)
        return self.compute_dc_stats(self.frq, self.vel, **kwargs)

    def append(self, frequency, velocity, identifier, **kwargs):
        """Method to append frequency and velocity data into the object.

        Parameters
        ----------
        TODO (jpv): Finish documentation.

        Returns
        -------
        None
            Updates object.

        """
        self.frq.append(np.array(frequency))
        self.vel.append(np.array(velocity))
        self.ids.append(str(identifier))

        for key, val in kwargs.items():
            cval = getattr(self, key)
            setattr(self, key, cval.append(np.array(val)))

    @classmethod
    def from_dicts(cls, peak_dicts):
        """Initialize `Peaks` from `dict`.

        Parameters
        ----------
        peak_dicts: dict or list
            Dictionary or list of dictionaries of the form
            {"identifier": {"frequency":freq, "velocity":vel,
            "kwarg1": kwarg1}} where identifier is a string identifying
            the data. freq is a list of floats denoting frequency
            values. vel is a list of floats denoting velocity values.
            kwarg1 is one of the optional keyword arguments,
            may use all of those listed in __init__.

        Returns
        -------
        Peaks
            Initialized `Peaks` instance.

        """
        if isinstance(peak_dicts, dict):
            peak_dicts = [peak_dicts]

        start = True
        for peak_dict in peak_dicts:
            for key, value in peak_dict.items():
                logging.debug(key)
                npts = len(value["frequency"])

                if start:
                    logging.debug(f"For {key} new object was started.")
                    obj = cls(value["frequency"],
                              value["velocity"],
                              key,
                              azi=value.get("azi") if value.get(
                                  "azi") else [0]*npts,
                              ell=value.get("ell") if value.get(
                                  "azi") else [0]*npts,
                              noi=value.get("noi") if value.get(
                                  "azi") else [0]*npts,
                              pwr=value.get("pwr") if value.get("azi") else [0]*npts)
                    start = False
                else:
                    logging.debug(f"{key} was appended.")
                    obj.append(value["frequency"],
                               value["velocity"],
                               key,
                               azi=value.get("azi") if value.get(
                                   "azi") else [0]*npts,
                               ell=value.get("ell") if value.get(
                                   "azi") else [0]*npts,
                               noi=value.get("noi") if value.get(
                                   "azi") else [0]*npts,
                               pwr=value.get("pwr") if value.get("azi") else [0]*npts)
        return obj

    @classmethod
    def from_jsons(cls, fnames):
        dicts = []
        for fname in fnames:
            with open(fname, "r") as f:
                dicts.append(json.load(f))
        return cls.from_dicts(dicts)

    def from_maxs(self, *args, **kwargs):
        msg = "Peaks.from_maxs has been deprecated, use PeaksPassive.from_max() or PeaksSuite.from_maxs() instead."
        raise DeprecationWarning(msg)

    # def write_to_txt_dinver(self, fname):
    #     pass

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

    def write_peak_json(self, fname):
        msg = ".write_peak_json is deprecated use .to_json instead."
        warnings.warn(msg, DeprecationWarning)
        self.to_json(fname)

    def to_json(self, fname):
        """Write `Peaks` to json file.

        Parameters
        ----------
        fname : str
            Output file name, can include a relative or the full path.

        Returns
        -------
        None
            Instead writes file to disk.

        """
        data = {}
        for num, (f, v, label) in enumerate(zip(self.frq, self.vel, self.ids)):
            data.update({label: {"frequency": f.tolist(),
                                 "velocity": v.tolist(),
                                 }})
            ext_dict = {}
            for key, value in self.ext.items():
                ext_dict.update({key: value[num].tolist()})
            data[label].update(ext_dict)

        with open(fname, "w") as f:
            json.dump(data, f)

    def plot(self, xtype="frequency", ax=None):
        """Create plot of dispersion data.

        Parameters
        ----------
        xtype : {'frequency', 'wavelength'}, optional
            Denote whether the x-axis should be either `frequency` or
            `wavelength`, default is `frequency`.
        ytype : {'velocity', 'slowness', '}
        ax : Axes, optional
            Pass an `Axes` on which to plot, default is `None` meaning
            a `Axes` will be generated on-the-fly.

        Returns
        -------
        None or tuple
            `None` if `ax` is provided, otherwise `tuple` of the form
            `(fig, ax)` where `fig` is the figure handle and `ax` is
            the axes handle.

        """
        ax_was_none=False
        if ax is None:
            ax_was_none = True
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3), dpi=150)

        if xtype.lower() == "wavelength":
            x = self.wav
            xlabel = "Wavelength (m)"
        else xtype.lower() == "frequency":
            x = self.frq
            xlabel = "Frequency (Hz)"
        else:
            raise ValueError(f"xtype = {xtype}, not recognized.")
        for x, v, color, label in zip(x, self.vel, _colors, self.ids):
            ax.plot(x, v, color=color, linestyle="none",
                    marker="o", markerfacecolor="none", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Phase Velocity (m/s)")
        ax.set_xscale("log")
        ax.legend()

        if ax_was_none:
            return (fig, ax)

    def plot_2pannel(self):
        fig, (axf, axw) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))
        for f, v, w, i, color in zip(self.frq, self.vel, self.wav, self.ids, _colors):
            axf.plot(f, v, color=color, linestyle="none",
                     marker="o", markerfacecolor="none", label=i)
            axw.plot(w, v, color=color, linestyle="none",
                     marker="o", markerfacecolor="none")
        axf.set_xlabel("Frequency (Hz)")
        axw.set_xlabel("Wavelength (m)")
        axf.set_ylabel("Phase Velocity (m/s)")
        axw.set_ylabel("Phase Velocity (m/s)")
        axw.set_xscale("log")
        return (fig, axf, axw)

    def _blitz(self, settings):
        """Remove all points outside the limits provided"""
        limits = settings.get("limits")
        if limits is None:
            msg = "_blitz returning with no removal, as limits does not exist."
            logger.info(msg)
            return

        vmin = limits.get("vmin")
        vmax = limits.get("vmax")

        if vmin is not None:
            for ith in range(len(self.frq)):
                keep_ids = np.where(self.vel[ith] > vmin)
                self.frq[ith] = self.frq[ith][keep_ids]
                self.vel[ith] = self.vel[ith][keep_ids]
                self.wav[ith] = self.wav[ith][keep_ids]

        if vmax is not None:
            for ith in range(len(self.frq)):
                keep_ids = np.where(self.vel[ith] < vmax)
                self.frq[ith] = self.frq[ith][keep_ids]
                self.vel[ith] = self.vel[ith][keep_ids]
                self.wav[ith] = self.wav[ith][keep_ids]

# TODO (jpv): Add test.
# TODO (jpv): Finish for wavelength and frequency.


        # Reference for weighted mean and standard deviation 
        # https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf



    def party_time(self, settings_file, klimits=None):
        with open(settings_file, "r") as f:
            settings = json.load(f)

        self._blitz(settings)

        cont = True
        cfig = 0
        while cont:
            if cfig:
                plt.close(cfig)
            self.mean_disp = self.compute_dc_stats(self.frq,
                                                   self.vel,
                                                   minp=settings["minval"],
                                                   maxp=settings["maxval"],
                                                   numbins=settings["nbins"],
                                                   binscale=settings["binscale"],
                                                   bintype=settings["bintype"])
            cfig = self.plot_dc_for_rmv(self.frq,
                                        self.vel,
                                        self.mean_disp,
                                        self.ids,
                                        klimits=klimits)
            self.rmv_dc_points(self.frq,
                               self.vel,
                               self.wav,
                               self.ids,
                               cfig,
                               extras=self.ext)

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

            # Ask user if they would like to continue cutting data
            while True:
                cont = input("Enter 1 to continue, 0 to quit: ")
                if cont == "":
                    continue
                else:
                    cont = int(cont)
                    break

    @staticmethod
    def compute_dc_stats(frequency, velocity, minp=5, maxp=100, numbins=96, binscale="linear", bintype="frequency", arrayweights=None):
        """Compute statistics on a set of frequency, velocity data"""

        if arrayweights is None:
            arrayweights = [1]*len(frequency)
        elif len(arrayweights) != len(frequency):
            raise ValueError(
                "len frequency velocity arrayweights must be equal.")

        frq, vel, wgt = np.array([]), np.array([]), np.array([])
        for ftmp, vtmp, wtmp in zip(frequency, velocity, arrayweights):
            if wtmp <= 0:
                raise ValueError("arrayweights must be > 0.")
            frq = np.concatenate((frq, ftmp))
            vel = np.concatenate((vel, vtmp))
            wgt = np.concatenate((wgt, wtmp*np.ones(frq.shape)))
        wav = vel/frq

        if binscale.lower() == "linear":
            binedges = np.linspace(minp, maxp, numbins+1)
        elif binscale.lower() == "log":
            binedges = np.logspace(np.log10(minp), np.log10(maxp), numbins+1)
        else:
            raise ValueError(f"Unknown binscale `{binscale}`.")
        logging.debug(f"binedges = {binedges}")

        if bintype.lower() == "frequency":
            bin_indices = np.digitize(frq, binedges)
        elif bintype.lower() == "wavelength":
            bin_indices = np.digitize(wav, binedges)
        else:
            raise ValueError(f"Unknown bintype `{bintype}")
        logging.debug(f"bin_indices = {bin_indices}")

        # bin_cnt = np.zeros(numbins)
        bin_wgt = np.zeros(numbins)
        bin_vel_mean = np.zeros(numbins)
        bin_vel_std = np.zeros(numbins)
        bin_slo_mean = np.zeros(numbins)
        bin_slo_std = np.zeros(numbins)
        bin_frq_mean = np.zeros(numbins)
        bin_wav_mean = np.zeros(numbins)

        for cbin in range(0, numbins):
            bin_id = np.where(bin_indices == (cbin+1))[0]
            cfrq = frq[bin_id]
            cvel = vel[bin_id]
            cslo = 1/cvel
            cwav = wav[bin_id]
            cwgt = wgt[bin_id]
            ccnt = len(bin_id)

            if ccnt != 0:

                logging.debug(f"ccnt = {ccnt}")
                logging.debug(f"bin_id = {bin_id}")
                logging.debug(f"cfrq = {cfrq}")
                logging.debug(f"cvel = {cvel}")
                logging.debug(f"cslo = {cslo}")
                logging.debug(f"cwav = {cwav}")
                logging.debug(f"cwgt = {cwgt}")

                bin_wgt[cbin] = sum(cwgt)

                bin_frq_mean[cbin] = sum(cfrq*cwgt) / sum(cwgt)
                bin_vel_mean[cbin] = sum(cvel*cwgt) / sum(cwgt)
                bin_slo_mean[cbin] = sum(cslo*cwgt) / sum(cwgt)
                bin_wav_mean[cbin] = sum(cwav*cwgt) / sum(cwgt)

                if ccnt > 1:
                    vres = sum(cwgt * ((cvel-bin_vel_mean[cbin])**2))
                    bin_vel_std[cbin] = np.sqrt(
                        (ccnt*vres)/((ccnt-1)*sum(cwgt)))
                    sres = sum(cwgt * ((cslo-bin_slo_mean[cbin])**2))
                    bin_slo_std[cbin] = np.sqrt(
                        (ccnt*sres)/((ccnt-1)*sum(cwgt)))
                else:
                    bin_vel_std[cbin] = 0
                    bin_slo_std[cbin] = 0

        keep_ids = np.where(bin_wgt > 0)[0]
        bin_frq_mean = bin_frq_mean[keep_ids]
        bin_vel_mean = bin_vel_mean[keep_ids]
        bin_vel_std = bin_vel_std[keep_ids]
        bin_slo_mean = bin_slo_mean[keep_ids]
        bin_slo_std = bin_slo_std[keep_ids]
        bin_wav_mean = bin_wav_mean[keep_ids]
        bin_wgt = bin_wgt[keep_ids]
        # cov = bin_vel_mean / bin_vel_std

        # mean_disp = np.vstack(
        #     (freqMean, velMean, velStd, slowMean, slowStd, waveMean, binWeight, cov))
        # mean_disp = mean_disp.transpose()
        # # Remove rows corresponding to empty bins (meanFreq==0)
        # z_ids = np.where(mean_disp[:, 0] == 0)[0]
        # mean_disp = np.delete(mean_disp, z_ids, 0)
        # return mean_disp

        meandisp = {"mean": {"frq": bin_frq_mean,
                             "vel": bin_vel_mean,
                             "slo": bin_slo_mean,
                             "wav": bin_wav_mean},
                    "std": {"vel": bin_vel_std,
                            "slo": bin_slo_std,
                            }
                    }
        return meandisp

    @staticmethod
    def plot_dc_for_rmv(frequency, velocity, mean_disp, legend, marker_type=None, color_spec=None, xscaletype="log", klimits=None):
        """Function to plot dispersion data along with averages and standard deviations.
        (Note that the min(klimits) and max(klimits) curves are used for passive-source FK processing,
        thus, min(klimits) and max(klimits) are set equal to NaN for MASW testing to avoid plotting.)
        """

        n_off = len(velocity)

        if marker_type is None:
            marker_type = ['o']*n_off
        if color_spec is None:
            color_spec = plot_tools.makecolormap(n_off)

        minf = np.min(mean_disp["mean"]["frq"])
        maxf = np.max(mean_disp["mean"]["frq"])
        maxv, maxw = 0, 0
        for vl, fr in zip(velocity, frequency):
            if max(vl) > maxv:
                maxv = max(vl)
            if max(vl/fr) > maxw:
                maxw = max(vl/fr)

        if klimits:
            freq_klim = np.logspace(np.log10(minf), np.log10(maxf), 100)
            vel_klimf = np.vstack((2*np.pi*freq_klim/max(klimits), 2*np.pi*freq_klim /
                                   (max(klimits)/2), 2*np.pi*freq_klim/min(klimits), 2*np.pi*freq_klim/(min(klimits)/2)))
            vel_klimf = vel_klimf.transpose()
            if not(np.isnan(max(klimits))):
                for j in range(np.shape(vel_klimf)[1]):
                    rmvID = np.where(vel_klimf[:, j] > maxv)[0]
                    vel_klimf[rmvID, j] = float('nan')
            wave_lim = np.hstack((2*np.pi/max(klimits)*np.array([[1], [1]]), 2*np.pi/(max(klimits)/2)*np.array(
                [[1], [1]]), 2*np.pi/min(klimits)*np.array([[1], [1]]), 2*np.pi/(min(klimits)/2)*np.array([[1], [1]])))
            vel_klimW = np.array([0, maxv])

        mwdth = 10
        mhght = 6
        fsize = 11
        cfig, (axf, axw) = plt.subplots(
            nrows=1, ncols=2, figsize=(mwdth, mhght))

        for fr, vl, mk, co in zip(frequency, velocity, marker_type, color_spec):
            axf.plot(fr,
                     vl,
                     marker=mk,
                     markersize=5,
                     markeredgecolor=co,
                     markerfacecolor="none",
                     linestyle="none")
        axf.errorbar(mean_disp["mean"]["frq"],
                     mean_disp["mean"]["vel"],
                     mean_disp["std"]["vel"],
                     marker="o",
                     markersize=5,
                     color="k",
                     linestyle="none")

        if klimits:
            axf.plot(freq_klim, vel_klimf[:, 0],
                     linestyle=":", color='#000000')
            axf.plot(freq_klim, vel_klimf[:, 1],
                     linestyle="-", color='#000000')
            axf.plot(freq_klim, vel_klimf[:, 2],
                     linestyle="--", color='#000000')
            axf.plot(freq_klim, vel_klimf[:, 3],
                     linestyle="-.", color='#000000')
        axf.set_xlabel("Frequency (Hz)", fontsize=fsize, fontname="arial")
        axf.set_ylabel("Velocity (m/s)", fontsize=fsize, fontname="arial")
        axf.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        axf.set_xscale(xscaletype)

        for fr, vl, mk, co, le in zip(frequency, velocity, marker_type, color_spec, legend):
            axw.plot(vl/fr,
                     vl,
                     marker=mk,
                     markersize=5,
                     markeredgecolor=co,
                     markerfacecolor="none",
                     linestyle="none",
                     label=le)
        axw.errorbar(mean_disp["mean"]["wav"],
                     mean_disp["mean"]["vel"],
                     mean_disp["std"]["vel"],
                     marker="o",
                     markersize=5,
                     color="k",
                     linestyle="none")

        if klimits:
            axw.plot(wave_lim[:, 0], vel_klimW, linestyle=":",
                     color='#000000', label='kmax')
            axw.plot(wave_lim[:, 1], vel_klimW, linestyle="-",
                     color='#000000', label='kmax/2')
            axw.plot(wave_lim[:, 2], vel_klimW,
                     linestyle="--", color='#000000', label='kmin')
            axw.plot(wave_lim[:, 3], vel_klimW, linestyle="-.",
                     color='#000000', label='kmin/2')

        # handles, labels = axw.get_legend_handles_labels()
        # axw.legend(handles, labels, loc='upper left')
        axw.legend(loc="upper left")
        axw.set_xlabel("Wavelength (m)", fontsize=fsize, fontname="arial")
        axw.set_xscale(xscaletype)
        # axw.set_xticklabels(axw.get_xticks(), fontsize=fsize, fontname="arial")
        # axw.set_yticklabels(axw.get_yticks(), fontsize=fsize, fontname="arial")
        axw.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        cfig.show()
        axf.set_autoscale_on(False)
        axw.set_autoscale_on(False)
        return cfig

    @staticmethod
    def rmv_dc_points(frequency, velocity, wavelength, offset, cfig, extras=None):
        """Function to prompt user to draw a box on a dispersion curve 
        figure. Data points inside of the box are removed and data 
        points outside of the box are kept.
        """
        while True:
            axclick = []

            def on_click(event):
                if event.inaxes is not None:
                    if len(axclick) < 2:
                        axclick.append(event.inaxes)
            cid = cfig.canvas.mpl_connect('button_press_event', on_click)

            axf = cfig.axes[0]
            axw = cfig.axes[1]

            fcursor = Cursor(axf, useblit=True, color='k', linewidth=1)
            wcursor = Cursor(axw, useblit=True, color='k', linewidth=1)

            rawBounds = np.asarray(cfig.ginput(2, timeout=0))
            cfig.canvas.mpl_disconnect(cid)
            xmin = np.min(rawBounds[:, 0])
            xmax = np.max(rawBounds[:, 0])
            ymin = np.min(rawBounds[:, 1])
            ymax = np.max(rawBounds[:, 1])

            n_rmv = 0
            # for g in range(len(frequency)):
            for bin_id, (f, v, w) in enumerate(zip(frequency, velocity, wavelength)):
                # Create arrays to store indices of data that will be kept and removed
                rmv_id = np.zeros(len(f), int)
                keep_id = np.zeros(len(f), int)

                # If user clicked on two different axes, warn user and return
                if axclick[0] != axclick[1]:
                    logging.warning("BOTH CLICKS MUST BE ON SAME AXIS")
                    return

                for i in range(len(f)):
                    condition1 = (axclick[0] == axf) and (
                        xmin < f[i] and f[i] < xmax and ymin < v[i] and v[i] < ymax)
                    condition2 = (axclick[0] == axw) and (
                        xmin < w[i] and w[i] < xmax and ymin < v[i] and v[i] < ymax)
                    # Points inside of selectd box are removed
                    if condition1 or condition2:
                        rmv_id[i] = i+1
                    # Points outside of selected box are kept
                    else:
                        keep_id[i] = i+1

                # Remove zeros from rmv_id and keep_id
                zid = np.where(rmv_id == 0)[0]
                rmv_id = np.delete(rmv_id, zid, 0)
                zid = np.where(keep_id == 0)[0]
                keep_id = np.delete(keep_id, zid, 0)

                frmv = f[(rmv_id-1)]
                vrmv = v[(rmv_id-1)]
                wrmv = w[(rmv_id-1)]
                n_rmv += len(vrmv)

                axf.plot(frmv, vrmv,
                         marker="x", color="k", markersize=5, linestyle="none")
                axw.plot(wrmv, vrmv,
                         marker="x", color="k", markersize=5, linestyle="none")
                cfig.canvas.draw_idle()

                logging.debug(f"f = {f}")
                logging.debug(f"v = {v}")
                logging.debug(f"w = {w}")

                frequency[bin_id] = f[(keep_id-1)]
                velocity[bin_id] = v[(keep_id-1)]
                wavelength[bin_id] = w[(keep_id-1)]

                logging.debug(f"f = {f}")
                logging.debug(f"v = {v}")
                logging.debug(f"w = {w}")
                logging.debug(f"extras = {extras}")

                for key, value in extras.items():
                    logging.debug(f"key = {key}")
                    logging.debug(f"value = {value}")
                    logging.debug(f"bin_id = {bin_id}")
                    logging.debug(f"keep_id-1 = {keep_id-1}")

                    extras[key][bin_id] = value[bin_id][(keep_id-1)]

            if n_rmv == 0:
                break

            del cid
