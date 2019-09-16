"""File defining the Peaks class that allows for read and modifying
peak values from the dispersion data."""

from utprocess import plot_tools
import re
import logging
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib as mpl
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.use('Qt4Agg')
logger = logging.getLogger(__name__)


class Peaks():
    """Class for handling peak dispersion data.

    Attributes:
        vel = List of np.arrays with velocity values (one per peak).
        frq = List of np.arrays with frequency values (one per peak).
        wav = List of np.arrays with wavelength values (one per peak).
        ids = List of strings to uniquely identity each array.
    """

    @staticmethod
    def __check_inputs(freqs, vels, ids):
        if isinstance(ids, str):
            ids = [ids]

        if isinstance(freqs, list) and (isinstance(freqs[0], float) or isinstance(freqs[0], int)):
            freqs = [np.array(freqs)]
            vels = [np.array(vels)]
        elif isinstance(freqs, list) and isinstance(freqs[0], list):
            freq_list, vel_list = freqs, vels
            freqs, vels = [], []
            for frq, vel in zip(freq_list, vel_list):
                freqs.append(np.array(frq))
                vels.append(np.array(vel))
        elif isinstance(freqs, list) and isinstance(freqs[0], np.ndarray):
            pass
        else:
            raise NotImplementedError("Unknown input type")
        return (freqs, vels, ids)

    def __init__(self, frequency_list, velocity_list, identifiers):
        """Initialize an instance of Peaks.

        Args:
            frequency_list = List of lists with frequency values
                (one per peaks).
            velocity_list = List of lists with velocity values (one
                per peak)
            identifiers = List of strings to uniquely identify each
                array.
        Returns:
            Instataiated Peaks object.

        Raises:
            This method raises no exceptions.
        """

        freqs, vels, self.ids = Peaks.__check_inputs(frequency_list,
                                                     velocity_list,
                                                     identifiers)
        logging.debug(freqs)
        logging.debug(vels)
        logging.debug(self.ids)
        self.frq, self.vel, self.wav = [], [], []
        for freq_in, vel_in in zip(freqs, vels):
            self.vel += [(vel_in[np.where(freq_in != 0)])]
            self.frq += [(freq_in[np.where(freq_in != 0)])]
            self.wav += [self.vel[-1]/self.frq[-1]]

        # TODO (jpv) remove this hack and fix mixed state issue
        # call calc mean disp once, to get a "intitial value".
        self.meanDisp = False

    @classmethod
    def from_peak_data_dicts(cls, peak_data_dicts):
        """Alternate constructor to initialize an instance of the Peaks
        class.

        Args:
            peak_data_dicts: List of dictionaries or a single dictionary
                of the form {"array_id":{"frequency":freq,
                "velocity":vel}} where:
                    array_id is a string denoting some unique info about
                        the array.
                    freq is a list of floats denoting frequency values.
                    vel is a list of floats denoting velocity values.
                TODO (jpv) Add the optional arguements.

        Returns:
            Initialized Peaks instance.

        Raises:
            This method raises no exceptions.
        """
        if isinstance(peak_data_dicts, dict):
            peak_data_dicts = [peak_data_dicts]

        ids, frq, vel = [], [], []
        for peak_data_dict in peak_data_dicts:
            for key, val in peak_data_dict.items():
                ids.append(key)
                frq.append(np.array(val["frequency"]))
                vel.append(np.array(val["velocity"]))
        return cls(frq, vel, ids)

    @classmethod
    def from_json(cls, fname):
        with open(fname, "r") as f:
            data = json.load(f)

        frequency, velocity, offset = [], [], []
        for key, value in data.items():
            # frq = np.array(value["frequency"])
            # vel = np.array(value["velocity"])
            # frequency += [frq[np.where(vel<max_vel)]]
            # velocity += [vel[np.where(vel<max_vel)]]
            frequency.append(np.array(value["frequency"]))
            velocity.append(np.array(value["velocity"]))
            offset.append(key)
        return cls(frequency, velocity, offset)

    @classmethod
    def from_hfk_historical(cls, fname):
        pass

    @classmethod
    def from_hfks(cls, fnames, array_names, rayleigh=True, love=False):
        """Alternate constructor for PeaksPassive object.

        Args:
            fname = String or list of string to denote the filename(s)
                for the .max. Can be with respect to the relative or
                full path.

            array_name = String or list of strings unqiuely identifying
                the dispersion data.

            rayleigh = Boolean to denote if rayleigh data should be
                extracted.

            love = Boolean to denote if love data should be extracted.

        Returns:
            Initialized PeaksPassive object.

        Raises:
            ValueError if both rayleigh and love are true or false.

        """
        if rayleigh and love:
            raise ValueError("`rayleigh` and `love` cannot both be True.")

        if not rayleigh and not love:
            raise ValueError("`rayleigh` and `love` cannot both be False.")

        if isinstance(fnames, str):
            fnames = [fnames]

        if isinstance(array_names, str):
            array_names = [array_names]

        disp_type = "Rayleigh" if rayleigh else "Love"
        pattern = r"^\d+\.?\d* (\d+\.?\d*) (Rayleigh|Love) (\d+\.?\d*) (\d+\.?\d*) (-?\d+\.?\d*) (\d+\.?\d*|-?inf|nan) (\d+\.?\d*) (0|1)$"
        frequency_list, velocity_list, azimuth_list, ell_list, noise_list, power_list = [
        ], [], [], [], [], []
        for fname in fnames:
            logging.debug(f"Attempting to Open File: {fname}")
            with open(fname, "r") as f:
                lines = f.read().splitlines()

            for line_number, line in enumerate(lines):
                if line.startswith("# BEGIN DATA"):
                    start_line = line_number + 3
                    break

            frqs, vels, azis, ells, nois, pwrs = [], [], [], [], [], []
            for line_number, line in enumerate(lines[start_line:]):
                # logging.debug(line)
                # logging.debug(re.findall(pattern, line)[0])
                frq, pol, slo, azi, ell, noi, pwr, valid = re.findall(pattern, line)[
                    0]
                if pol == disp_type and valid == "1":
                    frqs.append(float(frq))
                    vels.append(1/float(slo))
                    azis.append(float(azi))
                    ells.append(float(ell))
                    nois.append(float(noi))
                    pwrs.append(float(pwr))
                elif pol != disp_type:
                    continue
                elif valid == "0":
                    logging.warn(
                        f"Invalid point found! Line #{line_number+start_line+1}")
                else:
                    logging.debug(pol)
                    logging.debug(valid)
                    raise ValueError("Check line")

            frequency_list.append(frqs)
            velocity_list.append(vels)
            azimuth_list.append(azis)
            ell_list.append(ells)
            noise_list.append(nois)
            power_list.append(pwrs)

        return cls(frequency_list, velocity_list, array_names)

    def write_to_txt_dinver(self, fname):
        pass

    def write_stat_utinvert(self, fname):
        """Write statistics (mean and standard deviation) to csv file
        of the form accepted by utinvert.

        Args:
            fname = String for file name. Can be a relative or a full
                path.

        Returns:
            Method returns None, but saves file to disk.

        Raises:
            ValueError if part_time has not been run.
        """
        if isinstance(self.meanDisp, np.ndarray):
            if fname.endswith(".csv"):
                fname = fname[:-4]
            with open(fname+".csv", "w") as f:
                f.write("Frequency (Hz), Velocity (m/s), VelStd (m/s)\n")
                # for f_set, v_set, s_set in zip(self.meanDisp[:,0], self.meanDisp[:,1], self.meanDisp[:,2]):
                for fr, ve, st in zip(self.meanDisp[:, 0], self.meanDisp[:, 1], self.meanDisp[:, 2]):
                    f.write(f"{fr},{ve},{st}\n")
        else:
            raise ValueError("party_time not run self.meanDisp not defined.")

    def write_peak_json(self, fname):
        """Write peak dispersion points to json file.

        Args:
            fname = String for file name. Can be a relative or a full
                path.

        Returns:
            Method returns None, but saves file to disk.

        Raises:
            This method raises no exceptions.
        """
        if fname.endswith(".json"):
            fname = fname[:-4]

        data = {}
        for f, v, label in zip(self.frq, self.vel, self.ids):
            data.update({label: {"frequency": f.tolist(),
                                 "velocity": v.tolist()}})
        with open(fname+".json", "w") as f:
            json.dump(data, f)

    def plot(self, xtype="frequency"):
        """Create plot of dispersion data.

        Args:
            xtype = String denoting either frequency or velocity.

        Returns:
            Tuple of the form (fig, ax) where fig is the figure handle
            and ax is the axes handles.

        Raises:
            This method raises no exceptions.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        if xtype.lower() == "wavelength":
            x = self.wav
            xlabel = "Wavelength (m)"
        else:
            x = self.frq
            xlabel = "Frequency (Hz)"
        for x, v, color, label in zip(x, self.vel, colors, self.ids):
            ax.plot(x, v, color=color, linestyle="none",
                    marker="o", markerfacecolor="none", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Phase Velocity (m/s)")
        ax.set_xscale("log")
        plt.legend()
        return (fig, ax)

    def plot_2pannel(self):
        fig, (axf, axw) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))
        for f, v, w, i, color in zip(self.frq, self.vel, self.wav, self.ids, colors):
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

    def party_time(self, settings_file):
        with open(settings_file, "r") as f:
            settings = json.load(f)

        cont = True
        cfig = 0
        while cont:
            if cfig:
                plt.close(cfig)
            self.meanDisp = self.computeDCstats(self.frq,
                                                self.vel,
                                                minP=settings["minval"],
                                                maxP=settings["maxval"],
                                                numbins=settings["nbins"],
                                                binScale=settings["binscale"],
                                                binType=settings["bintype"])
            cfig = self.plotDCforRmv(self.frq,
                                     self.vel,
                                     self.meanDisp,
                                     self.ids)
            self.rmvDCpoints(self.frq,
                             self.vel,
                             self.wav,
                             self.ids, cfig)

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
    def computeDCstats(frequency, velocity, minP=5, maxP=100, numbins=96, binScale="linear", binType="frequency", arrayWeights=[]):

        # Combine all dispersion data into single vector for freq., vel., and wavel.
        vel = velocity[0]
        fr = frequency[0]
        if len(arrayWeights) != 0:
            wt = arrayWeights[0]*np.ones(len(velocity[0]))
        if len(vel) > 1:
            for p in range(1, len(velocity)):
                vel = np.concatenate((vel, velocity[p]))
                fr = np.concatenate((fr, frequency[p]))
                if len(arrayWeights) != 0:
                    wt = np.concatenate(
                        (wt, arrayWeights[p]*np.ones(len(velocity[p]))))
        wl = vel / fr

        # Bin edges
        if str.lower(binScale) == "linear":
            binEdges = np.linspace(minP, maxP, numbins+1)
        elif str.lower(binScale) == "log":
            binEdges = np.logspace(np.log10(minP), np.log10(maxP), numbins+1)
        else:
            raise ValueError("Invalid binScale")

        # Determine how many frequencies or wavelengths falls into each bin
        if str.lower(binType) == "frequency":
            whichBin = np.digitize(fr, binEdges)
        elif str.lower(binType) == "wavelength":
            whichBin = np.digitize(wl, binEdges)
        else:
            raise ValueError("Invalid binType")

        # Initialize variables
        weightPoints = np.zeros(numbins)
        binWeight = np.zeros(numbins)
        velMean = np.zeros(numbins)
        velStd = np.zeros(numbins)
        slowMean = np.zeros(numbins)
        slowStd = np.zeros(numbins)
        freqMean = np.zeros(numbins)
        waveMean = np.zeros(numbins)

        # Compute statistics for each bin
        for g in range(numbins):
            # Flag points in current bin
            flagPoints = np.where(whichBin == (g+1))[0]
            freqPoints = fr[flagPoints]
            wavePoints = wl[flagPoints]
            velPoints = vel[flagPoints]
            slowPoints = 1/velPoints

            # Compute averages and standard deviations
            # Set values equal to NaN if no points fall within current bin
            # Weighted calculations
            if len(flagPoints) != 0:
                if len(arrayWeights) != 0:
                    weightPoints[g] = wt[flagPoints]
                    binWeight[g] = sum(weightPoints)
                    velMean[g] = float(
                        sum(velPoints*weightPoints)) / sum(weightPoints)
                    velStd[g] = np.sqrt(
                        1.0/sum(weightPoints) * sum(((velPoints-velMean[g])**2) * weightPoints))
                    slowMean[g] = float(
                        sum(slowPoints*weightPoints)) / sum(weightPoints)
                    slowStd[g] = np.sqrt(
                        1.0/sum(weightPoints) * sum(((slowPoints-slowMean[g])**2) * weightPoints))
                    freqMean[g] = float(
                        sum(freqPoints*weightPoints)) / sum(weightPoints)
                    waveMean[g] = float(
                        sum(wavePoints*weightPoints)) / sum(weightPoints)
                # Unweighted calculations
                # (use unbiased sample standard deviation, with a normalization of 1/(n-1) or a ddof=1 )
                else:
                    binWeight[g] = len(velPoints)
                    velMean[g] = np.average(velPoints)
                    slowMean[g] = np.average(slowPoints)
                    if binWeight[g] > 1:
                        velStd[g] = np.std(velPoints, ddof=1)
                        slowStd[g] = np.std(slowPoints, ddof=1)
                    freqMean[g] = np.average(freqPoints)
                    waveMean[g] = np.average(wavePoints)

        # Remove zeros
        ids = np.where(freqMean > 0)[0]
        freqMean = freqMean[ids]
        velMean = velMean[ids]
        velStd = velStd[ids]
        slowMean = slowMean[ids]
        slowStd = slowStd[ids]
        waveMean = waveMean[ids]
        binWeight = binWeight[ids]

        # Calculate the coefficient of variation
        cov = velStd / velMean

        # Combine results into a single matrix
        meanDisp = np.vstack(
            (freqMean, velMean, velStd, slowMean, slowStd, waveMean, binWeight, cov))
        meanDisp = meanDisp.transpose()
        # Remove rows corresponding to empty bins (meanFreq==0)
        z_ids = np.where(meanDisp[:, 0] == 0)[0]
        meanDisp = np.delete(meanDisp, z_ids, 0)
        return meanDisp

    @staticmethod
    def plotDCforRmv(frequency, velocity, meanDisp, setLeg, markType=[], colorSpec=[], xScaleType="log", klimits=()):
        """Function to plot dispersion data along with averages and standard deviations.
        (Note that the min(klimits) and max(klimits) curves are used for passive-source FK processing,
        thus, min(klimits) and max(klimits) are set equal to NaN for MASW testing to avoid plotting.)

        """

        n_off = len(velocity)

        # Default markers and colors
        if not markType:
            markType = ['o']*n_off
        if not colorSpec:
            colorSpec = plot_tools.makecolormap(n_off)

        # Width and height (in) of plots
        mwdth = 10
        mhght = 6
        fsize = 11
        cfig = plt.figure(figsize=(mwdth, mhght))

        # Curves for min(klimits) and max(klimits) (if min(klimits) and max(klimits) are provided)
        minF = np.min(meanDisp[:, 0])
        maxF = np.max(meanDisp[:, 0])
        maxV = 0
        maxW = 0
        for k in range(n_off):
            if max(velocity[k]) > maxV:
                maxV = max(velocity[k])
            if max(velocity[k]/frequency[k]) > maxW:
                maxW = max(velocity[k]/frequency[k])

        if klimits:
            # min(klimits) and max(klimits) curves for frequency vs velocity
            freq_klim = np.logspace(np.log10(minF), np.log10(maxF), 100)
            vel_klimF = np.vstack((2*np.pi*freq_klim/max(klimits), 2*np.pi*freq_klim /
                                   (max(klimits)/2), 2*np.pi*freq_klim/min(klimits), 2*np.pi*freq_klim/(min(klimits)/2)))
            vel_klimF = vel_klimF.transpose()
            # Don't plot higher than maximum velocity of dispersion data
            if not(np.isnan(max(klimits))):
                for j in range(np.shape(vel_klimF)[1]):
                    rmvID = np.where(vel_klimF[:, j] > maxV)[0]
                    vel_klimF[rmvID, j] = float('nan')
            # min(klimits) and max(klimits) curves for wavelength vs velocity
            wave_lim = np.hstack((2*np.pi/max(klimits)*np.array([[1], [1]]), 2*np.pi/(max(klimits)/2)*np.array(
                [[1], [1]]), 2*np.pi/min(klimits)*np.array([[1], [1]]), 2*np.pi/(min(klimits)/2)*np.array([[1], [1]])))
            vel_klimW = np.array([0, maxV])

        # Velocity vs frequency plot
        axf = cfig.add_subplot(1, 2, 1)
        for r in range(len(velocity)):
            axf.plot(frequency[r],
                     velocity[r],
                     marker=markType[r],
                     markersize=5,
                     markeredgecolor=colorSpec[r],
                     markerfacecolor="none",
                     linestyle="none")
        axf.errorbar(meanDisp[:, 0],
                     meanDisp[:, 1],
                     meanDisp[:, 2],
                     marker="o",
                     markersize=5,
                     color="k",
                     linestyle="none")
        # min(klimits) and max(klimits) lines
        if klimits:
            axf.plot(freq_klim, vel_klimF[:, 0], linestyle=":")
            axf.plot(freq_klim, vel_klimF[:, 1], linestyle="-")
            axf.plot(freq_klim, vel_klimF[:, 2], linestyle="--")
            axf.plot(freq_klim, vel_klimF[:, 3], linestyle="-.")
        axf.set_xlabel("Frequency (Hz)", fontsize=fsize, fontname="arial")
        axf.set_ylabel("Velocity (m/s)", fontsize=fsize, fontname="arial")
        # axf.set_xticklabels(axf.get_xticks(), fontsize=fsize, fontname="arial")
        # axf.set_yticklabels(axf.get_yticks(), fontsize=fsize, fontname="arial")
        axf.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        axf.set_xscale(xScaleType)

        # Velocity vs wavelength
        axw = cfig.add_subplot(1, 2, 2)

        # Raw data and error bars
        for r in range(len(velocity)):
            axw.plot(velocity[r]/frequency[r], velocity[r], marker=markType[r], markersize=5,
                     markeredgecolor=colorSpec[r], markerfacecolor="none", linestyle="none", label=setLeg[r])
        axw.errorbar(meanDisp[:, 5], meanDisp[:, 1], meanDisp[:, 2],
                     marker="o", markersize=5, color="k", linestyle="none")

        if klimits:
            axw.plot(wave_lim[:, 0], vel_klimW, linestyle=":", label='kmax')
            axw.plot(wave_lim[:, 1], vel_klimW, linestyle="-", label='kmax/2')
            axw.plot(wave_lim[:, 2], vel_klimW, linestyle="--", label='kmin')
            axw.plot(wave_lim[:, 3], vel_klimW, linestyle="-.", label='kmin/2')

        handles, labels = axw.get_legend_handles_labels()
        axw.legend(handles, labels, loc='upper left')
        axw.set_xlabel("Wavelength (m)", fontsize=fsize, fontname="arial")
        axw.set_xscale(xScaleType)
        # axw.set_xticklabels(axw.get_xticks(), fontsize=fsize, fontname="arial")
        # axw.set_yticklabels(axw.get_yticks(), fontsize=fsize, fontname="arial")
        axw.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        cfig.show()
        axf.set_autoscale_on(False)
        axw.set_autoscale_on(False)
        return cfig

    @staticmethod
    def rmvDCpoints(frequency, velocity, wavelength, offset, cfig):
        """Function to prompt user to draw a box on a dispersion curve figure. Data points 
        inside of the box are removed and data points outside of the box are kept.
        """
        while True:
            axclick = []

            def on_click(event):
                if event.inaxes is not None:
                    if len(axclick) < 2:
                        axclick.append(event.inaxes)
            cid = cfig.canvas.mpl_connect('button_press_event', on_click)

            rawBounds = np.asarray(cfig.ginput(2, timeout=0))
            cfig.canvas.mpl_disconnect(cid)
            xmin = np.min(rawBounds[:, 0])
            xmax = np.max(rawBounds[:, 0])
            ymin = np.min(rawBounds[:, 1])
            ymax = np.max(rawBounds[:, 1])

            n_rmv = 0
            for g in range(len(frequency)):
                # Arrays containing data for current offset
                f = frequency[g]
                v = velocity[g]
                w = wavelength[g]
                # Create arrays to store indices of data that will be kept and removed
                rmv_id = np.zeros(len(f), int)
                keep_id = np.zeros(len(f), int)

                # If user clicked on two different axes, warn user and return
                if axclick[0] != axclick[1]:
                    logging.warning("BOTH CLICKS MUST BE ON SAME AXIS")
                    return

                axf = cfig.axes[0]
                axw = cfig.axes[1]
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

                # Removed data
                frmv = f[(rmv_id-1)]
                vrmv = v[(rmv_id-1)]
                wrmv = w[(rmv_id-1)]
                n_rmv += len(vrmv)
                # Plot deleted data with black x's
                axf.plot(frmv, vrmv, marker="x", color="k",
                         markersize=5, linestyle="none")
                axw.plot(wrmv, vrmv, marker="x", color="k",
                         markersize=5, linestyle="none")
                cfig.canvas.draw_idle()

                # Retained data
                fnew = f[(keep_id-1)]
                vnew = v[(keep_id-1)]
                wnew = w[(keep_id-1)]
                # Revise velocity, frequency, and wavelength cell arrays
                velocity[g] = vnew
                frequency[g] = fnew
                wavelength[g] = wnew

            if n_rmv == 0:
                break

            del cid
