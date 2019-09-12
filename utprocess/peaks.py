"""File defining the Peaks class that allows for read and modifying
peak values from the dispersion data."""

import logging
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from utprocess import plot_tools
mpl.use('Qt4Agg')
logger = logging.getLogger(__name__)


class Peaks():
    """Class for handling peak dispersion data."""

    def __init__(self, frequency, velocity, identifier):
        self.frq, self.vel, self.wav, self.std = [], [], [], []
        for freq_in, vel_in in zip(frequency, velocity):
            self.vel += [(vel_in[np.where(freq_in != 0)])]
            self.frq += [(freq_in[np.where(freq_in != 0)])]
            self.wav += [self.vel[-1]/self.frq[-1]]
        self.ids = identifier

    def write_to_txt_dinver(self, fname):
        pass

    def write_to_csv_utinvert(self, fname):
        with open(fname, "w") as f:
            f.write("Frequency (Hz), Velocity (m/s), VelStd (m/s)")
            for f_set, v_set, s_set in zip(self.frq, self.vel, self.std):
                for fr, ve, st in zip(f_set, v_set, s_set):
                    f.write(f"{fr},{ve},{st}")

    def write_to_json(self, fname):
        pass

    def party_time(self):
        cont = True
        cfig = 0
        while cont:
            if cfig:
                plt.close(cfig)
            meanDisp = self.computeDCstats(self.frq, self.vel)
            cfig = self.plotDCforRmv(self.frq, self.vel, meanDisp, self.ids)
            self.rmvDCpoints(self.frq, self.vel, self.wav, self.ids, cfig)

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
            cont = int(input("Enter 1 to continue, 0 to quit: "))


# # Create files with final processed data ***************************************
# # Compressed pickle file containing dispersion data from each offset
# f = gzip.open(outfile_path+"/"+outfile_name+".pklz", 'wb')
# pickle.dump(rawDC, f)
# f.close()
# # Text file containing frequency, slowness, slow std., and weight
# # (used in dinver software)
# rows = np.array([0, 3, 4, 6])
# np.savetxt(outfile_path+'/'+outfile_name+'.txt',
#            meanDisp[:, rows], fmt='%10.3f    %10.8f    %10.8f    %10.4f')

#   # Function to bin dispersion data from one or more arrays or offsets and compute
#   # statistics for each bin.

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


# Function to plot dispersion data along with averages and standard deviations.
# (Note that the min(klimits) and max(klimits) curves are used for passive-source FK processing,
# thus, min(klimits) and max(klimits) are set equal to NaN for MASW testing to avoid plotting.)
    @staticmethod
    def plotDCforRmv(frequency, velocity, meanDisp, setLeg, markType=[], colorSpec=[], xScaleType="log", klimits=()):

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
        axf.set_xticklabels(axf.get_xticks(), fontsize=fsize, fontname="arial")
        axf.set_yticklabels(axf.get_yticks(), fontsize=fsize, fontname="arial")
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
        axw.set_xticklabels(axw.get_xticks(), fontsize=fsize, fontname="arial")
        axw.set_yticklabels(axw.get_yticks(), fontsize=fsize, fontname="arial")
        axw.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        cfig.show()
        axf.set_autoscale_on(False)
        axw.set_autoscale_on(False)
        return cfig

    # Function to prompt user to draw a box on a dispersion curve figure. Data points
    # inside of the box are removed and data points outside of the box are kept.
    @staticmethod
    def rmvDCpoints(frequency, velocity, wavelength, offset, cfig):
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
                    logging.warning(
                        "WARNING: BOTH CLICKS MUST BE ON SAME AXIS")
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
