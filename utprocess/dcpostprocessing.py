
"""
    The functions contained in this file are used to post-process previously 
    computed raw experimental dispersion data. 

    Functions:

        computeDCstats:
        Bin dispersion data and compute statistics (e.g., mean and std. dev.).      
            
        plotDCforRmv: 
        Plot dispersion data for various source-offsets along with mean and std.
        deviation dispersion data.   
        
        rmvDCpoints:
        This program is used to manually remove "bad" dispersion data (e.g., 
        offline-noise, near-field effects, higher-modes, etc.).    


    This code was developed at the University of Texas at Austin.
    Copyright (C) 2016  David P. Teague, Clinton M. Wood, and Brady R. Cox 
 
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
 
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
 
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


# Specify backend for interactive plotting
import matplotlib as mpl
mpl.use('Qt4Agg')
# Import modules
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import dctypes
import shotgathers


# Width and height (in) of plots
mwdth = 10
mhght = 6
fsize = 11


# Function to bin dispersion data from one or more arrays or offsets and compute
# statistics for each bin.
def computeDCstats( rawDC, minP=5, maxP=100, numbins=96, binScale="linear", binType="frequency", arrayWeights=[] ):
    # Combine all dispersion data into single vector for freq., vel., and wavel.
    vel = rawDC.velocity[0]
    fr = rawDC.frequency[0]
    if len(arrayWeights)!=0:
        wt = arrayWeights[0]*np.ones(len(rawDC.velocity[0]))
    if len(vel)>1:
        for p in range( 1, len(rawDC.velocity) ):
            vel = np.concatenate((vel, rawDC.velocity[p]))
            fr = np.concatenate((fr, rawDC.frequency[p]))
            if len(arrayWeights)!=0:
                wt = np.concatenate((wt, arrayWeights[p]*np.ones(len(rawDC.velocity[p]))))
    wl = vel / fr
    
    # Bin edges
    if str.lower(binScale)=="linear":
        binEdges = np.linspace( minP, maxP, numbins+1 )
    elif str.lower(binScale)=="log":   
        binEdges = np.logspace( np.log10(minP), np.log10(maxP), numbins+1 )
    else:
        raise ValueError("Invalid binScale")
    
    # Determine how many frequencies or wavelengths falls into each bin
    if str.lower(binType)=="frequency":
        whichBin = np.digitize( fr, binEdges )
    elif str.lower(binType)=="wavelength":
        whichBin = np.digitize( wl, binEdges )
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
    for g in range( numbins ):
        # Flag points in current bin
        flagPoints = np.where( whichBin == (g+1) )[0]
        freqPoints = fr[flagPoints]
        wavePoints = wl[flagPoints]
        velPoints = vel[flagPoints]
        slowPoints = 1/velPoints
        
        # Compute averages and standard deviations
        # Set values equal to NaN if no points fall within current bin
        # Weighted calculations
        if len(flagPoints)!=0:
            if len(arrayWeights)!=0:
                weightPoints[g] = wt[flagPoints]
                binWeight[g] = sum(weightPoints)
                velMean[g] = float(sum(velPoints*weightPoints)) / sum(weightPoints)
                velStd[g] = np.sqrt( 1.0/sum(weightPoints) * sum( ((velPoints-velMean[g])**2) *weightPoints ) )
                slowMean[g] = float(sum(slowPoints*weightPoints)) / sum(weightPoints)
                slowStd[g] = np.sqrt( 1.0/sum(weightPoints) * sum( ((slowPoints-slowMean[g])**2) *weightPoints ) )
                freqMean[g] = float(sum(freqPoints*weightPoints)) / sum(weightPoints)
                waveMean[g] = float(sum(wavePoints*weightPoints)) / sum(weightPoints)
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
    ids=np.where( freqMean>0 )[0] 
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
    meanDisp = np.vstack((freqMean, velMean, velStd, slowMean, slowStd, waveMean, binWeight, cov))
    meanDisp = meanDisp.transpose()
    # Remove rows corresponding to empty bins (meanFreq==0)
    z_ids=np.where( meanDisp[:,0]==0 )[0]
    meanDisp = np.delete(meanDisp, z_ids, 0)
    return meanDisp
  

# Function to plot dispersion data along with averages and standard deviations.
# (Note that the kmin and kmax curves are used for passive-source FK processing,
# thus, kmin and kmax are set equal to NaN for MASW testing to avoid plotting.)
def plotDCforRmv( rawDC, meanDisp, setLeg, markType=[], colorSpec=[], xScaleType="log", kmin=float('nan'), kmax=float('nan') ):

    n_off = len(rawDC.velocity)

    # Default markers and colors
    if not markType:
        markType = ['o']*n_off
    if not colorSpec:
        colorSpec = shotgathers.create_ColorMap( n_off )
    
    # Set figure size equal to 2/3 screen size
    # Get screen size in mm and convert to in (25.4 mm per inch)
    #root = tk.Tk()
    #width = root.winfo_screenmmwidth() / 25.4 * 0.66
    #height = root.winfo_screenmmheight() / 25.4 * 0.66
    #cfig = plt.figure( figsize=(width,height) ) 
    cfig = plt.figure( figsize=(mwdth,mhght) ) 

    # Curves for kmin and kmax (if kmin and kmax are provided)
    minF = np.min(meanDisp[:,0])
    maxF = np.max(meanDisp[:,0])
    maxV = 0
    maxW = 0
    for k in range( n_off ):
        if max(rawDC.velocity[k]) > maxV:
            maxV = max(rawDC.velocity[k])
        if max(rawDC.velocity[k]/rawDC.frequency[k]) > maxW:
            maxW = max(rawDC.velocity[k]/rawDC.frequency[k])
    
    # kmin and kmax curves for frequency vs velocity
    freq_klim = np.logspace( np.log10(minF), np.log10(maxF), 100 )
    vel_klimF = np.vstack( ( 2*np.pi*freq_klim/kmax, 2*np.pi*freq_klim/(kmax/2), 2*np.pi*freq_klim/kmin, 2*np.pi*freq_klim/(kmin/2) ) )
    vel_klimF = vel_klimF.transpose()
    # Don't plot higher than maximum velocity of dispersion data
    if not(np.isnan(kmax)):
        for j in range( np.shape(vel_klimF)[1] ):
            rmvID = np.where( vel_klimF[:,j] > maxV )[0]
            vel_klimF[rmvID,j] = float('nan')    
    # kmin and kmax curves for wavelength vs velocity
    wave_lim = np.hstack( ( 2*np.pi/kmax*np.array([[1],[1]]), 2*np.pi/(kmax/2)*np.array([[1],[1]]), 2*np.pi/kmin*np.array([[1],[1]]), 2*np.pi/(kmin/2)*np.array([[1],[1]]) ) )
    vel_klimW = np.array([0,maxV])
    
    
    # Velocity vs frequency plot
    axf = cfig.add_subplot(1,2,1)
    for r in range( len(rawDC.velocity) ):
        axf.plot( rawDC.frequency[r], rawDC.velocity[r], marker=markType[r], markersize=5, markeredgecolor=colorSpec[r], markerfacecolor="none", linestyle="none" )
    axf.errorbar( meanDisp[:,0], meanDisp[:,1], meanDisp[:,2], marker="o", markersize=5, color="k", linestyle="none" )
    # kmin and kmax lines
    if not np.isnan(kmin):
        axf.plot( freq_klim, vel_klimF[:,0], linestyle=":" )
        axf.plot( freq_klim, vel_klimF[:,1], linestyle="-" )
        axf.plot( freq_klim, vel_klimF[:,2], linestyle="--" )
        axf.plot( freq_klim, vel_klimF[:,3], linestyle="-." )
    axf.set_xlabel( "Frequency (Hz)", fontsize=fsize, fontname="arial" )
    axf.set_ylabel( "Velocity (m/s)", fontsize=fsize, fontname="arial" ) 
    axf.set_xticklabels(axf.get_xticks(), fontsize=fsize, fontname="arial" )
    axf.set_yticklabels(axf.get_yticks(), fontsize=fsize, fontname="arial" )
    axf.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    axf.set_xscale( xScaleType )


    # Velocity vs wavelength
    axw = cfig.add_subplot(1,2,2)    
    # Raw data and error bars
    for r in range( len(rawDC.velocity) ):
        axw.plot( rawDC.velocity[r]/rawDC.frequency[r], rawDC.velocity[r], marker=markType[r], markersize=5, markeredgecolor=colorSpec[r], markerfacecolor="none", linestyle="none", label=setLeg[r] )
    axw.errorbar( meanDisp[:,5], meanDisp[:,1], meanDisp[:,2], marker="o", markersize=5, color="k", linestyle="none" )
    # kmin and kmax lines
    if not np.isnan(kmin):
        axw.plot( wave_lim[:,0], vel_klimW, linestyle=":", label='kmax' )
        axw.plot( wave_lim[:,1], vel_klimW, linestyle="-", label='kmax/2' )
        axw.plot( wave_lim[:,2], vel_klimW, linestyle="--", label='kmin' )
        axw.plot( wave_lim[:,3], vel_klimW, linestyle="-.", label='kmin/2' )
    handles, labels = axw.get_legend_handles_labels()
    axw.legend(handles, labels, loc='upper left')
    axw.set_xlabel( "Wavelength (m)", fontsize=fsize, fontname="arial" )
    axw.set_xscale( xScaleType )
    axw.set_xticklabels(axw.get_xticks(), fontsize=fsize, fontname="arial" )
    axw.set_yticklabels(axw.get_yticks(), fontsize=fsize, fontname="arial" ) 
    axw.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    cfig.show()

    axf.set_autoscale_on(False)
    axw.set_autoscale_on(False)

    # Return figure handle
    return cfig


# Function to prompt user to draw a box on a dispersion curve figure. Data points
# inside of the box are removed and data points outside of the box are kept. 
def rmvDCpoints( rawDC, cfig ):
    frequency = rawDC.frequency
    velocity = rawDC.velocity
    offset = rawDC.offset
    wavelength = []
    for m in range(len(frequency)):
        wavelength.append( velocity[m]/frequency[m] )

    # Point elimination loop
    while True:
        # Determine which axis was clicked
        axclick = []
        def on_click(event):
            if event.inaxes is not None:
                if len(axclick)<2:
                    axclick.append(event.inaxes)
        cid = cfig.canvas.mpl_connect('button_press_event', on_click)
        # Draw box, determine upper and lower bounds
        rawBounds = np.asarray(cfig.ginput(2, timeout=0))
        cfig.canvas.mpl_disconnect(cid)
        xmin = np.min(rawBounds[:,0])
        xmax = np.max(rawBounds[:,0])
        ymin = np.min(rawBounds[:,1])
        ymax = np.max(rawBounds[:,1])
        

        # Total removed points
        n_rmv = 0
        for g in range( len(frequency) ):
            # Arrays containing data for current offset
            f = frequency[g]
            v = velocity[g]
            w = wavelength[g]
            # Create arrays to store indices of data that will be kept and removed
            rmv_id = np.zeros( len(f), int )
            keep_id = np.zeros( len(f), int )

            # If user clicked on two different axes, warn user and return
            if axclick[0] != axclick[1]:
                print "WARNING: BOTH CLICKS MUST BE ON SAME AXIS"
                return

            # Identify axes
            axf = min(cfig.axes)
            axw = max(cfig.axes)
            if axf.get_xlabel() != 'Frequency (Hz)':
                axf = max(cfig.axes)
                axw = min(cfig.axes)
            # Determine if points fall within box
            for i in range( len(f) ):
                condition1 = (axclick[0]==axf) and ( xmin<f[i] and f[i]<xmax and ymin<v[i] and v[i]<ymax )
                condition2 = (axclick[0]==axw) and ( xmin<w[i] and w[i]<xmax and ymin<v[i] and v[i]<ymax )
                # Points inside of selectd box are removed
                if condition1 or condition2:
                    rmv_id[i] = i+1
                # Points outside of selected box are kept
                else:
                    keep_id[i] = i+1
            

            # Remove zeros from rmv_id and keep_id
            zid = np.where( rmv_id == 0 )[0]
            rmv_id = np.delete( rmv_id, zid, 0 )
            zid = np.where( keep_id == 0 )[0]
            keep_id = np.delete( keep_id, zid, 0 )
                            
            # Removed data
            frmv = f[(rmv_id-1)]
            vrmv = v[(rmv_id-1)]
            wrmv = w[(rmv_id-1)]
            n_rmv += len(vrmv)
            # Plot deleted data with black x's
            axf.plot(frmv, vrmv, marker="x", color="k", markersize=5, linestyle="none")
            axw.plot(wrmv, vrmv, marker="x", color="k", markersize=5, linestyle="none")
            cfig.canvas.draw_idle()

            # Retained data
            fnew = f[(keep_id-1)]
            vnew = v[(keep_id-1)]
            wnew = w[(keep_id-1)]
            # Revise velocity, frequency, and wavelength cell arrays
            velocity[g] = vnew
            frequency[g] = fnew
            wavelength[g] = wnew

        
        if n_rmv==0:
            break

        del cid
        
            
    rawDC = dctypes.RawDispersion( frequency, velocity, offset )
    return 
        

         
            
    