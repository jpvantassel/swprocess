"""
    Classes:
        DispersionPower: 
        Stores processed dispersion data for a given source-offset. This
        class stores the relevent processing paramters (e.g., frequency,
        trial values of k or v, kres, etc.). This class contains methods
        to plot (1) contours of power in various domians and/or (2) 1D 
        "slices" of power at user-defined frequencies.  

        RawDispersion:
        Stores the "raw" dispersion data (i.e., frequency, velocity, and offset) 
        for one or more source-offsets.     


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


# Import modues
import math
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
# import tkinter as tk
import shotgathers


# Width and height (in) of plots showing contours and/or "slices"
mwdth = 6
mhght = 4.25

# Class containing full dispersion processing results (including power)
# Results can be derived from FK, FDBF, slant-stack transform, or Park transform
class DispersionPower():
    
    # Class attributes**********************************************************
    def __init__(self, freq, peak_vals, trial_vals, val_type, kres, pnorm):
        self.freq = freq                # Frequencies considered in calculations
        self.peak_vals = peak_vals      # Wavenumber or velocity corresponding to peak power at each frequency
        self.trial_vals = trial_vals    # Wavenumbers or velocities considered in analysis
        self.val_type = val_type        # Type of values considered in analysis ("wavenumber" or "velocity")
        self.kres = kres                # Wavenumber above which spatial aliasing occurs
        self.pnorm = pnorm              # Normalized power for all frequencies and wavenumbers or velocities


    # Method to plot power in various domains***********************************
    def plotSpect( self, plotType="fv", plotLim=[] ):
        # Plotting parameters if wavenumbers were considered in analysis
        if str.lower(self.val_type)=="wavenumber":
            k_vals = self.trial_vals
            # Grid parameters
            freq_grid, wavenum_grid = np.meshgrid( self.freq, k_vals )
            vel_grid = 2*np.pi*freq_grid / wavenum_grid
            wavel_grid = 2*np.pi/wavenum_grid
            # Peaks
            k_peak = self.peak_vals
            wavel_peak = 2*np.pi / k_peak
            v_peak = wavel_peak*self.freq
        # Plotting parameters if velocities were considered in analysis
        elif str.lower(self.val_type)=="velocity": 
            v_vals = self.trial_vals
            # Grid parameters
            freq_grid, vel_grid = np.meshgrid( self.freq, v_vals )
            wavel_grid = vel_grid / freq_grid
            wavenum_grid = 2*np.pi / wavel_grid
            # Peaks
            v_peak = self.peak_vals
            k_peak = 2*np.pi*self.freq / v_peak
            wavel_peak = 2*np.pi / k_peak
        else:
            raise ValueError("Invalid val_type. Should be \"wavenumber\" or \"velocity\".")

        # Compute maximum power (for plotting purposes)
        maxZ = np.nanmax( np.abs( self.pnorm ) )

        # Set x- and y-axes based on plotType
        # Frequency-wavenumber
        if str.lower(plotType)=="fk":
            xgrid = freq_grid
            ygrid = wavenum_grid
            xpeak = self.freq
            ypeak = k_peak
            if len(plotLim) == 0:
                plotLim = [0, np.max(self.freq), 0, self.kres] 
            elif len(plotLim) != 4:
                raise ValueError("plotLim should be a four element list")
            xscale = "linear"
            yscale = "linear"
            xLabText = "Frequency (Hz)"
            yLabText = "Wavenumber (rad/m)"
        # Frequency-wavelength
        elif str.lower(plotType)=="fw":
            xgrid = freq_grid
            ygrid = wavel_grid
            xpeak = self.freq
            ypeak = wavel_peak
            if len(plotLim) == 0:
                plotLim = [0, np.max(self.freq), 1, 200] 
            elif len(plotLim) != 4:
                raise ValueError("plotLim should be a four element list")
            xscale = "linear"
            yscale = "log"
            xLabText = "Frequency (Hz)"
            yLabText = "Wavelength (m)" 
        # Frequency-velocity
        elif str.lower(plotType)=="fv":
            xgrid = freq_grid
            ygrid = vel_grid
            xpeak = self.freq
            ypeak = v_peak
            if len(plotLim) == 0:
                plotLim = [0, np.max(self.freq), 0, 1000] 
            elif len(plotLim) != 4:
                raise ValueError("plotLim should be a four element list")
            xscale = "linear"
            yscale = "linear"
            xLabText = "Frequency (Hz)"
            yLabText = "Velocity (m/s)"
        # Frequency-slowness
        elif str.lower(plotType)=="fp":
            xgrid = freq_grid
            ygrid = 1.0 / vel_grid
            xpeak = self.freq
            ypeak = 1.0 / v_peak
            if len(plotLim) == 0:
                plotLim = [0, np.max(self.freq), 1.0/1000, 1.0/100] 
            elif len(plotLim) != 4:
                raise ValueError("plotLim should be a four element list") 
            xscale = "linear"
            yscale = "linear"
            xLabText = "Frequency (Hz)"
            yLabText = "Slowness (s/m)"
        # Wavelength-velocity
        elif str.lower(plotType)=="wv":
            xgrid = wavel_grid
            ygrid = vel_grid
            xpeak = wavel_peak
            ypeak = v_peak            
            if len(plotLim) == 0:
                plotLim = [1, 200, 0, 1000] 
            elif len(plotLim) != 4:
                raise ValueError("plotLim should be a four element list")
            xscale = "log"
            yscale = "linear"
            xLabText = "Wavelength (m)"
            yLabText = "Velocity (m/s)" 

        # Ploting 
        # Set figure size equal to 2/3 screen size
        # Get screen size in mm and convert to in (25.4 mm per inch)
        #root = tk.Tk()
        #width = root.winfo_screenmmwidth() / 25.4 * 0.66
        #height = root.winfo_screenmmheight() / 25.4 * 0.66
        #fig = plt.figure( figsize=(width,height) )
        fig = plt.figure( figsize=(mwdth,mhght) )                
        ax = fig.add_axes([0.14, 0.14, 0.80, 0.80])
        plt.contourf( xgrid, ygrid, np.abs(self.pnorm), np.linspace(0, maxZ, 20), cmap=plt.cm.get_cmap("jet") )
        ax.plot( xpeak, ypeak, marker="o", markersize=5, markeredgecolor="w", markerfacecolor='none', linestyle="none" )
        ax.axis( plotLim )    
        ax.set_xlabel(xLabText, fontsize=12, fontname="arial")
        ax.set_ylabel(yLabText, fontsize=12, fontname="arial")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xticklabels(ax.get_xticks(), fontsize=12, fontname="arial" )
        ax.set_yticklabels(ax.get_yticks(), fontsize=12, fontname="arial" ) 
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        plt.colorbar(ticks=np.linspace(0, maxZ, 8))


    # Method to plot slices in various domains**********************************
    def plotSlices( self, plotType="fv", freqPlotValues=np.arange(6,22,1), xlims=[] ):
        # Determine appropriate number of panels and their arrangement        
        n_slices = len(freqPlotValues) 
        xFigDim = int( math.ceil( math.sqrt(n_slices) ) )
        if ( math.ceil( math.sqrt(n_slices) )*math.floor( math.sqrt(n_slices) ) < n_slices ):
            yFigDim = int( math.ceil( math.sqrt(n_slices) ) )
        else:
            yFigDim = int( math.floor( math.sqrt(n_slices) ) )

        # Create an array containing the row and column for each panel
        panel = 0
        panel_ids = np.zeros((n_slices,2))
        for r in range(yFigDim):
            for c in range(xFigDim):
                if (panel+1)<=n_slices:
                    panel_ids[panel,0] = r+1
                    panel_ids[panel,1] = c+1
                    panel += 1

        # Set figure size equal to 2/3 screen size
        # Get screen size in mm and convert to in (25.4 mm per inch)
        #root = tk.Tk()
        #width = root.winfo_screenmmwidth() / 25.4 * 0.66
        #height = root.winfo_screenmmheight() / 25.4 * 0.66
        #fig = plt.figure( figsize=(width,height) )
        fig = plt.figure( figsize=(mwdth,mhght) )

        # Loop through freqPlotValues
        for k in range(n_slices-1, -1, -1):

            # Find frequency closest to freqPlotValues(k)
            c_id = np.argmin( np.absolute(self.freq-freqPlotValues[k]) )
            cfreq = self.freq[c_id]

            # Plotting parameters
            if str.lower(self.val_type)=="wavenumber":
                k_vals = self.trial_vals
                k_peak = self.peak_vals
                v_vals = 2*np.pi*cfreq / k_vals
                v_peak = 2*np.pi*cfreq / k_peak
            elif str.lower(self.val_type)=="velocity":
                v_vals = self.trial_vals
                v_peak = self.peak_vals
                k_vals = 2*np.pi*cfreq / v_vals
                k_peak = 2*np.pi*cfreq / v_peak
            else:
                raise ValueError("Invalid value type, should be \"wavenumber\" or \"velocity\"")

            # Compute maximum power 
            maxY = np.nanmax( np.abs(self.pnorm[:,c_id]) )

            # Determine x-axis and corresponding limits based on chosen graph type
            if str.lower(plotType)=="fk":
                x = k_vals
                xp = k_peak[c_id]
                xLabText = "Wavenumber (rad/m)"
                if not xlims:
                    xlims = (0,self.kres)
                xscale = "linear"
                text_xloc = 0.66*(xlims[1] - xlims[0]) + xlims[0]
            elif str.lower(plotType)=="fw":
                x = 2*np.pi / k_vals
                xp = 2*np.pi / k_peak[c_id]               
                xLabText = 'Wavelength (m)'
                if not xlims:
                    xlims = (1,200)
                xscale = "log"
                text_xloc = math.pow( 10, (0.66*( math.log10(xlims[1]) - math.log10(xlims[0]) ) + math.log10(xlims[0])) )
            elif str.lower(plotType)=="fv":
                x = v_vals
                xp = v_peak[c_id]
                xLabText = "Velocity (m/s)"
                if not xlims:
                    xlims = (0,1000)
                xscale = "linear"
                text_xloc = 0.66*(xlims[1] - xlims[0]) + xlims[0]
            elif str.lower(plotType)=="fp":
                x = 1.0 / v_vals
                xp = 1.0 / v_peak[c_id]
                xLabText = "Slowness (s/m)"
                if (k+1)==n_slices:
                    minX = 1.0 / np.max(v_vals)
                if not xlims:
                    xlims = (minX, 1.0/100)
                xscale = "linear"
                text_xloc = 0.33*(xlims[1] - xlims[0]) + xlims[0]
            else:
                raise ValueError("Invalid plot type, should be \"fk\", \"fw\", \"fv\" or \"fp\"")

            # Plot power at current frequency
            ax = fig.add_subplot( yFigDim, xFigDim, k+1)
            ax.set_xlim( xlims )
            ax.set_ylim( ( 0, maxY ) )
            ax.set_xscale(xscale) 
            ax.plot( x, np.abs( self.pnorm[:,c_id] ) )
            ax.plot( xp, np.max( np.abs( self.pnorm[:,c_id] ) ), marker="*", color="r", markersize=10 )
            ax.set_xticklabels(ax.get_xticks(), fontsize=9, fontname='arial' )
            ax.set_yticklabels(ax.get_yticks(), fontsize=9, fontname='arial' )
            ax.xaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%d') )
            ax.yaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%d') )
            prfreq = '%.2f' % cfreq
            plt.text( text_xloc, 0.75*maxY, prfreq+" Hz", fontsize=9, fontname="arial" )
            if panel_ids[k,0] == yFigDim:
                ax.set_xlabel(xLabText, fontsize=9, fontname="arial") 
            if panel_ids[k,1] == 1:
                ax.set_ylabel("Normalized Amplitude", fontsize=9, fontname="arial")
   

            
# Class containing raw dispersion processing results for one or more source offsets    
class RawDispersion():

    # Class attributes**********************************************************
    def __init__(self, frequency, velocity, offset):
        self.frequency = frequency    # List containing frequency arrays (1 per offset)
        self.velocity = velocity      # List containing velocity arrays (1 per offset)
        self.offset = offset          # List containing offsets 

    # Method to remove data with excessively high/low Vs values*****************
    def rmvHighVs(self, Vs_cut=3500):  
        for k in range(np.shape(self.frequency)[0]):
            keep_id = np.logical_and( self.velocity[k] <= Vs_cut, self.velocity[k]>0 ) 
            self.velocity[k] = self.velocity[k][keep_id]
            self.frequency[k] = self.frequency[k][keep_id]
           