"""
    This script imports previously-processed dispersion data from one or 
    more source-offsets, bins the data, and computes statistics. The 
    analyst may manually remove data points that correspond to alternate 
    modes, near-field effects, offline noise, or outliers. Post-processed 
    dispersion data will be exported to .pklz and .txt files.


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

# INPUTS************************************************************************
# END OF INPUTS*****************************************************************

# Inputs/Output files...........................................................
infile_path = r'E:\New Zealand 2017\2_Wellington_June2017\Raw Active\CMPK'
infile_name = 'Delete_raw'
outfile_path = infile_path
outfile_name = 'Delete_processed'

# Binning parameters............................................................
# Bin in terms of 'frequency' or 'wavelength' on a 'linear' or 'log' scale
binType = 'frequency'
binScale = 'log'
# Min and max freq/wavelength and number of bins
minF = 5
maxF = 100
nBins = 30
# Array weights (Can assign different relative weights to each array, e.g. [1,2,2]
# assigns two times more weight to the second and third source-offsets. Set equal
# to [] for equal weighting).
arrayWt = []


# Plotting parameters........................................................... 
# Marker type and color for each source-offset
markType = [ 'o', 'v', 's', '*' ]
colorSpec = [ 'r', 'b', 'c', 'g' ]
# Manual legend entries for each source-offset (set to [] to list source-offset)
manualLeg = []
# Can view'linear' or 'log' x-axis
xScaleType = 'log'

# END OF INPUTS*****************************************************************
#*******************************************************************************


# Load modules
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import dcpostprocessing


# Load raw dispersion data
f = gzip.open(infile_path+'\\'+infile_name+'.pklz', 'r')
rawDC = pickle.load(f)
f.close()
if not manualLeg:
    setLeg = rawDC.offset
else:
    setLeg = manualLeg


# Post processing **************************************************************
cont = True
cfig = 0
while cont:
    # Close previous plot (if it exists)
    if cfig:
        plt.close(cfig)

    # Compute statistics    
    meanDisp = dcpostprocessing.computeDCstats( rawDC, minF, maxF, nBins, binScale, binType, [] )
    
    # Plot raw dispersion data
    cfig = dcpostprocessing.plotDCforRmv( rawDC, meanDisp, setLeg, markType, colorSpec, xScaleType )
    
    # Elimination of "bad" data
    dcpostprocessing.rmvDCpoints( rawDC, cfig  )

    # If all data is removed for a given offset, delete corresponding entries
    # (Only delete entries for one offset at a time because indices change after 
    # deletion. Continue while loop as long as emty entries are encountered). 
    prs = True
    while prs:
        n_empty = 0
        for k in range(len(rawDC.frequency)):
            if len(rawDC.frequency[k])==0:
                del rawDC.frequency[k]
                del rawDC.velocity[k]
                del rawDC.offset[k]
                n_empty += 1
                break
        if n_empty==0:
            prs = False

    # Ask user if they would like to continue cutting data
    cont = input('Enter 1 to continue cutting data, otherwise enter 0 to export data to text file: ')
                

# Create files with final processed data ***************************************
# Compressed pickle file containing dispersion data from each offset
f = gzip.open(outfile_path+"/"+outfile_name+".pklz", 'wb')
pickle.dump(rawDC, f)
f.close()
# Text file containing frequency, slowness, slow std., and weight
# (used in dinver software)
rows = np.array([0, 3, 4, 6])
np.savetxt(outfile_path+'/'+outfile_name+'.txt', meanDisp[:,rows], fmt='%10.3f    %10.8f    %10.8f    %10.4f')    