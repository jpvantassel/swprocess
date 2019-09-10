import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import dcpostprocessing
import importMAM


# Inputs
file_path = 'C:/DPT_Projects/Garner Valley/Passive/HFK'
inFileName = 'GV_C20_HFK'
outFileName = 'GV_C20_HFK_processed_python'
arrayDescript = 'C20 m'
kmin = 0.2503
kmax = 0.6000

# Inputs
#file_path =     input('Enter the file path: ')
#inFileName =    input('Enter the input file name (.max file): ')
#outFileName =   input('Enter desired output file name (.pcklz and .txt files):')
#arrayDescript = input('Enter a breif descriptio of the array') 
#kmin =           raw_input('Enter kmin [rad/m]: ')
#kmax =           raw_input('Enter kmax [rad/m]: ')

# If user did not enter kmin or kmax, set equal to NaN, otherwise convert to float
if kmin=='' or kmax=='':
    kmin = float('nan')
    kmax = float('nan')
else:
    kmin = float(kmin)
    kmax = float(kmax)


# Load dispesrion data
rawDC, binParam = importMAM.importHFKdc( file_path+'/'+inFileName )
# Cut excessively high velocities
rawDC.rmvHighVs()



# Post processing**************************************************************
cont = True
cfig = 0
while cont:
    # Close previous plot (if it exists)
    if cfig:
        plt.close(cfig)

    # Compute statistics    
    meanDisp = dcpostprocessing.computeDCstats( rawDC, binParam[0], binParam[1], int(binParam[2]), 'log', 'frequency', [] )
    
    # Plot raw dispersion data
    cfig = dcpostprocessing.plotDCforRmv( rawDC, meanDisp, [arrayDescript], [ 'o' ], [ 'r' ], 'log', kmin, kmax )
    
    # Elimination of "bad" data
    dcpostprocessing.rmvDCpoints( rawDC, cfig  )

    # Ask user if they would like to continue cutting data
    cont = input('Enter 1 to continue cutting data, otherwise enter 0: ')



# Create files with final processed data ***************************************
# Compressed pickle file containing dispersion data from each offset
f = gzip.open(file_path+"/"+outFileName+".pklz", 'wb')
pickle.dump(rawDC, f)
f.close()
# Text file containing frequency, slowness, slow std., and weight
# (used in dinver software)
rows = np.array([0, 3, 4, 6])
np.savetxt(file_path+'/'+outFileName+'.txt', meanDisp[:,rows], fmt='%10.4f    %10.8f    %10.8f    %10.4f')

