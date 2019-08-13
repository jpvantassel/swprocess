import numpy as np
import dctypes


# File to import data from HFK or FK output file from geopsy (.max format)
def importHFKdc(filename):
    f = open(filename+'.max', 'r')
    dataStruct = f.readlines()
    f.close()
    
    # Extract number of frequency bands (contained in line 9)
    s_id = len('# Number of freq bands: ')
    e_id = dataStruct[9].find('\n')
    noF = int( dataStruct[9][s_id:e_id] )
    
    # Minimum binning frequency (contained in line 10)
    s_id = dataStruct[10].find('center ') + len('center ')
    e_id = dataStruct[10].find(' upper')
    minF = float( dataStruct[10][s_id:e_id] )
    
    # Maximum binning frequency (contained in line 10 + noF - 1)
    s_id = dataStruct[(10+noF-1)].find('center ') + len('center ')
    e_id = dataStruct[(10+noF-1)].find(' upper') 
    maxF = float( dataStruct[(10+noF-1)][s_id:e_id] )
    
    # Import dispersion data
    s_line_id = 10 + noF + 1
    e_line_id = len(dataStruct) - 1
    dataMatrix = np.loadtxt(filename+'.max', skiprows=s_line_id-1)
    
    # Create vectors for each column
    sFromStart = dataMatrix[:,0]
    frequency = dataMatrix[:,1]
    slowness = dataMatrix[:,2]/1000
    velocity = 1/slowness
    azimuth = dataMatrix[:,3]
    mathPhi = dataMatrix[:,4]
    semblance = dataMatrix[:,5]
    beampower = dataMatrix[:,6]

    # Create a class containing raw dispesrion data
    rawDC = dctypes.RawDispersion( [frequency], [velocity], ['passive'] )
    # Create an array containing binning parameters [minF, maxF, noF]
    binParam = np.array([minF, maxF, noF])
    return [rawDC, binParam]

