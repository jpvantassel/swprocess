"""File for helpful plotting tools."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import logging

# Create an array, where each row contains an RGBA color equally spaced in a colormap
def makecolormap( N, maptype='jet' ):
    ccmap = plt.get_cmap( maptype ) 
    cNorm  = colors.Normalize(vmin=0, vmax=(N-1))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=ccmap)
    plotColors = np.zeros( (N,4) )
    for k in range( N ):
        plotColors[k,:] = scalarMap.to_rgba( k )
    return plotColors