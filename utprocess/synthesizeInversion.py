"""
    This file imports layered earth models, dispersion curves, and ellipticity 
    curves from text files, which were previously obtained using the 
    extractFromReport.py script. Theoretical linear, visco-elastic transfer 
    functions are also computed for each layered earth model.


    This code was developed at the University of Texas at Austin.
    Copyright (C) 2017  David P. Teague 
 
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

import sys, os
sys.path.append('c:/Python Files')
import numpy as np
import invpost, site_res


# FILE AND FOLDER NAMES
# (infilename=prefix_LR_suffix_type.npz and outfilename=prefix_LR_suffix_type.npz)
inpath = 'C:\DPT_Projects\dinver_log_files\Garner Valley\North\Best_TF-1-9\GV_North_Best_Ell_1-78_LR3_reports'
prefix = 'GV_North_LR'
LR = ['3']
suffix = 'best1000'
outfolder = os.getcwd()+'/best1000'

# TRANSFER FUNCTION PARAMETERS
# Depth at which ground motions were recorded (use 0 for outcrop, use 'within' if
# ground motions were recorded in bedrock, or enter depth is within soil layer)
d_tf = 'within'
# Damping ratio (%) of soil (set to [] to use Darendeli (2001) damping)
dsoil = ''
# Damping ratio (%) of rock
drock = 0.5
# Frequency values to calculate transfer function
freq_tf=np.logspace(-1,1,512)


# Loop through layering ratios
for k in range(len(LR)):

    # Store new miniseed files in folder titled "Array Miniseed"
    if not os.path.isdir(outfolder):
        print 'Creating directory: '+outfolder
        os.mkdir(outfolder)

    # Import layered earth models
    thk, Vp1, Vs1, rho1, misfit = invpost.txt2npz_EarthModel( inpath+'/'+prefix+'_'+LR[k]+'_'+suffix+'_GM', createOutFile=True, outfile=[], outpath=outfolder )

    # Import dispersion curves
    frequency, velocity, wavelength = invpost.txt2npz_DCorEll( inpath+'/'+prefix+'_'+LR[k]+'_'+suffix+'_DC', 'disp', createOutFile=True, outfile=[], outpath=outfolder )

    # Import ellipticity curves
    frequency, ellipticity = invpost.txt2npz_DCorEll( inpath+'/'+prefix+'_'+LR[k]+'_'+suffix+'_Ell', 'ell', createOutFile=True, outfile=[], outpath=outfolder )
    
    # Calculate and save theoretical linear, viscoelastic transfer functions
    depth = invpost.thick2depth(thk)
    nl,npr = np.shape(depth)
    depth1 = depth[np.arange(0,nl,2),:]
    tf, frequency = site_res.calc_LETF_forGM(depth1, Vs1, Vp1, dsoil, drock, 'outcrop', 'rock', freq_tf )
    np.savez(outfolder+'/'+prefix+LR[k]+'_'+'_'+suffix+'_TF.npz', frequency=freq_tf, tf=tf)
    




