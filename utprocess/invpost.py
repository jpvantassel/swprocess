
"""
    The functions contained in this file are used to post-process the results 
    from a surface wave inversion performed using Geopsy.

    Functions:

        split_filename:
        Separate filename from path and remove .txt (if included in filename)

        V1toV2:
        Convert Vs or Vp from 1-row format (V1) to 2-row (V2) format

        thk2depth:
        Convert thickness to depth (2-row format)

        thk2middepth:
        Convert thickness to middepth (1-row format)

        Row2toRow1:
        Convert variable from 2-row format to 1-row format

        resampleV:
        Resample Vs or Vp profile along specified depth vector

        assign_unit_wt:
        Assign unit weights to layers using Vs relationship in Mayne (2001)

        mean_eff_stress:
        Calculate the mean effective stress at the mid point of each layer.

        txt2npz_EarthModel:
        Import layered earth models from text file and export to npz file

        txt2npz_DCorEll:
        Import dispersion or ellipticity data from text file and export to npz file

        subplotLoc:
        Calculates the positions of axes in a subplot using a specified number
        of rows, columns, and white-space. All units are normalized (i.e., 0 to
        1). 


    References:
        Mayne, P. (2001). "Stress-strength-flow parameters from enhanced in-situ
        tests." Proceedings, International Conference on In-Situ Measurement of
        Soil Properties & Case Histories [In-Situ 2001], Bali,Indonesia, May
        21-24, 2001, pp. 27-48.


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


# Import modules
import numpy as np
import os


# Global variables used in calculations
water_gamma = 9.81
atm_pressure = 101.325


# Separate filename from path and remove txt (if included in filename)
def split_filename(fname):
    txtID = fname.rfind('.txt')
    if txtID>0:
        fname = fname[0:txtID]
    fnameID = fname.rfind('/')
    if fnameID>=0:
        fpath = fname[0:fnameID]
        fname = fname[(fnameID+1)::]
    else:
        fpath = os.getcwd()
    return [fname, fpath]


# Convert Vs or Vp from 1-row format (V1) to 2-row (V2) format
def V1toV2( V1 ):
    # V1 data contained in an array
    if type(V1) is np.ndarray:
        [Nl, Np] = np.shape(V1) 
        V2 = np.zeros( (2*Nl,Np), float )
        for k in range(Nl):
            V2[ (k*2), : ] = V1[k,:]
            V2[ (k*2+1), : ] = V1[k,:]
        return V2
    # V1 data contained in a list of arrays (use recursion)
    if type(V1) is list:
        Nj = len(V1)
        V2 = []
        for j in range(Nj):
            V2j = V1toV2( V1[j] )
            V2.append( V2j )
        return V2


# Convert thickness to depth (2-row format)
def thick2depth( thk, maxd=100000, dpen=[] ):
    # thk data contained in an array
    if type(thk) is np.ndarray:
        [Nl, Np] = np.shape(thk)
        depth = np.zeros( (2*Nl, Np), float )
        depth[1,:] = thk[0,:]
        for k in range(1,Nl):
            depth[ (2*k), : ] = depth[ (2*k-1), : ]
            depth[ (2*k+1), : ] = depth[ (2*k), : ] + thk[ k, : ]
        if dpen:
            depth[ (2*Nl-1), : ] = depth[ (2*Nl-1), : ]+dpen
        else: 
            depth[ (2*Nl-1), : ] = maxd
        return depth
    # thk data contained in a list of arrays (use recursion)
    if type(thk) is list:
        Nj = len(thk)
        depth = []
        for j in range( Nj ):
            depthj = thick2depth( thk[j] )
            depth.append( depthj )
        return depth
 

# Convert depth (1-row format) to mid-depth (1-row format)
def depth2middepth( depth1, dpen=20 ):
    # depth1 data contained in an array
    if type(depth1) is np.ndarray:
        [Nl,Np] = np.shape(depth1)
        mid_d = np.zeros( (Nl,Np), float )
        for k in range(0,Nl-1):
            mid_d[k,:] = 0.5*(depth1[k,:] + depth1[k+1,:])
        mid_d[Nl-1,:] = mid_d[Nl-2,:] + dpen
        return mid_d
    # depth1 data contained in a list of arrays
    if type(depth1) is list:
        Nj = len(depth1)
        mid_d = []
        for j in range( Nj ):
            mid_dj = depth2middepth( depth1[j], dpen )
            mid_d.append( mid_dj )
        return mid_d
        
           
# Convert variable from 2-row format (var2) to 1-row format (var1)
# Set loc='even' to extract even array entries or 'odd' to extract odd entries
def Row2toRow1( var2, loc='even' ):
    # If var2 data is stored in an array
    if type(var2) is np.ndarray:
        [nr2, nc] = np.shape(var2)
        nr = nr2/2
        var1 = np.zeros( (nr, nc) ) 
        if str.lower(loc)=='even':
            ids = range(0,nr2,2)
        else:
            ids = range(1,nr2,2)
        var1 = var2[ ids, : ]
        return var1
    # If var2 data is stored in a list of arrays (use recursion)
    if type(var2) is list:
        Nj = len(var2)
        var1 = []
        for j in range( Nj ):
            var1j = Row2toRow1( var2[j], loc )
            var1.append( var1j )
        return var1
    

# Resample Vs or Vp profile (V1 and d1, 1-row format, depths are to bottom of layer) along specified depth vector (dr)
def resampleV( V1, d1, dr ):
    Nre = len(dr)
    # If V1 and d1 are stored in arrays
    if type(V1) is np.ndarray:
        [Nl, Np] = np.shape( V1 )
        Vr = np.zeros( (Nre, Np) )
        for p in range( Np ):
            Vr[:,p] = V1[0,p]
            for k in range(1,Nl):
                ids = np.where( (dr<=d1[k,p]) * (dr>d1[k-1,p]) )
                Vr[ids,p] = V1[k,p]
        return Vr
    # If V1 and d1 are stored as a list of arrays (use recursion)
    if type(V1) is list:
        Nj = len(V1)
        Vr = []
        for j in range( Nj ):
            Vri = resampleV( V1[j], d1[j], dr )
            Vr.append( Vri )
        return Vr


# Assign unit weight based on shear wave velocity using equation 32 of Mayne 
# (2001). If the soil is unsaturated, the total unit weight will be assumed to 
# be 90% of the total unit weight.
def assign_unit_wt( depth1, Vs1, Vp1orGWT, gamma_max=23, dpen=20 ):
    # Number of layers and profiles
    Nl,Np = np.shape(depth1)
    # Calculate mid-depth of the layer
    mid_depth = np.zeros( (Nl,Np) )
    for k in range(Nl-1):
        mid_depth[k,:] = 0.5*( depth1[k,:] + depth1[k+1,:] )
    # Last layer is assumed to be infinitely thick, compute unit wt based on a depth dpen into the layer
    mid_depth[Nl-1,:] = depth1[Nl-1,:]+dpen
 
    # Saturated unit weight (Eq. 32 of Mayne 2001)
    gamma = 8.32*np.log10(Vs1) - 1.61*np.log10(mid_depth)
 
    # Adjust values based on whether or not material is saturated
    for k in range(Np):
        # If Vp1orGWT is scalar, then it is equal to the GWT depth
        if np.isscalar(Vp1orGWT):
            unsat_id = np.where( depth1[:,k] < Vp1orGWT)[0]
        # If Vp1orGWT is a vector or matrix, then it is equal to Vp
        # (Vp > 1400 m/s is assumed to be saturated)
        else:
            unsat_id = np.where( Vp1orGWT[:,k] < 1400 )[0]
        # Assume unsaturated unit weight is 90% of saturated unit weight
        gamma[unsat_id,k] = 0.9*gamma[unsat_id,k]
         
    # Ensure that no values are above gamma_max
    gamma[gamma>gamma_max] = gamma_max
 
    return gamma


# Calculate mean effective stress at mid-depth of each layer 
# (excluding bottommost layer, which is assumed infinite)
def mean_eff_stress( depth1, gamma1, Vp1orGWT, K=0.5 ):
    # Number of layers and profiles    
    Nl,Np = np.shape(depth1)

    # Thickness, mid-depth, and increase in stress in each layer
    thk = np.diff( depth1, axis=0 )
    mid_depth = depth1[0:(Nl-1),:] + 0.5*thk
    d_sigma = thk*gamma1[0:(Nl-1),:]

    # Total vertical stress at midpoint of layer
    mid_sigma = np.zeros( (Nl-1,Np) )
    for k in range(Nl-1):
        mid_sigma[k,:] = np.sum( d_sigma[0:k,:], axis=0 ) + 0.5*d_sigma[k,:] 

    # Pore pressure at middle of layer
    mid_u = np.zeros( (Nl-1,Np) )
    for m in range(Np):
        # If Vp1orGWT corresponds to the GWT
        if np.isscalar(Vp1orGWT):
            gwt_depth = Vp1orGWT
        else:
            sat_id = np.where( Vp1orGWT[:,m] > 1400 )[0]
            if np.shape(sat_id):
                gwt_depth = depth1[sat_id[0],m]
            else:
                gwt_depth = np.inf

        for k in range(Nl-1):
            if mid_depth[k,m] > gwt_depth:
                mid_u[k,m] = water_gamma*(mid_depth[k,m] - gwt_depth)

    # Vertical effective stress at middle of layer
    mid_sigma_eff = mid_sigma - mid_u

    # Mean vertical effective stress at middle of layer
    mid_mean_sigma_eff = ( (1 + 2*K)/3 * mid_sigma_eff ) / atm_pressure
    return mid_mean_sigma_eff


# Import layered earth models from text file and export to npz file
def txt2npz_EarthModel( fname, createOutFile=True, outfile=[], outpath=os.getcwd() ):
    
    # Separate filename from path and remove txt (if included in filename)
    [fname,fpath] = split_filename(fname)

    # Load file (each row is a string)
    f = open(fpath  +'/' + fname + '.txt', 'r')
    dataStruct = f.readlines() 
    
    # Identify the misfit values and start-line associated with each profile
    misfit = []
    sline = []
    Nl = 0
    for k in range( len(dataStruct) ):
        mid = dataStruct[k].find('value=')
        if mid>0:
            misfit.append( float(dataStruct[k][(mid+6):]) )
            sline.append( k+2 )
            if Nl==0:
                Nl = int( dataStruct[k+1] )
    # Convert to numpy arrays
    misfit = np.array( misfit )
    sline = np.array( sline )
    # Number of profiles
    Np = len( misfit )
    
    # Initialize matrix to store ground models
    thk = np.zeros( (Nl, Np) )
    Vp1 = np.zeros( (Nl, Np) )
    Vs1 = np.zeros( (Nl, Np) )
    rho1 = np.zeros( (Nl, Np) )
    
    for k in range( Np ):
        temp = np.zeros( (Nl, 4) )
        for j in range( Nl ):
            temp[j,:] = np.array( dataStruct[sline[k]+j].split(), float )
        thk[:,k] = temp[:,0]
        Vp1[:,k] = temp[:,1]
        Vs1[:,k] = temp[:,2]
        rho1[:,k] = temp[:,3]
        
    
    # Default output filename
    if not outfile:
        outfile = fname
    if not outpath:
        outpath = fpath 
    # Create output file
    if createOutFile:
        np.savez(outpath+'/'+outfile+'.npz', thk=thk, Vp1=Vp1, Vs1=Vs1, rho1=rho1, misfit=misfit)
    
    # Return results
    return [thk, Vp1, Vs1, rho1, misfit]



# Import dispersion or ellipticity data from text file and export to npz file
def txt2npz_DCorEll( fname, tag='disp', createOutFile=True, outfile=[], outpath=os.getcwd() ): 
    
    # Separate filename from path and remove txt (if included in filename)
    [fname,fpath] = split_filename(fname) 
    
    # Load file (each row is a string)
    f = open(fpath  +'/' + fname + '.txt', 'r')
    dataStruct = f.readlines()  
    
    # Determine Nmodes
    for k in range( len(dataStruct) ):
        if str.lower(tag)=='disp':
            eid = dataStruct[k].find( ' Rayleigh dispersion mode(s)' )
        elif str.lower(tag)=='ell':
            eid = dataStruct[k].find( ' Rayleigh ellipticity mode(s)' )
        if eid > 0:
            Nmodes = int( dataStruct[k][1:eid] )
            break   
    
    # Determine Nprofiles and locate any failed calculations
    Np = 0
    Nfail = 0
    fail_ids = []
    for k in range( len(dataStruct) ):
        ntest = dataStruct[k].find( 'value=' )
        ftest = dataStruct[k].find( 'Failed:' )
        if ntest > 0:
            Np += 1
        if ftest > 0:
            Nfail += 1
            fail_ids.append( Np )   
    fail_ids = np.array(fail_ids)
    # Close file
    f.close()
    
    
    # Reload file (only numeric values are extracted using loadtxt command)    
    dataArray = np.loadtxt(fname+'.txt')
    # Max frequency extracted
    maxF = np.max( dataArray[:,0] )
    # Location of max frequency (i.e., last data point for each mode/profile)
    eids = np.where( dataArray[:,0]==maxF )[0]+1
    # Number of frequencies
    Nf = eids[0]
    
    # Fundamental mode (i.e., 0 to eids[0]) is defined at all frequencies
    frequency = dataArray[0:eids[0],0]
    # Initialize variable to store dispersion data or ellipticity
    var = np.empty((Nmodes,Nf,Np))
    var[:] = np.NAN
     
    # Populate dispersion or ellipticity data (avoid any failed computations)
    c = 0
    for k in range(Np):
        if not sum( k == fail_ids ):
            for m in range(Nmodes):
                if c==0:
                    ids = range(0,eids[c])
                else:
                    ids = range(eids[c-1],eids[c])
                sid = Nf - len(ids)
                var[m,sid::,k] = dataArray[ids,1]
                c += 1
    
    # Default output filename
    if not outfile:
        outfile = fname
    if not outpath:
        outpath = fpath
    
    # Return and save results
    if str.lower(tag)=='disp':
        velocity = 1.0/var
        # Tile frequency values for element-by-element division
        ftile = np.tile(frequency, (Np,1) ).transpose()
        ftile = np.tile(ftile, (Nmodes,1,1) )
        wavelength = velocity/ftile
        if createOutFile:
            np.savez(outpath+'/'+outfile+'.npz', frequency=frequency, velocity=velocity, wavelength=wavelength)
        return [frequency, velocity, wavelength]
    elif str.lower(tag)=='ell':
        ellipticity = var
        if createOutFile:
            np.savez(outpath+'/'+outfile+'.npz', frequency=frequency, ellipticity=ellipticity)
        return [frequency, ellipticity]
    else:
        raise Exception('Invalid tag')
    
    
def subplotLoc(n_row, n_col, xL, xR, xI, yL, yU, yI, direction='lr'):
    x_fig = (1 - (xL + (n_col-1)*xI + xR)) / n_col
    y_fig = (1 - (yU + (n_row-1)*yI + yL)) / n_row
    
    # Initialize variables
    ax_rc = ['']*n_row*n_col
    ax_position = []
    for a in range(n_row):
        ax_position.append(['']*n_col)

    # Subplot locations
    for a in range(n_row):
        for b in range(n_col):
            left = xL + (b)*(x_fig+xI)
            bottom = 1 - (yU + y_fig + (a)*(y_fig+yI))
            ax_position[a][b] = [left, bottom, x_fig, y_fig]
            if direction.lower() == 'lr':
                ax_rc[(a)*n_col+b] = [a,b]
            else:
                ax_rc[(b)*n_row+a] = [a,b]
    return ax_position, ax_rc
            
            

    
     
     
  
  
             
        
    

