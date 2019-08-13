
"""
    The functions contained in this file are used to calculate the 1D site 
    response for layered earth models.

    Functions:

        calc_wave_coeff:
        Calculates the up-going (A) and down-going (B) wave coefficients for 
        vertically propagating, horizontally polarized shear waves in a multi-
        layered earth. Each layer is defined by a thickness, shearing stiffness,
        unit weight, and damping ratio.

        calc_LETF_forGM:
        Calculates the linear, visco-elastic transfer fucntions for layered 
        earth models derived from a surface wave inversion.

    References:
        
        Kramer, S.L. (1996). Geotechnical Earthquake Engineering. Prentice Hall:
        New Jersey, pp. 257-270.
        
        Kottke, A. and Rathje, E. (2009). "Technical Manual for Strata." Report
        No. 2008/10, Pacific Earthquake Engineering Research Center, Berkeley, 
        California.


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


import modulus_and_damping as mad
import invpost
import numpy as np



# Calculate up-going and down-going wave coefficients...........................
def calc_wave_coeff(freq, depth, G, d, gamma):

    # Ensure that dimensions of array are correct (one-dimensional arrays)
    depth = np.reshape(depth,len(depth))
    G = np.reshape(G,len(G))
    d = np.reshape(d,len(d))
    gamma = np.reshape(gamma,len(gamma)) 

    # Number of layers
    nl = len(depth)
    nf = len(freq)
    # Angular frequency
    omega = 2*np.pi*freq
    # Thickness
    h = np.diff(depth)
    h = np.append( h, 0 )
    # Mass density
    rho = gamma/9.80665
    # Complex shear modulus (Eq. 2.4 in Kottke and Rathje 2009)
    compG = G*(1 - 2*d*d + 1j*2*d*np.sqrt(1-d*d))
    # Complex shear wave velocity (Eq. 7.9 in Kramer 1996)
    compVs = np.sqrt(compG/rho)
    
    # Calculate complex wavenumber for all layers and frequencies (Eq. 7-10 in Kramer)
    temp1, temp2 = np.meshgrid(omega, compVs)
    comp_k = temp1/temp2
    del temp1, temp2
    
    # Calculate complex impedance ratio for all layers and frequencies (Eq. 7.35 in Kramer)
    alpha = np.zeros(np.shape(comp_k), complex)
    for m in range(nl-1):
        alpha[m,:] = (comp_k[m,:] * compG[m]) / (comp_k[m+1,:] * compG[m+1])
    
    # Calculate i*comp_k*h for use in subsequent calculations
    hrep = np.zeros((nl,nf))
    for k in range(nl):
        hrep[k,:] = h[k]
    expTerm = 1j*comp_k*hrep
    
    # Calculate A and B coefficients for each layer and frequency
    waveA = np.ones((nl, nf), complex) 
    waveB = np.ones((nl, nf), complex) 
    for m in range(nl-1):
        # Upgoing wave (equation 7.34a in Kramer)
        waveA[m+1,:] = 0.5*waveA[m,:]*(1 + alpha[m,:])*np.exp(expTerm[m,:]) + 0.5*waveB[m,:]*(1 - alpha[m,:])*np.exp(-expTerm[m,:])
        # Upgoing wave (equation 7.34b in Kramer)
        waveB[m+1,:] = 0.5*waveA[m,:]*(1 - alpha[m,:])*np.exp(expTerm[m,:]) + 0.5*waveB[m,:]*(1 + alpha[m,:])*np.exp(-expTerm[m,:])

    return waveA, waveB


# Calculate linear-viscoelastic transfer function between the surface and an arbitrary depth for layered earth model
def calc_LETF_forGM(depth, Vs, Vp, d_soil='', d_rock=0.5, gm_type='outcrop', d_tf='rock', freq=np.logspace(-1,1,512)):

    # Ensure that data type of d_soil is numpy array (if d_soil is specified and non-scalar)
    if isinstance(d_soil,list) and len(d_soil)>0:
        d_soil = np.array(d_soil)
    
    # Number of layers, profiles, and frequency values
    if np.shape(freq):
        nf = len(freq)
    else:
        nf = 1
    nl, npr = np.shape(depth)

    # Assign unit weight based on Vs and water table depth (or Vp)    
    gamma = invpost.assign_unit_wt(depth, Vs, Vp)
    
    # Initialize variable for surface transfer function
    surfaceTF = np.zeros((npr,nf))
    
    # Compute small strain shear modulus
    G = (Vs**2)*(gamma/9.80665)

    # Compute damping
    if not isinstance(d_soil, str):
        if not np.shape(d_soil): 
            # If a scalar value is provided
            d_min = float(d_soil)/100 * np.ones((nl, npr))
        else:                               
            # If a 1-item array (i.e., a scalar value) is provided
            if len(d_soil)==1:
                d_min = float(d_soil[0])/100 * np.ones((nl, npr))
            else:
                # If damping is a vector (1 value per soil layer)
                d_min = np.zeros((nl,npr))
                for k in range(nl-1):
                    d_min[k,:] = d_soil[k]
    else:
        # Compute damping from Darendeli (2001) if a string value is provided
        mean_eff_stress = invpost.mean_eff_stress(depth, gamma, Vp)
        d_min = np.zeros((nl,npr))
        for k in range(npr):
            _, damping, _ = mad.darendeliCalc( mean_eff_stress[:,k], 1e-4 )
            d_min[0:nl-1,k] = damping[:,0]/100
    # Append d_min value for bedrock
    d_min[-1:,:] = float(d_rock)/100

    # Loop through Vs profiles
    for k in range(npr):
        c_depth = depth[:,k]
        c_gamma = gamma[:,k]
        c_G = G[:,k]
        c_d = d_min[:,k]
        
        # If d_tf exceeds depth to rock, add layer above d_tf whose properties are equal to rock
        if (not isinstance(d_tf,str)) and (d_tf>c_depth[-1]):
            c_depth = np.append(c_depth, d_tf)
            c_gamma = np.append(c_gamma, c_gamma[-1])
            c_G = np.append(c_G, c_G[-1])
            c_d = np.append(c_d, c_d[-1])
            lid = len(c_depth)-1
        # if d_tf is shallower than depth to rock, truncate profile
        elif (not isinstance(d_tf,str)) and (d_tf<c_depth[-1]):
            lid = np.where(c_depth>d_tf)[0][0]
            c_depth = c_depth[0:lid+1]
            c_depth[-1] = d_tf
            c_G = c_G[0:lid+1]
            c_G[-1] = c_G[lid-1]
            c_d = c_d[0:lid+1]
            c_d[-1] = c_d[lid-1]
            c_gamma = c_gamma[0:lid+1]
            c_gamma[-1] = c_gamma[lid-1]
        elif (isinstance(d_tf,str) and d_tf.lower()=='rock') or ( (not isinstance(d_tf,str)), d_tf==c_depth[-1] ):
            lid = len(c_depth)-1
    
        # Calculate upgoing and downgoing wave coefficients
        waveA, waveB = calc_wave_coeff( freq, c_depth, c_G, c_d, c_gamma )

        # Calculate transfer function
        if gm_type.lower() == 'outcrop':
            surfaceTF[k,:] = np.abs( (waveA[0,:]+waveB[0,:]) / (2*waveA[lid,:]) )
        elif gm_type.lower() == 'within':
            surfaceTF[k,:] = np.abs( (waveA[0,:]+waveB[0,:]) / (waveA[lid,:]+waveB[lid,:]) )
        else:
            raise ValueError('Invalid ground motion type')

    # Transpose so that rows correspond to frequency and columns to Vs profiles
    surfaceTF = np.transpose(surfaceTF)    
    return surfaceTF, freq
            
        
        
        
    

    
