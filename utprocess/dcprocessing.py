
"""
    The functions contained in this file compute experimental dispersion data 
    for a given source-offset (i.e., for an instance of the 'shotgather' class).

    Functions:

        fk:
        Compute dispersion data using a standard frequency-wavenumber 
        transformation (i.e., convert the data from x-t to f-k domain using a 
        two-dimensional Fast Fourier Transformation).      
            
        fdbf: 
        Compute dispersion data using frequency domain beamformer. Multiple
        receiver weighting techniques are available. Refer to Zywicki (1999) and
        Foti et al. (2014).   
        
        phase_shift:
        Compute dispersion data using phase-shift transformation described in
        Park et al. (1998).

        tau_p:
        Compute dispersion data using a slant-stack (linear Radon) transform. 
        Refer to McMechan and Yedlin (1981), Sacchi and Ulrych(1995), and 
        Louie (2001). 

        nextpow2:
        Function to find exponent that will result in 2^exponent >= value.      

    
    References:
        Foti, S., Lai, C.G., Rix, G.R., and Strobbia, C. (2014). Surface
            Wave Methods for Near-Surface Characterization. CRC Press.
        Louie, John N. (2001) "Faster, Better: Shear Wave Velocity to 100 
            Meters Depth From Refraction Microtremor Arrays." Bulletin of 
            Seismological Society of America, Vol. 91 (2), pp. 347-364
        Park, C. B., R. D. Miller, and J. Xia (1998). "Imaging dispersion curves of
            surface waves on multichannel record", 68th Annual International
            Soc. Explor. Geophys., New Orleans, Sept. 13-18, Expanded Abstracts,
            p. 1377-1380.
	Sacchi M.D. and Ulrych T.J. (1995). "High Resolution velocity gathers
            and offset space reconstruction", Geophysics, Vol. 60, pp. 1169-
            1177.
        Zywicki, D.J. and Rix, G.J. (2005). Mitigation of Near-Field Effects
            for Seismic Surface Wave Velocity Estimation with Cylindrical
            Beamformers. Journal of Geotechnical and Geoenvironmental
            Engineering, 131(8), pp. 970-977.
        Zywicki, D.J. (1999). Advanced signal processing methods applied to 
            engineering analysis of seismic surface waves. Ph.D. 
            Dissertation, School of Civil and Environmental Engineering, 
            Georgia Institute of Technology, Atlanta, GA, p. 357.


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
import numpy as np
from scipy import signal
from scipy import special
import dctypes

  
#*******************************************************************************
# Frequency-Wavenumber Transformation (two-dimensional Fourier Transform)
def fk( shotGather, numk=2048, min_frequency=5, max_frequency=100 ):

    # Check that spacing is uniform.............................................
    if sum( np.diff(shotGather.position) - np.mean(np.diff(shotGather.position)) )!=0:
        raise ValueError('Receiver spacing must be uniform for FK transform')
    else: 
        spacing = shotGather.position[1]-shotGather.position[0]

    # Sampling parameters.......................................................
    # Time
    freq = np.arange( 0, shotGather.fnyq+shotGather.df, shotGather.df )
    # Space
    dk = 2*np.pi / (numk*spacing)
    k_vals = np.arange( dk, shotGather.kres, dk )

    # Perform two-dimensional FFT...............................................
    fk = np.fft.fft2( shotGather.timeHistories, s=(shotGather.n_samples, numk) )
    # Reverse order so that F[:,n] corresponds to kvals[n]
    fk = np.fliplr( fk )
    # Remove k = 0 column
    fk = fk[ 0:len(freq), 1:: ]

    # Remove frequencies above/below specificied max/min frequencies and downsample (if required by zero padding)
    fminID = np.argmin( np.absolute(freq-min_frequency)  )
    fmaxID = np.argmin( np.absolute(freq-max_frequency)  )
    freq_id = range(fminID,(fmaxID+1), shotGather.multiple)
    freq = freq[freq_id]
    fk = fk[freq_id, :]
    
    # Identify wavenumber associated with maximum in fk domain..................
    pnorm = np.zeros( np.shape(fk) )
    k_peak = np.zeros( np.shape(freq) )
    for k in range( np.shape(fk)[0] ):
        # Normalize by largest number in fk domain at each frequency
        pnorm[k,:] = np.abs( fk[k,:] ) / np.max( np.abs( fk[k,:] ) )
        # Find peak
        pk_id = np.argmax( pnorm[k,:] )
        k_peak[k] = k_vals[pk_id]

    # Create instance of DispersionPower class..................................
    dispersionPower = dctypes.DispersionPower( freq, k_peak, k_vals, 'wavenumber', shotGather.kres, np.transpose(pnorm) )
    return dispersionPower 
    


#*******************************************************************************      
# Frequency Domain Beamformer             
def fdbf( shotGather, weightType='none', steeringVector='plane', numv=2048, min_vel=1, max_vel=1000, min_frequency=5, max_frequency=100 ):

    # Ensure that min_velocity is greater than zero for numerical stability
    if min_vel<1:
        min_vel = 1

    # Spatiospectral correlation matrix.........................................
    R = np.zeros(( (shotGather.n_samples/2+1), shotGather.n_channels, shotGather.n_channels), complex)
    for m in range(shotGather.n_channels):
        for n in range(shotGather.n_channels):
            freq,R[:,m,n] = signal.csd( shotGather.timeHistories[:,m], shotGather.timeHistories[:,n], 
                            fs=1.0/shotGather.dt, window='boxcar', nperseg=shotGather.n_samples )

    # Remove frequencies above/below specificied max/min frequencies and downsample (if required by zero padding)
    fminID = np.argmin( np.absolute(freq-min_frequency)  )
    fmaxID = np.argmin( np.absolute(freq-max_frequency)  )
    freq_id = range(fminID,(fmaxID+1), shotGather.multiple)
    R = R[freq_id,:,:]
    freq = freq[freq_id]

    # Weighting matrices........................................................
    W = np.zeros( np.shape(R) ) 
    # Sqare root of distance from source   
    if str.lower(weightType) == 'sqrt': 
        W[:,:,:] = np.diag( np.sqrt(abs(shotGather.offset) + shotGather.position) )
    # 1/|A(f,x)|, where A is Fourier Transform of a(t,x)
    elif str.lower(weightType) == 'invamp':
        freqFFT = np.concatenate([np.arange(0, shotGather.fnyq+shotGather.df, shotGather.df), np.arange(-(shotGather.fnyq-shotGather.df), 0, shotGather.df) ])
        Af = np.fft.fft(shotGather.timeHistories, axis=0)
        for bb in range(len(freq)):
            freq_id = np.argmin( np.absolute(freqFFT-freq[bb]) )
            weight = 1.0 / np.absolute( Af[freq_id,:] ) 
            W[bb,:,:] = np.diag( weight )
    # No weighting  
    else: 
        W[:,:,:] = np.eye( shotGather.n_channels ) 

    # Beamforming...............................................................
    v_vals = np.linspace(min_vel, max_vel, numv)
    # Pre-allocate variables for efficiency
    power = np.zeros( (numv, len(freq)), complex )
    pnorm = np.zeros( np.shape(power), complex ) 
    v_peak = np.zeros( np.shape(freq) )
    # Loop through all frequency values, compute power at all trial wavenumbers
    for m in range( len(freq) ):
        # Convert trial velocities to wavenumbers (set equal to 0 for k > kres)
        k_vals = 2*np.pi*freq[m] / v_vals
        alias_id = np.where( k_vals > shotGather.kres )[0]
        # Weighting matrix for current frequency
        Wf = W[m,:,:]
        for k in range( numv ):
            # Steering vector
            if str.lower(steeringVector) == 'cylindrical': 
                pos = shotGather.position
                # If x[0]=0, set equal to arbitrarilly small number for stability                
                if pos[0]==0:
                    pos[0] = 1e-16
                H0 = special.j0( k_vals[k] * pos ) + 1j*special.y0( k_vals[k] * pos )
                expterm = np.exp( 1j * np.angle(H0) )
            else:
                expterm = np.exp( 1j * k_vals[k] * shotGather.position )
            # power[k,m] = expterm' * Wf * R[m,:,:] * Wf' * expterm
            power[k,m] = np.dot( np.dot( np.dot( np.dot( np.conj(expterm).transpose(),Wf ),R[m,:,:] ),Wf.transpose()),expterm)
            power[alias_id,m] = 0
        # Index of wavenumber corresponding to maximum power at freq[m]
        max_id = np.argmax( np.abs(power[:,m]) ) 
        # Normalize all power values at freq[m] by the maximum power at freq[m]
        pnorm[:,m] = np.abs(power[:,m]) / np.max(np.abs(power[:,m])) 
        pnorm[alias_id,m] = float( 'nan' )
        # Wavenumber corresponding to max power at freq[m]
        v_peak[m] = v_vals[max_id]

    # Create instance of DispersionPower class
    dispersionPower = dctypes.DispersionPower( freq, v_peak, v_vals, 'velocity', shotGather.kres, pnorm )
    return dispersionPower    



#*******************************************************************************
# Phase-Shift Transformation
def phase_shift( shotGather, num_vel=2048, min_frequency=5, max_frequency=100, min_velocity=1, max_velocity=1000 ):

    # Ensure that min_velocity is greater than zero for numerical stability
    if min_velocity < 1:
        min_velocity = 1
    
    # Frequency vector
    freq = np.arange(0, shotGather.fnyq, shotGather.df)

    # FFT of timeHistories (Equation 1 of Park et al. 1998).....................
    U = np.fft.fft(shotGather.timeHistories, axis=0)
    
    # Remove frequencies above/below specificied max/min frequencies and downsample (if required by zero padding)
    fminID = np.argmin( np.absolute(freq-min_frequency) )
    fmaxID = np.argmin( np.absolute(freq-max_frequency) )
    freq_id = range(fminID,(fmaxID+1), shotGather.multiple)
    freq = freq[freq_id]
    U = U[freq_id, :]

    # Trial velocities
    v_vals = np.linspace( min_velocity, max_velocity, num_vel )

    # Initialize variables
    v_peak = np.zeros( np.shape(freq) )
    V = np.zeros( (np.shape(v_vals)[0], len(freq)) )
    pnorm = np.zeros( np.shape(V) )

    # Transformation ...........................................................
    # Loop through frequencies
    for c in range( len(freq) ):
        # Loop through trial velocities at each frequency
        for r in range( np.shape(v_vals)[0] ):
            # Set power equal to NaN at wavenumbers > kres
            if v_vals[r] < (2*np.pi*freq[c]/shotGather.kres):
                V[r,c] = float( 'nan' )
            # (Equation 4 in Park et al. 1998)
            else:
                V[r,c] = np.abs( np.sum( U[c,:]/np.abs(U[c,:]) * np.exp( 1j*2*np.pi*freq[c]*shotGather.position / v_vals[r] ) ) )

        # Identify index associated with peak power at current frequency
        max_id = np.nanargmax( V[:,c] )
        pnorm[:,c] = V[:,c] / V[max_id,c]
        v_peak[c] = v_vals[max_id]

    # Create instance of DispersionPower class..................................
    dispersionPower = dctypes.DispersionPower( freq, v_peak, v_vals, 'velocity', shotGather.kres, pnorm )
    return dispersionPower 
                
    
    
#*******************************************************************************
# Slant-stack transform
def tau_p( shotGather, num_vel=2048, min_frequency=5, max_frequency=100, min_velocity=1, max_velocity=1000 ):
        
    # Ensure that min_velocity is greater than zero
    if min_velocity<80:
        print 'Warning: minimum velocity may result in loss of precision, 80 m/s recommended for slant-stack transform'
        if min_velocity<1:
            min_velocity = 1
    
    # Processing parameters.......................................................
    freq = np.arange(0, shotGather.fnyq+shotGather.df, shotGather.df)
    h = shotGather.position

    # Processing parameters.....................................................
    p_max = 1.0 / min_velocity
    # Trial slowness values (positive and negative for two-directions, positive for one-direction)
    q = np.linspace(0, p_max ,num_vel+1)    
    # Number of trial slowness values and location of p=0
    nq = len(q)


    # Perform slant-stack (Radon) transform in freq. domain for efficiency......
    # Refer to Sacchi and Ulrych (1995)
    # d = observed data
    # D = FFT of observed data
    # m = data transformed to the tau-p domain
    # M = FFT of m
    nfft = 2**nextpow2(shotGather.n_samples)
    D = np.fft.fft( shotGather.timeHistories, n=nfft, axis=0 )
    M = np.zeros( (nfft, nq), complex )
    
    klow = 1
    khigh = int( np.floor(nfft/2) )

    for kfreq in range(klow, khigh+1): 
        omega = 2*np.pi*kfreq / nfft / shotGather.dt
        # Equations 5 and 6 in Sacchi (1995)
        L = np.exp( -1j*omega*np.matrix(h).H * np.matrix(q) )
        y = np.matrix(D[kfreq,:]).transpose() 
        x = L.H * y
        M[kfreq,:] = np.array( x.transpose() )
        M[nfft-kfreq,:] = np.array( x.conj().transpose() )

    M[nfft/2,:] = np.zeros( (1,nq) )
    m = np.real( np.fft.ifft(M, axis=0) )
    m = m[0:shotGather.n_samples,:] 

    # Compute dispersion data as described in Louie (2001), but only using positive slowness values

    # Compute FFT on p-f trace (equation 4)
    Fa = np.fft.fft(m, axis=0)

    # Compute power spectrum for p (equaton 5)
    Sa = np.real( Fa[0:(shotGather.n_samples/2),:] * Fa[0:(shotGather.n_samples/2),:].conj() )

    # Normalize by the maximum power at a given frequency
    pnorm = np.zeros( np.shape(Sa)  )
    for j in range( 0, np.shape(Sa)[0] ):
        pnorm[j,:] = Sa[j,:] / np.max(Sa[j,:])

    # Remove frequencies above/below specified max/min frequencies and downsample (if required by zero padding)
    fminID = np.argmin( np.absolute(freq-min_frequency) )
    fmaxID = np.argmin( np.absolute(freq-max_frequency) )
    freq_id = range(fminID,(fmaxID+1), shotGather.multiple)
    freq = freq[freq_id]
    pnorm = pnorm[freq_id, :]


    # Find peaks in power spectrum..............................................
    
    # Pre-allocate memory for variables 
    v_alias = np.zeros( np.shape(freq) )
    p_alias = np.zeros( np.shape(freq) )
    p_peak =  np.zeros( np.shape(freq) )
    max_power = np.zeros( np.shape(freq) ) 

    # Compute peaks in power spectrum (avoid aliased velocites)
    for n in range( len(freq) ):
        # Aliasing criteria
        v_alias[n] = freq[n]*2*np.pi / shotGather.kres
        p_alias[n] = 1.0 / v_alias[n]
        # Set power associated with all aliased wavenumbers equal to NaN
        aliased_id = np.where( q > p_alias[n] )[0]
        pnorm[n, aliased_id] = float('nan')
        # Find maximum power
        max_id = np.nanargmax( pnorm[n,:] )
        max_power[n] = pnorm[n,max_id]
        # If maximum is associated with slowness=0, set p_peak equal to arbitrary high number (for numerical stability)
        p_peak[n] = np.max(np.array([q[max_id], 1e-10]))
    v_peak = 1.0/p_peak


    # Transpose matrices so that frequency is on x-axis and remove column for slowness=0
    q = q[1::]    
    v_vals = 1.0/q
    pnorm = pnorm[:,1::].transpose()
        

    # Create instance of DispersionPower class..................................
    dispersionPower = dctypes.DispersionPower( freq, v_peak, v_vals, 'velocity', shotGather.kres, pnorm )
    return dispersionPower        
            
    
    

#*******************************************************************************
# Auxillary functions

# Function to find exponent that will result in 2^exponent >= value
def nextpow2( val ):
    exponent = 0
    while 2**exponent < val:
        exponent += 1
    return exponent 
