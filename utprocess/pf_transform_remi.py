import math
import numpy as np
from scipy import signal
import dctypes


# Slant-stack transform that can be used for ReMi Testing
def pfTransform( shotGather, num_vel=2048, min_frequency=5, max_frequency=100, min_velocity=1, max_velocity=1000, normType='max', rLimit=7, one_direction=True ):
        
    # Ensure that min_velocity is greater than zero
    if min_velocity<=0:
        min_velocity = 1
        print 'Warning: minimum velocity must be greater than zero. Set equal to 1'
    
    # Sampling parameters.......................................................
    # Time
    fnyq = 1/(2*shotGather.dt)
    df = 1/(shotGather.n_samples*shotGather.dt)
    freq = np.arange(0, fnyq+df, df)
    # Space
    h = shotGather.position
    kres = 2*np.pi / min(np.diff(shotGather.position))    

    # Processing parameters.....................................................
    p_max = 1.0 / min_velocity
    # Trial slowness values (positive and negative for two-directions, positive for one-direction)
    temp = np.linspace(0, p_max ,num_vel+1)    
    if one_direction:
        q = temp 
        zero_id = 0
    else:
        q = np.concatenate( (-temp[::-1], temp[1::]) )
        zero_id = len(temp)-1
    # Number of trial slowness values and location of p=0
    nq = len(q)


    # Perform slant-stack (Radon) transform in freq. domain for efficiency......
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

    # Compute dispersion data as described in Louie (2001)......................

    # Compute FFT on p-f trace (equation 4)
    Fa = np.fft.fft(m, axis=0)

    # Compute power spectrum for p (equaton 5)
    Sa1 = np.real( Fa[0:(shotGather.n_samples/2),:] * Fa[0:(shotGather.n_samples/2),:].conj() )

    if not one_direction:
        # Compute power spectrum for |p| (equation 6)
        Sa1 = Sa1 + np.fliplr(Sa1[:,:])
        # Don't double count for p=0
        Sa1[:,zero_id] = Sa1[:,zero_id]/2
    
    # Truncate 
    Sa = Sa1[:,(zero_id+1)::]
    q = q[(zero_id+1)::]

    # Normalization
    pnorm = np.zeros( np.shape(Sa)  )
    if str.lower(normType)=='louie':
        # Normalize by the average power over all the slowness values at a given frequency
        # (equation 7)
        Stotal = np.sum(Sa, axis=1)
        for j in range( 0, np.shape(Sa)[0] ):
            pnorm[j,:] = Sa[j,:] / Stotal[j] * len(q)
    elif str.lower(normType)=='max':
        # Normalize by the maximum power at a given frequency
        for j in range( 0, np.shape(Sa)[0] ):
            pnorm[j,:] = Sa[j,:] / np.max(Sa[j,:])
    else:
        raise ValueError("Invalid normalization type")


    # Remove frequencies above/below specified max/min frequencies and downsample (if required by zero padding)
    fminID = np.where( np.absolute(freq-min_frequency) == min(np.absolute(freq-min_frequency)) )[0][0] 
    fmaxID = np.where( np.absolute(freq-max_frequency) == min(np.absolute(freq-max_frequency)) )[0][0] 
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
        v_alias[n] = freq[n]*2*np.pi / kres
        p_alias[n] = 1/v_alias[n]
        # Set power associated with all aliased wavenumbers equal to NaN
        aliased_id = np.where( q > p_alias[n] )[0]
        pnorm[n, aliased_id] = float('nan')
        # Find maximum power
        max_id = np.nanargmax( pnorm[n,:] )
        max_power[n] = pnorm[n,max_id]
        p_peak[n] = q[max_id]
    v_peak = 1/p_peak


    # Additional normalization for plotting purposes............................
    # (only needed for normalization described by Louie 2001)
    if str.lower(normType)=='louie':
        if str.lower(str(rLimit))=='none':
            rLimit = math.ceil( np.max(max_power) )
        for r in range( np.shape(pnorm)[0] ): 
            for c in range( np.shape(pnorm)[1] ):
                if not np.isnan(pnorm[r,c]) and (pnorm[r,c]>rLimit):
                    pnorm[r,c] = rLimit


    # Transpose matrices so that frequency is on x-axis and remove column for slowness=0
    v_vals = 1/q
    pnorm = pnorm.transpose()
        

    # Create instance of DispersionPower class..................................
    dispersionPower = dctypes.DispersionPower( freq, v_peak, v_vals, "velocity", kres, pnorm )
    return dispersionPower        