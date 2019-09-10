import numpy as np
import warnings


# DARENDELI (2001)..............................................................

# Global variables used in Darendeli (2001) and Menq calculations

# See 9.2 in Darendeli (2001) for constants phi[1] through phi[12]
# Add phi[0]=0 so that indexing is consistent with notation in dissertation
phi = np.array( [0, 0.0352, 0.0010, 0.3246, 0.3483, 0.9190, 0.8005,
                 0.0129, -0.1069, -0.2889, 0.2919, 0.6329,
                 -0.0057, -4.2300, 3.6200, -5.0000, -0.2500] )



def darendeliCalc( meStress, strain=np.logspace(-4,1,45), PI=-1, ocr=-1, freq=1, ncycles=10, curve_type=1 ):

    # See 9.3 in Darendeli (2001) for equations for c1, c2, and c3
    #c1 = -1.1143*(phi[5]**2) + 1.8618*(phi[5]) + 0.2523
    c1 = 1.0221998776999999
    #c2 = 0.0805*(phi[5]**2) -0.071*phi[5] -0.0095
    c2 = -0.0067618394999999967
    #c3 = -0.0005*(phi[5]**2) + 0.0002*phi[5] + 0.0003
    c3 = 6.1519499999999905e-05

    # If mean effective stress and strain are scalars, convert to arrays
    if len(np.shape(meStress))==0:
        meStress = np.array([[meStress]])
    elif len(np.shape(meStress))==1:
        meStress = np.reshape(meStress,[len(meStress),1])
    if len(np.shape(strain))==0:
        strain = np.array([strain])

    (nl, npr) = np.shape(meStress)
    if npr>1:
        warnings.warn("Warning: darendeliCalc is intended for one soil profile at a time. Only using first profile.")
        meStress = meStress[:,0] 

    if np.isscalar(PI) and PI<0:
        PI = np.zeros( np.shape(meStress) )
    elif len(np.shape(PI))>1:
        PI = PI[:,0]

    if np.isscalar(ocr) and ocr<0:
        ocr = np.ones( np.shape(meStress) )
    elif len(np.shape(ocr))>1:
        ocr = ocr[:,0]   
    
         
    # Eq 9.1d
    b = phi[11] + phi[12]*np.log(ncycles)    
    # Eq. 9.1a
    lam_r = ( phi[1] + phi[2]*PI*ocr**phi[3] ) * meStress**phi[4]

    # Number of strain values and number of layers
    nl = len(meStress)
    ns = len(strain)
        

    # G/Gmax curves
    ggmax = np.zeros( (nl,ns) )
    ggmax1 = np.zeros( (nl,ns) )
    ggmaxStd = np.zeros( (nl,ns) )

    for j in range(nl):
        for k in range(ns):
            # Eq. 12.1a
            ggmax1[j,k] = ( 1/( 1 + (strain[k]/lam_r[j])**phi[5] ) )
            # Eq. 12.2e
            ggmaxStd[j,k] = np.exp(phi[13]) + np.sqrt(  ( 0.25/np.exp(phi[14]) ) - ( (ggmax1[j,k]-0.5)**2 / np.exp(phi[14]) )  )

            # curvetypes 1, 2, and 3 represent the mean, mean+std, mean-std
            if curve_type==1:
                ggmax[j,k] = ggmax1[j,k]
            elif curve_type==2:
                ggmax[j,k] = ggmax1[j,k] + ggmaxStd[j,k]
            elif curve_type==3:
                ggmax[j,k] = ggmax1[j,k] - ggmaxStd[j,k]
                
            # Ensure that G/Gmax is not less than 0.005 or greater than 1
            if ggmax[j,k]<0.005:
                ggmax[j,k] = 0.005
            elif ggmax[j,k]>1:
                ggmax = 1.0


    # Damping curves
    # Eq.12.2c        
    dmin = (phi[6] + phi[7]*PI*(ocr**phi[8])) * (meStress**phi[9]) * (1+(phi[10]*np.log(freq))) 
    
    dmas1 = np.zeros( (nl,ns) )
    dmas = np.zeros( (nl,ns) )
    damp1 = np.zeros( (nl,ns) )
    dampStd = np.zeros( (nl,ns) )
    damping = np.zeros( (nl,ns) )
    for j in range(nl):
        for k in range(ns):
            # (see 9.3 for the following equations)
            # Dmasing for a=1.0
            dmas1[j,k] = ( (100/np.pi) * (  4*(  (strain[k]-lam_r[j]*np.log( (strain[k]+lam_r[j])/lam_r[j]) ) / ( (strain[k]**2)/(strain[k]+lam_r[j]) )  ) - 2  ) )
            # Dmasing
            dmas[j,k] = c1*dmas1[j,k] + c2*dmas1[j,k]**2 + c3*dmas1[j,k]**3
            # Eq. 12.1b
            damp1[j,k] = dmin[j] + ( dmas[j,k]*( b * ( ggmax1[j,k]**0.1 ) ) )
            # Eq. 12.2f
            dampStd[j,k] = np.exp(phi[15]) + ( np.exp(phi[16])*( damp1[j,k]**(1/2) ) )

            if curve_type==1:
                damping[j,k] = damp1[j,k]
            elif curve_type==2:
                damping[j,k] = damp1[j,k] + dampStd[j,k]
            elif curve_type==3:
                damping[j,k] = damp1[j,k] - dampStd[j,k]

            # Ensure that damping is not less than 0.1%
            if damping[j,k]<0.1:
                damping[j,k] = 0.1

            # Ensure that damping is greater than for previous strain value
            if k>0:
                if damping[j,k]<damping[j,k-1]:
                    damping[j,k] = damping[j,k-1] 

    return ggmax, damping, strain
                
                

    
    
    
    
    