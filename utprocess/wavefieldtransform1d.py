"""This file contains a class for performing and manipulating a
wavefield transformation for a 1D array."""

from utprocess import DispersionPower
import json
import numpy as np


class WavefieldTransform1D():
    """Class for performing and manipulating wavefield 
    transformations for a 1D array.

    Attributues:
        timelength: Length of time record used in seconds.

        method: Type of wavefield transformation.

        f_trial: Dictionary of the form {'min': ,'max': , 'npts': } which
            defines the minimum, maximum, and number of frequency points.

        weighting: Type of weighting vector, can be ['none', ...]

        steering_vector: Type of steering vector, can be ['plane', ...]

    """

    def __init__(self, array, settings_file):
        """Initialize an instance of the WavefieldTransformation1D class 
        from an instance of the Array1D class.

        Args:
            array: Instance of an Array1D class.

            settings: Name of a .json file that descibes the settings to be
                used for the 1D wavefield transformation.

        Returns:


        Raises:

        """
        self.array = array

        with open(settings_file, "r") as f:
            settings = json.load(f)

        if settings["type"] == "fk":
            numk = settings["general"]["n_trial"]
            if numk % 2 != 0:
                numk += 1
            # TODO (jpv) generalize numk to be n_trial, so move above outside of if statement

            freq = np.arange(0, self.array.fnyq+self.array.df, self.array.df)

            kres = self.array.kres
            dk = 2*np.pi / (numk*self.array.spacing)
            k_vals = np.arange(dk, kres, dk)
            
            fk = np.fft.fft2(self.array.timeseriesmatrix,
                             s=(self.array.nsamples, numk))
            fk = np.fliplr(np.abs(fk))
            fk = fk[0:len(freq), 1::]

            # Remove frequencies above/below specificied max/min frequencies and downsample (if required by zero padding)
            # TODO (jpv): I think there is cleaner syntax for this.
            # fminID = np.argmin(np.absolute(freq-min_frequency))
            # fmaxID = np.argmin(np.absolute(freq-max_frequency))
            # freq_id = range(fminID, (fmaxID+1), shotGather.multiple)
            # freq = freq[freq_id]
            # fk = fk[freq_id, :]

            # Identify wavenumber associated with maximum in fk domain..................
            pnorm = np.zeros(np.shape(fk))
            k_peak = np.zeros(np.shape(freq))
            for k in range(np.shape(fk)[0]):
                pnorm[k, :] = fk[k, :] / np.max(fk[k, :])
                k_peak[k] = k_vals[np.argmax(pnorm[k, :])]

            self.disp_power = DispersionPower(freq=freq,
                                              peak_vals=k_peak,
                                              trial_vals=k_vals,
                                              val_type='wavenumber',
                                              kres=kres,
                                              pnorm=np.transpose(pnorm))
