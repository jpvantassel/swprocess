"""Non-linear least-square inversion."""

import numpy as np


def leastsquare_iterativealgorithm(p0, pm, covp0p0, d0, dm, covd0d0, dgdp):
    """Least square algorithm based on Tarantola and Valette (1982).

    Parameters
    ----------
    p0 : ndarray
        Prior parameter vector of shape `(j,1)` where `j` is the number
        of unknown parameters.
    pm : ndarray
        Parameter vector from prior iteration of shape `(j,1)` where
        `j` is the number of unknown parameters.
    covp0p0 : ndarray
        Covariance matrix of prior parameter information of shape
        `(j, j)` where `j` is the number of unknown parameters.
    d0 : ndarray
        Data vector of shape `(k, 1)` where `k` is the number of known
        data values.
    dm : ndarray
        Predicted data vector from the previous iteration [i.e., g(pm)
        where the function g converts pm from the parameter domain to
        the data domain] of shape `(k, 1)` where `k` is the
        number of known data values.
    covd0d0 : ndarray
        Covariance matrix of data, of shape `(k, k)` where `k` is the
        number of known data values.
    dgdp : ndarray
        Matrix of partial derivatives of `g(pm)` with respect to `p`
        of shape `(k, j)`.

    Returns
    -------
    ndarray
        Updated estimate of the model parameters (i.e., pm+1).

    """
    a = np.matmul(covp0p0, dgdp.T)
    b = np.linalg.inv(covd0d0 + np.matmul(np.matmul(dgdp, covp0p0), dgdp.T))
    c = d0 - dm + np.matmul(dgdp, (pm-p0))
    return p0 + np.matmul(np.matmul(a, b), c)
