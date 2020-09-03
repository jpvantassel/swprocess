"""Class definitions for various wavefield transforms."""

from register import WavefieldTransformRegistry
from abstract_transform import AbstractTransform

# @WavefieldTransformRegistry.register('fk')
class FK(AbstractTransform):
    
    def __init__(self):
        """Perform Frequency-Wavenumber (fk) transform.

        The FK approach utilizes a 2D Fourier Transform to transform
        data from the time-space domain to the frequency-wavenumber
        domain. The FK approach was adapted by Gabriels et al. (1987)
        for linear arrays from the FK approach developed by Nolet and
        Panza (1976) for 2D arrays.

        Parameters
        ----------
        array : Array1d
            One-dimensional array object.
        nwave : int
            Number of wavenumbers to consider.
        fmin, fmax : float
            Minimum and maximum frequency of interest in the
            transformation.

        Returns
        -------
        Tuple
            Of the form `(frqs, domain, ks, pnorm, kpeaks)`.

        """
        # Frequency vector
        sensor = array.sensors[0]
        frqs = np.arange(sensor.nsamples) * sensor.df

        # Perform 2D FFT
        if array._flip_required:
            tseries = np.flipud(array.timeseriesmatrix)
        else:
            tseries = array.timeseriesmatrix
        fk = np.fft.fft2(tseries, s=(nwave, sensor.nsamples))
        fk = np.abs(fk[-2::-1, 0:len(frqs)])

        # Trim frequencies and downsample (if required by zero padding)
        fmin_ids = np.argmin(np.abs(frqs-fmin))
        fmax_ids = np.argmin(np.abs(frqs-fmax))
        freq_ids = range(fmin_ids, (fmax_ids+1), sensor.multiple)
        frqs = frqs[freq_ids]
        fk = fk[:, freq_ids]

        # Wavenumber vector
        kres = array.kres
        dk = 2*kres / nwave
        ks = np.arange(dk, 2*kres, dk)

        # Normalize power and find peaks
        pnorm = np.empty_like(fk)
        kpeaks = np.empty_like(frqs)
        for k, _fk in enumerate(fk.T):
            normed_fk = np.abs(_fk/np.max(_fk))
            pnorm[:, k] = normed_fk
            kpeaks[k] = ks[np.argmax(normed_fk)]

        return (frqs, "wavenumber", ks, pnorm, kpeaks)

    def plot(self, *args, **kwargs):
        raise NotImplementedError

@WavefieldTransformRegistry.register('slantstack')
class SlantStack(AbstractTransform):

    def _slant_stack(array, velocities):
        """Perform a slant-stack on the given wavefield data.

        Parameters
        ----------
        array : Array1d
            One-dimensional array object.

        Returns
        -------
        tuple
            Of the form `(tau, slant_stack)` where `tau` is an ndarray
            of the attempted intercept times and `slant_stack` are the
            slant-stacked waveforms.

        """
        if array._flip_required:
            tmatrix = np.flipud(array.timeseriesmatrix)
        else:
            tmatrix = array.timeseriesmatrix

        position = np.array(array.position)
        position -= np.min(position)
        nchannels = array.nchannels
        diff = position[1:] - position[:-1]
        diff = diff.reshape((len(diff), 1))
        dt = array.sensors[0].dt
        npts = tmatrix.shape[1]
        ntaus = npts - int(np.max(position)*np.max(1/velocities)/dt) - 1
        slant_stack = np.empty((len(velocities), ntaus))
        rows = np.tile(np.arange(nchannels).reshape(nchannels, 1), (1, ntaus))
        cols = np.tile(np.arange(ntaus).reshape(1, ntaus), (nchannels, 1))

        pre_float_indices = position.reshape(nchannels, 1)/dt
        previous_lower_indices = np.zeros((nchannels, 1), dtype=int)
        for i, velocity in enumerate(velocities):
            float_indices = pre_float_indices/velocity
            lower_indices = np.array(float_indices, dtype=int)
            delta = float_indices - lower_indices
            cols += lower_indices - previous_lower_indices
            amplitudes = tmatrix[rows, cols]*(1-delta) + tmatrix[rows, cols+1]*delta
            integral = 0.5*diff*(amplitudes[1:, :] + amplitudes[:-1, :])
            summation = np.sum(integral, axis=0)
            slant_stack[i, :] = summation

            previous_lower_indices[:] = lower_indices
        taus = np.arange(ntaus)*dt
        return (taus, slant_stack)



    """Perform the Slant-Stack transform.

    Parameters
    ----------
    array : Array1d
        One-dimensional array object.


    Returns
    -------

    """
    _, tau_p = WavefieldTransform1D._slant_stack(array, velocities)

    # Frequency vector
    sensor = array.sensors[0]
    ntaus = tau_p.shape[1]
    df = 1/(ntaus*sensor.dt)
    frqs = np.arange(ntaus) * df

    # TODO (jpv): Adjust zero padding for the slant-stack. Need to
    # be padding in the tau-p domain rather than in the x-t domain.

    # Fourier Transform of Slant Stack
    fp = np.fft.fft(tau_p)

    # Trim frequencies and downsample (if required by zero padding)
    fmin_ids = np.argmin(np.abs(frqs-fmin))
    fmax_ids = np.argmin(np.abs(frqs-fmax))
    freq_ids = range(fmin_ids, (fmax_ids+1), sensor.multiple)
    frqs = frqs[freq_ids]
    fp = fp[:, freq_ids]

    # Normalize power and find peaks
    pnorm = np.empty(fp.shape)
    vpeaks = np.empty_like(frqs)
    # fp = np.abs(fp/np.max(fp))
    abs_fp = np.abs(fp)
    for k, _fp in enumerate(abs_fp.T):
        normed_fp = _fp/np.max(_fp)
        pnorm[:, k] = normed_fp
        vpeaks[k] = velocities[np.argmax(normed_fp)]

    return (frqs, "velocity", velocities, pnorm, vpeaks)

@WavefieldTransformRegistry.register('phaseshift')
class PhaseShift(AbstractTransform):
    """Perform the Phase-Shift transform.

    Parameters
    ----------
    array : Array1d
        One-dimensional array object.
    fmin, fmax : float
        Minimum and maximum frequency of interest in the
        transformation.
    vmin, vmax : float
        Minimum and maximum velocity of interest in the
        transformation.
    nvel : int
        Number of trial velocities to attempt between vmin and vmax.

    Returns
    -------
    Tuple
        Of the form `(frqs, domain, vels, pnorm, vpeaks)`.

    """
    # Frequency vector
    sensor = array.sensors[0]
    frqs = np.arange(sensor.nsamples) * sensor.df

    # u(x,t) -> FFT -> U(x,f)
    if array._flip_required:
        offsets = array.offsets[::-1]
        tmatrix = np.flipud(array.timeseriesmatrix)
    else:
        offsets = array.offsets
        tmatrix = array.timeseriesmatrix
    offsets = np.array(offsets)
    trans = np.fft.fft(tmatrix)

    # Trim frequencies and downsample (if required by zero padding)
    fmin_ids = np.argmin(np.abs(frqs-fmin))
    fmax_ids = np.argmin(np.abs(frqs-fmax))
    freq_ids = range(fmin_ids, (fmax_ids+1), sensor.multiple)
    frqs = frqs[freq_ids]
    trans = trans[:, freq_ids]

    # Integrate across the array offsets
    power = np.empty((len(frqs), nvel))
    vels = np.linspace(vmin, vmax, nvel)
    dx = offsets[1:] - offsets[:-1]
    for f_index, frq in enumerate(frqs):
        for v_index, vel in enumerate(vels):
            shift = np.exp(1j * 2*np.pi*frq/vel * offsets)
            inner = shift*trans[:, f_index]/np.abs(trans[:, f_index])
            power[f_index, v_index] = np.abs(
                np.sum(0.5*dx*(inner[:-1] + inner[1:])))

    # Normalize power and find peaks
    pnorm = np.empty_like(power)
    vpeaks = np.empty_like(frqs)
    pnorm = power/np.max(power)
    for k, _power in enumerate(pnorm):
        # normed_power = np.abs(_power/np.max(_power))
        # pnorm[k, :] = normed_power
        vpeaks[k] = vels[np.argmax(_power)]

    return (frqs, "velocity", vels, pnorm.T, vpeaks)

@WavefieldTransformRegistry.register('fdbf')
class FDBF(AbstractTransform):

        @staticmethod
    def _spatiospectral_correlation_matrix(tmatrix, frq_ids=None, weighting=None):
        """Compute the spatiospectral correlation matrix.

        Parameters
        ----------
        tmatrix : ndarray
            Three-dimensional matrix of shape
            `(samples_per_block, nblocks, nchannels)`. 
        fmin, fmax : float, optional
            Minimum and maximum frequency of interest.

        Returns
        -------
        ndarray
            Of size `(nchannels, nchannels, nfrqs)` containing the
            spatiospectral correlation matrix.

        """
        nchannels, samples_per_block, nblocks = tmatrix.shape

        # Perform FFT
        transform = np.fft.fft(tmatrix, axis=1)

        # Trim FFT
        if frq_ids is not None:
            transform = transform[:, frq_ids, :]

        # Define weighting matrix
        if weighting == "invamp":
            _, nfrqs, _ = transform.shape
            weighting = 1/np.abs(np.mean(transform, axis=-1))

            for i in range(nfrqs):
                w = weighting[:, i]
                for b in range(nblocks):
                    transform[:, i, b] *= w

        # Calculate spatiospectral correlation matrix
        nchannels, nfrqs, nblocks = transform.shape
        spatiospectral = np.empty((nchannels, nchannels, nfrqs), dtype=complex)
        scm = np.zeros((nchannels, nchannels), dtype=complex)
        tslice = np.zeros((nchannels,1), dtype=complex)
        tslice_h = np.zeros((1, nchannels), dtype=complex)
        for i in range(nfrqs):
            scm[:, :] = 0
            for j in range(nblocks):
                tslice[:, 0] = transform[:, i, j]
                tslice_h[0, :] = np.conjugate(tslice)[:,0]
                scm += np.dot(tslice, tslice_h)
            scm /= nblocks
            spatiospectral[:, :, i] = scm[:]

        return spatiospectral

    if array._flip_required:
        offsets = array.offsets[::-1]
        tmatrix = np.flipud(array.timeseriesmatrix)
    else:
        offsets = array.offsets
        tmatrix = array.timeseriesmatrix
    offsets = np.array(offsets)
    sensor = array.sensors[0]
    tmatrix = tmatrix.reshape(array.nchannels, sensor.nsamples, 1)

    # Frequency vector
    frqs = np.arange(sensor.nsamples) * 1/(sensor.nsamples*sensor.dt)
    fmin_ids = np.argmin(np.abs(frqs-fmin))
    fmax_ids = np.argmin(np.abs(frqs-fmax))
    frq_ids = range(fmin_ids, (fmax_ids+1), sensor.multiple)
    frqs = frqs[frq_ids]

    spatiospectral = WavefieldTransform1D._spatiospectral_correlation_matrix(tmatrix, frq_ids, weighting=weighting)

    # Weighting
    if weighting is None:
        w = np.ones((array.nchannels, array.nchannels) )
    elif weighting == "sqrt":
        offsets_n = offsets.reshape(array.nchannels, 1)
        offsets_h = np.transpose(np.conjugate(offsets_n))
        w = np.dot(offsets_n, offsets_h)
    elif weighting == "invamp":
        w = np.ones((array.nchannels, array.nchannels) )
    else:
        raise NotImplementedError

    # Steering
    if steering == "cylindrical":
        def create_steering(kx):
            return np.exp(-1j * np.angle(special.j0(kx) + 1j*special.y0(kx)))
    elif steering == "plane":
        def create_steering(kx):
            return np.exp(-1j * kx)
    else:
        raise NotImplementedError

    steering = np.empty((array.nchannels, 1), dtype=complex)
    power = np.empty((len(velocities), len(frqs)), dtype=complex)
    for i, f in enumerate(frqs):
        weighted_spatiospectral = spatiospectral[:, :, i]*w
        for j, v in enumerate(velocities):
            kx = 2*np.pi*f/v * offsets
            steering[:, 0] = create_steering(kx)[:]
            _power = np.dot(np.dot(np.transpose(np.conjugate(steering)), weighted_spatiospectral), steering)
            power[j, i] = _power

    # Normalize power and find peaks
    pnorm = np.empty(power.shape)
    vpeaks = np.empty_like(frqs)
    for k, _fp in enumerate(power.T):
        normed_p = np.abs(_fp)/np.max(np.abs(_fp))
        pnorm[:, k] = normed_p
        vpeaks[k] = velocities[np.argmax(normed_p)]

    return (frqs, "velocity", velocities, pnorm, vpeaks)
