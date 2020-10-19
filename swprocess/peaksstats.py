"""PeaksStats class definition."""

import numpy as np


class PeaksStats():

    def __init__(self, xx, data):
        self.xx = np.array(xx, dtype=float)
        self.data = np.array(data, dtype=float)

    @classmethod
    def from_peakssuite(cls, xx, peakssuite, xtype="wavelength",
                        ytype="velocity"):
        xx, data = peakssuite._create_data_matrix(xtype, ytype, xx)
        cls(xx, data)

    def _statistics(self, xtype, ytype, xx, ignore_corr=True):
        """Determine the statistics of the `PeaksSuite`.

        Parameters
        ----------
        xtype : {"frequency","wavelength"}
            Axis along which to calculate statistics.
        ytype : {"velocity", "slowness"}
            Axis along which to define uncertainty.
        xx : iterable
            Values in `xtype` units where statistics are to be
            calculated.
        ignore_corr : bool, optional
            Ignore calculation of data's correlation coefficients,
            default is `True`.

        Returns
        -------
        tuple
            Of the form `(xx, mean, std, corr)` where `mean` and
            `std` are the mean and standard deviation at each point and
            `corr` are the correlation coefficients between every point
            and all other points.

        """
        npeaks = len(self.peaks)
        if npeaks < 3:
            msg = f"Cannot calculate statistics on fewer than 3 `Peaks`."
            raise ValueError(msg)

        xx = np.array(xx)
        data_matrix = np.empty((len(xx), npeaks))

        for col, _peaks in enumerate(self.peaks):
            # TODO (jpv): Allow assume_sorted should improve speed.
            interpfxn = interp1d(getattr(_peaks, xtype),
                                 getattr(_peaks, ytype),
                                 copy=False, bounds_error=False,
                                 fill_value=np.nan)
            data_matrix[:, col] = interpfxn(xx)

        xx, data_matrix = self._drop(xx, data_matrix.T,
                                     drop_sample_if_fewer_percent=0.,
                                     drop_observation_if_fewer_percent=0.,
                                     drop_sample_if_fewer_count=3)
        data_matrix = data_matrix.T

        mean = np.nanmean(data_matrix, axis=1)
        std = np.nanstd(data_matrix, axis=1, ddof=1)
        corr = None if ignore_corr else np.corrcoef(data_matrix)

        return (xx, mean, std, corr)
