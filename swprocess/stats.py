"""Statistics class definition."""

import numpy as np


class Statistics():

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

    @staticmethod
    def _sort_data(data_matrix):
        """Reorganize rows to increase point density."""
        nrows, ncols = data_matrix.shape
        tmp_row = np.empty(ncols, dtype=data_matrix.dtype)
        row_ids = np.arange(nrows, dtype=int)
        c_row = 0
        for column in data_matrix.T:
            val_ids = np.argwhere(np.logical_and(~np.isnan(column),
                                                 row_ids >= c_row))
            for val_id in val_ids.flatten():
                tmp_row[:] = data_matrix[c_row][:]
                data_matrix[c_row][:] = data_matrix[val_id][:]
                data_matrix[val_id][:] = tmp_row[:]
                c_row += 1
        return data_matrix

    @staticmethod
    def _identify_regions(data_matrix, density_threshold=0.9):
        """Find the largest regions of the data_matrix with at least
        the minimum density_threshold."""
        nrows, ncols = data_matrix.shape
        s_row, s_col = 0, 0

        regions = []
        for _ in range(100):
            c_row, c_col = int(s_row), int(s_col)
    
            n_vals, totaln = 1, 1

            remaining_rows, remaining_cols = (nrows-s_row), (ncols-s_col)
            for _ in range(remaining_rows + remaining_cols):

                # Check next row (if it exists).
                if c_row + 1 < nrows:
                    row = data_matrix[c_row+1, s_col:c_col+1]
                    # print(row)

                    total = len(row)
                    n_nan = np.sum(np.isnan(row))
                    n_val = total - n_nan
                    density = (n_vals + n_val)/(totaln + total)
                    # print(density)

                    if density >= density_threshold:
                        row_failed = False
                        n_vals += n_val
                        totaln += total
                        c_row += 1
                    else:
                        row_failed = True
                else:
                    row_failed = True

                # Check next column (if it exists).
                if c_col + 1 < ncols:
                    col = data_matrix[s_row:c_row+1, c_col+1]
                    # print(col)

                    total = len(col)
                    n_nan = np.sum(np.isnan(col))
                    n_val = total - n_nan
                    density = (n_vals + n_val)/(totaln + total)
                    # print(density)

                    if density >= density_threshold:
                        col_failed = False
                        n_vals += n_val
                        totaln += total
                        c_col += 1
                    else:
                        col_failed = True
                else:
                    col_failed = True

                if row_failed and col_failed:
                    regions.append(((s_row, s_col), (c_row+1, c_col+1)))
                    break

            else: # pragma: no cover
                raise ValueError("Iteration stopped without breaking.")

            if c_row+1 == nrows and c_col+1 == ncols:
                break
            else:
                s_col = int(c_col)+1
                s_row = np.argwhere(~np.isnan(data_matrix[:, s_col]))[0][0]

        else: # pragma: no cover
            raise ValueError("Number of regions exceeded 100.")
            

        return regions

    @staticmethod
    def _calc_density(data_matrix, tl_corner, br_corner):
        """Calculate value density given a start and stop location."""
        b_row, b_col = tl_corner
        e_row, e_col = br_corner
        n_nums = np.sum(~np.isnan(data_matrix[b_row:e_row+1, b_col:e_col+1]))
        n_poss = (e_col - b_col + 1) * (e_row - b_row + 1)
        return n_nums/n_poss
