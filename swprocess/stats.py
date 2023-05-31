# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Statistics class definition."""

import logging

import numpy as np
from numpy.random import default_rng, PCG64
from scipy.optimize import curve_fit

logger = logging.getLogger("swprocess.stats")

class Statistics():

    def __init__(self, xx, mean, stddev, corr):
        self.xx = np.array(xx, dtype=float)
        self.mean = np.array(mean, dtype=float)
        self.stddev = np.array(stddev, dtype=float)
        self.corr = np.array(corr, dytpe=float)

    @classmethod
    def from_peakssuite(cls, peakssuite, xx, xtype="wavelength",
                        ytype="velocity"):
        xx, array = peakssuite.to_array(xtype, ytype, xx)
        cls.from_array(xx, array)

    @classmethod
    def from_data_matrix(cls, xx, data_matrix, drop_kwargs=None,
                         density_threshold=0.8, replacement_attempts=10):
        """Create `Statistics` object from an array of data.

        Parameters
        ----------
        xx : iterable
            Iterable containing the x-locations associated with each
            column of `array`.
        data_matrix : ndarray
            Two-dimensional array where each row denotes an observation
            and each column a separate sampling location (e.g.,
            wavelength). The value of each entry denotes the parameter
            being estimated statistically (e.g., velocity), missing
            values are denoted with `np.nan`.

        Returns
        -------
        Statistics
            An initialized `Statistics` object.

        """
        # Drop missing data following drop_kwargs.
        drop_kwargs = {} if drop_kwargs is None else drop_kwargs
        default_drop_kwargs = dict(drop_observation_if_fewer_percent=0,
                                   drop_sample_if_fewer_percent=0,
                                   drop_sample_if_fewer_count=3)
        drop_kwargs = {**default_drop_kwargs, **drop_kwargs}
        xx, data_matrix = cls._drop(xx, data_matrix, **drop_kwargs)

        # Sort data_matrix to increase point density.
        data_matrix = cls._sort(data_matrix)

        # Calculate statistics (assume normal distribution).
        mean, stddev = cls._calc_stat(data_matrix)

        # Identify regions of highest density.
        regions = cls._identify_regions(data_matrix,
                                        density_threshold=density_threshold)

        # Random replacement.
        nan_mask = np.isnan(data_matrix)
        for _ in range(replacement_attempts):
            for region in regions:
                ((s_row, s_col), (e_row, e_col)) = region
                data_matrix[s_row:e_row, s_col:e_col] = cls._fill_data(data_matrix[s_row:e_row, s_col:e_col],
                                                                       means=mean[s_col:e_col],
                                                                       stddevs=stddev[s_col:e_col])

            new_mean, new_stddev = cls._calc_stat(data_matrix)

            p_diff_mean = np.max(np.abs((new_mean - mean)/mean))
            diff_cov = np.max(np.abs(new_stddev/new_mean - stddev/mean))
            # diff_stddev = np.max(np.abs(new_stddev - stddev))
            # p_diff_stddev = np.max(np.abs((new_stddev - stddev)/stddev))

            logger.info(f"  p_diff_mean = {p_diff_mean}")
            logger.info(f"  diff_cov = {diff_cov}")

            if p_diff_mean < 0.05 and diff_cov < 0.01:
                break
            else:
                data_matrix[nan_mask] = np.nan
        else:
            msg = f"Replacement attempts exceeded {replacement_attempts}."
            raise ValueError(msg)

        # Calculate correlation coefficients.
        corr = np.empty_like(data_matrix)
        # nan_mask = np.isnan(data_matrix)
        for region in regions:
            ((s_row, s_col), (e_row, e_col)) = region
            corr[s_row:e_row, s_col:e_col] = np.corrcoef(data_matrix[s_row:e_row, s_col:e_col],
                                                         rowvar=False)

        # Fill remaining correlation coefficients.
        corr = cls._fill_corr(xx, corr)

        return cls(xx, mean, stddev, corr)

    @staticmethod
    def _drop(xx, data_matrix,
              drop_observation_if_fewer_percent=0.8,
              drop_sample_if_fewer_percent=0.4,
              drop_sample_if_fewer_count=3):
        """Procedure for removing problematic observations/samplings.

        Parameters
        ----------
        xx : ndarray
            Statistic sampling locations.
        data_matrix : ndarray
            Of shape `(# observations, # samples)` each entry's
            value indicates the parameters value (e.g., velocity)
            the presence of `np.nan` indicates missing data.
        drop_observation_if_fewer_percent : {0. - 1.}, optional
            Remove observations if the number of valid entries is
            fewer than the specified fraction times the total
            possible, default is 0.8.
        drop_sample_if_fewer_percent : {0. - 1.}, optional
            Remove statistic sample if the number of valid entries
            is fewer than the specified fraction times the total
            possible, default is 0.4.
        drop_sample_if_fewer_count : int, optional
            Remove statistic sample if the number of valid entries
            is fewer than the specified number, default is 3.

        Returns
        -------
        tuple
            Of the form `(xx, data_matrix)` where `xx` and
            `data_matrix` are the permuted inputs.

        """
        # Initial
        i_nans = np.sum(np.isnan(data_matrix))
        i_nums = data_matrix.size - i_nans

        # Option 1: Drop columns then rows.
        drop_cols = Statistics._drop_indices(data_matrix.T,
                                             drop_if_fewer_percent=drop_sample_if_fewer_percent,
                                             drop_if_fewer_count=drop_sample_if_fewer_count)
        xx_1 = np.delete(xx, drop_cols)
        data_matrix_1 = np.delete(data_matrix, drop_cols, axis=1)
        drop_rows = Statistics._drop_indices(data_matrix_1,
                                             drop_if_fewer_percent=drop_observation_if_fewer_percent,
                                             drop_if_fewer_count=0)
        data_matrix_1 = np.delete(data_matrix_1, drop_rows, axis=0)
        r_nans = np.sum(np.isnan(data_matrix_1))
        r_nums = data_matrix_1.size - r_nans
        utility_option_1 = (i_nans - r_nans) / (i_nums - r_nums + 1)

        # Option 2: Drop rows then columns.
        drop_rows = Statistics._drop_indices(data_matrix,
                                             drop_if_fewer_percent=drop_observation_if_fewer_percent,
                                             drop_if_fewer_count=0)
        data_matrix_2 = np.delete(data_matrix, drop_rows, axis=0)
        drop_cols = Statistics._drop_indices(data_matrix_2.T,
                                             drop_if_fewer_percent=drop_sample_if_fewer_percent,
                                             drop_if_fewer_count=drop_sample_if_fewer_count)
        xx_2 = np.delete(xx, drop_cols)
        data_matrix_2 = np.delete(data_matrix_2, drop_cols, axis=1)
        r_nans = np.sum(np.isnan(data_matrix_2))
        r_nums = data_matrix_2.size - r_nans
        utility_option_2 = (i_nans - r_nans) / (i_nums - r_nums + 1)

        logger.debug(
            f"utility_option_1={utility_option_1}, utility_option_2={utility_option_2}")
        if utility_option_1 > utility_option_2:
            return (xx_1, data_matrix_1)
        else:
            return (xx_2, data_matrix_2)

    @staticmethod
    def _drop_indices(data_matrix, drop_if_fewer_percent, drop_if_fewer_count):
        """Iterate by row, return rejection indices."""
        if data_matrix.size == 0:
            return np.array([], dtype=int)

        drop_indices = []
        for index, row in enumerate(data_matrix):
            n_nan = np.sum(np.isnan(row))
            n_tot = len(row)
            n_num = n_tot - n_nan
            p_num = n_num/n_tot
            if n_num < drop_if_fewer_count or p_num < drop_if_fewer_percent:
                drop_indices.append(index)
        return np.array(drop_indices, dtype=int)

    @staticmethod
    def _sort(data_matrix):
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
    def _identify_regions(data_matrix, density_threshold=0.9, max_regions=100):
        """Find the largest regions of the data_matrix with at least
        the minimum density_threshold."""
        nrows, ncols = data_matrix.shape
        s_row, s_col = 0, 0

        regions = []
        for _ in range(max_regions):
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

            else:  # pragma: no cover
                raise ValueError("Iteration stopped without breaking.")

            if c_row+1 == nrows and c_col+1 == ncols:
                break
            else:
                s_col = int(c_col)+1
                s_row = np.argwhere(~np.isnan(data_matrix[:, s_col]))[0][0]

        else:  # pragma: no cover
            raise ValueError(f"Number of regions exceeded {max_regions}.")

        return regions

    @staticmethod
    def _fill_data(data_matrix, means=None, stddevs=None, rng=None):
        """Fill a matrix using random assignment.

        Parameters
        ----------
        data_matrix : ndarray
            Data values in a 2D `ndarray`. Each row is an observation
            and each column is a different sampling location. Values to
            be filled are denoted with `np.nan`.
        means : iterable, optional
            Iterable of means (one per column), default is `None`
            indicating the mean should be calculated from the
            `data_matrix` using `np.nanmean()`.
        stddevs : iterable, optional
            Iterable of standard deviations (one per column), default is
            `None` indicating the standard deviation should be
            calculated from the `data_matrix` using `np.nanstd()`.

        Returns
        -------
        ndarray
            With the missing data (i.e., `np.nan`) replaced with values
            from random assignment.

        """
        try:
            getattr(rng, "normal")
        except AttributeError:
            rng = default_rng(PCG64())

        if means is None:
            means = np.nanmean(data_matrix, axis=0)
        if stddevs is None:
            stddevs = np.nanstd(data_matrix, axis=0, ddof=1)

        for col, (data, mean, stddev) in enumerate(zip(data_matrix.T, means, stddevs)):
            nan_ids = np.flatnonzero(np.isnan(data))
            for _id in nan_ids:
                value = rng.normal(mean, stddev)
                data_matrix[_id, col] = value

        return data_matrix

    @staticmethod
    def _fill_corr(xx, corr_matrix):
        """Fill correlation coefficient matrix.

        Parameters
        ----------
        xx : ndarray
            Location of x samples.
        corr_matrix : ndarray
            Correlation coefficient matrix, where missing values are
            denoted with `np.nan`.

        Returns
        -------
        ndarray
            Where all missing values have been replaced using the
            specified filling procedure.

        """
        def exponential_decay(x, y0, decay_factor):
            return y0*np.power(decay_factor, x)

        def filler(xs, ys, function):
            invalid_ids = np.flatnonzero(np.isnan(xs))
            valid_ids = np.flatnonzero(~np.isnan(xs))

            # Fit functional form to valid entries.
            popt, _ = curve_fit(f=function,
                                xdata=xs[valid_ids],
                                ydata=ys[valid_ids])

            # Replace invalid entries with functional approximation.
            ys[invalid_ids] = function(xs[invalid_ids], *popt)

            return ys

        log_xx = np.log(xx)
        for row, corr_row in enumerate(corr_matrix):
            # Fill right
            xs = log_xx[row:]
            ys = corr_row[row:]
            if np.sum(np.isnan(ys)) > 0:
                corr_matrix[row, row:] = filler(
                    xs - xs[0], ys, function=exponential_decay)

            # Fill left
            xs = log_xx[:row][::-1]
            ys = corr_row[:row][::-1]
            if np.sum(np.isnan(ys)) > 0:
                corr_matrix[row, :row] = filler(
                    xs - xs[0], ys, function=exponential_decay)[::-1]

        return corr_matrix

    @staticmethod
    def _calc_density(data_matrix, tl_corner, br_corner):
        """Calculate value density given a start and stop location."""
        b_row, b_col = tl_corner
        e_row, e_col = br_corner
        n_nums = np.sum(~np.isnan(data_matrix[b_row:e_row+1, b_col:e_col+1]))
        n_poss = (e_col - b_col + 1) * (e_row - b_row + 1)
        return n_nums/n_poss

    @staticmethod
    def _calc_stat(data_matrix):
        mean = np.nanmean(data_matrix, axis=0)
        stddev = np.nanstd(data_matrix, axis=0, ddof=1)
        return (mean, stddev)
