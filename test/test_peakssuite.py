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

"""Tests for PeaksSuite class."""

import json
import os
import logging
import warnings
from unittest.mock import patch, MagicMock, call

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import swprocess
from testtools import unittest, TestCase, get_path

logger = logging.getLogger("swprocess")
logger.setLevel(logging.ERROR)


class Test_PeaksSuite(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = get_path(__file__)

    def test_init(self):
        p0 = swprocess.Peaks([1, np.nan, 3], [4, np.nan, 6], "p0")
        suite = swprocess.PeaksSuite(p0)

        self.assertEqual(suite[0], p0)
        self.assertEqual(suite.ids[0], p0.identifier)

    def test_append(self):
        # Simple append operation.
        p0 = swprocess.Peaks([1, 2, 3], [4, 5, 6], "p0")
        p1 = swprocess.Peaks([[7, np.nan, 9], [4, 5, 6]],
                             [[1, np.nan, 2], [7, 8, 9]], "p1")
        suite = swprocess.PeaksSuite(p0)
        suite.append(p1)

        for returned, expected in zip(suite, [p0, p1]):
            self.assertEqual(expected, returned)

        # Bad: replicated identifier
        self.assertRaises(KeyError, suite.append, p1)

        # Bad: append non-Peaks object
        self.assertRaises(TypeError, suite.append, "not a Peaks object")

    def test_reject_limits_outside(self):
        # Create PeaksSuite.
        peak0 = swprocess.Peaks(
            [1, 2, np.nan, 3], [4, 5, np.nan, 6], identifier="0")
        peak1 = swprocess.Peaks(
            [1, 2, np.nan, 3], [4, 5, np.nan, 6], identifier="1")
        suite = swprocess.PeaksSuite(peak0)
        suite.append(peak1)

        # Perform rejections.
        suite.reject_limits_outside("frequency", (0.5, 2.5))
        expected = swprocess.Peaks([1, 2], [4, 5])

        # Check result.
        for returned, _id in zip(suite, suite.ids):
            expected.identifier = _id
            self.assertEqual(expected, returned)

    def test_reject_box_inside(self):
        # Create PeaksSuite
        peak0 = swprocess.Peaks([1, 3, 5, np.nan, 7], [1, 3, 5, np.nan, 7],
                                identifier="0")
        peak1 = swprocess.Peaks([1, 3, 5, np.nan, 7], [1, 3, 5, np.nan, 7],
                                identifier="1")
        suite = swprocess.PeaksSuite(peak0)
        suite.append(peak1)

        # Perform rejections.
        suite.reject_box_inside("frequency", (4, 8), "velocity", (4, 8))
        expected = swprocess.Peaks([1, 3], [1, 3])

        # Check result.
        for returned, _id in zip(suite, suite.ids):
            expected.identifier = _id
            self.assertEqual(expected, returned)

    def test_calc_resolution_limit(self):
        # Shorten method.
        calc_res_limits = swprocess.PeaksSuite.calc_resolution_limits

        # xtype == frequency
        xtype, xs = "frequency", np.array([1., 100.])

        #   attribute == wavelength
        attribute, limits = "wavelength", (2., 50.)

        #     ytype == velocity -> v=fl
        ytype, ys = "velocity", np.array([0, 0])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayEqual(np.array([1., 100.]), x1)
        self.assertArrayEqual(np.array([1., 100.]), x2)
        self.assertArrayEqual(np.array([1., 100.])*2., y1)
        self.assertArrayEqual(np.array([1., 100.])*50., y2)

        #     ytype == wavenumber -> k=2pi/l
        ytype, ys = "wavenumber", np.array([0, 0])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayEqual(np.array([1., 100.]), x1)
        self.assertArrayEqual(np.array([1., 100.]), x2)
        self.assertArrayEqual(np.array([1., 1.])*2*np.pi/2., y1)
        self.assertArrayEqual(np.array([1., 1.])*2*np.pi/50., y2)

        #     ytype == slowness -> p=1/(fl)
        ytype, ys = "slowness", np.array([0, 0])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayEqual(np.array([1., 100.]), x1)
        self.assertArrayEqual(np.array([1., 100.]), x2)
        self.assertArrayEqual(1/(np.array([1., 100.])*2.), y1)
        self.assertArrayEqual(1/(np.array([1., 100.])*50.), y2)

        #     ytype == other -> NotImplementedError
        for _ytype in ["frequency", "wavelength"]:
            self.assertRaises(NotImplementedError, calc_res_limits, xtype,
                              attribute, _ytype, limits, xs, ys)

        #   attribute == "wavenumber"
        attribute, limits = "wavenumber", (2*np.pi/2., 2*np.pi/50.)

        #     ytype == velocity -> v=2*pi*f/k
        ytype, ys = "velocity", np.array([0, 0])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayEqual(np.array([1., 100.]), x1)
        self.assertArrayEqual(np.array([1., 100.]), x2)
        self.assertArrayEqual(np.array([1., 100.])*2.*np.pi/limits[0], y1)
        self.assertArrayEqual(np.array([1., 100.])*2.*np.pi/limits[1], y2)

        #     ytype == wavenumber -> k=k
        ytype, ys = "wavenumber", np.array([0, 0])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayEqual(np.array([1., 100.]), x1)
        self.assertArrayEqual(np.array([1., 100.]), x2)
        self.assertArrayEqual(np.array([1., 1.])*limits[0], y1)
        self.assertArrayEqual(np.array([1., 1.])*limits[1], y2)

        #     ytype == slowness -> p=k/2*pi*f
        ytype, ys = "slowness", np.array([0, 0])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayEqual(np.array([1., 100.]), x1)
        self.assertArrayEqual(np.array([1., 100.]), x2)
        self.assertArrayEqual(limits[0]/(np.array([1., 100.])*2.*np.pi), y1)
        self.assertArrayEqual(limits[1]/(np.array([1., 100.])*2.*np.pi), y2)

        #     ytype == other -> NotImplementedError
        for _ytype in ["frequency", "wavelength"]:
            self.assertRaises(NotImplementedError, calc_res_limits, xtype,
                              attribute, _ytype, limits, xs, ys)

        #     attribute == other -> NotImplementedError
        for _attribute in ["frequency", "slowness"]:
            self.assertRaises(NotImplementedError, calc_res_limits, xtype,
                              _attribute, ytype, limits, xs, ys)

        # xtype == wavelength
        xtype, xs = "wavelength", np.array([2., 50.])

        #   attribute == wavelength
        attribute, limits = "wavelength", (2., 50.)

        #     ytype == velocity -> v=v, l=l
        ytype, ys = "velocity", np.array([100., 500.])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayEqual(np.array([2., 2.]), x1)
        self.assertArrayEqual(np.array([50., 50.]), x2)
        self.assertArrayEqual(np.array([100., 500.]), y1)
        self.assertArrayEqual(np.array([100., 500.]), y2)

        #     ytype == slowness -> p=p, l=l
        ytype, ys = "slowness", np.array([1/100., 1/500.])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayEqual(np.array([2., 2.]), x1)
        self.assertArrayEqual(np.array([50., 50.]), x2)
        self.assertArrayEqual(np.array([1/100., 1/500.]), y1)
        self.assertArrayEqual(np.array([1/100., 1/500.]), y2)

        #     ytype == other -> NotImplementedError
        for _ytype in ["frequency", "wavelength"]:
            self.assertRaises(NotImplementedError, calc_res_limits, xtype,
                              attribute, _ytype, limits, xs, ys)

        #   attribute == other -> NotImplementedError
        for _attribute in ["velocity", "frequency"]:
            self.assertRaises(NotImplementedError, calc_res_limits, xtype,
                              _attribute, ytype, limits, xs, ys)

        #   attribute == wavenumber
        attribute, limits = "wavenumber", (2*np.pi/2., 2*np.pi/50.)

        #     ytype == velocity -> v=v, k=k
        ytype, ys = "velocity", np.array([100., 500.])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayAlmostEqual(np.array([2., 2.]), x1)
        self.assertArrayAlmostEqual(np.array([50., 50.]), x2)
        self.assertArrayEqual(np.array([100., 500.]), y1)
        self.assertArrayEqual(np.array([100., 500.]), y2)

        #     ytype == slowness -> p=p, l=l
        ytype, ys = "slowness", np.array([1/100., 1/500.])
        (x1, y1), (x2, y2) = calc_res_limits(xtype, attribute, ytype, limits,
                                             xs, ys)
        self.assertArrayAlmostEqual(np.array([2., 2.]), x1)
        self.assertArrayAlmostEqual(np.array([50., 50.]), x2)
        self.assertArrayEqual(np.array([1/100., 1/500.]), y1)
        self.assertArrayEqual(np.array([1/100., 1/500.]), y2)

        #     ytype == other -> NotImplementedError
        for _ytype in ["frequency", "wavelength"]:
            self.assertRaises(NotImplementedError, calc_res_limits, xtype,
                              attribute, _ytype, limits, xs, ys)

        #   attribute == other -> NotImplementedError
        for _attribute in ["velocity", "frequency"]:
            self.assertRaises(NotImplementedError, calc_res_limits, xtype,
                              _attribute, ytype, limits, xs, ys)

        # xtype == other -> NotImplementedError
        for _xtype in ["wavenumber", "slowness"]:
            self.assertRaises(NotImplementedError, calc_res_limits, _xtype,
                              attribute, ytype, limits, xs, ys)

    # def test_plot_resolution_limits(self):
    #     ax = MagicMock(spec=plt.Axes)
    #     ax.get_xlim.return_value = (1, 10)
    #     ax.get_xscale.return_value = "log"
    #     ax.get_ylim.return_value = (100, 500)
    #     ax.get_yscale.return_value = "linear"

    #     default_kwargs = dict(color="#000000", linestyle="--",
    #                           linewidth=0.75, label="limit")

    #     # Standard plot.
    #     ret_val = (([1, 2, 3], [4, 5, 6]), ([0, 1, 2], [3, 4, 5]))
    #     with patch("swprocess.peakssuite.PeaksSuite.calc_resolution_limits", return_value=ret_val):
    #         swprocess.PeaksSuite.plot_resolution_limits(ax=ax, xtype="frequency",
    #                                                     ytype="velocity",
    #                                                     attribute="wavelength",
    #                                                     limits=(5, 100))
    #         calls = [call(*limit_pair, **default_kwargs) for limit_pair in ret_val]
    #         ax.plot.assert_has_calls(calls, any_order=False)

    #     # No plot.
    #     def side_effect(*args, **kwargs):
    #         raise NotImplementedError

    #     with patch("swprocess.peakssuite.PeaksSuite.calc_resolution_limits", side_effect=side_effect):
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             swprocess.PeaksSuite.plot_resolution_limits(ax=ax,
    #                                                         xtype="frequency",
    #                                                         ytype="velocity",
    #                                                         attribute="wavelength",
    #                                                         limits=(5, 100))

    def test_plot(self):
        # Default
        fname = self.path / "data/peak/suite_raw.json"
        suite = swprocess.PeaksSuite.from_json(fname)

        _, ax = suite.plot(xtype=["frequency", "wavelength", "frequency"],
                             ytype=["velocity", "velocity", "slowness"],
                             )

        # With a provided Axes.
        _, ax = plt.subplots()
        result = suite.plot(ax=ax, xtype="frequency", ytype="velocity")
        self.assertTrue(result is None)

        # With a provided Axes (wrong size).
        _, ax = plt.subplots(ncols=3)
        self.assertRaises(IndexError, suite.plot, ax=ax, xtype="frequency",
                          ytype="velocity")

        # With a provided mask (wrong size).
        self.assertRaises(IndexError, suite.plot, xtype="frequency",
                          ytype="velocity", mask=[[0], [1]])

        # With a name provided.
        _, ax = plt.subplots()
        suite.plot(ax=ax, xtype="frequency", ytype="velocity",
                   plot_kwargs=dict(label="tada"))
        _, labels = ax.get_legend_handles_labels()
        self.assertListEqual(["tada"], labels)

        plt.show(block=False)
        plt.close("all")

    def test_prepare_plot_kwargs(self):
        # Apply to all.
        plot_kwargs = dict(color="color", label="label")
        returned = swprocess.PeaksSuite._prepare_plot_kwargs(plot_kwargs,
                                                             ncols=3)
        expected = [plot_kwargs]*3
        self.assertListEqual(expected, returned)

        # Apply one by index.
        plot_kwargs = dict(color=["c0", "c1", "c2"], label="label")
        returned = swprocess.PeaksSuite._prepare_plot_kwargs(plot_kwargs,
                                                             ncols=3)
        expected = [dict(plot_kwargs) for _ in range(3)]
        for index, _dict in enumerate(expected):
            _dict["color"] = f"c{index}"
        self.assertListEqual(expected, returned)

        # Bad input -> NotImplementedError
        plot_kwargs = dict(color=5)
        self.assertRaises(NotImplementedError,
                          swprocess.PeaksSuite._prepare_plot_kwargs,
                          plot_kwargs, ncols=3)

    def test_interactive_trimming(self):
        # Create simple suite, composed of two Peaks.
        peaks_a = swprocess.Peaks(frequency=[0.5, 0.5], velocity=[0.5, 1.5],
                                  identifier="a")
        peaks_b = swprocess.Peaks(frequency=[1.5, 1.5], velocity=[0.5, 1.5],
                                  identifier="b")
        suite = swprocess.PeaksSuite(peaks_a)
        suite.append(peaks_b)

        # Create a response generator.
        def response_generator(responses):
            index = 0
            while index < len(responses):
                yield responses[index]
                index += 1

        # Use a closure to wrap generator.
        def wrap_generator(generator):
            def wrapper(*args, **kwargs):
                return next(generator)
            return wrapper

        # Define generator to replace _draw_box()
        xlims, ylims, axclicked = (1., 2.), (0., 1.), 0
        response_0 = (xlims, ylims, axclicked)
        xlims, ylims, axclicked = (1., 1.), (1., 1.), 0
        response_1 = (xlims, ylims, axclicked)
        pick_generator = response_generator([response_0, response_1])
        _draw_box_responses = wrap_generator(generator=pick_generator)

        with patch("swprocess.peakssuite.PeaksSuite._draw_box", side_effect=_draw_box_responses):
            with patch('builtins.input', return_value="0"):
                suite.interactive_trimming(xtype="frequency",
                                           ytype="velocity")
        self.assertArrayEqual(np.array([0.5, 0.5]), peaks_a.frequency)
        self.assertArrayEqual(np.array([0.5, 1.5]), peaks_a.velocity)
        self.assertArrayEqual(np.array([1.5]), peaks_b.frequency)
        self.assertArrayEqual(np.array([1.5]), peaks_b.velocity)

        # Redefine generator for _draw_box()
        pick_generator = response_generator([response_0, response_1])
        _draw_box_responses = wrap_generator(generator=pick_generator)

        # Define generator to replace input()
        input_generator = response_generator(["5", "0"])
        _input_responses = wrap_generator(generator=input_generator)

        # Check with bad user entry at input().
        with patch("swprocess.peakssuite.PeaksSuite._draw_box", side_effect=_draw_box_responses):
            with patch("builtins.input", side_effect=_input_responses):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    suite.interactive_trimming(xtype="frequency",
                                               ytype="velocity")

        # Redefine generator for _draw_box()
        pick_generator = response_generator([response_1])
        _draw_box_responses = wrap_generator(generator=pick_generator)
        mock = MagicMock()

        # Check with bad user entry at input().
        with patch("swprocess.peakssuite.PeaksSuite._draw_box", side_effect=_draw_box_responses):
            with patch("builtins.input", return_value="0"):
                with patch("swprocess.peakssuite.PeaksSuite.plot_resolution_limits", side_effect=mock):
                    suite.interactive_trimming(xtype="frequency",
                                               ytype="velocity",
                                               resolution_limits=["wavelength", (1, 10)])
                    mock.assert_called()

    # TODO (jpv): Draw box needs to be refactored.
    # def test_draw_box(self):
    #     # Create a response generator.
    #     def response_generator(responses):
    #         index = 0
    #         while index < len(responses):
    #             yield responses[index]
    #             index += 1

    #     # Use a closure to wrap generator.
    #     def wrap_generator(generator):
    #         def wrapper(*args, **kwargs):
    #             return next(generator)
    #         return wrapper

    #     # Create PeakSuite object.
    #     peaks = swprocess.Peaks(frequency=[0,1,2], velocity=[0,1,2])
    #     suite = swprocess.PeaksSuite(peaks)

    #     # Patch ginput
    #     fig, ax = plt.subplots()
    #     # fig.canvas.mpl_connect = MagicMock()
    #     # fig.canvas.mpl_diconnect = MagicMock()

    #     click_generator = response_generator([((0, 0), (1, 1))])
    #     ginput_response = wrap_generator(generator=click_generator)
    #     fig.ginput = MagicMock(side_effect=[ginput_response, on_click])
    #     suite._draw_box(fig=fig)

    def test_statistics(self):
        # 1D: no missing data.
        values = np.array([[1, 2, 3, 4, 5],
                           [4, 5, 7, 8, 9],
                           [4, 3, 6, 4, 2]])
        frq = [1, 2, 3, 4, 5]
        peaks = [swprocess.Peaks(frq, values[k], str(k)) for k in range(3)]
        suite = swprocess.PeaksSuite.from_peaks(peaks)
        rfrq, rmean, rstd, rcorr = suite.statistics(xtype="frequency",
                                                    ytype="velocity",
                                                    xx=frq,
                                                    ignore_corr=False)
        self.assertArrayEqual(np.array(frq), rfrq)
        self.assertArrayEqual(np.mean(values, axis=0), rmean)
        self.assertArrayEqual(np.std(values, axis=0, ddof=1), rstd)
        self.assertArrayEqual(np.corrcoef(values.T), rcorr)

        # 1D: fewer than three Peaks in PeaksSuite -> ValueError.
        peaks = [swprocess.Peaks(frq, values[k], str(k)) for k in range(2)]
        suite = swprocess.PeaksSuite.from_peaks(peaks)
        self.assertRaises(ValueError, suite.statistics, xtype="frequency",
                          ytype="velocity", xx=frq)

        # 1D: drop_sample_if_fewer_count = 3.
        values = np.array([[np.nan]*6,
                           [np.nan, 1, 2, 3, 4, 5],
                           [0, 4, 5, 7, 8, 9],
                           [0, 4, 3, 6, 4, 2]])
        frq = [0.2, 1, 2, 3, 4, 5]

        valid = np.array([[1, 2, 3, 4, 5],
                          [4, 5, 7, 8, 9],
                          [4, 3, 6, 4, 2]])
        valid_frq = frq[1:]
        peaks = [swprocess.Peaks(frq, values[k], str(k)) for k in range(4)]
        suite = swprocess.PeaksSuite.from_peaks(peaks)

        rfrq, rmean, rstd, rcorr = suite.statistics(xtype="frequency",
                                                    ytype="velocity",
                                                    xx=frq,
                                                    ignore_corr=False)
        self.assertArrayEqual(np.array(valid_frq), rfrq)
        self.assertArrayEqual(np.mean(valid, axis=0), rmean)
        self.assertArrayEqual(np.std(valid, axis=0, ddof=1), rstd)
        self.assertArrayEqual(np.corrcoef(valid.T), rcorr)

        # 2D: with missing data.
        f0 = [[1, 3, 5, 7, np.nan],
              [np.nan, 3, 5, 7, 9],
              [1, 3, 5, 7, 9]]
        v0 = [[4, 6, 7, 1, np.nan],
              [np.nan, 3, 2, 5, 4],
              [1, 3, 5, 1, 2]]
        p0 = [[4, 3, 5, 1, np.nan],
              [np.nan, 3, 7, 5, 4],
              [1, 6, 7, 1, 2]]
        # --> 4, 3, 2, 5, 4
        peaks0 = swprocess.Peaks(f0, v0, identifier="0", power=p0)

        f1 = [[1, 3, 5, 7, np.nan],
              [np.nan, 3, 5, 7, 9],
              [1, 3, 5, 7, 9]]
        v1 = [[8, 8, 7, 7, np.nan],
              [np.nan, 2, 1, 3, 8],
              [4, 1, 4, 1, 4]]
        p1 = [[4, 6, 7, 1, np.nan],
              [np.nan, 3, 7, 5, 4],
              [1, 3, 5, 1, 2]]
        # --> 8, 8, 7, 3, 8
        peaks1 = swprocess.Peaks(f1, v1, identifier="1", power=p1)

        f2 = [[1, 3, 5, 7, np.nan],
              [np.nan, 3, 5, 7, 9],
              [1, 3, 5, 7, 9]]
        v2 = [[2, 4, 1, 1, np.nan],
              [np.nan, 8, 1, 3, 2],
              [5, 2, 9, 4, 8]]
        p2 = [[4, 6, 7, 1, np.nan],
              [np.nan, 3, 7, 5, 4],
              [1, 3, 5, 1, 2]]
        # --> 2, 4, 1, 3, 2
        peaks2 = swprocess.Peaks(f2, v2, identifier="2", power=p2)

        suite = swprocess.PeaksSuite.from_peaks([peaks0, peaks1, peaks2])
        valid = np.array([[4, 3, 2, 5, 4],
                          [8, 8, 7, 3, 8],
                          [2, 4, 1, 3, 2]], dtype=float)
        frq = [1, 3, 5, 7, 9]
        rfrq, rmean, rstd, rcorr = suite.statistics(xtype="frequency",
                                                    ytype="velocity",
                                                    xx=frq,
                                                    ignore_corr=False)
        self.assertArrayEqual(np.array(frq), rfrq)
        self.assertArrayEqual(np.mean(valid, axis=0), rmean)
        self.assertArrayEqual(np.std(valid, axis=0, ddof=1), rstd)
        self.assertArrayEqual(np.corrcoef(valid.T), rcorr)

    def test_plot_statistics(self):
        # Mock ax
        ax = MagicMock(spec=plt.Axes)
        suite = swprocess.PeaksSuite(swprocess.Peaks([1, 2, 3], [0, 1, 2]))
        suite.plot_statistics(ax, [1, 2, 3], [0, 1, 2], [4, 5, 6])
        ax.errorbar.assert_called_once()

    def test_drop(self):
        # Full matrix -> No drop
        xx = np.array([1, 2, 3])
        data_matrix = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9],
                                [0, 1, 2]])
        rxx, rdata_matrix = swprocess.PeaksSuite._drop(xx, data_matrix)
        self.assertArrayEqual(xx, rxx)
        self.assertArrayEqual(data_matrix, rdata_matrix)

        # Remove single empty column regardless of threshold.
        xx = np.array([1, 2, 3, 4])
        data_matrix = np.array([[1, 2, 3, np.nan],
                                [4, 5, 6, np.nan],
                                [7, 8, 9, np.nan],
                                [0, 1, 2, np.nan]])
        for drop_observation in [0., 0.5, 1.]:
            rxx, rdata_matrix = swprocess.PeaksSuite._drop(xx, data_matrix,
                                                           drop_observation_if_fewer_percent=drop_observation)
            self.assertArrayEqual(xx[:-1], rxx)
            self.assertArrayEqual(data_matrix[:, :-1], rdata_matrix)

        # Remove single bad observation.
        xx = np.array([1, 2, 3])
        data_matrix = np.array([[1, 2, 3],
                                [7, 8, 9],
                                [0, 1, 2],
                                [np.nan, np.nan, np.nan]])

        for drop_observation in [0.1, 0.5, 1.]:
            rxx, rdata_matrix = swprocess.PeaksSuite._drop(xx, data_matrix,
                                                           drop_observation_if_fewer_percent=drop_observation)
            self.assertArrayEqual(xx, rxx)
            self.assertArrayEqual(data_matrix[:-1, :], rdata_matrix)

        # Remove sample b/c too few data points.
        xx = np.array([1, 2, 3, 4, 5])
        data_matrix = np.array([[1, 2, 3, 4, 5],
                                [7, 8, 9, 0, 1],
                                [1, 2, 3, 4, np.nan]])

        rxx, rdata_matrix = swprocess.PeaksSuite._drop(xx, data_matrix,
                                                       drop_sample_if_fewer_count=3)
        self.assertArrayEqual(xx[:-1], rxx)
        self.assertArrayEqual(data_matrix[:, :-1], rdata_matrix)

    def test_to_and_from_json(self):
        # 2D: Two keyword arguments with nan.
        frequency = [[1., np.nan, 5, 6, 1, 5, 3],
                     [2., 7, np.nan, 2, 8, 9, 3]]
        velocity = [[5., np.nan, 2, 1, 1, 5, 4],
                    [2., 7, np.nan, 4, 8, 7, 4]]
        azimuth = [[1., np.nan, 5, 2, 5, 1, 4],
                   [4., 5, np.nan, 8, 7, 9, 5]]
        power = [[1., np.nan, 5, 2, 5, 1, 2],
                 [3., 4, np.nan, 7, 6, 3, 4]]

        # Create Peak instances and append to list and json.
        peak_list = []
        fnames = []
        for num in range(3):
            fname = self.path / f"data/tmp_id{num}.json"
            peaks = swprocess.Peaks(frequency, velocity, str(num),
                                    azimuth=azimuth, power=power)
            peaks.to_json(fname)
            fnames.append(fname)
            peak_list.append(peaks)

        # Compare from_peaks and from_json.
        returned = swprocess.PeaksSuite.from_json(fnames=fnames)
        expected = swprocess.PeaksSuite.from_peaks(peak_list)
        self.assertEqual(expected, returned)

        for fname in fnames:
            os.remove(fname)

        # Compare from_peaks and to_json -> from_json.
        fname = "tmp.json"
        expected.to_json(fname)
        returned = swprocess.PeaksSuite.from_json(fname)
        self.assertEqual(expected, returned)
        os.remove(fname)

    def test_from_dict(self):
        # 1D: Single dictionary
        data = {"test": {"frequency": [1, 2, 3], "velocity": [4, 5, 6]}}
        suite = swprocess.PeaksSuite.from_dict(data)
        peaks = swprocess.Peaks.from_dict(data["test"], "test")
        self.assertEqual(peaks, suite[0])

    def test_from_max(self):
        # rayleigh, nmaxima=3, nblocksets=3, samples=10
        path = self.path / "data/rtbf/rtbf_nblockset=3_nmaxima=3"
        fname_max = str(path) + ".max"
        fname_csvs = [str(path) + f"_r_bs{bs}_parsed.csv" for bs in range(3)]

        peaksuite = swprocess.PeaksSuite.from_max(fname_max,
                                                  wavetype="rayleigh")
        for peak, fname_csv in zip(peaksuite, fname_csvs):
            df = pd.read_csv(fname_csv)
            for attr in ["frequency", "slowness", "azimuth", "ellipticity", "noise", "power"]:
                expected = getattr(df, attr).to_numpy()
                index = 0
                for returned in getattr(peak, f"_{attr}").flatten(order="F"):
                    if np.isnan(returned):
                        continue
                    else:
                        self.assertAlmostEqual(expected[index],
                                               returned, places=4)
                        index += 1

        # love, nmaxima=3, nblocksets=3, samples=10
        path = self.path / "data/rtbf/rtbf_nblockset=3_nmaxima=3"
        fname_max = str(path) + ".max"
        fname_csvs = [str(path) + f"_l_bs{bs}_parsed.csv" for bs in range(3)]

        peaksuite = swprocess.PeaksSuite.from_max(fname_max, wavetype="love")
        for peak, fname_csv in zip(peaksuite, fname_csvs):
            df = pd.read_csv(fname_csv)
            for attr in ["frequency", "slowness", "azimuth", "ellipticity", "noise", "power"]:
                expected = getattr(df, attr).to_numpy()
                index = 0
                for returned in getattr(peak, f"_{attr}").flatten(order="F"):
                    if np.isnan(returned):
                        continue
                    else:
                        self.assertAlmostEqual(
                            expected[index], returned, places=4)
                        index += 1

    def test_from_peakssuite(self):
        frq = [0, 1, 2, 3]
        vel = [1, 2, 3, 4]

        # suites[0]
        peaks0 = [swprocess.Peaks(frq, vel, identifier=str(num))
                  for num in range(2)]
        suite0 = swprocess.PeaksSuite.from_peaks(peaks0)

        # Create new PeaksSuite from old, no copy.
        new_suite = swprocess.PeaksSuite.from_peakssuite(suite0)
        for expected, returned in zip(peaks0, new_suite.peaks):
            self.assertIs(expected, returned)

        # Create several PeaksSuite objects.
        # suites[1]
        peaks1 = [swprocess.Peaks(frq, vel, identifier=str(num))
                  for num in range(3, 5)]
        suite1 = swprocess.PeaksSuite.from_peaks(peaks1)

        # Create new PeaksSuite from old, no copy.
        new_suite = swprocess.PeaksSuite.from_peakssuite([suite0, suite1])
        for expected, returned in zip(peaks0 + peaks1, new_suite.peaks):
            self.assertIs(expected, returned)

        # Rename suite1 so there is a naming conflict -> KeyError
        peaks2 = [swprocess.Peaks(frq, vel, identifier=str(num))
                  for num in range(2)]
        suite2 = swprocess.PeaksSuite.from_peaks(peaks2)
        self.assertRaises(KeyError,
                          swprocess.PeaksSuite.from_peakssuite,
                          [suite0, suite2])

    def test_eq(self):
        p0 = swprocess.Peaks([1, 2, 3], [4, 5, 6], "0")
        p1 = swprocess.Peaks([1, 2, 3], [7, 8, 9], "1")
        p2 = swprocess.Peaks([1, 2, 3], [0, 1, 2], "2")

        suite_a = swprocess.PeaksSuite.from_peaks([p0, p1, p2])
        suite_b = "I am not a PeakSuite"
        suite_c = swprocess.PeaksSuite.from_peaks([p1, p2])
        suite_d = swprocess.PeaksSuite.from_peaks([p1])
        suite_e = swprocess.PeaksSuite.from_peaks([p1, p0, p2])
        suite_f = swprocess.PeaksSuite.from_peaks([p0, p1, p2])

        self.assertTrue(suite_a == suite_a)
        self.assertFalse(suite_a == suite_b)
        self.assertFalse(suite_a == suite_c)
        self.assertFalse(suite_a == suite_d)
        self.assertFalse(suite_a == suite_e)
        self.assertTrue(suite_a == suite_f)


if __name__ == "__main__":
    unittest.main()
