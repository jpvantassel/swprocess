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
from swprocess.peaks import Peaks
from testtools import unittest, TestCase, get_full_path

logger = logging.getLogger("swprocess")
logger.setLevel(logging.ERROR)


class Test_PeaksSuite(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_init(self):
        p0 = Peaks([1, 2, 3], [4, 5, 6], "p0")
        suite = swprocess.PeaksSuite(p0)

        self.assertEqual(suite[0], p0)
        self.assertEqual(suite.ids[0], p0.identifier)

    def test_append(self):
        p0 = Peaks([1, 2, 3], [4, 5, 6], "p0")
        p1 = Peaks([7, 8, 9], [10, 11, 12], "p1")
        suite = swprocess.PeaksSuite(p0)
        suite.append(p1)

        for returned, expected in zip(suite, [p0, p1]):
            self.assertEqual(expected, returned)

        # Bad: replicated identifier
        self.assertRaises(KeyError, suite.append, p1)

        # Bad: append non-Peaks object
        self.assertRaises(TypeError, suite.append, "not a Peaks object")

    def test_blitz(self):
        # Create mock PeaksSuite
        peak0 = MagicMock(spec=Peaks)
        peak0.identifier = "p0"
        peak1 = MagicMock(spec=Peaks)
        peak1.identifier = "p1"
        suite = swprocess.PeaksSuite(peak0)
        suite.append(peak1)

        # Call and assert call with args.
        args = ("attribute", ("low", "high"))
        suite.blitz(*args)
        for peak in suite:
            peak.blitz.assert_called_once_with(*args)

    def test_reject(self):
        # Create mock PeaksSuite
        peak0 = MagicMock(spec=Peaks)
        peak0.identifier = "p0"
        peak0.reject_ids.return_value = [0, 1, 2]
        peak1 = MagicMock(spec=Peaks)
        peak1.identifier = "p1"
        peak1.reject_ids.return_value = [2, 1, 0]
        suite = swprocess.PeaksSuite(peak0)
        suite.append(peak1)

        # Call reject and assert call with args.
        args = ("xtype", ("xlow", "xhigh"), "ytype", ("ylow", "yhigh"))
        suite.reject(*args)
        for peak in suite:
            peak.reject.assert_called_once_with(*args)

        # Call reject_ids and assert call with args.
        args = ("xtype", ("xlow", "xhigh"), "ytype", ("ylow", "yhigh"))
        reject_ids = suite.reject_ids(*args)
        for peak in suite:
            peak.reject_ids.assert_called_once_with(*args)
        self.assertListEqual([[0, 1, 2], [2, 1, 0]], reject_ids)

        # Call _reject and assert call with args.
        arg = ["reject_peak0", "reject_peak1"]
        suite._reject(arg)
        for peak, _arg in zip(suite, arg):
            peak._reject.assert_called_once_with(_arg)

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

    def test_plot_resolution_limits(self):
        ax = MagicMock(spec=plt.Axes)
        ax.get_xlim.return_value = (1, 10)
        ax.get_xscale.return_value = "log"
        ax.get_ylim.return_value = (100, 500)
        ax.get_yscale.return_value = "linear"

        default_kwargs = dict(color="#000000", linestyle="--",
                              linewidth=0.75, label="limit")

        # Standard plot.
        ret_val = (([1, 2, 3], [4, 5, 6]), ([0, 1, 2], [3, 4, 5]))
        with patch("swprocess.peakssuite.PeaksSuite.calc_resolution_limits", return_value=ret_val):
            swprocess.PeaksSuite.plot_resolution_limits(ax=ax, xtype="frequency",
                                                        ytype="velocity",
                                                        attribute="wavelength",
                                                        limits=(5, 100))
            calls = [call(*limit_pair, **default_kwargs)
                     for limit_pair in ret_val]
            ax.plot.has_calls(calls, any_order=False)

        # No plot.
        def side_effect(*args, **kwargs):
            raise NotImplementedError

        with patch("swprocess.peakssuite.PeaksSuite.calc_resolution_limits", side_effect=side_effect):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                swprocess.PeaksSuite.plot_resolution_limits(ax=ax,
                                                            xtype="frequency",
                                                            ytype="velocity",
                                                            attribute="wavelength",
                                                            limits=(5, 100))

    def test_plot(self):
        # Default
        fname = self.full_path + "data/peak/suite_raw.json"
        suite = swprocess.PeaksSuite.from_json(fname)

        fig, ax = suite.plot(xtype=["frequency", "wavelength", "frequency"],
                             ytype=["velocity", "velocity", "slowness"],
                             )

        # With a provided Axes.
        fig, ax = plt.subplots()
        result = suite.plot(ax=ax, xtype="frequency", ytype="velocity")
        self.assertTrue(result is None)

        # With a provided Axes (wrong size).
        fig, ax = plt.subplots(ncols=3)
        self.assertRaises(IndexError, suite.plot, ax=ax, xtype="frequency",
                          ytype="velocity")

        # With a provided indices (wrong size).
        self.assertRaises(IndexError, suite.plot, xtype="frequency",
                          ytype="velocity", indices=[[0], [1]])

        # With a name provided.
        fig, ax = plt.subplots()
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
        peaks_a = Peaks(frequency=[0.5, 0.5], velocity=[0.5, 1.5],
                        identifier="a")
        peaks_b = Peaks(frequency=[1.5, 1.5], velocity=[0.5, 1.5],
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
    #     peaks = Peaks(frequency=[0,1,2], velocity=[0,1,2])
    #     suite = swprocess.PeaksSuite(peaks)

    #     # Patch ginput
    #     fig, ax = plt.subplots()
    #     # fig.canvas.mpl_connect = MagicMock()
    #     # fig.canvas.mpl_diconnect = MagicMock()

    #     click_generator = response_generator([((0, 0), (1, 1))])
    #     ginput_response = wrap_generator(generator=click_generator)
    #     fig.ginput = MagicMock(side_effect=[ginput_response, on_click])
    #     suite._draw_box(fig=fig)

    # def test_statistics(self):
    #     # No missing data
    #     values = np.array([[1, 2, 3, 4, 5],
    #                        [4, 5, 7, 8, 9],
    #                        [4, 3, 6, 4, 2]])
    #     frq = [1, 2, 3, 4, 5]
    #     peaks = [Peaks(frq, values[k], str(k)) for k in range(3)]
    #     suite = swprocess.PeaksSuite.from_peaks(peaks)
    #     rfrq, rmean, rstd, rcorr = suite.statistics(xtype="frequency",
    #                                                 ytype="velocity",
    #                                                 xx=frq,
    #                                                 ignore_corr=False)
    #     self.assertArrayEqual(np.array(frq), rfrq)
    #     self.assertArrayEqual(np.mean(values, axis=0), rmean)
    #     self.assertArrayEqual(np.std(values, axis=0, ddof=1), rstd)
    #     self.assertArrayEqual(np.corrcoef(values.T), rcorr)

    #     # Fewer than three peaks in PeaksSuite -> ValueError
    #     peaks = [Peaks(frq, values[k], str(k)) for k in range(2)]
    #     suite = swprocess.PeaksSuite.from_peaks(peaks)
    #     self.assertRaises(ValueError, suite.statistics, xtype="frequency",
    #                       ytype="velocity", xx=frq)

        # # missing_data_procedure="drop"
        # values = np.array([[np.nan]*6,
        #                    [np.nan, 1, 2, 3, 4, 5],
        #                    [0, 4, 5, 7, 8, 9],
        #                    [0, 4, 3, 6, 4, 2]])
        # frq = [0.2, 1, 2, 3, 4, 5]

        # valid = np.array([[1, 2, 3, 4, 5],
        #                   [4, 5, 7, 8, 9],
        #                   [4, 3, 6, 4, 2]])
        # valid_frq = frq[1:]
        # peaks = [Peaks(frq, values[k], str(k)) for k in range(4)]
        # suite = swprocess.PeaksSuite.from_peaks(peaks)
        # rfrq, rmean, rstd, rcorr = suite.statistics(frq,
        #                                             xtype="frequency",
        #                                             ytype="velocity",
        #                                             missing_data_procedure="drop")
        # self.assertArrayEqual(np.array(valid_frq), rfrq)
        # self.assertArrayEqual(np.mean(valid, axis=0), rmean)
        # self.assertArrayEqual(np.std(valid, axis=0, ddof=1), rstd)
        # self.assertArrayEqual(np.corrcoef(valid.T), rcorr)

    def test_plot_statistics(self):
        # Mock ax
        ax = MagicMock(spec=plt.Axes)
        suite = swprocess.PeaksSuite(Peaks([1, 2, 3], [0, 1, 2]))
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
        # Advanced Case: Two keyword arguments
        frequency = [100., 50, 30, 10, 5, 3]
        velocity = [100., 120, 130, 140, 145, 150]
        azi = [10., 15, 20, 35, 10, 20]
        pwr = [10., 15, 20, 35, 11, 20]

        peak_list = []
        fnames = []
        for num in range(3):
            fname = f"{self.full_path}data/tmp_id{num}.json"
            peaks = Peaks(frequency, velocity, str(num), azi=azi, pwr=pwr)
            peaks.to_json(fname)
            fnames.append(fname)
            peak_list.append(peaks)

        # Compare from_peaks and from_json
        returned = swprocess.PeaksSuite.from_json(fnames=fnames)
        expected = swprocess.PeaksSuite.from_peaks(peak_list)
        self.assertEqual(expected, returned)

        for fname in fnames:
            os.remove(fname)

        # Compare from_peaks and to/from_json
        fname = "tmp.json"
        expected.to_json(fname)
        returned = swprocess.PeaksSuite.from_json(fname)
        self.assertEqual(expected, returned)

    def test_from_dict(self):
        # Simple Case: Single dictionary
        data = {"test": {"frequency": [1, 2, 3], "velocity": [4, 5, 6]}}
        suite = swprocess.PeaksSuite.from_dict(data)
        peaks = Peaks.from_dict(data["test"], "test")
        self.assertEqual(peaks, suite[0])

    def test_from_max(self):
        # rayleigh, nmaxima=3, nblocksets=3, samples=10
        path = self.full_path + "data/rtbf/"
        fname_max = path + "rtbf_nblockset=3_nmaxima=3.max"
        fname_csvs = [
            path + f"rtbf_nblockset=3_nmaxima=3_r_bs{bs}_parsed.csv" for bs in range(3)]

        peaksuite = swprocess.PeaksSuite.from_max(
            fname_max, wavetype="rayleigh")
        for peak, fname_csv in zip(peaksuite, fname_csvs):
            df = pd.read_csv(fname_csv)
            for attr in ["frequency", "slowness", "azimuth", "ellipticity", "noise", "power"]:
                expected = getattr(df, attr).to_numpy()
                index = 0
                for returned in getattr(peak, attr).flatten(order="F"):
                    if np.isnan(returned):
                        continue
                    else:
                        self.assertAlmostEqual(
                            expected[index], returned, places=4)
                        index += 1

        # love, nmaxima=3, nblocksets=3, samples=10
        path = self.full_path + "data/rtbf/"
        fname_max = path + "rtbf_nblockset=3_nmaxima=3.max"
        fname_csvs = [
            path + f"rtbf_nblockset=3_nmaxima=3_l_bs{bs}_parsed.csv" for bs in range(3)]

        peaksuite = swprocess.PeaksSuite.from_max(fname_max, wavetype="love")
        for peak, fname_csv in zip(peaksuite, fname_csvs):
            df = pd.read_csv(fname_csv)
            for attr in ["frequency", "slowness", "azimuth", "ellipticity", "noise", "power"]:
                expected = getattr(df, attr).to_numpy()
                index = 0
                for returned in getattr(peak, attr).flatten(order="F"):
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
        peaks0 = [Peaks(frq, vel, identifier=str(num)) for num in range(2)]
        suite0 = swprocess.PeaksSuite.from_peaks(peaks0)

        # Create new PeaksSuite from old, no copy.
        new_suite = swprocess.PeaksSuite.from_peakssuite(suite0)
        for expected, returned in zip(peaks0, new_suite.peaks):
            self.assertIs(expected, returned)

        # Create several PeaksSuite objects.
        # suites[1]
        peaks1 = [Peaks(frq, vel, identifier=str(num)) for num in range(3, 5)]
        suite1 = swprocess.PeaksSuite.from_peaks(peaks1)

        # Create new PeaksSuite from old, no copy.
        new_suite = swprocess.PeaksSuite.from_peakssuite([suite0, suite1])
        for expected, returned in zip(peaks0 + peaks1, new_suite.peaks):
            self.assertIs(expected, returned)

        # Rename suite1 so there is a naming conflict -> KeyError
        peaks2 = [Peaks(frq, vel, identifier=str(num)) for num in range(2)]
        suite2 = swprocess.PeaksSuite.from_peaks(peaks2)
        self.assertRaises(
            KeyError, swprocess.PeaksSuite.from_peakssuite, [suite0, suite2])

    def test_eq(self):
        p0 = Peaks([1, 2, 3], [4, 5, 6], "0")
        p1 = Peaks([1, 2, 3], [7, 8, 9], "1")
        p2 = Peaks([1, 2, 3], [0, 1, 2], "2")

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
