"""Tests for PeaksSuite class."""

import json
import os
import logging
import warnings
from unittest.mock import patch, MagicMock, call

import numpy as np
import matplotlib.pyplot as plt

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
                suite.interactive_trimming(xtype="frequency", ytype="velocity")
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
                    suite.interactive_trimming(
                        xtype="frequency", ytype="velocity")

        # Redefine generator for _draw_box()
        pick_generator = response_generator([response_1])
        _draw_box_responses = wrap_generator(generator=pick_generator)
        mock = MagicMock()

        # Check with bad user entry at input().
        with patch("swprocess.peakssuite.PeaksSuite._draw_box", side_effect=_draw_box_responses):
            with patch("builtins.input", return_value="0"):
                with patch("swprocess.peakssuite.PeaksSuite.plot_resolution_limits", side_effect=mock):
                    suite.interactive_trimming(xtype="frequency", ytype="velocity",
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

    def test_statistics(self):
        # No missing data
        values = np.array([[1, 2, 3, 4, 5],
                           [4, 5, 7, 8, 9],
                           [4, 3, 6, 4, 2]])
        frq = [1, 2, 3, 4, 5]
        peaks = [Peaks(frq, values[k], str(k)) for k in range(3)]
        suite = swprocess.PeaksSuite.from_peaks(peaks)
        rfrq, rmean, rstd, rcorr = suite.statistics(frq,
                                                    xtype="frequency",
                                                    ytype="velocity")
        self.assertArrayEqual(np.array(frq), rfrq)
        self.assertArrayEqual(np.mean(values, axis=0), rmean)
        self.assertArrayEqual(np.std(values, axis=0, ddof=1), rstd)
        self.assertArrayEqual(np.corrcoef(values.T), rcorr)

        # missing_data_procedure="drop"
        values = np.array([[np.nan]*6,
                           [np.nan, 1, 2, 3, 4, 5],
                           [0, 4, 5, 7, 8, 9],
                           [0, 4, 3, 6, 4, 2]])
        frq = [0.2, 1, 2, 3, 4, 5]

        valid = np.array([[1, 2, 3, 4, 5],
                          [4, 5, 7, 8, 9],
                          [4, 3, 6, 4, 2]])
        valid_frq = frq[1:]
        peaks = [Peaks(frq, values[k], str(k)) for k in range(4)]
        suite = swprocess.PeaksSuite.from_peaks(peaks)
        rfrq, rmean, rstd, rcorr = suite.statistics(frq,
                                                    xtype="frequency",
                                                    ytype="velocity",
                                                    missing_data_procedure="drop")
        self.assertArrayEqual(np.array(valid_frq), rfrq)
        self.assertArrayEqual(np.mean(valid, axis=0), rmean)
        self.assertArrayEqual(np.std(valid, axis=0, ddof=1), rstd)
        self.assertArrayEqual(np.corrcoef(valid.T), rcorr)

    def test_from_dict(self):
        # Simple Case: Single dictionary
        data = {"test": {"frequency": [1, 2, 3], "velocity": [4, 5, 6]}}
        suite = swprocess.PeaksSuite.from_dict(data)
        peaks = Peaks.from_dict(data["test"], "test")
        self.assertEqual(peaks, suite[0])

    def test_from_json(self):
        # Advanced Case: Two keyword arguements
        frequency = [100., 50, 30, 10, 5, 3]
        velocity = [100., 120, 130, 140, 145, 150]
        azi = [10., 15, 20, 35, 10, 20]
        pwr = [10., 15, 20, 35, 11, 20]

        peak_list = []
        fnames = []
        for num in range(0, 3):
            fname = f"{self.full_path}data/tmp_id{num}.json"
            peaks = Peaks(
                frequency, velocity, str(num), azi=azi, pwr=pwr)
            peaks.to_json(fname)
            fnames.append(fname)
            peak_list.append(peaks)

        returned = swprocess.PeaksSuite.from_json(fnames=fnames)
        expected = swprocess.PeaksSuite.from_peaks(peak_list)
        self.assertEqual(expected, returned)

        for fname in fnames:
            os.remove(fname)

    def test_from_max(self):
        # Check Rayleigh (2 files, 2 lines per file)
        fnames = [self.full_path +
                  f"data/mm/test_hfk_line2_{x}.max" for x in range(2)]
        returned = swprocess.PeaksSuite.from_max(fnames=fnames,
                                                 wavetype="rayleigh")

        r0 = {"16200": {"frequency": [20.000000000000106581, 19.282217609815102577],
                        "velocity": [1/0.0068859013683322750979, 1/0.0074117944332218188563],
                        "azimuth": [144.53791572557310019, 143.1083743693494057],
                        "ellipticity": [1.0214647665926679387, 1.022287917081338593],
                        "noise": [8.9773778053801098764, 7.3044365524672443257],
                        "power": [2092111.2367646128405, 2074967.9391639579553],
                        }}

        r1 = {"20700": {"frequency": [5.0000000000000266454],
                        "velocity": [1/0.0062049454559321261596],
                        "azimuth": [336.6469768619966203],
                        "ellipticity": [-1.6318692462645167929],
                        "noise": [0],
                        "power": [6040073199.9762010574],
                        }}

        r2 = {"20760.040000000000873": {"frequency": [5.0000000000000266454],
                                        "velocity": [1/0.005256207300441733711],
                                        "azimuth": [77.740031601074633727],
                                        "ellipticity": [-1.8969709764459561363],
                                        "noise": [0],
                                        "power": [17030488659.287288666],
                                        }}

        ray_dicts = [r0, r1, r2]
        expected = swprocess.PeaksSuite.from_dict(ray_dicts)

        self.assertEqual(expected, returned)

        # Check Love (2 files, 2 lines per file)
        fnames = [self.full_path +
                  f"data/mm/test_hfk_line2_{x}.max" for x in range(2)]
        returned = swprocess.PeaksSuite.from_max(fnames=fnames,
                                                 wavetype="love")

        l0 = {"16200": {"frequency": [20.000000000000106581, 19.282217609815102577],
                        "velocity": [1/0.0088200863560403078983, 1/0.0089530611050798007688],
                        "azimuth": [252.05441718438927978, 99.345595852002077208],
                        "ellipticity": [0, 0],
                        "noise": [0, 0],
                        "power": [3832630.8840260845609, 4039408.6602126094513],
                        }}

        l1 = {"20700": {"frequency": [5.0000000000000266454],
                        "velocity": [1/0.001522605285544077541],
                        "azimuth": [92.499974198256211366],
                        "ellipticity": [0],
                        "noise": [0],
                        "power": [232295.4010567846417],
                        }}

        l2 = {"20760.040000000000873": {"frequency": [5.0000000000000266454],
                                        "velocity": [1/0.004102623000517897911],
                                        "azimuth": [174.33483725071613435],
                                        "ellipticity": [0],
                                        "noise": [0],
                                        "power": [422413.79929310601437],
                                        }}

        lov_dicts = [l0, l1, l2]
        expected = swprocess.PeaksSuite.from_dict(lov_dicts)

        self.assertEqual(expected, returned)

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
