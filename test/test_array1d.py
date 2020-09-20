"""Tests for Array1D class."""

import warnings
import logging

import obspy
import numpy as np
import matplotlib.pyplot as plt

from testtools import TestCase, unittest, get_full_path
import swprocess

logging.basicConfig(level=logging.ERROR)


class Test_Array1D(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)
        cls.vuws_path = cls.full_path + "../examples/sample_data/vuws/"

        cls.sensor_0 = swprocess.Sensor1C(amplitude=[1, 2, 3], dt=1, x=0, y=0, z=0,
                                          nstacks=1, delay=0)
        cls.sensor_1 = swprocess.Sensor1C(amplitude=[1, 2, 3], dt=1, x=1, y=0, z=0,
                                          nstacks=1, delay=0)
        cls.sensor_5 = swprocess.Sensor1C(amplitude=[1, 2, 3], dt=1, x=5, y=0, z=0,
                                          nstacks=1, delay=0)
        cls.sensor_6 = swprocess.Sensor1C(amplitude=[1, 2, 3], dt=1, x=6, y=0, z=0,
                                          nstacks=1, delay=0)

    @staticmethod
    def dummy_array(amp, dt, nstacks, delay, nsensors, spacing, source_x):
        """Make simple dummy array from timeseries for testing."""
        sensors = []
        for i in range(nsensors):
            sensor = swprocess.Sensor1C(amp, dt, x=i*spacing, y=0, z=0,
                                        nstacks=nstacks, delay=delay)
            sensors.append(sensor)
        source = swprocess.Source(x=source_x, y=0, z=0)
        return swprocess.Array1D(sensors=sensors, source=source)

    def test_init(self):
        # Basic
        source = swprocess.Source(x=-5, y=0, z=0)
        sensors = [self.sensor_0, self.sensor_1]
        array = swprocess.Array1D(sensors=sensors, source=source)
        self.assertEqual(array.source, source)
        self.assertListEqual(array.sensors, sensors)


        # # Normalize positions
        # sensor_5 = swprocess.Sensor1C.from_sensor1c(self.sensor_5)
        # sensor_6 = swprocess.Sensor1C.from_sensor1c(self.sensor_6)
        # source = swprocess.Source(x=-5, y=0, z=0)
        # array = swprocess.Array1D(sensors=[sensor_5, sensor_6],
        #                           source=source, normalize_positions=True)
        # self.assertEqual(2, array.nchannels)
        # self.assertListEqual([0, 1], array.position)
        # self.assertEqual(5, array.absolute_minus_relative)
        # self.assertEqual(-10, array.source.x)

        # Bad: Invalid sensors
        self.assertRaises(ValueError, swprocess.Array1D, sensors=[self.sensor_5, self.sensor_5],
                          source=source)

        # Bad: Incompatable sensors
        sensor_bad = swprocess.Sensor1C(amplitude=[1, 2, 3, 4], dt=1, x=7, y=0, z=0,
                                        nstacks=1, delay=0)
        self.assertRaises(ValueError, swprocess.Array1D, sensors=[self.sensor_5, sensor_bad],
                          source=source)

    def test_timeseriesmatrix(self):
        source = swprocess.Source(x=-5, y=0, z=0)
        sensors = [self.sensor_0, self.sensor_1]
        array = swprocess.Array1D(sensors=sensors, source=source)
        expected = np.array([[1., 2., 3.], [1., 2., 3.]])
        returned = array.timeseriesmatrix
        self.assertArrayEqual(expected, returned)

    def test_source_inside(self):
        # _source_inside
        sensors = [self.sensor_0, self.sensor_6]

        # _source_inside -> True
        source = swprocess.Source(x=3, y=0, z=0)
        array = swprocess.Array1D(sensors, source)
        self.assertTrue(array._source_inside)

        # _source_inside -> False
        source = swprocess.Source(x=-10, y=0, z=0)
        array = swprocess.Array1D(sensors, source)
        self.assertFalse(array._source_inside)

    def test_flip_required(self):
        sensors = [self.sensor_0, self.sensor_1]

        # _flip_required -> True
        source = swprocess.Source(x=3, y=0, z=0)
        array = swprocess.Array1D(sensors, source)
        self.assertTrue(array._flip_required)

        # _flip_required -> False
        source = swprocess.Source(x=-5, y=0, z=0)
        array = swprocess.Array1D(sensors, source)
        self.assertFalse(array._flip_required)

    def test_from_files(self):
        # Single File : SEG2
        fname = self.vuws_path + "1.dat"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            known = obspy.read(fname)
        test = swprocess.Array1D.from_files(fname)
        self.assertArrayEqual(known.traces[0].data,
                              test.timeseriesmatrix[0, :])

        # Single File : SU
        fname = self.full_path + "data/denise/v1.2_y.su.shot2"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            known = obspy.read(fname)
        test = swprocess.Array1D.from_files(fname)
        self.assertArrayEqual(known.traces[0].data,
                              test.timeseriesmatrix[0, :])

        # Multiple Files
        fnames = [f"{self.vuws_path}{x}.dat" for x in range(1, 5)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp_stream = obspy.read(fnames[0])
            expected = np.zeros(tmp_stream.traces[0].data.size)
            for fname in fnames:
                tmp = obspy.read(fname).traces[0]
                expected += tmp.data
            expected /= len(fnames)
        returned = swprocess.Array1D.from_files(fnames)[0].amp
        self.assertArrayAlmostEqual(expected, returned, places=2)

        # Bad : incompatable sources
        fnames = [f"{self.vuws_path}{x}.dat" for x in range(1, 10)]
        self.assertRaises(ValueError, swprocess.Array1D.from_files, fnames)

        # Bad : miniseed
        fname = self.full_path+"data/custom/0101010.miniseed"
        self.assertRaises(NotImplementedError,
                          swprocess.Array1D.from_files, fname)

    def test_plot_waterfall(self):
        # Single shot (near-side)
        fname = self.vuws_path+"1.dat"
        array1 = swprocess.Array1D.from_files(fname)
        array1.waterfall()

        # Multiple shots (near-side)
        fnames = [f"{self.vuws_path}{x}.dat" for x in range(1, 6)]
        array2 = swprocess.Array1D.from_files(fnames)
        array2.waterfall()

        # Single shot (far-side)
        fname = self.vuws_path+"16.dat"
        array3 = swprocess.Array1D.from_files(fname)
        array3.waterfall()

        # Multiple shots (near-side)
        fnames = [f"{self.vuws_path}{x}.dat" for x in range(16, 20)]
        array4 = swprocess.Array1D.from_files(fnames)
        array4.waterfall()
        array4.waterfall(time_along="x")

        # Bad : time_along
        self.assertRaises(ValueError, array4.waterfall, time_along="z")

        # plt.close('all')
        plt.show()

    # def test_plot_array(self):
    #     # Basic case (near-side, 2m spacing)
    #     fname = self.vuws_path+"1.dat"
    #     swprocess.Array1D.from_files(fname).plot()

    #     # Non-linear spacing
    #     sensors = [swprocess.Sensor1C(
    #         [1, 2, 3], dt=1, x=x, y=0, z=0,) for x in [0, 1, 3]]
    #     source = swprocess.Source(x=-5, y=0, z=0)
    #     array = swprocess.Array1D(sensors=sensors, source=source)
    #     array.plot()

    #     # Basic case (far-side, 2m spacing)
    #     fname = self.vuws_path+"20.dat"
    #     swprocess.Array1D.from_files(fname).plot()
    #     plt.close("all")
    #     # plt.show()

    # def test_trim_timeseries(self):
    #     # Standard case (1s delay, 1s record -> 0.5s record)
    #     array = self.dummy_array(amp=np.sin(2*np.pi*1*np.arange(-1, 1, 0.01)),
    #                              dt=0.01, nstacks=1, delay=-1, nsensors=2,
    #                              spacing=2, source_x=-5)
    #     self.assertEqual(-1, array.sensors[0].delay)
    #     self.assertEqual(200, array.sensors[0].nsamples)
    #     array.trim(0, 0.5)
    #     self.assertEqual(0, array.sensors[0].delay)
    #     self.assertEqual(51, array.sensors[0].nsamples)

    #     # Long record (-1s delay, 2s record -> 1s record)
    #     array = self.dummy_array(amp=np.sin(2*np.pi*1*np.arange(-1, 2, 0.01)),
    #                              dt=0.01, nstacks=1, delay=-1, nsensors=2,
    #                              spacing=2, source_x=-5)
    #     self.assertEqual(-1, array.sensors[0].delay)
    #     self.assertEqual(300, array.sensors[0].nsamples)
    #     array.trim(0, 1)
    #     self.assertEqual(0, array.sensors[0].delay)
    #     self.assertEqual(101, array.sensors[0].nsamples)

    #     # Bad trigger (-0.5s delay, 0.5s record -> 0.2s record)
    #     array = self.dummy_array(amp=np.sin(2*np.pi*1*np.arange(-0.5, 0.5, 0.01)),
    #                              dt=0.01, nstacks=1, delay=-0.5, nsensors=2,
    #                              spacing=2, source_x=-5)
    #     self.assertEqual(-0.5, array.sensors[0].delay)
    #     self.assertEqual(100, array.sensors[0].nsamples)
    #     array.trim(-0.1, 0.1)
    #     self.assertEqual(-0.1, array.sensors[0].delay)
    #     self.assertEqual(21, array.sensors[0].nsamples)

    # def test_auto_pick_first_arrivals(self):
    #     s1 = swprocess.Sensor1C(np.concatenate((np.zeros(100),
    #                                             np.array([0.1, 0, 0]),
    #                                             np.zeros(100))),
    #                             dt=1, x=1, y=0, z=0)
    #     s2 = swprocess.Sensor1C(np.concatenate((np.zeros(100),
    #                                             np.array([0, 0.2, 0]),
    #                                             np.zeros(100))),
    #                             dt=1, x=2, y=0, z=0)
    #     source = swprocess.Source(0, 0, 0)
    #     array = swprocess.Array1D([s1, s2], source)

    #     # algorithm = "threshold"
    #     position, times = array.auto_pick_first_arrivals(algorithm="threshold")
    #     self.assertListEqual(array.position, position)
    #     self.assertListEqual([100, 101], times)

    #     # algorithm = "bad"
    #     self.assertRaises(NotImplementedError, array.auto_pick_first_arrivals,
    #                       algorithm="bad")

    # # def test_manual_pick_first_arrivals(self):
    # #     # fnames = self.full_path + "data/denise/v1.2_y.su.shot1"
    # #     fnames = self.full_path + "../examples/sample_data/vuws/10.dat"

    # #     array = swprocess.Array1D.from_files(fnames=fnames)
    # #     #  map_x=lambda x:x/1000,
    # #     #  map_y=lambda y:y/1000)

    # #     array.waterfall()
    # #     array.interactive_mute()
    # #     # array.mute(pre_mute=((0, 0.0), (46, 0.2)), post_mute=((0, 0.2), (46, 0.7)),
    # #     #            shape="tukey")
    # #     array.waterfall()
    # #     # plt.show()
    # #     # distance, time = array.manual_pick_first_arrivals()
    # #     # print(distance, time)

    # def test_from_array1d(self):
    #     source = swprocess.Source(1, 0, 0)

    #     # Non-normalized
    #     sensors = [self.sensor_1, self.sensor_5]
    #     expected = swprocess.Array1D(sensors, source)
    #     returned = swprocess.Array1D.from_array1d(expected)
    #     self.assertEqual(expected, returned)

    #     # Normalized
    #     sensors = [swprocess.Sensor1C.from_sensor1c(
    #         getattr(self, f"sensor_{num}")) for num in [1, 5]]
    #     expected = swprocess.Array1D(sensors, source, normalize_positions=True)
    #     returned = swprocess.Array1D.from_array1d(expected)
    #     self.assertEqual(expected, returned)

    # def test_eq(self):
    #     source_0 = swprocess.Source(0, 0, 0)
    #     source_1 = swprocess.Source(1, 0, 0)
    #     array_a = swprocess.Array1D([self.sensor_0, self.sensor_1], source_0)
    #     array_b = "array1d"
    #     array_c = swprocess.Array1D([self.sensor_0], source_0)
    #     array_d = swprocess.Array1D([self.sensor_0, self.sensor_1], source_1)
    #     array_e = swprocess.Array1D([self.sensor_5, self.sensor_6], source_0)
    #     array_f = swprocess.Array1D([self.sensor_0, self.sensor_1], source_0)

    #     self.assertFalse(array_a == array_b)
    #     self.assertFalse(array_a == array_c)
    #     self.assertFalse(array_a == array_d)
    #     self.assertFalse(array_a == array_e)
    #     self.assertFalse(array_a != array_f)


if __name__ == '__main__':
    unittest.main()
