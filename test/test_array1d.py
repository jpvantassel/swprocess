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
        # Successful __init__
        sensor_1 = swprocess.Sensor1C(amplitude=[1, 2, 3], dt=1, x=0, y=0, z=0,
                                      nstacks=1, delay=0)
        sensor_2 = swprocess.Sensor1C(amplitude=[1, 2, 3], dt=1, x=1, y=0, z=0,
                                      nstacks=1, delay=0)
        source = swprocess.Source(x=-5, y=0, z=0)
        array = swprocess.Array1D(sensors=[sensor_1, sensor_2],
                                  source=source)
        self.assertEqual(2, array.nchannels)

        expected = np.array([[1., 2., 3.], [1., 2., 3.]])
        returned = array.timeseriesmatrix
        self.assertArrayEqual(expected, returned)

    def test_from_files(self):
        # Single File
        fname = self.full_path + "data/vuws/1.dat"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            known = obspy.read(fname)
        test = swprocess.Array1D.from_files(fname)
        self.assertArrayEqual(known.traces[0].data,
                              test.timeseriesmatrix[0, :])

        # Multiple Files
        fnames = [f"{self.full_path}data/vuws/{x}.dat" for x in range(1, 5)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp_stream = obspy.read(fname)
            expected = np.zeros(tmp_stream.traces[0].data.size)
            for fname in fnames:
                tmp = obspy.read(fname).traces[0]
                expected += tmp.data
            expected /= len(fnames)
        returned = swprocess.Array1D.from_files(fnames)[0].amp
        self.assertArrayAlmostEqual(expected, returned, places=2)

    def test_plot_waterfall(self):
        # Single shot (near-side)
        fname = self.full_path+"data/vuws/1.dat"
        array1 = swprocess.Array1D.from_files(fname)
        array1.waterfall()

        # Multiple shots (near-side)
        fnames = [f"{self.full_path}data/vuws/{x}.dat" for x in range(1, 6)]
        array2 = swprocess.Array1D.from_files(fnames)
        array2.waterfall()

        # Single shot (far-side)
        fname = self.full_path+"data/vuws/16.dat"
        array3 = swprocess.Array1D.from_files(fname)
        array3.waterfall()

        # Multiple shots (near-side)
        fnames = [f"{self.full_path}data/vuws/{x}.dat" for x in range(16, 20)]
        array4 = swprocess.Array1D.from_files(fnames)
        array4.waterfall()
        plt.close('all')
        # plt.show()

    def test_plot_array(self):
        # Basic case (near-side, 2m spacing)
        fname = self.full_path+"data/vuws/1.dat"
        swprocess.Array1D.from_files(fname).plot()

        # Non-linear spacing
        sensors = [swprocess.Sensor1C(
            [1, 2, 3], dt=1, x=x, y=0, z=0,) for x in [0, 1, 3]]
        source = swprocess.Source(x=-5, y=0, z=0)
        array = swprocess.Array1D(sensors=sensors, source=source)
        array.plot()

        # Basic case (far-side, 2m spacing)
        fname = self.full_path+"data/vuws/20.dat"
        swprocess.Array1D.from_files(fname).plot()
        plt.close("all")
        # plt.show()

    def test_trim_timeseries(self):
        # Standard case (1s delay, 1s record -> 0.5s record)
        array = self.dummy_array(amp=np.sin(2*np.pi*1*np.arange(-1, 1, 0.01)),
                                 dt=0.01, nstacks=1, delay=-1, nsensors=2,
                                 spacing=2, source_x=-5)
        self.assertEqual(-1, array.sensors[0].delay)
        self.assertEqual(200, array.sensors[0].nsamples)
        array.trim(0, 0.5)
        self.assertEqual(0, array.sensors[0].delay)
        self.assertEqual(51, array.sensors[0].nsamples)

        # Long record (-1s delay, 2s record -> 1s record)
        array = self.dummy_array(amp=np.sin(2*np.pi*1*np.arange(-1, 2, 0.01)),
                               dt=0.01, nstacks=1, delay=-1, nsensors=2,
                               spacing=2, source_x=-5)
        self.assertEqual(-1, array.sensors[0].delay)
        self.assertEqual(300, array.sensors[0].nsamples)
        array.trim(0, 1)
        self.assertEqual(0, array.sensors[0].delay)
        self.assertEqual(101, array.sensors[0].nsamples)

        # Bad trigger (-0.5s delay, 0.5s record -> 0.2s record)
        array = self.dummy_array(amp=np.sin(2*np.pi*1*np.arange(-0.5, 0.5, 0.01)),
                               dt=0.01, nstacks=1, delay=-0.5, nsensors=2,
                               spacing=2, source_x=-5)
        self.assertEqual(-0.5, array.sensors[0].delay)
        self.assertEqual(100, array.sensors[0].nsamples)
        array.trim(-0.1, 0.1)
        self.assertEqual(-0.1, array.sensors[0].delay)
        self.assertEqual(21, array.sensors[0].nsamples)


if __name__ == '__main__':
    unittest.main()
