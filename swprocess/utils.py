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

"""Surface wave processing utilities."""

import os
import datetime
import logging
import warnings
import pathlib

import numpy as np
import obspy
import pandas as pd

logger = logging.getLogger("swprocess.utils")


def extract_mseed(startend_fname, network, data_dir="./", output_dir="./", extension="mseed"):
    """Extract specific time blocks from a set of miniseed files.

    Reads a large set of miniseed files, trims out specified time
    block(s), and writes the trimmed block(s) to disk. Useful for
    condensing a large dataset consisting of miniseed files written at
    the end of each hour to a single file that spans several hours.
    Stations which share an array name will appear in a common
    directory.

    Parameters
    ----------
    startend_fname : str
        Name of .csv file with start and end times. An example file is
        provided `here <https://github.com/jpvantassel/swprocess/blob/main/examples/extract/extract_startandend.csv>`_
    network : str
        Short string of characters to identify the network. Exported
        files will utilize this network code as its prefix.
    data_dir : str, optional
        The full or a relative file path to the directory containing the
        miniseed files, default is the current directory.
    output_dir : str, optional
        The full or a relative file path to the location to place the
        output miniseed files, default is the current directory.
    extension : {"mseed", "miniseed"}, optional
        Extension used for miniSEED format, default is `"mseed"`.

    Returns
    -------
    None
        Writes folder and files to disk.

    """
    data_dir = pathlib.Path(data_dir)
    output_dir = pathlib.Path(output_dir)

    # Read start and end times.
    dtype = {"folder name": str, "array name": str, "station number": int,
             "start year": int, "start month": int, "start day": int,
             "start hour": int, "start minute": int, "start second": int,
             "end year": int, "end month": int, "end day": int,
             "end hour": int, "end minute": int, "end second": int,
             "notes": str}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(startend_fname, dtype=dtype)
    except:
        raise NotImplementedError("File type not recognized.")

    # Loop through across defined timeblocks.
    logger.info("Begin iteration across dataframe ...")
    total = df["folder name"].count()
    for index, series in df.iterrows():
        logger.debug(f"\tindex={index} series={series}")

        # Start and end time.
        starttime = datetime.datetime(year=series["start year"],
                                      month=series["start month"],
                                      day=series["start date"],
                                      hour=series["start hour"],
                                      tzinfo=datetime.timezone.utc)
        logging.debug(f"\t\tstarttime={starttime}")
        currenttime = starttime

        endtime = datetime.datetime(year=series["end year"],
                                    month=series["end month"],
                                    day=series["end date"],
                                    hour=series["end hour"],
                                    tzinfo=datetime.timezone.utc)
        logging.debug(f"\t\tendtime={endtime}")

        # Avoid nonsensical time blocks.
        if endtime < starttime:
            msg = f"endtime={endtime} is less than starttime={starttime}."
            raise ValueError(msg)

        # Loop across the required hours and merge traces.
        append = False
        dt = datetime.timedelta(hours=1)
        reads = None
        while currenttime <= endtime:

            # miniSEED file name: NW.STNSN_SENSOR_YYYYMMDD_HH0000.miniseed
            fname = f"{network}.STN{str(series['station number']).zfill(2)}_{currenttime.year}{str(currenttime.month).zfill(2)}{str(currenttime.day).zfill(2)}_{str(currenttime.hour).zfill(2)}0000.{extension}"

            # Read current file and append if necessary
            if append:
                reads += obspy.read(str(data_dir / fname))
            else:
                reads = obspy.read(str(data_dir / fname))
                append = True

            currenttime += dt

        reads = reads.merge(method=1)

        # Trim merged traces between specified start and end times
        trim_start = obspy.UTCDateTime(series["start year"], series["start month"],
                                       series["start date"], series["start hour"],
                                       series["start minute"], series["start second"])
        trim_end = obspy.UTCDateTime(series["end year"], series["end month"],
                                     series["end date"], series["end hour"],
                                     series["end minute"], series["end second"])
        reads.trim(starttime=trim_start, endtime=trim_end)

        # Store new miniseed files in folder titled "Array Miniseed"
        folder = series["folder name"]
        if not os.path.isdir(f"{output_dir / folder}"):
            logger.info(f"Creating folder: {output_dir / folder}")
            os.mkdir(f"{output_dir / folder}")

        # Unmask masked array.
        for tr in reads:
            if isinstance(tr.data, np.ma.masked_array):
                tr.data = tr.data.filled()
                msg = f"{folder}/{network}.STN{str(series['station number']).zfill(2)} was a masked array."
                warnings.warn(msg)

        # Write trimmed file to disk.
        fname_out = output_dir / folder / f"{network}.STN{str(series['station number']).zfill(2)}.{series['array name']}.{extension}"
        logger.info(
            f"Extracted {index+1} of {total}. Extracting data from station {str(series['station number']).zfill(2)}. Creating file: {fname_out}.")

        reads.write(fname_out, format="mseed")
