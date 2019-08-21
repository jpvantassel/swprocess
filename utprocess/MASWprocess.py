"""
    This script calculates experimental dispersion data for one or more
    source-offsets. Input file(s) for each source offset are assumed to be in
    seg2 (.dat) format.

    Note that the data acquisition paramters (e.g., receiver spacing, source-
    offset, etc) are automatically read from the seg2 file. Thus, if the data
    acquisition parameters were not properly entered into the data acquisition
    system, then these values should be manually overwritten by setting the
    optional processing inputs (see below).


    This code was developed at the University of Texas at Austin.
    Copyright (C) 2016  David P. Teague, Clinton M. Wood, and Brady R. Cox

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

# # INPUTS************************************************************************
# #*******************************************************************************

# # Processing inputs.............................................................

# # Input file numbers (e.g., [ [1,10], [11, 20], [21,30] ] , where shots
# # 1-10 correspond to one offset, 11-20 correspond to another offset, etc.)
# infile_numbers = [[6,10],[11,15],[16,20]]

# # Directory containing input seg2 (.dat) files. (Enter pwd for current directory)
# infile_path = r'E:\New Zealand 2017\2_Wellington_June2017\Raw Active\CMPK'

# # Files to exclude from calculations (enter [] to include all files)
# exclude_files=[]

# # Name output compressed pickle (.pklz) file
# outfile_name = 'Delete_raw'

# # Path of output compressed pickle (.pklz) file
# outfile_path = infile_pathp

# # Length of recorded trace to use in calculations [s]
# timelength = 1

# # Desired spacing of dispersion data points [Hz]
# deltaf = 0.5

# # Min and max frequency to consider in calculations
# min_freq = 5
# max_freq = 60

# # Min velocity (FDBF, phase-shift, and tau-p) and max velocity (FDBF and phase-shift) to consider in calculations
# min_vel = 80
# max_vel = 1600

# # Number of velocities or wavenumbers to consider in calculations
# n_trial = 2048

# # Processing method (can specify single method or one for each source-offset)
# #   'fk':           Frequency-wavenumber transformation
# #   'fdbf':         Frequency domain beamformer
# #   'phase-shift':  Phase-shift transformation
# #   'slant-stack':  Slant-stack (linear Radon) transformation
# processMethod = [ 'fk' ]

# # Weighting technique for FDBF (similar to processing method, can specify multiple)
# #   'none' for equal weighting
# #   'sqrt' to weight each receiver trace by the square root of the distance from the source
# #   'invamp' to weight each receiver by 1/|A(f,x)|
# weightType = [ 'none' ]

# # Steering vector type to use in FDBF calculations (similar to processing method, can specify multiple)
# #   'plane' for plane wave steering vector
# #   'cylindrical' for cylindrical wave steering vector
# steeringVector = ['plane']

# # Signal-to-noise ratio inputs
# # Noise location ('pretrigger' if pretrigger delay was used or 'end' to obtain from end of record)
# noise_location = 'pretrigger'


# # OPTIONAL processing inputs....................................................
# # (ONLY USE IF SEG2 HEADER FIELDS ARE INCORRECT)

# # Set flag_overwrite=True to overwrite automatically read parameters
# # Otherwise dt, n_channels, position, offset, and delay variables will be ignored
# flag_overwrite = False
# # Sample rate [s]
# dt = [0.004, 0.004, 0.004, 0.004]
# # Number of receivers
# n_channels = [48, 48, 48, 48]
# # Vector containing positions of receivers
# position = [range(0,96,2), range(0,96,2), range(0,96,2), range(0,96,2)]
# # Source offset from first or last receiver
# offset = [5,10,20,40]
# # Pre-trigger delay
# delay = [0,0,0,0]


# # Plotting inputs ..............................................................
# # Set flag_SNR or flag_waterfall equal to True to plot signal-to-noise ratio or waterfall plots, respectively
# flag_SNR = False
# flag_waterfall = True
# # Contour plots (Set equal to True to plot, otherwise False)
# flag_con_fk = False                # Frequency-wavenumber
# flag_con_fw = False                # Frequency-wavelength
# flag_con_fv = True                 # Frequency-phase velocity
# flag_con_fp = False                # Frequency-slowness
# flag_con_wv = False                # Wavelength-phase velocity
# # Slices in various domains (Set equal to True to plot, otherwise False)
# f_plot = range(10,50,10)          # Frequencies to plot slices in chosen domain
# flag_slices_fk = False            # Wavenumber-power at select frequencies
# flag_slices_fw = False            # Wavelength-power at select frequencies
# flag_slices_fv = True            # Velocity-power at select frequencies
# flag_slices_fp = False            # Slowness-power at select frequencies

# END OF INPUTS*****************************************************************
# *******************************************************************************


# Load modules
import json
import os
import time
import numpy as np
import pickle
import gzip
import shotgathers
import dcprocessing
import dctypes
import matplotlib.pyplot as plt
plt.ion()

# while True:
# fname = input("Enter desired processing settings: ")
fname = r"test\processing_settings.json"
if os.path.isfile(fname):
    print(fname)
    print(fname[-4:])
    if fname[-4:] != ".json":
        with open(fname, "r") as f:
            settings = json.load(f)
            # break
    else:
        print("File must be of type .json")
else:
    print("File {} does not exist.".format(fname))


# Initialize lists for storing processed dispersion data
offsetRaw = []
frequencyRaw = []
velocityRaw = []
wavelengthRaw = []

# Time the processing
tic = time.clock()

# Compare signal-to-noise ratios for various offsets
if settings["plot"]["signal_to_noise"]:
    shotgathers.compareSNR(infile_numbers=settings["input_file"]["numbers"],
                           infile_path=settings["input_file"]["path"],
                           noise_location=settings["processing_settings"]["noise_location"],
                           timelength=settings["processing_settings"]["timelength"],
                           exclude_files=settings["input_file"]["exclude_files"],
                           deltaf=settings["processing_settings"]["df"],
                           xlmts=(0, settings["processing_settings"]["max_freq"]))

# Loop through all offsets
print(settings["input_file"]["numbers"])
for k in range(len(settings["input_file"]["numbers"])):

    # Processing***************************************************************
    # TODO (jpv): Move this logic into the import and stack
    # Stack records for current offset location
    # if len(settings["input_file"]["numbers"][k]) == 1:
    #     cShotGather = shotgathers.importAndStackWaveforms(
    #         start_file=settings["input_file"]["numbers"][k][0],
    #         end_file=settings["input_file"]["numbers"][k][0],
    #         file_path=settings["input_file"]["filepath"],
    #         exclude_files=settings["input_file"]["exclude_files"],
    #         print_status='no')
    # else:
    cShotGather = shotgathers.importAndStackWaveforms(
        start_file=settings["input_file"]["numbers"][k][0],
        end_file=settings["input_file"]["numbers"][k][1],
        file_path=settings["input_file"]["path"],
        exclude_files=settings["input_file"]["exclude_files"],
        print_status='no')

    # Manually overwrite automatically-read parameters (optional)
    if settings["overwrite"]["perform_overwrite"]:
        cShotGather.dt = settings["overwrite"]["dt"]
        cShotGather.n_channels = settings["overwrite"]["dt"]
        cShotGather.position == list(range(
            settings["position"]["start"], settings["position"]["stop"], settings["position"]["step"])),
        cShotGather.offset = settings["offset"]["dt"]
        cShotGather.delay = settings["offset"]["delay"]
    cShotGather.print_attributes()
    # Cut time records at timelength
    cShotGather.cut(settings["processing_settings"]["timelength"])
    # Generate waterfall plot
    if settings["plot"]["waterfall"]:
        cShotGather.plot()
    # Padd zeros to acheive desired df
    cShotGather.zero_pad(settings["processing_settings"]["df"])

    # User-defined processing method for current offset
    # if len(processMethod) == 1:
    #     cpMethod=processMethod[0]
    # else:
    #     cpMethod=processMethod[k]
    cpMethod = settings["processing_settings"]["method"]

    # Process dispersion data based on user-defined processing method
    n_trial = settings["processing_settings"]["n_trial"]
    min_freq = settings["processing_settings"]["min_freq"]
    max_freq = settings["processing_settings"]["max_freq"]
    min_vel = settings["processing_settings"]["min_vel"]
    max_vel = settings["processing_settings"]["max_vel"]

    if str.lower(cpMethod) == 'fdbf':
        # User-defined weighting technique for fdbf
        # if len(settings["processing_settings"]["weighting"])weightType) == 1:
        #     cWtType = weightType[0]
        # else:
        #     cWtType = weightType[k]

        # if len(steeringVector) == 1:
        #     cSvec = steeringVector[0]
        # else:
        #     cSvec = steeringVector[k]

        cDCpower = dcprocessing.fdbf(
            cShotGather, settings["processing_settings"]["weighting"],
            settings["processing_settings"]["steering_vector"],
            n_trial, min_vel, max_vel, min_freq, max_freq)

    elif str.lower(cpMethod) == 'fk':
        cDCpower = dcprocessing.fk(cShotGather, n_trial, min_freq, max_freq)

    elif str.lower(cpMethod) == 'phase-shift':
        cDCpower = dcprocessing.phase_shift(
            cShotGather, n_trial, min_freq, max_freq, min_vel, max_vel)

    elif str.lower(cpMethod) == 'slant-stack':
        cDCpower = dcprocessing.tau_p(
            cShotGather, n_trial, min_freq, max_freq, min_vel, max_vel)

    else:
        raise ValueError('Invalid processing method')

    # Store dispersion data in lists
    offsetRaw.append(cShotGather.offset)
    frequencyRaw.append(cDCpower.freq)
    if str.lower(cDCpower.val_type) == 'wavenumber':
        velocityRaw.append(2*np.pi*cDCpower.freq / cDCpower.peak_vals)
        wavelengthRaw.append(2*np.pi / cDCpower.peak_vals)
    elif str.lower(cDCpower.val_type) == 'velocity':
        velocityRaw.append(cDCpower.peak_vals)
        wavelengthRaw.append(cDCpower.peak_vals / cDCpower.freq)
    else:
        raise ValueError(
            'Invalid val_type. Should be \"wavenumber\" or \"velocity\".')

    # Plotting******************************************************************
    # Contour plots

    if settings["plot"]["contour"]["frequency-wavenumber"]:
        cDCpower.plotSpect("fk", [0, max_freq, 0, cDCpower.kres])
    if settings["plot"]["contour"]["frequency-wavelength"]:
        cDCpower.plotSpect(
            "fw", [0, max_freq, 0.5*cShotGather.position[1], 2*cShotGather.position[-1]])
    if settings["plot"]["contour"]["frequency-velocity"]:
        cDCpower.plotSpect("fv", [0, max_freq, 0, max_vel])
    if settings["plot"]["contour"]["frequency-slowness"]:
        cDCpower.plotSpect("fp", [0, max_freq, 1.0/max_vel, 1.0/min_vel])
    if settings["plot"]["contour"]["frequency-wavelength"]:
        cDCpower.plotSpect(
            "wv", [0.5*cShotGather.position[1], 2*cShotGather.position[-1], 0, max_vel])

    # Slices in various domains
    f_plot = list(range(settings["plot"]["slice"]["min_freq"],
                        settings["plot"]["slice"]["max_freq"],
                        settings["plot"]["slice"]["n_freq"]))

    if settings["plot"]["slice"]["type"]["frequency-wavenumber"]:
        cDCpower.plotSlices("fk", f_plot, (0, cDCpower.kres))
    if settings["plot"]["slice"]["type"]["frequency-wavelength"]:
        cDCpower.plotSlices(
            "fw", f_plot, (0.5*cShotGather.position[1], 2*cShotGather.position[-1]))
    if settings["plot"]["slice"]["type"]["frequency-velocity"]:
        cDCpower.plotSlices("fv", f_plot, (0, max_vel))
    if settings["plot"]["slice"]["type"]["frequency-slowness"]:
        cDCpower.plotSlices("fp", f_plot, (1.0/max_vel, 1.0/min_vel))

# Create class containing all raw dispersion data
cRawDC = dctypes.RawDispersion(frequencyRaw, velocityRaw, offsetRaw)
# Automatically cut data with excessively high velocities
cRawDC.rmvHighVs(max_vel)
# Save results to a compressed pickle file
f = gzip.open(settings["output_file"]["path"]+"/" +
              settings["output_file"]["name"]+".pklz",
              'wb')
pickle.dump(cRawDC, f)
f.close()


# Print execution time
toc = time.clock()
print("Elapsed time: "+str(toc - tic)+" seconds")
plt.show(block=True)
