import os
import numpy as np
import obspy

aName = 'VUWS_L100'
s_time = [2017,06,02,23,45,0]
e_time = [2017,06,03,00,30,0]
stations = [11,12,13,14,15,16,17,18,19,20]
searchDates = [02,03]
searchHours = [23,00]

# Loop through stations
for e in range( len(stations) ):
    print 'Extracting data from station '+str(stations[e])

    # Count number of files for each station
    n_files = 0

    # Loop through potential dates
    for p in range( len(searchDates) ):

        # Loop through potential hours
        for f in range( len(searchHours) ):
            # Current miniseed file named as follows: UT.STN01_YYYYMMDD_HH0000.miniseed
            # (e.g. UT.STN04_20160107_150000.miniseed) 
            currentFile = 'UT.STN'+str(stations[e]).zfill(2) + '_' + str(searchDates[p]) + '_' + str(searchHours[f]).zfill(2) + '0000.miniseed'  
        #if os.path.isfile(currentFile): 
                # Import data from current file
            if n_files==0:
                XX = obspy.read(currentFile)
            else: 
                XX += obspy.read(currentFile)
            n_files += 1

    # Trim stream object to be between specified start and end times
    s_timeS = obspy.UTCDateTime(s_time[0], s_time[1], s_time[2], s_time[3], s_time[4], s_time[5])
    e_timeS = obspy.UTCDateTime(e_time[0], e_time[1], e_time[2], e_time[3], e_time[4], e_time[5])
    XX.trim( starttime=s_timeS, endtime=e_timeS )

    # Store new miniseed files in folder titled "Array Miniseed"
    if not os.path.isdir('./' + aName):
        print 'Creating \"Array Miniseed\" directory'
        os.mkdir('./' + aName) 

    # Create new miniseed file containing only the data between specified start and end times
    filename = './'+aName+'/UT.STN'+ str(stations[e]).zfill(2) + '.' + aName + '.miniseed'
    print 'Creating file :' + filename
    XX.write( filename, format="MSEED" )