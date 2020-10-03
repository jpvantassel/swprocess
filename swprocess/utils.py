"""Surface wave processing utilities."""

import os

import obspy
import pandas as pd

logger = logging.getLogger("swprocess.utils")

def extract_mseed(startend_fname, network):
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
        Name of .xlsx or .csv file with start and end times. An example
        file is provided here TODO (jpv): Add link to example file.
    network : str
        Short string of characters to identify the network. Exported
        files will utilize this network code as its prefix.

    Returns
    -------
    None
        Writes folder and files to disk.

    """
    # Read start and end times.
    try:
        df = pd.read_excel(startend_fname)
    except:
        raise NotImplementedError("To implement csv parsing")

    # Loop through across defined timeblocks.
    logger.info("Begin iteration across dataframe ...")
    for index, series in df.iterrows():
        logger.debug(f"\tindex={index} series={series}")

        # Range of years required.
        years = np.range(series["start year"], series["end years"] + 1)
        logger.debug(f"\t\tyears={years}")
        
        # Loop through potential months
        for m in range(len(searchYears)):
            if db['Start Month'][n]==db['End Month'][n]:
                searchMonths=[db['Start Month'][n]]
            else:
                error('Different start and end months!')
                
            # Loop through potential dates  
            for o in range(len(searchMonths)):
                if db['Start Date'][n]==db['End Date'][n]:
                    searchDates=[db['Start Date'][n]]
                else:
                    searchDates = list(range(db['Start Date'][n],db['End Date'][n]+1))
                    
                # Loop through potential hours
                for p in range(len(searchDates)):
                    if (len(searchDates)==1) & (db['Start Hour'][n]==db['End Hour'][n]):
                        searchHours=[db['Start Hour'][n]]
                    elif (len(searchDates)==1) & (db['Start Hour'][n]<db['End Hour'][n]):
                        searchHours=list(range(db['Start Hour'][n],db['End Hour'][n]+1))
                    else:
                        if p==0:
                            searchHours=list(range(db['Start Hour'][n],24))
                        elif p==len(searchDates)-1:
                            searchHours = list(range(0,db['End Hour'][n]+1))
                        else:
                            searchHours=list(range(0,24))
                            
                    for q in range(len(searchHours)):
                
    #                     # Current miniseed file named as follows: UT.STN01_YYYYMMDD_HH0000.miniseed
    #                     currentFile = station_code+'.STN'+str(db.Station[n]).zfill(2)+'_'+str(searchYears[m])+str(searchMonths[o]).zfill(2)\
    #                     +str(searchDates[p]).zfill(2)+'_'+str(searchHours[q]).zfill(2)+'0000.miniseed'
                        
                        # Current miniseed file named as follows: CD.STN01_SENSOR_YYYYMMDD_HH0000.miniseed
                        currentFile = station_code+'.STN'+str(db.Station[n]).zfill(2)+'_'+str(db.Sensor[n])+'_'+str(searchYears[m])+str(searchMonths[o]).zfill(2)\
                        +str(searchDates[p]).zfill(2)+'_'+str(searchHours[q]).zfill(2)+'0000.miniseed'

                        # Read current file and append if necessary
                        data_dir = f"{searchYears[m]}_{str(searchMonths[o]).zfill(2)}_{str(searchDates[p]).zfill(2)}"
                        if n_files==0:
                            XX = obspy.read(data_dir+'\\'+currentFile)
                            n_files += 1
                        else:
                            XX += obspy.read(data_dir+'\\'+currentFile)
                            
        XX = XX.merge(method=1)
        
    #     # Rename station for all components
    #     for r in range(3):
    #         XX.traces[r].stats.station = str(n).zfill(3)
                        
        # Trim stream object to be between specified start and end times
        s_timeS = obspy.UTCDateTime(db['Start Year'][n], db['Start Month'][n], db['Start Date'][n], db['Start Hour'][n],\
                                    db['Start Minute'][n], db['Start Second'][n])
        e_timeS = obspy.UTCDateTime(db['End Year'][n], db['End Month'][n], db['End Date'][n], db['End Hour'][n],\
                                    db['End Minute'][n], db['End Second'][n])
        XX.trim( starttime=s_timeS, endtime=e_timeS)
                
        # Store new miniseed files in folder titled "Array Miniseed"
        aName = db.ShortName[n]
        if not os.path.isdir('./' + aName):
            print("Creating Directory".format(aName))
            os.mkdir('./' + aName) 
        
        # In case somehow it becomes a masked array. Why this sometimes happens is unclear
        for tr in XX:
            if isinstance(tr.data, np.ma.masked_array):
                tr.data = tr.data.filled()
                print(str(db.ShortName[n])+str(n).zfill(2)+' was a masked array')

        # Create new miniseed file containing only the data between specified start and end times   
    #     filename='SSHV_'+str(n).zfill(3)+'.miniseed'
    #     print('SSHV '+str(n+1)+' of '+str(total)+' Extracting data from station '+str(db.Station[n]).zfill(2)+' Creating file: ' + filename)

        filename = './'+aName+'/'+station_code+'.STN'+str(db.Station[n]).zfill(2) + '.' + aName + '.miniseed'
        print('Extracted '+str(n+1)+' of '+str(total)+' Extracting data from station '+str(db.Station[n]).zfill(2)+' Creating file: ' + filename)

        XX.write( filename, format="MSEED")