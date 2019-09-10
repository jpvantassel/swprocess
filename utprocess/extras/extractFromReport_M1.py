""" 
    This script extracts a user-defined number of layered earth models, 
    dispersion curves, and ellipticity curves from the output file from a Geopsy
    inversion (i.e., a .report file). Extracted data are output to .txt files.
    This specific script is intended for analyses wherein the misfit is to be 
    re-computed (e.g., a misfit <= 1 analysis). Results are extracted from the 
    original .report file, misfits are re-computed, and results corresponding to
    both the lowest misfit models and the last n models are exported. 
    
    This code was developed at the University of Texas at Austin.
    
    Copyright (C) 2017  David P. Teague
 
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

# INPUTS************************************************************************
#*******************************************************************************

# Input report file
report_file = 'run_01'
# Input target file (for re-computing misfit)
target_file = 'GV_North_Best_NoEll'
# Ellipticity flag (True to consider ellipticity peak in misfit calc, otherwise False)
flag_ell=False

# Name of output files 
#( Prefix_LR_bestNp_type.txt and Prefix_LR_Np_type.txt, where Np is the number  
# of profiles and type is GM for Ground Model, DC for dispersion data, or Ell 
# for ellipticity data)
LR = '3-5'
prefix = 'GV_North_LR'

# Number of profiles to extract and recompute misfit
Nbest = 10000
# Number of lowest-misfit GM/DC/Ell to extract
Np = 1000
# Number of dispersion modes
Nmodes = 3
# Minimum, maximum, and number of frequency samples, respectively
minF = 0.1
maxF = 100
Nsamples = 100

# END OF INPUTS*****************************************************************
#*******************************************************************************



# Import modules
import shlex, subprocess, os


# Extract Nbest profiles (presumably with misfit <= 1.0)
command = 'c:/geopsy.org/bin/gpdcreport ' + report_file + '.report -best ' + str(Nbest) + ' -gm ' + ' -o temp1.report' 
p1 = subprocess.call( shlex.split(command) )

# Recalculate misfit for Nbest profiles
if flag_ell:
    command = 'c:/geopsy.org/bin/gpdcmisfit -report temp1.report -target '+ target_file+ '.target -all 0 -o temp2.report' 
else:
    command = 'c:/geopsy.org/bin/gpdcmisfit -report temp1.report -target '+ target_file+ '.target -disp 0 -o temp2.report'
p1 = subprocess.call( shlex.split(command) ) 


# Commands to extract profiles from report (used multiple times)
command1a = 'c:/geopsy.org/bin/gpdcreport temp2.report -best ' + str(Np) + ' -gm' 
command1b = 'c:/geopsy.org/bin/gpdcreport temp2.report -n ' + str(Np) + ' -gm'


# Extract Np profiles with lowest misfits from the report file
command2a = 'c:/geopsy.org/bin/gpdcreport temp2.report -best ' + str(Np) + ' -gm' 
command2b = 'c:/geopsy.org/bin/gpdcreport temp2.report -n ' + str(Np) + ' -gm' 
ofile_a = prefix+ '_'+ LR+ '_best' + str(Np) + '_GM.txt'
ofile_b = prefix+ '_'+ LR+ '_' + str(Np) + '_GM.txt'
# Best 1000
with open(ofile_a, 'w') as outfile:
    subprocess.call( shlex.split(command2a), stdout=outfile )
# Last 1000
with open(ofile_b, 'w') as outfile:
    subprocess.call( shlex.split(command2b), stdout=outfile )


# Extract Np dispersion curves with lowest misfits from the report file
command2 = 'c:/geopsy.org/bin/gpdc -R '+ str(Nmodes) + ' -min ' + str(minF)+ ' -max ' + str(maxF) + ' -n '+ str(Nsamples)+ ' -f' 
ofile_a = prefix + '_'+ LR+ '_best'+str(Np)+'_DC.txt'
ofile_b = prefix + '_'+ LR+ '_'+str(Np)+'_DC.txt'
# Best 1000
p1a = subprocess.Popen( shlex.split(command1a), stdout = subprocess.PIPE )
with open(ofile_a, 'w') as outfile:
    subprocess.call( shlex.split(command2), stdin=p1a.stdout, stdout=outfile )
p1a.stdout.close()
# Last 1000
p1b = subprocess.Popen( shlex.split(command1b), stdout = subprocess.PIPE )
with open(ofile_b, 'w') as outfile:
    subprocess.call( shlex.split(command2), stdin=p1b.stdout, stdout=outfile )
p1b.stdout.close()


# Extract Np ellipticity curves with lowest misfits from the report file
command2 = 'c:/geopsy.org/bin/gpell -R 1' + ' -min ' + str(minF) + ' -max ' + str(maxF) +  ' -n ' + str(Nsamples)
ofile_a = prefix + '_'+ LR+ '_best'+str(Np)+'_Ell.txt'
ofile_b = prefix + '_'+ LR+ '_'+str(Np)+'_Ell.txt'
# Best 1000
p1a = subprocess.Popen( shlex.split(command1a), stdout = subprocess.PIPE )
with open(ofile_a, 'w') as outfile:
    subprocess.call( shlex.split(command2), stdin=p1a.stdout, stdout=outfile )
p1a.stdout.close()
# Last 1000
p1b = subprocess.Popen( shlex.split(command1b), stdout = subprocess.PIPE )
with open(ofile_b, 'w') as outfile:
    subprocess.call( shlex.split(command2), stdin=p1b.stdout, stdout=outfile )
p1b.stdout.close()


# Remove temporary report files
os.remove( 'temp1.report' )
os.remove( 'temp2.report' )
