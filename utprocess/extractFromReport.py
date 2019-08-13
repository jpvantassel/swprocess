""" 
    This script extracts a user-defined number of layered earth models, 
    dispersion curves, and ellipticity curves from the output file from a Geopsy
    inversion (i.e., a .report file). Extracted data are output to .txt files.
    
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

# Name of .report file
report_file = 'run_01'
# Name of output files ( Prefix_LR_bestNp_type.txt, where Np is the number of 
# profiles and type is GM for Ground Model, DC for dispersion data, or Ell for 
# ellipticity data)
LR = '3'
prefix = 'GV_North_LR'


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
import shlex, subprocess

# Command to extract profiles from report (used multiple times)
command1 = 'c:/geopsy.org/bin/gpdcreport ' + report_file + '.report -best '+ str(Np)

# Extract Np profiles with lowest misfits from the report file
command2 = 'c:/geopsy.org/bin/gpdcreport ' + report_file + '.report -best ' + str(Np) + ' -gm' 
ofile = prefix+ '_'+ LR+ '_best' + str(Np) + '_GM.txt'
with open(ofile, 'w') as outfile:
    p1 = subprocess.Popen( shlex.split(command2), stdout=outfile )

# Extract Np dispersion curves with lowest misfits from the report file
command2 = 'c:/geopsy.org/bin/gpdc -R '+ str(Nmodes) + ' -min ' + str(minF)+ ' -max ' + str(maxF) + ' -n '+ str(Nsamples)+ ' -f' 
ofile = prefix + '_'+ LR+ '_best'+str(Np)+'_DC.txt'
p1 = subprocess.Popen( shlex.split(command1), stdout = subprocess.PIPE )
with open(ofile, 'w') as outfile:
    p2 = subprocess.Popen( shlex.split(command2), stdin=p1.stdout, stdout=outfile )
p1.stdout.close()

# Extract Np ellipticity curves with lowest misfits from the report file
command2 = 'c:/geopsy.org/bin/gpell -R 1' + ' -min ' + str(minF) + ' -max ' + str(maxF) +  ' -n ' + str(Nsamples)
ofile = prefix + '_'+ LR+ '_best'+str(Np)+'_Ell.txt'
p1 = subprocess.Popen( shlex.split(command1), stdout = subprocess.PIPE )
with open(ofile, 'w') as outfile:
    p2 = subprocess.Popen( shlex.split(command2), stdin=p1.stdout, stdout=outfile )
p1.stdout.close()



