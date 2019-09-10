import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import string
import invpost

#INPUTS*************************************************************************

# Layering ratios and filename information......................................
infoler = 'C:/DPT_Projects/dinver_log_files/Garner Valley/North/Best_TF-1-9/GV_North_Best_Ell_1-78_LR3_reports'
prefix = 'GV_North_LR'
suffix = 'best1000'

LR = ['3']
LR_leg_entries = [r'$\Xi$=3.0']
LR_colors = ['gray']

# Number of profiles, modes, etc................................................
n_profiles = 100
n_modes = 2
flag_tf = True
flag_misfit = True


# Data to plot with inversion results...........................................
# Dispersion data to plot with theoretical curves
# (Format: frequency, slowness, slowness std. dev., and weight)
dc_files = ['C:/DPT_Projects/Garner Valley/Inversion/North/Post-processing/GV_North_R0v2_ExpDC.txt']
dcfile_leg_entries = ['Exp. Disp. Data']
dcfile_colors = ['black']

# HVSR data to plot with ellipticity curves or transfer functions
# (Format: frequency, mean HVSR, mean HVSR - std. dev., mean HVSR + std. dev.)
hv_files = ['C:/DPT_Projects/Garner Valley/Inversion/North/Post-processing/GV_North_HVSR.txt']
hvfile_leg_entries = ['HVSR Data']
hvfile_colors = ['black']

# Alternative/previously developed Vs profiles to plot with inversion Vs profiles
vs_files = []
vsfile_leg_entries = []
vsfile_colors = []


# Plotting size inputs..........................................................
w_fig = 7
h_fig = 7

n_row = 2
n_col = 2

yUp = 0.07
yLow = 0.12
yInt = 0.14

xLeft = 0.09
xRight = 0.08
xInt = 0.18

fname = 'arial'
fsize = 10

tickl = [5,2.5]
tickw = [1,1]
tickdir = 'in'
alphabet = list(string.ascii_lowercase)


# Plotting axes inputs.......................................................... 
nplots = 4

# x-limits (dispersion frequency, tf/ell frequency, Vs, Vs)
xlabels = ['Frequency [Hz]', 'Frequency [Hz]', 'Vs [m/s]', 'Vs [m/s]']
xlims = [[0.5,100], [0.1,10], [0,3500], [0,3500]]
xscale_types = ['log', 'log', 'linear', 'linear']
xminorticks = [True, True, False, False]
flag_dual_axx = [True, True, True, True]
dual_ax_factorx = [1, 1, 3.28084, 3.28084] 
xlabels_dual = ['', '', 'Vs [ft/s]', 'Vs [ft/s]']


# y-limits (phase velocity, tf/ell amplitude, depth, depth)
ylabels = ['Phase Velocity [m/s]', 'Amplitude', 'Depth [m]', 'Depth [m]']
ylims = [[100,4000], [0.1,10], [150,0], [300,0]]
yscale_types = ['log', 'log', 'linear', 'linear']
yminorticks = [True, True, False, False]
flag_dual_axy = [True, False, True, True]
dual_ax_factory = [3.28084, 3.28084, 3.28084, 3.28084]
ylabels_dual = ['Phase Velocity [ft/s]', 'Amplitude', 'Depth [ft]', 'Depth [ft]']



# END OF INPUTS*****************************************************************


# Create figure and axes........................................................

# Create figure
fig = plt.figure(figsize=(w_fig, h_fig))
fig.show()
ax_position, ax_rc = invpost.subplotLoc(n_row, n_col, xLeft, xRight, xInt, yLow, yUp, yInt, 'lr')

# Create axes
figax = []
for k in range(nplots):
    ax = fig.add_axes( ax_position[ax_rc[k][0]][ax_rc[k][1]] )
    ax.set_xlim(xlims[k])
    ax.set_ylim(ylims[k])
    ax.set_xscale(xscale_types[k])
    ax.set_yscale(yscale_types[k])
    ax.set_xticklabels(ax.get_xticks(), fontsize=fsize, fontname=fname )
    ax.set_yticklabels(ax.get_yticks(), fontsize=fsize, fontname=fname )
    ax.tick_params(which='major', direction=tickdir, length=tickl[0], width=tickw[0], labelsize=fsize)
    ax.tick_params(which='minor', direction=tickdir, length=tickl[1], width=tickw[1])
    if ylims[k][0]>1 or ylims[k][1]>=1000:
        ax.yaxis.set_major_formatter( FormatStrFormatter('%d') )
    if xlims[k][0]>1 or xlims[k][1]>=1000:
        ax.xaxis.set_major_formatter( FormatStrFormatter('%d') )
    ax.set_xlabel(xlabels[k], fontsize=fsize, fontname=fname )
    ax.set_ylabel(ylabels[k], fontsize=fsize, fontname=fname )
    # Dual x-axis
    if flag_dual_axx[k]:
        axx = ax.twiny()
        axx.set_xlim([xlims[k][0]*dual_ax_factorx[k], xlims[k][1]*dual_ax_factorx[k]])
        axx.set_xscale(xscale_types[k])
        axx.set_xticklabels(axx.get_xticks(), fontsize=fsize, fontname=fname )
        axx.tick_params(which='major', direction=tickdir, length=tickl[0], width=tickw[0], labelsize=fsize)
        axx.tick_params(which='minor', direction=tickdir, length=tickl[1], width=tickw[1])
        if xlims[k][0]*dual_ax_factorx[k] > 1 or xlims[k][1]*dual_ax_factorx[k] >= 1000:
            axx.xaxis.set_major_formatter( FormatStrFormatter('%d') )
        axx.set_xlabel(xlabels_dual[k], fontsize=fsize, fontname=fname )
    # Dual y-axis
    if flag_dual_axy[k]:
        axy = ax.twinx()
        axy.set_ylim([ylims[k][0]*dual_ax_factory[k], ylims[k][1]*dual_ax_factory[k]])
        axy.set_yscale(xscale_types[k])
        axy.set_yticklabels(axy.get_yticks(), fontsize=fsize, fontname=fname )
        axy.tick_params(which='major', direction=tickdir, length=tickl[0], width=tickw[0], labelsize=fsize)
        axy.tick_params(which='minor', direction=tickdir, length=tickl[1], width=tickw[1])
        if ylims[k][0]*dual_ax_factory[k] > 1 or ylims[k][1]*dual_ax_factory[k] >= 1000:
            axy.yaxis.set_major_formatter( FormatStrFormatter('%d') )
        axy.set_ylabel(ylabels_dual[k], fontsize=fsize, fontname=fname )
    figax.append(ax)


# Plot dispersion curves along with experimental dispersion data



