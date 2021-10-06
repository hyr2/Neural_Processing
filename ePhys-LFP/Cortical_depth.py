# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.io import loadmat
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt
import os

def nan_helper(y):
	"""Helper to handle indices and logical indices of NaNs.

	Input:
		- y, 1d numpy array with possible NaNs
	Output:
		- nans, logical indices of NaNs
		- index, a function, with signature indices= index(logical_indices),
		  to convert logical indices of NaNs to 'equivalent' indices
	Example:
		>>> # linear interpolation of NaNs
		>>> nans, x= nan_helper(y)
		>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
	"""

	return np.isnan(y), lambda z: z.nonzero()[0]

def interpol_4shank(depth_x,depth_mat_in):
	## Performing interpolation on MUA-mean
	xa = depth_x
	xb = depth_x
	xc = depth_x
	xd = depth_x
	
	# splitting arrays
	a,b,c,d = np.split(depth_mat_in,4,axis = 1)
	a = np.reshape(a,(len(a),))
	b = np.reshape(b,(len(b),))
	c = np.reshape(c,(len(c),))
	d = np.reshape(d,(len(d),))
	
	# Removing nans
	xa = xa[~np.isnan(a)]
	xb = xb[~np.isnan(b)]
	xc = xc[~np.isnan(c)]
	xd = xd[~np.isnan(d)]
	a = a[~np.isnan(a)]
	b = b[~np.isnan(b)]
	c = c[~np.isnan(c)]
	d = d[~np.isnan(d)]
	
	za = interp1d(xa, a,kind = 'quadratic',copy = True, fill_value = 'extrapolate')
	zb = interp1d(xb, b,kind = 'quadratic',copy = True,fill_value = 'extrapolate')
	zc = interp1d(xc, c,kind = 'quadratic',copy = True,fill_value = 'extrapolate')
	zd = interp1d(xd, d,kind = 'quadratic',copy = True, fill_value = 'extrapolate')
	
	x_new = np.linspace(25,800,32)
	
	za_new = za(x_new)
	zb_new = zb(x_new)
	zc_new = zc(x_new)
	zd_new = zd(x_new)
#    zd_new = np.zeros((zc_new.shape))
	depth_mat = np.vstack((za_new,zb_new,zc_new,zd_new))
	depth_mat = np.transpose(depth_mat)
	return depth_mat

# Files and folders
source_dir = input('Enter the source directory: \n')
output_dir_cortical_depth = os.path.join(source_dir,'Processed','Cortical-Depth')
dir_data_mat = os.path.join(source_dir,'Processed','Spectrogram_mat')
dir_expsummary = os.path.join(source_dir,'exp_summary.xlsx')
file_cortical_depth = os.path.join(source_dir,'Processed','Cortical-Depth','Spectrogram-py-data.mat')

# Loading Extra data (experimental and processing parameters)
df_exp_summary = pd.read_excel(dir_expsummary)
arr_exp_summary = df_exp_summary.to_numpy()
stim_start_time = arr_exp_summary[4,1]  # Stimulation start
dir_data_mat = os.path.join(dir_data_mat,os.listdir(dir_data_mat)[0])
df_temp = loadmat(dir_data_mat)
delta_t = df_temp['time_MUA']


# Loading Data
df = loadmat(file_cortical_depth)
MUA_depth_mean = df['MUA_depth_mean']
MUA_depth_peak = df['MUA_depth_peak']
MUA_depth_mean_post = df['MUA_depth_mean_post']
LFP_depth_mean = df['LFP_depth_mean']
LFP_depth_peak = df['LFP_depth_peak']
LFP_depth_mean_post = df['LFP_depth_mean_post']
depth_shank = df['depth_shank']
depth_shank = np.reshape(depth_shank,(depth_shank.size,))

# Data categorization
MUA_depth_peak = MUA_depth_peak[:,:,0]
MUA_depth_peak_time = MUA_depth_peak[:,:,1]
LFPHigh_depth_mean = LFP_depth_mean[:,:,3]
LFPHigh_depth_peak = LFP_depth_peak[:,:,3]
LFPHigh_depth_mean_post = LFP_depth_mean_post[:,:,3]
Gamma_depth_mean = LFP_depth_mean[:,:,2]
Gamma_depth_peak = LFP_depth_peak[:,:,2]
Gamma_depth_mean_post = LFP_depth_mean_post[:,:,2]
Beta_depth_mean = LFP_depth_mean[:,:,1]
Beta_depth_peak = LFP_depth_peak[:,:,1]
Beta_depth_mean_post = LFP_depth_mean[:,:,1]
Alpha_depth_mean = LFP_depth_mean[:,:,0]
Alpha_depth_peak = LFP_depth_peak[:,:,0]
Alpha_depth_mean_post = LFP_depth_mean[:,:,0]

# Interpolation
# MUA
MUA_depth_mean_interp = interpol_4shank(depth_shank,MUA_depth_mean)
MUA_depth_peak_interp = interpol_4shank(depth_shank,MUA_depth_peak) 
MUA_depth_mean_post_interp = interpol_4shank(depth_shank,MUA_depth_mean_post) 
# LFP-High
LFPHigh_depth_mean_interp = interpol_4shank(depth_shank,LFPHigh_depth_mean)
LFPHigh_depth_peak_interp = interpol_4shank(depth_shank,LFPHigh_depth_peak)
LFPHigh_depth_mean_post_interp = interpol_4shank(depth_shank,LFPHigh_depth_mean_post) 
# Gamma
Gamma_depth_mean_interp = interpol_4shank(depth_shank,Gamma_depth_mean)
Gamma_depth_peak_interp = interpol_4shank(depth_shank,Gamma_depth_peak)
Gamma_depth_mean_post_interp = interpol_4shank(depth_shank,Gamma_depth_mean_post) 
# Beta
Beta_depth_mean_interp = interpol_4shank(depth_shank,Beta_depth_mean)
Beta_depth_peak_interp = interpol_4shank(depth_shank,Beta_depth_peak)
Beta_depth_mean_post_interp = interpol_4shank(depth_shank,Beta_depth_mean_post) 
# Alpha
Alpha_depth_mean_interp = interpol_4shank(depth_shank,Alpha_depth_mean)
Alpha_depth_peak_interp = interpol_4shank(depth_shank,Alpha_depth_peak)
Alpha_depth_mean_post_interp = interpol_4shank(depth_shank,Alpha_depth_mean_post) 

# Avg during stimulation
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1,ncols = 5,sharex=True, sharey=True)
fig.suptitle('Avg normalized change in baseline during stimulation', fontsize=14)
max_lim = 1.0
min_lim = -0.1
# max_lim = np.nanmean(MUA_depth_mean) + 2*np.nanstd(MUA_depth_mean)
# min_lim = np.nanmean(MUA_depth_mean) - 1*np.nanstd(MUA_depth_mean)
im1 = ax1.imshow(MUA_depth_mean, vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 2.0
min_lim = -0.2
# max_lim = np.nanmean(LFPHigh_depth_mean) + 2*np.nanstd(LFPHigh_depth_mean)
# min_lim = np.nanmean(LFPHigh_depth_mean) - 1*np.nanstd(LFPHigh_depth_mean)
im2 = ax2.imshow(LFPHigh_depth_mean,vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 6.5
min_lim = 0.1
# max_lim = np.nanmean(Gamma_depth_mean) + 2*np.nanstd(Gamma_depth_mean)
# min_lim = np.nanmean(Gamma_depth_mean) - 1*np.nanstd(Gamma_depth_mean)
im3 = ax3.imshow(Gamma_depth_mean,vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 7.0
min_lim = -0.1
# max_lim = np.nanmean(Beta_depth_mean) + 2*np.nanstd(Beta_depth_mean)
# min_lim = np.nanmean(Beta_depth_mean) - 1*np.nanstd(Beta_depth_mean)
im4 = ax4.imshow(Beta_depth_mean,vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 9.0
min_lim = 0.1
# max_lim = np.nanmean(Alpha_depth_mean) + 2*np.nanstd(Alpha_depth_mean)
# min_lim = np.nanmean(Alpha_depth_mean) - 1*np.nanstd(Alpha_depth_mean)
im5 = ax5.imshow(Alpha_depth_mean,vmax = max_lim, vmin = min_lim, cmap = 'jet')
plt.setp(ax1, xticks = [0,1,2,3], xticklabels=['A','B','C','D'])
plt.setp(ax1, yticks = [0,15,31], yticklabels = [0,400,800])
#plt.setp(ax2,yticks = [0,15,31], yticklabels = [250,650,1050])
#plt.setp(ax3,yticks = [0,15,31], yticklabels = [500,900,1300])
#plt.setp(ax4,yticks = [0,15,31], yticklabels = [500,900,1300])
ax1.title.set_text('MUA')
ax2.title.set_text('LFP-high')
ax3.title.set_text(r'$\gamma$')
ax4.title.set_text(r'$\beta$')
ax5.title.set_text(r'$\alpha$')
ax1.set_xlabel('Shank #')
ax1.set_ylabel('Cortical depth (um)')
fig.set_size_inches((10, 6), forward=False)
cbar = fig.colorbar(im5, ax = ax5,label =  'Avg ' + r'$\Delta$'+r'$P_n$')
cbar.set_ticks([])
filename = 'Avg-duringStim' + '.png'
filename = os.path.join(output_dir_cortical_depth,filename)
plt.savefig(filename,format = 'png')
plt.clf()
plt.cla()

# Avg post stimulation
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1,ncols = 5,sharex=True, sharey=True)
fig.suptitle('Avg normalized change in baseline 160 ms post stimulation', fontsize=14)
max_lim = 1.0
min_lim = -0.1
# max_lim = np.nanmean(MUA_depth_mean) + 2*np.nanstd(MUA_depth_mean)
# min_lim = np.nanmean(MUA_depth_mean) - 1*np.nanstd(MUA_depth_mean)
im1 = ax1.imshow(MUA_depth_mean_post, vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 1.0
min_lim = -0.2
# max_lim = np.nanmean(LFPHigh_depth_mean) + 2*np.nanstd(LFPHigh_depth_mean)
# min_lim = np.nanmean(LFPHigh_depth_mean) - 1*np.nanstd(LFPHigh_depth_mean)
im2 = ax2.imshow(LFPHigh_depth_mean_post,vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 4.0
min_lim = -0.1
# max_lim = np.nanmean(Gamma_depth_mean) + 2*np.nanstd(Gamma_depth_mean)
# min_lim = np.nanmean(Gamma_depth_mean) - 1*np.nanstd(Gamma_depth_mean)
im3 = ax3.imshow(Gamma_depth_mean_post,vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 8.0
min_lim = -0.1
# max_lim = np.nanmean(Beta_depth_mean) + 2*np.nanstd(Beta_depth_mean)
# min_lim = np.nanmean(Beta_depth_mean) - 1*np.nanstd(Beta_depth_mean)
im4 = ax4.imshow(Beta_depth_mean_post,vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 10.0
min_lim = 0.1
# max_lim = np.nanmean(Alpha_depth_mean) + 2*np.nanstd(Alpha_depth_mean)
# min_lim = np.nanmean(Alpha_depth_mean) - 1*np.nanstd(Alpha_depth_mean)
im5 = ax5.imshow(Alpha_depth_mean_post,vmax = max_lim, vmin = min_lim, cmap = 'jet')
plt.setp(ax1, xticks = [0,1,2,3], xticklabels=['A','B','C','D'])
plt.setp(ax1, yticks = [0,15,31], yticklabels = [0,400,800])
#plt.setp(ax2,yticks = [0,15,31], yticklabels = [250,650,1050])
#plt.setp(ax3,yticks = [0,15,31], yticklabels = [500,900,1300])
#plt.setp(ax4,yticks = [0,15,31], yticklabels = [500,900,1300])
ax1.title.set_text('MUA')
ax2.title.set_text('LFP-high')
ax3.title.set_text(r'$\gamma$')
ax4.title.set_text(r'$\beta$')
ax5.title.set_text(r'$\alpha$')
ax1.set_xlabel('Shank #')
ax1.set_ylabel('Cortical depth (um)')
fig.set_size_inches((10, 6), forward=False)
cbar = fig.colorbar(im5, ax = ax5,label =  'Avg ' + r'$\Delta$'+r'$P_n$')
cbar.set_ticks([])
filename = 'Avg-postStim' + '.png'
filename = os.path.join(output_dir_cortical_depth,filename)
plt.savefig(filename,format = 'png')
plt.clf()
plt.cla()

# Peak during stimulation
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1,ncols = 5,sharex=True, sharey=True)
fig.suptitle('Peak normalized change in baseline during stimulation', fontsize=14)
max_lim = 4.5
min_lim = 0.3
# max_lim = np.nanmean(MUA_depth_mean) + 2*np.nanstd(MUA_depth_mean)
# min_lim = np.nanmean(MUA_depth_mean) - 1*np.nanstd(MUA_depth_mean)
im1 = ax1.imshow(MUA_depth_peak, vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 12.0
min_lim = 1.5
# max_lim = np.nanmean(LFPHigh_depth_mean) + 2*np.nanstd(LFPHigh_depth_mean)
# min_lim = np.nanmean(LFPHigh_depth_mean) - 1*np.nanstd(LFPHigh_depth_mean)
im2 = ax2.imshow(LFPHigh_depth_peak,vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 18.0
min_lim = 2
# max_lim = np.nanmean(Gamma_depth_mean) + 2*np.nanstd(Gamma_depth_mean)
# min_lim = np.nanmean(Gamma_depth_mean) - 1*np.nanstd(Gamma_depth_mean)
im3 = ax3.imshow(Gamma_depth_peak,vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 18.0
min_lim = 2
# max_lim = np.nanmean(Beta_depth_mean) + 2*np.nanstd(Beta_depth_mean)
# min_lim = np.nanmean(Beta_depth_mean) - 1*np.nanstd(Beta_depth_mean)
im4 = ax4.imshow(Beta_depth_peak,vmax = max_lim, vmin = min_lim, cmap = 'jet')
max_lim = 15.0
min_lim = 1.5
# max_lim = np.nanmean(Alpha_depth_mean) + 2*np.nanstd(Alpha_depth_mean)
# min_lim = np.nanmean(Alpha_depth_mean) - 1*np.nanstd(Alpha_depth_mean)
im5 = ax5.imshow(Alpha_depth_peak,vmax = max_lim, vmin = min_lim, cmap = 'jet')
plt.setp(ax1, xticks = [0,1,2,3], xticklabels=['A','B','C','D'])
plt.setp(ax1, yticks = [0,15,31], yticklabels = [0,400,800])
#plt.setp(ax2,yticks = [0,15,31], yticklabels = [250,650,1050])
#plt.setp(ax3,yticks = [0,15,31], yticklabels = [500,900,1300])
#plt.setp(ax4,yticks = [0,15,31], yticklabels = [500,900,1300])
ax1.title.set_text('MUA')
ax2.title.set_text('LFP-high')
ax3.title.set_text(r'$\gamma$')
ax4.title.set_text(r'$\beta$')
ax5.title.set_text(r'$\alpha$')
ax1.set_xlabel('Shank #')
ax1.set_ylabel('Cortical depth (um)')
fig.set_size_inches((10, 6), forward=False)
cbar = fig.colorbar(im5, ax = ax5,label =  'Avg ' + r'$\Delta$'+r'$P_n$')
cbar.set_ticks([])
filename = 'pk-duringStim' + '.png'
filename = os.path.join(output_dir_cortical_depth,filename)
plt.savefig(filename,format = 'png')
plt.clf()
plt.cla()

# Peak time during stimulation
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols = 2,sharex=True, sharey=True)
fig.suptitle('Time to peak during stimulation', fontsize=14)
max_lim = 50
min_lim = 1
im1 = ax1.imshow(MUA_depth_peak_time, cmap = 'jet_r')
MUA_depth_peak_time = MUA_depth_peak_time * delta_t + stim_start_time
min_lim = stim_start_time
max_lim = stim_start_time + 1
im2 = ax2.imshow(MUA_depth_peak_time,vmax = max_lim, vmin = min_lim, cmap = 'jet_r')
plt.setp(ax1, xticks = [0,1,2,3], xticklabels=['A','B','C','D'])
plt.setp(ax1, yticks = [0,15,31], yticklabels = [0,400,800])
#plt.setp(ax2,yticks = [0,15,31], yticklabels = [250,650,1050])
#plt.setp(ax3,yticks = [0,15,31], yticklabels = [500,900,1300])
#plt.setp(ax4,yticks = [0,15,31], yticklabels = [500,900,1300])
ax1.title.set_text('MUA')
ax2.title.set_text('MUA')
ax1.set_xlabel('Shank #')
ax1.set_ylabel('Cortical depth (um)')
fig.set_size_inches((10, 6), forward=False)
cbar = fig.colorbar(im2, ax = ax2,label =  'Peak Time (s)')
filename = 'pkTime-duringStim' + '.png'
filename = os.path.join(output_dir_cortical_depth,filename)
plt.savefig(filename,format = 'png')
plt.clf()
plt.cla()

