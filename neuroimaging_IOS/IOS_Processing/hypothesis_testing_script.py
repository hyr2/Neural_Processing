# -*- coding: utf-8 -*-
"""
Created on Thu May 19 21:41:03 2022

@author: Haad-Rathore
"""

# This code will take all the single trial data of rICT and do the statistical significance test 
# during stimulation vs pre-stimulation

from scipy.io import loadmat, savemat
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
import os, sys
import seaborn as sns
import pandas as pd

def read_stimtxt(matlabTXT):
    # Read .txt file
    Txt_File = open(matlabTXT,"r")
    str1 = Txt_File.readlines()
    stim_start_time = float(str1[3])-0.05        # Stimulation start time
    stim_num = int(str1[6])                 # Number of stimulations
    seq_period = float(str1[4])             # Period of each sequence consisting of 10 to 15 frames
    num_trials = int(str1[7])               # total number of trials
    # FramePerSeq = int(str1[2])              # frames per sequence. Important info to read the data.timing file
    # total_seq = num_trials * len_trials
    # len_trials_arr = list(range(1,total_seq,len_trials))
    n_seq_per_trial = int(str1[1])
    Txt_File.close()
    return n_seq_per_trial, num_trials, stim_start_time, stim_num, seq_period

files_list = os.listdir(os.getcwd())
whisker_stim_txt_dir = files_list[files_list.index('extras')]
n_seq_per_trial, num_trials, stim_start_time, stim_duration, seq_period = read_stimtxt(os.path.join(whisker_stim_txt_dir,'whisker_stim.txt'))

# Creating a dictionary in the for loop
data_raw = {}
iter_local = 0
for file_name in files_list:
    if file_name[0] == file_name[2]:
        local_filepath = os.path.join(os.getcwd(),file_name,'ROI.mat')
        data_raw[iter_local] = loadmat(local_filepath)
        iter_local += 1
        
# Only considering Artery data (ie ROI1)    
artery_rICT = np.empty([num_trials,n_seq_per_trial])
for iter_local in range(len(data_raw)):
    artery_rICT[iter_local,:] = data_raw[iter_local]['rICT'][:,0]

# Extracting pre stim and stim data points and concatenating the data
pre_stim_indx = np.arange(int(stim_start_time/seq_period)-50,int(stim_start_time/seq_period),1)
stim_indx = np.arange(int(stim_start_time/seq_period),int(stim_start_time/seq_period)+50)
bucket_pre_stim_all_trials = np.array([])
bucket_stim_all_trials = np.array([])
for iter_local in range(3):
    bucket_pre_stim_all_trials = np.concatenate((bucket_pre_stim_all_trials,artery_rICT[iter_local,pre_stim_indx]))
    bucket_stim_all_trials = np.concatenate((bucket_stim_all_trials,artery_rICT[iter_local,stim_indx]))
    
# Fudge here
bucket_stim_all_trials_orig = np.copy(bucket_stim_all_trials)
bucket_stim_all_trials[0:50] = bucket_stim_all_trials[0:50]

# Create dataframe
stim_bin = np.ones([len(bucket_stim_all_trials)])
pre_stim_bin = np.zeros([len(bucket_pre_stim_all_trials)])
data_list =  {'rICT': np.append(bucket_pre_stim_all_trials,bucket_stim_all_trials) ,
        'stimulation':np.append(pre_stim_bin,stim_bin)}
df = pd.DataFrame(data_list)

# Performing the pairwise t-test
pre_stim_p_val = stats.shapiro(bucket_pre_stim_all_trials)      # testing for gaussian nature of the distribution
stim_p_val = stats.shapiro(bucket_stim_all_trials)              # testing for gaussian nature of the distribution
t_test = stats.ttest_rel(bucket_stim_all_trials_orig,bucket_pre_stim_all_trials)  # t_test for pair
Wilcoxon_signed_rank = stats.wilcoxon(bucket_stim_all_trials_orig,bucket_pre_stim_all_trials, alternative = 'greater') #

# Plotting of box and whisker 
sns.set(font_scale=2)
fig, axes = plt.subplots(1,2, figsize=(14,10), dpi=100)
axes = axes.flatten()
ax = sns.boxplot(x='stimulation', y='rICT', data = df, linewidth=2, showmeans=True, meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},ax=axes[0])
#* tick params

axes[0].set_xticklabels(['Baseline','Stimulation'], rotation=0)
axes[0].set(xlabel=None)
axes[0].set_ylabel('rICT')
axes[0].grid(alpha=0.5)
axes[0].legend(loc = 'lower right')
 
#*set edge color = black
for b in range(len(ax.artists)):
    ax.artists[b].set_edgecolor('black')
    ax.artists[b].set_alpha(0.8)
   
#* statistical tests
x1, x2 = -1,1
y, h, col = -1 , 1, 'k'
axes[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
axes[1].text(0,-0.05, 'Baseline Shapiro-Wilk test for normality: ' + str(pre_stim_p_val[1]), ha='center', va='bottom', color= col, fontsize = 10)
axes[1].text(0,-0.2, 'Stimulation Shapiro-Wilk test for normality: ' + str(stim_p_val[1]), ha='center', va='bottom', color= col,fontsize = 10)
if pre_stim_p_val[1] < 0.05 or stim_p_val[1] < 0.05:
    axes[1].text(0,-0.4, 'Wilcoxon signed-rank test: ' + str(Wilcoxon_signed_rank[1]), ha='center', va='bottom', color= col,fontsize = 10)
else:
    axes[1].text(0,-0.4, ' T-test:' + str(t_test[1]), ha='center', va='bottom', color= col, fontsize = 10)

filename_save = os.path.join(os.getcwd(),'box_whisker_stats.svg')
fig.savefig(filename_save,format = 'svg')
filename_save = os.path.join(os.getcwd(),'box_whisker_stats.png')
fig.savefig(filename_save,format = 'png')