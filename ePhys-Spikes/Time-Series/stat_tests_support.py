#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:07:36 2023

@author: hyr2-office
"""
import pandas as pd
import scipy.stats as sstats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np








# Script to perform statistical testing on the neural data (only for spiking data)
def stat_test_spikes(filename_dfL300,filename_dfG300,filename_S2,filename_dfSummary):
    # Input:
        # filename_dfL300 : h5 file containing the dataframe for the peri-infarct region
        # filename_dfG300 : h5 file containing the dataframe for the spared barrel cortex
        # filename_S2 : h5 file containing the dataframe for the secondary somatosensory cortex
        # filename_dfSummary : h5 file containing the dataframe for the summary of the three regions
    
    
    df_less300_full = pd.read_hdf(filename_dfL300)
    df_great300_full = pd.read_hdf(filename_dfL300)
    df_S2_full = pd.read_hdf(filename_dfL300)
    df_all_summary = pd.read_hdf(filename_dfL300)
    df_nonless300 = pd.concat([df_great300_full,df_S2_full],axis = 0)
    
    # Statistical tests on the I/E ratio
    
    
    
    
    
    # # More statistical tests
    # stat_test = np.zeros([3,2])
    # group1_non300 = df_nonless300[(df_nonless300['day'] == 28) | (df_nonless300['day'] == 35) | (df_nonless300['day'] == 42)]       # recovery
    # group1_300 = df_less300_full[(df_less300_full['day'] == 28) | (df_less300_full['day'] == 35) | (df_less300_full['day'] == 42)]     
    # group2_non300 = df_nonless300[df_nonless300['day'] == 2]  # sub acute
    # group2_300 = df_less300_full[df_less300_full['day'] == 2]          
    # group3_non300 = df_nonless300[(df_nonless300['day'] == 7) | (df_nonless300['day'] == 14)]   # post sub acute
    # group3_300 = df_less300_full[(df_less300_full['day'] == 7) | (df_less300_full['day'] == 14)]       
    # stat_test[0,0] = sstats.ranksums(group1_300['inhibitory'],group1_non300['inhibitory'],alternative = 'greater')[1]      # recovery
    # stat_test[1,0]= sstats.ranksums(group2_300['inhibitory'],group2_non300['inhibitory'],alternative = 'less')[1]      # sub acute
    # stat_test[2,0] = sstats.ranksums(group3_300['inhibitory'],group3_non300['inhibitory'],alternative = 'two-sided')[1]     # post sub acute        
    # stat_test[0,1] = sstats.ranksums(group1_300['excitatory'],group1_non300['excitatory'],alternative = 'two-sided')[1]      # recovery
    # stat_test[1,1] = sstats.ranksums(group2_300['excitatory'],group2_non300['excitatory'],alternative = 'less')[1]      # sub acute
    # stat_test[2,1] = sstats.ranksums(group3_300['excitatory'],group3_non300['excitatory'],alternative = 'less')[1]            # post sub acute
    # # filename_stat = os.path.join(output_folder,'stats.xlsx')
    # df_stat_test = pd.DataFrame(stat_test, columns = ['inhibitory','excitatory'], index = ['Day28-42', 'Day2','Day7-14'])

    # ## Performing tests (Welch and T-test and Z-Tests) ---------------------------

    # # Between neuron types/response types/cell classifications
    # # days_for_testing = np.array([2,7,14,21,42])
    # stat_test = np.zeros([5,4])
    # num = 0
    # iter_local = np.array([7,14])
    # # for (num,iter_local) in enumerate(days_for_testing):
    # group_data = df_less300_full[(df_less300_full['day'] == 7) | (df_less300_full['day'] == 14)] # day 7,14 (sub Ac)
    # stat_test[1,0] = sstats.wilcoxon(group_data['activated'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]            # **
    # stat_test[1,1] = sstats.wilcoxon(group_data['activated'],group_data['suppressed'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[1,2] = sstats.wilcoxon(group_data['suppressed'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[1,3] = sstats.wilcoxon(group_data['excitatory'],group_data['inhibitory'],alternative = 'less',zero_method = 'wilcox')[1]  
    # # stat_test[1,3] = sstats.wilcoxon(group_data['excitatory'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # group_data = df_less300_full[df_less300_full['day'] == 2]       # day 2 
    # stat_test[0,0] = sstats.wilcoxon(group_data['activated'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[0,1] = sstats.wilcoxon(group_data['activated'],group_data['suppressed'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[0,2] = sstats.wilcoxon(group_data['suppressed'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[0,3] = sstats.wilcoxon(group_data['excitatory'],group_data['inhibitory'],alternative = 'less',zero_method = 'wilcox')[1]    # **
    # # stat_test[0,3] = sstats.wilcoxon(group_data['excitatory'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # group_data = df_less300_full[(df_less300_full['day'] == 21) | (df_less300_full['day'] == 28)]           # days 21 and 28 (recovery)
    # stat_test[2,0] = sstats.wilcoxon(group_data['activated'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[2,1] = sstats.wilcoxon(group_data['activated'],group_data['suppressed'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[2,2] = sstats.wilcoxon(group_data['suppressed'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[2,3] = sstats.wilcoxon(group_data['excitatory'],group_data['inhibitory'],alternative = 'less',zero_method = 'wilcox')[1]    # **
    # # stat_test[2,3] = sstats.wilcoxon(group_data['excitatory'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # group_data = df_less300_full[(df_less300_full['day'] == 7) | (df_less300_full['day'] == 14) | (df_less300_full['day'] == 21)]           # days 7 to 21 (recovery)
    # stat_test[3,0] = sstats.wilcoxon(group_data['activated'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[3,1] = sstats.wilcoxon(group_data['activated'],group_data['suppressed'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[3,2] = sstats.wilcoxon(group_data['suppressed'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[3,3] = sstats.wilcoxon(group_data['excitatory'],group_data['inhibitory'],alternative = 'less',zero_method = 'wilcox')[1]    # **
    # # stat_test[3,3] = sstats.wilcoxon(group_data['excitatory'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # group_data = df_less300_full[(df_less300_full['day'] == 28) | (df_less300_full['day'] == 35) | (df_less300_full['day'] == 42)]           # days 28 to 42 (recovery)
    # stat_test[4,0] = sstats.wilcoxon(group_data['activated'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[4,1] = sstats.wilcoxon(group_data['activated'],group_data['suppressed'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[4,2] = sstats.wilcoxon(group_data['suppressed'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
    # stat_test[4,3] = sstats.wilcoxon(group_data['excitatory'],group_data['inhibitory'],alternative = 'less',zero_method = 'wilcox')[1]    # **
    # stat_test[4,3] = sstats.wilcoxon(group_data['excitatory'],group_data['nor'],alternative = 'less',zero_method = 'wilcox')[1]  
        
    # filename_stat = os.path.join(output_folder,'stats.xlsx')
    # df_stat_test = pd.DataFrame(stat_test, columns = ['act-NoR','act-sup','sup-NoR','excit-inhib'], index = ['Day2', 'Day7-14','Day21-28','Day7-21','Day28-42'])
    # # append_df_to_excel(filename_stat, df_stat_test)

    # # Between Sessions (using the box and whisker plots as the references for the statistical tests) --------------------------
    # stat_test_s = np.zeros([5,4])
    # stat_test_s[0,0] = sstats.f_oneway(df[df['day'] == 'Pre']['activated'],df[df['day'] == 'SubAcute']['activated'],df[df['day'] == 'PostSubAcute']['activated'],df[df['day'] == 'Recovery']['activated'])[1]
    # stat_test_s[1,0] = sstats.f_oneway(df[df['day'] == 'Pre']['suppressed'],df[df['day'] == 'SubAcute']['suppressed'],df[df['day'] == 'PostSubAcute']['suppressed'],df[df['day'] == 'Recovery']['suppressed'])[1]
    # stat_test_s[2,0] = sstats.f_oneway(df[df['day'] == 'Pre']['nor'],df[df['day'] == 'SubAcute']['nor'],df[df['day'] == 'PostSubAcute']['nor'],df[df['day'] == 'Recovery']['nor'])[1]
    # stat_test_s[3,0] = sstats.f_oneway(df[df['day'] == 'Pre']['excitatory'],df[df['day'] == 'SubAcute']['excitatory'],df[df['day'] == 'PostSubAcute']['excitatory'],df[df['day'] == 'Recovery']['excitatory'])[1]
    # stat_test_s[4,0] = sstats.f_oneway(df[df['day'] == 'Pre']['inhibitory'],df[df['day'] == 'SubAcute']['inhibitory'],df[df['day'] == 'PostSubAcute']['inhibitory'],df[df['day'] == 'Recovery']['inhibitory'])[1]

    # tukey = pairwise_tukeyhsd(endog=df['activated'],groups=df['day'],alpha=0.05)
    # # append_df_to_excel(filename_stat, tukey, sheet_name = 'session')
    # tukey = pairwise_tukeyhsd(endog=df['suppressed'],groups=df['day'],alpha=0.05)
    # # append_df_to_excel(filename_stat, tukey, sheet_name = 'session')
    # tukey = pairwise_tukeyhsd(endog=df['nor'],groups=df['day'],alpha=0.05)
    # # append_df_to_excel(filename_stat, tukey, sheet_name = 'session')
    # tukey = pairwise_tukeyhsd(endog=df['excitatory'],groups=df['day'],alpha=0.05)
    # # append_df_to_excel(filename_stat, tukey, sheet_name = 'session')
    # tukey = pairwise_tukeyhsd(endog=df['inhibitory'],groups=df['day'],alpha=0.05)
    # append_df_to_excel(filename_stat, tukey, sheet_name = 'session')
    # group_suppressed = df_less300_full[df_less300_full['day'] == iter_local]['suppressed'] 
    # group_excitatory= df_less300_full[df_less300_full['day'] == iter_local]['excitatory'] 
    # group_inhibitory= df_less300_full[df_less300_full['day'] == iter_local]['inhibitory'] 
    # group_nor= df_less300_full[df_less300_full['day'] == iter_local]['nor'] 
    # print(group_data)
    # stat_test[num,1,:] = sstats.ttest_ind(group_data['activated'], group_data['suppressed'],equal_var=False)
    # stat_test[num,0,:] = sstats.ttest_ind(group_data['excitatory'], group_data['inhibitory'],equal_var=False)
    # stat_test[num,2,:] = sstats.ttest_ind(group_data['nor'], group_data['activated'],equal_var=False)
    # stat_test[num,3,:] = sstats.ttest_ind(group_data['nor'], group_data['inhibitory'],equal_var=False)
    # print(stat_test[num,:,:])
    
    print('Statistical Tests Done')
    print('Results saved')
    