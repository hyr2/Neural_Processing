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
import os
import sys
import seaborn as sns



# Script to perform statistical testing on the neural data (only for spiking data) [use after running: "combined_FR_TS_pop_versionA"]





# Script to perform statistical testing on the neural data (only for spiking data) [use after running: "combined_FR_TS_pop_versionA"]
def stat_test_spikes(source_dir):
    # Input:
        
    
    # SINGLE CLUSTER LEVEL STATISTICAL TESTING (BY BRAIN REGION) ----------------
    df_act = pd.read_pickle(os.path.join(source_dir,'dataframe_act.pkl'))
    df_S = pd.read_pickle(os.path.join(source_dir,'dataframe_sup.pkl'))
    df_E = pd.read_pickle(os.path.join(source_dir,'dataframe_E.pkl'))
    df_I = pd.read_pickle(os.path.join(source_dir,'dataframe_I.pkl'))
    df_Es = pd.read_pickle(os.path.join(source_dir,'dataframe_Eact.pkl'))
    df_Is = pd.read_pickle(os.path.join(source_dir,'dataframe_Iact.pkl'))
    
    
    # rejecting false positives for activated clusters
    df_act_new = df_act[df_act['delta_S'] > 0]
    df_act_new_DL300 = df_act_new.groupby('region').get_group('DL300')
    df_act_new_DG300 = df_act_new.groupby('region').get_group('DG300')
    df_act_new_S2 = df_act_new.groupby('region').get_group('S2')
    # running tests: 1) mu_DL300_recovery == mu_DL300_chronic == mu_DL300_bsl 2) mu_DG300_recovery == mu_DG300_chronic == mu_DG300_bsl 3) mu_S2_recovery == mu_S2_chronic == mu_S2_bsl
    group0 = df_act_new_DL300[(df_act_new_DL300['day'] == -3) | (df_act_new_DL300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DL300[df_act_new_DL300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DL300[(df_act_new_DL300['day'] == 7) | (df_act_new_DL300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DL300[(df_act_new_DL300['day'] == 21) | (df_act_new_DL300['day'] == 28) | (df_act_new_DL300['day'] == 42) | (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'greater')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'greater')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'greater')
    group0 = df_act_new_DG300[(df_act_new_DG300['day'] == -3) | (df_act_new_DG300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DG300[df_act_new_DG300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DG300[(df_act_new_DG300['day'] == 7) | (df_act_new_DG300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DG300[(df_act_new_DG300['day'] == 21) | (df_act_new_DG300['day'] == 28) | (df_act_new_DG300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'greater')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'less')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'less')
    group0 = df_act_new_S2[(df_act_new_S2['day'] == -3) | (df_act_new_S2['day'] == -2)]                                         # baseline
    group1 = df_act_new_S2[df_act_new_S2['day'] == 2]                                                                           # sub-acute
    group2 = df_act_new_S2[(df_act_new_S2['day'] == 7) | (df_act_new_S2['day'] == 14)]                                          # recovery
    group3 = df_act_new_S2[(df_act_new_S2['day'] == 21) | (df_act_new_S2['day'] == 28) | (df_act_new_S2['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]          # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'less')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'less')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    
    # rejecting false positives for suppressed clusters
    df_S = df_S.dropna()
    df_act_new = df_S[df_S['delta_S'] < 0]
    df_act_new_DL300 = df_act_new.groupby('region').get_group('DL300')
    df_act_new_DG300 = df_act_new.groupby('region').get_group('DG300')
    df_act_new_S2 = df_act_new.groupby('region').get_group('S2')
    # running tests: 1) mu_DL300_recovery == mu_DL300_chronic == mu_DL300_bsl 2) mu_DG300_recovery == mu_DG300_chronic == mu_DG300_bsl 3) mu_S2_recovery == mu_S2_chronic == mu_S2_bsl
    group0 = df_act_new_DL300[(df_act_new_DL300['day'] == -3) | (df_act_new_DL300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DL300[df_act_new_DL300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DL300[(df_act_new_DL300['day'] == 7) | (df_act_new_DL300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DL300[(df_act_new_DL300['day'] == 21) | (df_act_new_DL300['day'] == 28) | (df_act_new_DL300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'less')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    group0 = df_act_new_DG300[(df_act_new_DG300['day'] == -3) | (df_act_new_DG300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DG300[df_act_new_DG300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DG300[(df_act_new_DG300['day'] == 7) | (df_act_new_DG300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DG300[(df_act_new_DG300['day'] == 21) | (df_act_new_DG300['day'] == 28) | (df_act_new_DG300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'less')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'greater')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'greater')
    group0 = df_act_new_S2[(df_act_new_S2['day'] == -3) | (df_act_new_S2['day'] == -2)]                                         # baseline
    group1 = df_act_new_S2[df_act_new_S2['day'] == 2]                                                                           # sub-acute
    group2 = df_act_new_S2[(df_act_new_S2['day'] == 7) | (df_act_new_S2['day'] == 14)]                                          # recovery
    group3 = df_act_new_S2[(df_act_new_S2['day'] == 21) | (df_act_new_S2['day'] == 28) | (df_act_new_S2['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]          # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'greater')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    
    # Excitatory Cells (Mostly Pyramidal)
    df_act_new = df_E.dropna()
    df_act_new_DL300 = df_act_new.groupby('region').get_group('DL300')
    df_act_new_DG300 = df_act_new.groupby('region').get_group('DG300')
    df_act_new_S2 = df_act_new.groupby('region').get_group('S2')
    # running tests: 1) mu_DL300_recovery == mu_DL300_chronic == mu_DL300_bsl 2) mu_DG300_recovery == mu_DG300_chronic == mu_DG300_bsl 3) mu_S2_recovery == mu_S2_chronic == mu_S2_bsl
    group0 = df_act_new_DL300[(df_act_new_DL300['day'] == -3) | (df_act_new_DL300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DL300[df_act_new_DL300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DL300[(df_act_new_DL300['day'] == 7) | (df_act_new_DL300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DL300[(df_act_new_DL300['day'] == 21) | (df_act_new_DL300['day'] == 28) | (df_act_new_DL300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'greater')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'greater')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'greater')
    group0 = df_act_new_DG300[(df_act_new_DG300['day'] == -3) | (df_act_new_DG300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DG300[df_act_new_DG300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DG300[(df_act_new_DG300['day'] == 7) | (df_act_new_DG300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DG300[(df_act_new_DG300['day'] == 21) | (df_act_new_DG300['day'] == 28) | (df_act_new_DG300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    group0 = df_act_new_S2[(df_act_new_S2['day'] == -3) | (df_act_new_S2['day'] == -2)]                                         # baseline
    group1 = df_act_new_S2[df_act_new_S2['day'] == 2]                                                                           # sub-acute
    group2 = df_act_new_S2[(df_act_new_S2['day'] == 7) | (df_act_new_S2['day'] == 14)]                                          # recovery
    group3 = df_act_new_S2[(df_act_new_S2['day'] == 21) | (df_act_new_S2['day'] == 28) | (df_act_new_S2['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]          # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    
    # Excitatory Cells Stimulus Locked (Mostly Pyramidal) [False positives are removed]
    df_Es = df_Es.dropna()
    df_act_new = df_Es[df_Es['delta_S'] > 0]
    df_act_new_DL300 = df_act_new.groupby('region').get_group('DL300')
    df_act_new_DG300 = df_act_new.groupby('region').get_group('DG300')
    df_act_new_S2 = df_act_new.groupby('region').get_group('S2')
    # running tests: 1) mu_DL300_recovery == mu_DL300_chronic == mu_DL300_bsl 2) mu_DG300_recovery == mu_DG300_chronic == mu_DG300_bsl 3) mu_S2_recovery == mu_S2_chronic == mu_S2_bsl
    group0 = df_act_new_DL300[(df_act_new_DL300['day'] == -3) | (df_act_new_DL300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DL300[df_act_new_DL300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DL300[(df_act_new_DL300['day'] == 7) | (df_act_new_DL300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DL300[(df_act_new_DL300['day'] == 21) | (df_act_new_DL300['day'] == 28) | (df_act_new_DL300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    group0 = df_act_new_DG300[(df_act_new_DG300['day'] == -3) | (df_act_new_DG300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DG300[df_act_new_DG300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DG300[(df_act_new_DG300['day'] == 7) | (df_act_new_DG300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DG300[(df_act_new_DG300['day'] == 21) | (df_act_new_DG300['day'] == 28) | (df_act_new_DG300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    group0 = df_act_new_S2[(df_act_new_S2['day'] == -3) | (df_act_new_S2['day'] == -2)]                                         # baseline
    group1 = df_act_new_S2[df_act_new_S2['day'] == 2]                                                                           # sub-acute
    group2 = df_act_new_S2[(df_act_new_S2['day'] == 7) | (df_act_new_S2['day'] == 14)]                                          # recovery
    group3 = df_act_new_S2[(df_act_new_S2['day'] == 21) | (df_act_new_S2['day'] == 28) | (df_act_new_S2['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]          # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    
    # Inhibitory Cells (Mostly FS/PV)
    df_act_new = df_I.dropna()
    df_act_new_DL300 = df_act_new.groupby('region').get_group('DL300')
    df_act_new_DG300 = df_act_new.groupby('region').get_group('DG300')
    df_act_new_S2 = df_act_new.groupby('region').get_group('S2')
    # running tests: 1) mu_DL300_recovery == mu_DL300_chronic == mu_DL300_bsl 2) mu_DG300_recovery == mu_DG300_chronic == mu_DG300_bsl 3) mu_S2_recovery == mu_S2_chronic == mu_S2_bsl
    group0 = df_act_new_DL300[(df_act_new_DL300['day'] == -3) | (df_act_new_DL300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DL300[df_act_new_DL300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DL300[(df_act_new_DL300['day'] == 7) | (df_act_new_DL300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DL300[(df_act_new_DL300['day'] == 21) | (df_act_new_DL300['day'] == 28) | (df_act_new_DL300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    group0 = df_act_new_DG300[(df_act_new_DG300['day'] == -3) | (df_act_new_DG300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DG300[df_act_new_DG300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DG300[(df_act_new_DG300['day'] == 7) | (df_act_new_DG300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DG300[(df_act_new_DG300['day'] == 21) | (df_act_new_DG300['day'] == 28) | (df_act_new_DG300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    group0 = df_act_new_S2[(df_act_new_S2['day'] == -3) | (df_act_new_S2['day'] == -2)]                                         # baseline
    group1 = df_act_new_S2[df_act_new_S2['day'] == 2]                                                                           # sub-acute
    group2 = df_act_new_S2[(df_act_new_S2['day'] == 7) | (df_act_new_S2['day'] == 14)]                                          # recovery
    group3 = df_act_new_S2[(df_act_new_S2['day'] == 21) | (df_act_new_S2['day'] == 28) | (df_act_new_S2['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]          # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    
    # Inhibitory Cells (Mostly FS/PV) Stimulus Locked [False positives are removed]
    df_Is = df_Is.dropna()
    df_act_new = df_Is[df_Is['delta_S'] > 0]
    df_act_new_DL300 = df_act_new.groupby('region').get_group('DL300')
    df_act_new_DG300 = df_act_new.groupby('region').get_group('DG300')
    df_act_new_S2 = df_act_new.groupby('region').get_group('S2')
    # running tests: 1) mu_DL300_recovery == mu_DL300_chronic == mu_DL300_bsl 2) mu_DG300_recovery == mu_DG300_chronic == mu_DG300_bsl 3) mu_S2_recovery == mu_S2_chronic == mu_S2_bsl
    group0 = df_act_new_DL300[(df_act_new_DL300['day'] == -3) | (df_act_new_DL300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DL300[df_act_new_DL300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DL300[(df_act_new_DL300['day'] == 7) | (df_act_new_DL300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DL300[(df_act_new_DL300['day'] == 21) | (df_act_new_DL300['day'] == 28) | (df_act_new_DL300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    group0 = df_act_new_DG300[(df_act_new_DG300['day'] == -3) | (df_act_new_DG300['day'] == -2)]                                     # baseline
    group1 = df_act_new_DG300[df_act_new_DG300['day'] == 2]                                                                          # sub-acute
    group2 = df_act_new_DG300[(df_act_new_DG300['day'] == 7) | (df_act_new_DG300['day'] == 14)]                                      # recovery
    group3 = df_act_new_DG300[(df_act_new_DG300['day'] == 21) | (df_act_new_DG300['day'] == 28) | (df_act_new_DG300['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]   # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    group0 = df_act_new_S2[(df_act_new_S2['day'] == -3) | (df_act_new_S2['day'] == -2)]                                         # baseline
    group1 = df_act_new_S2[df_act_new_S2['day'] == 2]                                                                           # sub-acute
    group2 = df_act_new_S2[(df_act_new_S2['day'] == 7) | (df_act_new_S2['day'] == 14)]                                          # recovery
    group3 = df_act_new_S2[(df_act_new_S2['day'] == 21) | (df_act_new_S2['day'] == 28) | (df_act_new_S2['day'] == 42)| (df_act_new_DL300['day'] == 35) | (df_act_new_DL300['day'] == 49)]          # chronic
    sstats.ranksums(x = group0['delta_S'],y = group1['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group2['delta_S'], alternative = 'two-sided')
    sstats.ranksums(x = group0['delta_S'],y = group3['delta_S'], alternative = 'two-sided')
    
    # Connectivities STATISTICAL TESTING (BY BRAIN REGION) ----------------
    df = pd.read_pickle(os.path.join(source_dir,'dataframe_df.pkl'))
    df_S2 = df.groupby('region').get_group('S2')
    df_L300 = df.groupby('region').get_group('D<300')
    df_G300 = df.groupby('region').get_group('D>300')
    
    sstats.ranksums(x = df_L300[df_L300['day'] == 'Pre']['avg_synch'],y = df_L300[df_L300['day'] == 'SubAcute']['avg_synch'].fillna(0),alternative='greater')
    sstats.ranksums(x = df_L300[df_L300['day'] == 'Pre']['avg_synch'],y = df_L300[df_L300['day'] == 'Recovery']['avg_synch'].fillna(0),alternative='greater')
    sstats.ranksums(x = df_L300[df_L300['day'] == 'Pre']['avg_synch'],y = df_L300[df_L300['day'] == 'Chronic']['avg_synch'].fillna(0),alternative='greater')
    
    sstats.ranksums(x = df_G300[df_G300['day'] == 'Pre']['avg_synch'],y = df_G300[df_G300['day'] == 'SubAcute']['avg_synch'].fillna(0),alternative='greater')
    sstats.ranksums(x = df_G300[df_G300['day'] == 'Pre']['avg_synch'],y = df_G300[df_G300['day'] == 'Recovery']['avg_synch'].fillna(0),alternative = 'less')
    sstats.ranksums(x = df_G300[df_G300['day'] == 'Pre']['avg_synch'],y = df_G300[df_G300['day'] == 'Chronic']['avg_synch'].fillna(0), alternative = 'less')
    
    sstats.ranksums(x = df_S2[df_S2['day'] == 'Pre']['avg_synch'],y = df_S2[df_S2['day'] == 'SubAcute']['avg_synch'].fillna(0))
    sstats.ranksums(x = df_S2[df_S2['day'] == 'Pre']['avg_synch'],y = df_S2[df_S2['day'] == 'Recovery']['avg_synch'].fillna(0))
    sstats.ranksums(x = df_S2[df_S2['day'] == 'Pre']['avg_synch'],y = df_S2[df_S2['day'] == 'Chronic']['avg_synch'].fillna(0))
    
    # monosyn E
    sstats.ranksums(x = df_L300[df_L300['day'] == 'Pre']['num_monosyn_e'],y = df_L300[df_L300['day'] == 'SubAcute']['num_monosyn_e'].fillna(0))
    sstats.ranksums(x = df_L300[df_L300['day'] == 'Pre']['num_monosyn_e'],y = df_L300[df_L300['day'] == 'Recovery']['num_monosyn_e'].fillna(0))
    sstats.ranksums(x = df_L300[df_L300['day'] == 'Pre']['num_monosyn_e'],y = df_L300[df_L300['day'] == 'Chronic']['num_monosyn_e'].fillna(0))
    
    sstats.ranksums(x = df_G300[df_G300['day'] == 'Pre']['num_monosyn_e'],y = df_G300[df_G300['day'] == 'SubAcute']['num_monosyn_e'].fillna(0))
    sstats.ranksums(x = df_G300[df_G300['day'] == 'Pre']['num_monosyn_e'],y = df_G300[df_G300['day'] == 'Recovery']['num_monosyn_e'].fillna(0))
    sstats.ranksums(x = df_G300[df_G300['day'] == 'Pre']['num_monosyn_e'],y = df_G300[df_G300['day'] == 'Chronic']['num_monosyn_e'].fillna(0))
    
    sstats.ranksums(x = df_S2[df_S2['day'] == 'Pre']['num_monosyn_e'],y = df_S2[df_S2['day'] == 'SubAcute']['num_monosyn_e'].fillna(0))
    sstats.ranksums(x = df_S2[df_S2['day'] == 'Pre']['num_monosyn_e'],y = df_S2[df_S2['day'] == 'Recovery']['num_monosyn_e'].fillna(0))
    sstats.ranksums(x = df_S2[df_S2['day'] == 'Pre']['num_monosyn_e'],y = df_S2[df_S2['day'] == 'Chronic']['num_monosyn_e'].fillna(0))
    
    # monosyn I
    sstats.ranksums(x = df_L300[df_L300['day'] == 'Pre']['num_monosyn_i'],y = df_L300[df_L300['day'] == 'SubAcute']['num_monosyn_i'].fillna(0))
    sstats.ranksums(x = df_L300[df_L300['day'] == 'Pre']['num_monosyn_i'],y = df_L300[df_L300['day'] == 'Recovery']['num_monosyn_i'].fillna(0))
    sstats.ranksums(x = df_L300[df_L300['day'] == 'Pre']['num_monosyn_i'],y = df_L300[df_L300['day'] == 'Chronic']['num_monosyn_i'].fillna(0))
    
    sstats.ranksums(x = df_G300[df_G300['day'] == 'Pre']['num_monosyn_i'],y = df_G300[df_G300['day'] == 'SubAcute']['num_monosyn_i'].fillna(0))
    sstats.ranksums(x = df_G300[df_G300['day'] == 'Pre']['num_monosyn_i'],y = df_G300[df_G300['day'] == 'Recovery']['num_monosyn_i'].fillna(0))
    sstats.ranksums(x = df_G300[df_G300['day'] == 'Pre']['num_monosyn_i'],y = df_G300[df_G300['day'] == 'Chronic']['num_monosyn_i'].fillna(0))
    
    sstats.ranksums(x = df_S2[df_S2['day'] == 'Pre']['num_monosyn_i'],y = df_S2[df_S2['day'] == 'SubAcute']['num_monosyn_i'].fillna(0))
    sstats.ranksums(x = df_S2[df_S2['day'] == 'Pre']['num_monosyn_i'],y = df_S2[df_S2['day'] == 'Recovery']['num_monosyn_i'].fillna(0))
    sstats.ranksums(x = df_S2[df_S2['day'] == 'Pre']['num_monosyn_i'],y = df_S2[df_S2['day'] == 'Chronic']['num_monosyn_i'].fillna(0))
    

    # Statistical testing and all population plot
    df_L300 = pd.read_pickle(os.path.join(source_dir,'dataframe_L300.pkl'))
    df_G300 = pd.read_pickle(os.path.join(source_dir,'dataframe_G300.pkl'))
    df_S2 = pd.read_pickle(os.path.join(source_dir,'dataframe_S2.pkl'))
    
    
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



source_dir = input('Results directory: ')
source_dir = str(source_dir)
