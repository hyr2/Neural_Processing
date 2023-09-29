import pandas as pd
import scipy.stats as sstats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import os
import sys
import seaborn as sns
from matplotlib import pyplot as plt

# Statistical tests for all_clusters_analysis_*.py type data

# input_file = os.path.join(r'C:\Rice_2023\Data\Results\09_26_2023_13_08','all_animal_summary.pkl')   # input directory
input_file = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results/09_26_2023_13_08/all_animal_summary.pkl')   # input_directory
df_all_clusters_main = pd.read_pickle(input_file)

# Folder output: NumSpikes*******************************************************
df_all_clusters_main['deltaN'] = df_all_clusters_main['N_stim'] - df_all_clusters_main['N_bsl']
df_tmp = df_all_clusters_main.copy()
df_tmp.reset_index(inplace = True)
i = df_tmp[df_tmp.region == 'NaN'].index         # these units dont belong to any region
df_tmp.drop(i,inplace = True)                    # these units dont belong to any region hence dropped

i = df_tmp[(df_tmp.spont_FR < 1.5)].index         # these units fire too sparsely
df_tmp.drop(i,inplace = True)                    # dropped
df_tmp.reset_index(drop=True,inplace = True)

df_tmp = df_tmp.loc[:,['day','shank','response','region','celltype','N_bsl','N_stim','deltaN']]
df_tmp.reset_index(drop = True,inplace = True)
i = df_tmp[(df_tmp.day == 56) | (df_tmp.day == 49)].index
df_tmp.drop(i,inplace = True)
df_tmp.reset_index(drop=True,inplace = True)
# time buckets
df_tmp.loc[ (df_tmp['day'] >= 21)  & (df_tmp['day'] < 56) , 'day' ] = 'chro'
df_tmp.loc[ (df_tmp['day'] == -3) | (df_tmp['day'] == -2) , 'day' ] = 'bsl'
df_tmp.loc[ df_tmp['day'] == 2  , 'day'] = 'SA'
df_tmp.loc[ (df_tmp['day'] == 7) | (df_tmp['day'] == 14) , 'day' ] = 'rec'

# stat test for pyramidal cells  ---
df_bsl = df_tmp.loc[(df_tmp['day'] == 'bsl') & (df_tmp['celltype'] == 'P') & (df_tmp['region'] == 'L300'),['N_bsl','N_stim','deltaN','shank']]
df_SA = df_tmp.loc[(df_tmp['day'] == 'SA') & (df_tmp['celltype'] == 'P') & (df_tmp['region'] == 'L300'),['N_bsl','N_stim','deltaN','shank']]
df_rec = df_tmp.loc[(df_tmp['day'] == 'rec') & (df_tmp['celltype'] == 'P') & (df_tmp['region'] == 'L300'),['N_bsl','N_stim','deltaN','shank']]
df_chr = df_tmp.loc[(df_tmp['day'] == 'chro') & (df_tmp['celltype'] == 'P') & (df_tmp['region'] == 'L300'),['N_bsl','N_stim','deltaN','shank']]
# baseline ie tonic
sstats.mannwhitneyu(df_bsl['N_bsl'].to_numpy(),df_SA['N_bsl'].to_numpy(),nan_policy ='omit',alternative = 'greater' )   # with sub acute
sstats.mannwhitneyu(df_bsl['N_bsl'].to_numpy(),df_rec['N_bsl'].to_numpy(),nan_policy ='omit',alternative = 'less' )  # with recovery
sstats.mannwhitneyu(df_bsl['N_bsl'].to_numpy(),df_chr['N_bsl'].to_numpy(),nan_policy ='omit',alternative = 'greater' )  # with chronic
# stim ie phasic
sstats.mannwhitneyu(df_bsl['N_stim'].to_numpy(),df_SA['N_stim'].to_numpy(),nan_policy ='omit',alternative = 'greater' )   # with sub acute
sstats.mannwhitneyu(df_bsl['N_stim'].to_numpy(),df_rec['N_stim'].to_numpy(),nan_policy ='omit',alternative = 'greater' )  # with recovery
sstats.mannwhitneyu(df_bsl['N_stim'].to_numpy(),df_chr['N_stim'].to_numpy(),nan_policy ='omit',alternative = 'greater' )  # with chronic
# change in FR
sstats.mannwhitneyu(df_bsl['deltaN'].to_numpy(),df_SA['deltaN'].to_numpy(),nan_policy ='omit',alternative = 'greater' )   # with sub acute
sstats.mannwhitneyu(df_bsl['deltaN'].to_numpy(),df_rec['deltaN'].to_numpy(),nan_policy ='omit',alternative = 'greater' )  # with recovery
sstats.mannwhitneyu(df_bsl['deltaN'].to_numpy(),df_chr['deltaN'].to_numpy(),nan_policy ='omit',alternative = 'greater' )  # with chronic


# stat test for FS/PV cells  ---
df_bsl = df_tmp.loc[(df_tmp['day'] == 'bsl') & (df_tmp['celltype'] == 'NI'),['N_bsl','N_stim']]
df_SA = df_tmp.loc[(df_tmp['day'] == 'SA') & (df_tmp['celltype'] == 'NI'),['N_bsl','N_stim']]
df_rec = df_tmp.loc[(df_tmp['day'] == 'rec') & (df_tmp['celltype'] == 'NI'),['N_bsl','N_stim']]
df_chr = df_tmp.loc[(df_tmp['day'] == 'chro') & (df_tmp['celltype'] == 'NI'),['N_bsl','N_stim']]
# baseline ie tonic
sstats.mannwhitneyu(df_bsl['N_bsl'].to_numpy(),df_SA['N_bsl'].to_numpy(),nan_policy ='omit',alternative = 'greater' )   # with sub acute
sstats.mannwhitneyu(df_bsl['N_bsl'].to_numpy(),df_rec['N_bsl'].to_numpy(),nan_policy ='omit',alternative = 'greater' )  # with recovery
sstats.mannwhitneyu(df_bsl['N_bsl'].to_numpy(),df_chr['N_bsl'].to_numpy(),nan_policy ='omit',alternative = 'greater' )  # with chronic
# stim ie phasic
sstats.mannwhitneyu(df_bsl['N_stim'].to_numpy(),df_SA['N_stim'].to_numpy(),nan_policy ='omit',alternative = 'greater' )   # with sub acute
sstats.mannwhitneyu(df_bsl['N_stim'].to_numpy(),df_rec['N_stim'].to_numpy(),nan_policy ='omit',alternative = 'two-sided' )  # with recovery
sstats.mannwhitneyu(df_bsl['N_stim'].to_numpy(),df_chr['N_stim'].to_numpy(),nan_policy ='omit',alternative = 'two-sided' )  # with chronic


