import pandas as pd
import scipy.stats as sstats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import os
import sys
import seaborn as sns
from matplotlib import pyplot as plt

# Statistical tests for all_clusters_analysis_*.py type data

input_file = os.path.join(r'C:\Rice_2023\Data\Results\09_26_2023_13_08','all_animal_summary.pkl')   # input directory
df_all_clusters_main = pd.read_pickle(input_file)

# Folder output: NumSpikes*******************************************************
df_tmp = df_all_clusters_main.copy()
df_tmp.reset_index(inplace = True)
i = df_tmp[df_tmp.region == 'NaN'].index         # these units dont belong to any region
df_tmp.drop(i,inplace = True)                    # these units dont belong to any region hence dropped

i = df_tmp[(df_tmp.spont_FR < 0.75) ].index         # these units fire too sparsely
df_tmp.drop(i,inplace = True)                    # dropped
df_tmp.reset_index(drop=True,inplace = True)

df_tmp = df_tmp.loc[:,['day','celltype','N_bsl','N_stim']]
df_tmp.reset_index(drop = True,inplace = True)
i = df_tmp[df_tmp.day == 56].index
df_tmp.drop(i,inplace = True)
df_tmp.reset_index(drop=True,inplace = True)
# time buckets
df_tmp.loc[ (df_tmp['day'] >= 21)  & (df_tmp['day'] < 56) , 'day' ] = 'chro'
df_tmp.loc[ (df_tmp['day'] == -3) | (df_tmp['day'] == -2) , 'day' ] = 'bsl'
df_tmp.loc[ df_tmp['day'] == 2  , 'day'] = 'SA'
df_tmp.loc[ (df_tmp['day'] == 7) | (df_tmp['day'] == 14) , 'day' ] = 'rec'

# stat test for pyramidal cells  ---
df_bsl = df_tmp.loc[(df_tmp['day'] == 'bsl') & (df_tmp['celltype'] == 'P'),['N_bsl','N_stim']]
df_SA = df_tmp.loc[(df_tmp['day'] == 'SA') & (df_tmp['celltype'] == 'P'),['N_bsl','N_stim']]
df_rec = df_tmp.loc[(df_tmp['day'] == 'rec') & (df_tmp['celltype'] == 'P'),['N_bsl','N_stim']]
df_chr = df_tmp.loc[(df_tmp['day'] == 'chro') & (df_tmp['celltype'] == 'P'),['N_bsl','N_stim']]
# baseline ie tonic
sstats.mannwhitneyu(df_bsl['N_bsl'].to_numpy(),df_SA['N_bsl'].to_numpy(),nan_policy ='omit',alternative = 'greater' )   # with sub acute
sstats.mannwhitneyu(df_bsl['N_bsl'].to_numpy(),df_rec['N_bsl'].to_numpy(),nan_policy ='omit',alternative = 'two-sided' )  # with recovery
sstats.mannwhitneyu(df_bsl['N_bsl'].to_numpy(),df_chr['N_bsl'].to_numpy(),nan_policy ='omit',alternative = 'two-sided' )  # with chronic
# stim ie phasic
sstats.mannwhitneyu(df_bsl['N_stim'].to_numpy(),df_SA['N_stim'].to_numpy(),nan_policy ='omit',alternative = 'greater' )   # with sub acute
sstats.mannwhitneyu(df_bsl['N_stim'].to_numpy(),df_rec['N_stim'].to_numpy(),nan_policy ='omit',alternative = 'two-sided' )  # with recovery
sstats.mannwhitneyu(df_bsl['N_stim'].to_numpy(),df_chr['N_stim'].to_numpy(),nan_policy ='omit',alternative = 'two-sided' )  # with chronic

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


