# Population coupling script

import numpy as np
import os, re, pickle
from matplotlib import pyplot as plt
import scipy.ndimage as sc_i
import scipy.signal as sc_s
import scipy.stats as sc_ss
import copy, time
import pandas as pd
import seaborn as sns

import scipy.stats as sc_st

def get_lower_tri_heatmap(df, output):
    # function used to plot lower half of a triangle using seaborn
    
    df = df.astype('float')
    df[df < 0.09] = 0     # to create a binary mask for this matrix: 0.1 means 0.05 for one-sided test
    
    # mask = np.zeros_like(df, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    # mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.dark_palette("xkcd:golden", 8)
    cmap = sns.dark_palette("#79C",as_cmap=True).reversed()

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(data=df,cmap=cmap,vmax = 0.05,vmin = 0.00,square=True,linewidths= 2,cbar_kws={"shrink": .5})
    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(output,dpi = 300)

def func_invariant_params(f_i_M):   
    # function computes the 2N invariant parameters after shuffling. N is the number of neurons (row index)
    tmp_sum_temporal = np.sum(f_i_M,dtype=np.int32,axis = 0)
    custom_bins = np.linspace(0,6,7)
    hist, bin_edges = np.histogram(tmp_sum_temporal, bins=custom_bins)
    bin_edges = bin_edges[:-1]

    return (bin_edges,hist,np.sum(f_i_M,axis=1))

def replace_submatrix(mat, ind1, ind2, mat_replace):
    # mat is the input matrix
    # ind1 is the rows
    # ind2 is the columns
    # mat_replace is the replacement matrix of size ind1 x ind2 sizes
  for i, index in enumerate(ind1):
    mat[index, ind2] = mat_replace[i, :]
  return mat
    
def process_single_shank(folder_input,session_list):
    A1 = np.load(os.path.join(folder_input,'all_clus_property.npy'),allow_pickle=True)
    A2 = np.load(os.path.join(folder_input,'all_clus_property_star.npy'),allow_pickle=True)
    # with open('saved_dictionary.pkl', 'rb') as f:
    #     df_spk_times = pickle.load(f)
    
    num_units_on_shank = len(A1)
    num_sessions = len(A2)
    
    arr_pop_rate = []    # for all sessions     (sampling rate: 1000/decimate_f)
    arr_temporal = []    # time bins for all sessions
    arr_mean_pop_rate = [] # mean pop rate for all sessions

    decimate_f = 5

    # Population Coupling Coefficients with the population 
    session_ids = ['session' + str(iter_l) for iter_l in range(num_sessions)]
    c_i = []    # population coupling of unit i
    df_c_i_pr = pd.DataFrame(data=None, index=range(num_units_on_shank) , columns = [session_list])
    spk_c_i = []   # mean firing rate of this unit
    for iter_s in range(num_sessions):
        for iter_i in range(num_units_on_shank):
            
            f_i = sc_i.gaussian_filter1d(A1[iter_i]['FR_session'][iter_s],10.19/np.sqrt(2))
            f_i_mod = A1[iter_i]['spike_count_session'][iter_s]
            
            f_j_sum = np.zeros(f_i.shape,dtype = np.float64)    #initialize array
            for iter_j in range(num_units_on_shank):            # this is the summation over the units except for i
                if iter_j == iter_i:
                    continue
                filtered_signal = sc_i.gaussian_filter1d(A1[iter_j]['FR_session'][iter_s],10.19/np.sqrt(2))
                # avg_fr_local = A1[iter_j]['spike_count_session'][iter_s] / A1[iter_j]['length_session'][iter_s]
                # f_j = filtered_signal - avg_fr_local
                f_j = filtered_signal
                f_j_sum = f_j_sum + f_j
            f_j_sum = f_j_sum/(num_units_on_shank-1)    
            # c_i_local = np.dot(f_i,f_j_sum) / f_i_mod
            # df_c_i.iat[iter_i,iter_s] = c_i_local
            df_c_i_pr.iat[iter_i,iter_s] = sc_ss.pearsonr(f_j_sum,f_i)[0]        # definition used is: DOI: https://doi.org/10.7554/eLife.56053
            
            # c_i.append(c_i_local)
            spk_c_i.append(f_i_mod)
    return df_c_i_pr

def full_panel_plot_fig6(result_df,output_folder,str_filesave_flag):
    # significance matrix for baseline vs chronic
    df_sig_matrix = pd.DataFrame(data = None, columns = [0,21,28,42,49],index = [0,21,28,42,49])
    df_sig_array = pd.Series(data = None,index = [0,1])
    # df_sig_matrix_ks = pd.Series(data=None, index = [21,28,42,49])
    
    result_df.fillna(0,inplace = True)
    delta_c_bsl = result_df.iloc[:,0] - result_df.iloc[:,1]
    delta_c_bsl = delta_c_bsl.to_numpy(dtype=float)
    bsl_avg = (result_df.iloc[:,0] + result_df.iloc[:,1])/2

    delta_c_bsl_21 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,2].to_numpy(dtype = float)
    delta_c_bsl_28 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,3].to_numpy(dtype = float)
    delta_c_bsl_42 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,4].to_numpy(dtype = float)
    delta_c_bsl_49 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,5].to_numpy(dtype = float)
    
    
    # df_sig_matrix.at[0,14] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_14)[1]
    df_sig_matrix.at[0,21] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_21)[1]
    df_sig_matrix.at[0,28] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_28)[1]
    # sc_st.ttest_rel(delta_c_bsl,delta_c_bsl_35)     # this one is bad for some reason. Remove day 35 since its not common to all animals and has mostly nans
    df_sig_matrix.at[0,42] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_42)[1]
    df_sig_matrix.at[0,49] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_49)[1]
    # df_sig_matrix.at[14,21] = sc_st.ttest_ind(delta_c_bsl_14,delta_c_bsl_21)[1]
    # df_sig_matrix.at[14,28] = sc_st.ttest_ind(delta_c_bsl_14,delta_c_bsl_28)[1]
    # df_sig_matrix.at[14,42] = sc_st.ttest_ind(delta_c_bsl_14,delta_c_bsl_42)[1]
    # df_sig_matrix.at[14,49] = sc_st.ttest_ind(delta_c_bsl_14,delta_c_bsl_49)[1]
    df_sig_matrix.at[21,28] = sc_st.ttest_ind(delta_c_bsl_21,delta_c_bsl_28)[1]
    df_sig_matrix.at[21,42] = sc_st.ttest_ind(delta_c_bsl_21,delta_c_bsl_42)[1]
    df_sig_matrix.at[21,49] = sc_st.ttest_ind(delta_c_bsl_21,delta_c_bsl_49)[1]
    df_sig_matrix.at[28,42] = sc_st.ttest_ind(delta_c_bsl_28,delta_c_bsl_42)[1]
    df_sig_matrix.at[28,49] = sc_st.ttest_ind(delta_c_bsl_28,delta_c_bsl_49)[1]
    df_sig_matrix.at[42,49] = sc_st.ttest_ind(delta_c_bsl_42,delta_c_bsl_49)[1]
    get_lower_tri_heatmap(df_sig_matrix,os.path.join(output_folder,f'{str_filesave_flag}_significance_matrix_variation.png'))     # plotting significance matrix
    
    # Lan modified statistical test :
    delta_c_bsl = result_df.iloc[:,0] - result_df.iloc[:,1]     # 1st
    delta_c_bsl = delta_c_bsl.to_numpy(dtype=float)
    bsl_avg = (result_df.iloc[:,0] + result_df.iloc[:,1])/2     

    delta_c_bsl_21 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,2].to_numpy(dtype = float)
    delta_c_bsl_28 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,3].to_numpy(dtype = float)
    delta_c_bsl_42 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,4].to_numpy(dtype = float)
    delta_c_bsl_49 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,5].to_numpy(dtype = float)

    # df_sig_matrix.at[0,14] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_14)[1]
    df_sig_matrix.at[0,21] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_21)[1]
    df_sig_matrix.at[0,28] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_28)[1]
    df_sig_matrix.at[0,42] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_42)[1]
    df_sig_matrix.at[0,49] = sc_st.ttest_ind(delta_c_bsl,delta_c_bsl_49)[1]
    df_sig_matrix.at[21,28] = sc_st.ttest_ind(delta_c_bsl,result_df.iloc[:,3].to_numpy(dtype = float) - result_df.iloc[:,2].to_numpy(dtype = float))[1]
    df_sig_matrix.at[21,42] = sc_st.ttest_ind(delta_c_bsl,result_df.iloc[:,4].to_numpy(dtype = float) - result_df.iloc[:,2].to_numpy(dtype = float))[1]
    df_sig_matrix.at[21,49] = sc_st.ttest_ind(delta_c_bsl,result_df.iloc[:,5].to_numpy(dtype = float) - result_df.iloc[:,2].to_numpy(dtype = float))[1]
    df_sig_matrix.at[28,42] = sc_st.ttest_ind(delta_c_bsl,result_df.iloc[:,4].to_numpy(dtype = float) - result_df.iloc[:,3].to_numpy(dtype = float))[1]
    df_sig_matrix.at[28,49] = sc_st.ttest_ind(delta_c_bsl,result_df.iloc[:,5].to_numpy(dtype = float) - result_df.iloc[:,3].to_numpy(dtype = float))[1]
    df_sig_matrix.at[42,49] = sc_st.ttest_ind(delta_c_bsl,result_df.iloc[:,5].to_numpy(dtype = float) - result_df.iloc[:,4].to_numpy(dtype = float))[1]
    
    get_lower_tri_heatmap(df_sig_matrix,os.path.join(output_folder,f'{str_filesave_flag}_significance_matrix_variation_lan.png'))     # plotting significance matrix
    
    # Plotting scatter plot
    result_df.replace(0, np.nan,inplace=True)
    chrr1 = result_df.iloc[:,2:].mean(axis=1).to_numpy()
    # chrr2 = result_df.iloc[:,4:6].mean(axis=1).to_numpy()
    bsl = result_df.iloc[:,0:2].to_numpy()
    bsl = bsl.astype(dtype=float)
    x = np.linspace(-0.1,0.45)
    y = x
    y1 = x + np.nanmean(bsl[:,0] - bsl[:,1]) + np.nanstd(bsl[:,0] - bsl[:,1])
    y2 = x - np.nanmean(bsl[:,0] - bsl[:,1]) - np.nanstd(bsl[:,0] - bsl[:,1])
    fg,ax = plt.subplots(1,1)
    ax.plot(x,y1,'--',c = 'k',alpha = 0.2)
    ax.plot(x,y2,'--',c = 'k',alpha = 0.2)
    ax.plot(x,y,'#3a3a3a')
    ax.scatter(np.nanmean(bsl,axis=1),chrr1,alpha = 0.3,s = 35,c = '#4873b7',linewidth=0)
    fg.set_size_inches(3,3)
    sns.despine()
    filename_l = f'Fig6_{str_filesave_flag}_scatter_bsl_vs_chr.png'
    fg.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    # ax[1].plot(x,y1,'--',c = '#3a3a3a',alpha = 0.3)
    # ax[1].plot(x,y2,'--',c = '#3a3a3a',alpha = 0.3)
    # ax[1].plot(x,y,'#3a3a3a')
    # ax[1].scatter(np.nanmean(bsl,axis=1),chrr2,alpha = 0.5)
    
    # Bar plot and statistical test
    # zero c_i typically means that the sesssion was missing from this animal/shank OR the unit had zero spikes in the session ie not found in the longitudinal tracking
    result_df_melted = result_df.melt(var_name = 'session',value_name = 'c_i')
    session_axis = result_df_melted.session.unique()
    result_df_melted.loc[(result_df_melted.loc[:,'session'] >= 21),'session'] = 'chronic'
    result_df_melted.loc[(result_df_melted.loc[:,'session'] == -3) | (result_df_melted.loc[:,'session'] == -2),'session'] = 'bsl'
    df_chronic = result_df_melted.loc[result_df_melted['session'] == 'chronic','c_i']
    df_bsl = result_df_melted.loc[result_df_melted['session'] == 'bsl','c_i']
    df_chronic.dropna(axis = 0,inplace=True)
    df_bsl.dropna(axis = 0,inplace=True)
    fig,axes = plt.subplots(1,1)
    sns.barplot(data=result_df_melted, x="session", y="c_i",errorbar = 'se',ax = axes,linewidth=3.5,color = '#a9a8a9',alpha=0.9,edgecolor='k',width=0.9)
    axes.set(xlabel=None, ylabel = None,title = None)
    axes.legend().set_visible(False)
    axes.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right=False,      # ticks along the bottom edge are off
        left=True) # labels along the bottom edge are off
    # axes.set_ylim(bottom = 0,top = 0.12)
    axes.set_yticks([0,0.05,0.1])
    fig.set_size_inches((2.5, 3), forward=True)
    sns.despine()
    filename_l = f'Ci_{str_filesave_flag}_barplot_bsl_vs_chr.png'
    fig.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    df_sig_array[0] = sc_st.ttest_ind(df_bsl,df_chronic)[1]  # stat test for bar plot
    
    # prob distribution of the coupling coefficients  CDF
    fg,axes = plt.subplots(1,1)
    sns.kdeplot(data=result_df_melted, x = 'c_i',hue = 'session',palette = ['#4f85b0','#060606'],linewidth = 3,ax = axes,cumulative=True,common_norm=False, common_grid=True)     # CDF
    axes.set_xlim(-0.1,0.35)
    axes.set(xlabel=None, ylabel = None,title = None)
    axes.legend().set_visible(False)
    axes.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right=False,      # ticks along the bottom edge are off
        left=True) # labels along the bottom edge are off
    axes.set_ylim(bottom = 0,top = 0.12)
    axes.set_xticks([-0.1,0,0.1,0.2,0.3])
    axes.set_yticks([0,0.5,1])
    axes.set_ylim(0.0,1.05)
    sns.despine()
    fg.set_size_inches(4,2.5)
    filename_l = f'Ci_{str_filesave_flag}_CDF_bsl_vs_chr.png'
    fg.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    df_sig_array[1] = sc_st.kstest(df_bsl,df_chronic,method = 'exact')[1]  # Stat tests KS Test to check if the distribution is the same or not

    # prob distribution of the coupling coefficients PDF
    fg,axes = plt.subplots(1,1)
    sns.kdeplot(data=result_df_melted, x = 'c_i',hue = 'session',palette = ['#4f85b0','#060606'],ax = axes,common_norm=False, common_grid=False,linewidth = 3)     # PDF
    # sns.histplot(data=result_df_melted, x = 'c_i',hue = 'session',palette = ['#4f85b0','#060606'],ax = axes,common_norm=False, linewidth = 3)     # PDF
    axes.set(xlabel=None, ylabel = None,title = None)
    axes.legend().set_visible(False)
    axes.set_xlim(-0.1,0.3)
    axes.set_xticks([-0.1,0,0.1,0.2,0.3])
    # axes.set_ylim(0.0,7.5)
    fg.set_size_inches(3,2.5)
    sns.despine()
    filename_l = f'Ci_{str_filesave_flag}_PDF_bsl_vs_chr.png'
    fg.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    
    # Change to averaged only counts histogram (non aggregated days)
    result_df_bsl = result_df.iloc[:,0:2].mean(axis = 1,skipna = True)
    result_df_chr = result_df.iloc[:,2:].mean(axis = 1,skipna = True)
    result_df_new = pd.DataFrame(data = None, columns = ['bsl','chr'])
    result_df_new.loc[:,'bsl'] = result_df_bsl
    result_df_new.loc[:,'chr'] = result_df_chr
    result_df_melted = result_df_new.melt(var_name = 'session',value_name = 'c_i')
    fg,axes = plt.subplots(1,1)
    # sns.histplot(data=result_df_melted, x = 'c_i',hue = 'session',palette = ['#4f85b0','#060606'],ax = axes,common_norm=False, linewidth = 2,fill=False )     # PDF
    sns.histplot(data=result_df_melted, x = 'c_i',hue = 'session',palette = ['#4f85b0','#060606'],ax = axes,common_norm=False, linewidth = 2,fill=False, kde=True, line_kws = {'linewidth':2.5},alpha = 0.9)     # PDF
    # sns.histplot(data=result_df_melted, x = 'c_i',hue = 'session',palette = ['#4f85b0','#060606'],ax = axes,common_norm=False, linewidth = 0, kde=True, line_kws = {'linewidth':3.5},fill=False )     # PDF
    axes.set(xlabel=None, ylabel = None,title = None)
    axes.legend().set_visible(False)
    axes.set_xlim(-0.1,0.65)
    axes.set_xticks([-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6])
    # axes.set_ylim(0.0,7.5)
    fg.set_size_inches(3,2.5)
    sns.despine()
    filename_l = f'Ci_{str_filesave_flag}_PDF_bsl_vs_chr_modified_lan.png'
    fg.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    
    # save statistical tests
    df_sig_array.index = ['barplot','CDF']
    df_sig_array.to_csv(os.path.join(output_folder,f'c_avg_stats_{str_filesave_flag}.csv'))
    df_sig_matrix.to_csv(os.path.join(output_folder,f'delta_c_stats_{str_filesave_flag}.csv'))
    
    plt.close('all')

def full_panel_plot_fig6_celltype(result_df_p,result_df_ni,output_folder,str_filesave_flag):
    
    i_index = [('NI'),('P')]
    # multi_index = pd.MultiIndex.from_tuples(multi_index, names = ['celltype'])
    df_stat_test = pd.DataFrame({
    },columns = ['mean','dist'],index = i_index)
    
    # For figure 6F cell type analysis with population coupling
    df_sig_array = pd.Series(data = None,index = [0,1])

    chrr_p = result_df_p.iloc[:,2:].mean(axis=1).to_numpy()
    chrr_ni = result_df_ni.iloc[:,2:].mean(axis=1).to_numpy()
    bsl_p = result_df_p.iloc[:,0:2].to_numpy()
    bsl_ni = result_df_ni.iloc[:,0:2].to_numpy()
    bsl_ni = bsl_ni.astype(dtype=float)
    bsl_p = bsl_p.astype(dtype=float)
    x = np.linspace(-0.1,0.45)
    y = x
    # y1 = x + np.nanmean(bsl[:,0] - bsl[:,1]) + np.nanstd(bsl[:,0] - bsl[:,1])
    # y2 = x - np.nanmean(bsl[:,0] - bsl[:,1]) - np.nanstd(bsl[:,0] - bsl[:,1])
    fg,ax = plt.subplots(1,1)
    ax.plot(x,y,'#3a3a3a')
    ax.scatter(np.nanmean(bsl_p,axis=1),chrr_p,alpha = 0.35,s = 60,c = '#cc1b1b',linewidth=0)
    ax.scatter(np.nanmean(bsl_ni,axis=1),chrr_ni,alpha = 0.35,s = 60,c = '#0f36e2',linewidth=0)
    fg.set_size_inches(3,3)
    sns.despine()
    filename_l = f'Fig6_{str_filesave_flag}_scatter_bsl_vs_chr.png'
    fg.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    # ax[1].plot(x,y1,'--',c = '#3a3a3a',alpha = 0.3)
    # ax[1].plot(x,y2,'--',c = '#3a3a3a',alpha = 0.3)
    # ax[1].plot(x,y,'#3a3a3a')
    # ax[1].scatter(np.nanmean(bsl,axis=1),chrr2,alpha = 0.5)
    
    # Bar plot and statistical test for pyr
    result_df_melted_p = result_df_p.melt(var_name = 'session',value_name = 'c_i')
    session_axis = result_df_melted_p.session.unique()
    result_df_melted_p.loc[(result_df_melted_p.loc[:,'session'] >= 21),'session'] = 'chronic'
    result_df_melted_p.loc[(result_df_melted_p.loc[:,'session'] == -3) | (result_df_melted_p.loc[:,'session'] == -2),'session'] = 'bsl'
    df_chronic = result_df_melted_p.loc[result_df_melted_p['session'] == 'chronic','c_i']
    df_bsl = result_df_melted_p.loc[result_df_melted_p['session'] == 'bsl','c_i']
    df_chronic.dropna(axis = 0,inplace=True)
    df_bsl.dropna(axis = 0,inplace=True)
    df_bsl = df_bsl.astype('float')
    df_chronic = df_chronic.astype('float')
    fig,axes = plt.subplots(1,1)
    sns.barplot(data=result_df_melted_p, x="session", y="c_i",errorbar = 'se',ax = axes,linewidth=3.5,color = '#cc1b1b',alpha=0.9,edgecolor='k',width=0.9)
    axes.set(xlabel=None, ylabel = None,title = None)
    axes.legend().set_visible(False)
    axes.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right=False,      # ticks along the bottom edge are off
        left=True) # labels along the bottom edge are off
    axes.set_ylim(bottom = 0,top = 0.12)
    axes.set_yticks([0,0.05,0.1])
    fig.set_size_inches((2.5, 5), forward=True)
    sns.despine()
    filename_l = f'Ci_pyr_barplot_bsl_vs_chr.png'
    fig.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    df_stat_test.loc['P','mean'] = sc_st.ttest_ind(df_bsl,df_chronic)[1]  # stat test for bar plot
    # Bar plot and statistical test for NI
    result_df_melted_ni = result_df_ni.melt(var_name = 'session',value_name = 'c_i')
    session_axis = result_df_melted_ni.session.unique()
    result_df_melted_ni.loc[(result_df_melted_ni.loc[:,'session'] >= 21),'session'] = 'chronic'
    result_df_melted_ni.loc[(result_df_melted_ni.loc[:,'session'] == -3) | (result_df_melted_ni.loc[:,'session'] == -2),'session'] = 'bsl'
    df_chronic = result_df_melted_ni.loc[result_df_melted_ni['session'] == 'chronic','c_i']
    df_bsl = result_df_melted_ni.loc[result_df_melted_ni['session'] == 'bsl','c_i']
    df_chronic.dropna(axis = 0,inplace=True)
    df_bsl.dropna(axis = 0,inplace=True)
    df_bsl = df_bsl.astype('float')
    df_chronic = df_chronic.astype('float')
    fig,axes = plt.subplots(1,1)
    sns.barplot(data=result_df_melted_ni, x="session", y="c_i",errorbar = 'se',ax = axes,linewidth=3.5,color = '#0f36e2',alpha=0.9,edgecolor='k',width=0.9)
    axes.set(xlabel=None, ylabel = None,title = None)
    axes.legend().set_visible(False)
    axes.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right=False,      # ticks along the bottom edge are off
        left=True) # labels along the bottom edge are off
    axes.set_ylim(bottom = 0,top = 0.175)
    axes.set_yticks([0,0.05,0.1,0.15,0.175])
    fig.set_size_inches((2.5, 5), forward=True)
    sns.despine()
    filename_l = f'Ci_NI_barplot_bsl_vs_chr.png'
    fig.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    df_stat_test.loc['NI','mean'] = sc_st.ttest_ind(df_bsl,df_chronic)[1]  # stat test for bar plot
    
    
    # prob distribution of the coupling coefficients  CDF
    result_df_melted_p['celltype'] = ['P']*result_df_melted_p.shape[0]
    result_df_melted_ni['celltype'] = ['NI']*result_df_melted_ni.shape[0]
    result_df_melted = pd.concat((result_df_melted_p,result_df_melted_ni),axis = 0)
    result_melted_df_bsl = result_df_melted.loc[result_df_melted['session'] == 'bsl',:]
    result_melted_df_bsl.reset_index(inplace=True)
    result_melted_df_chr = result_df_melted.loc[result_df_melted['session'] == 'chronic',:]
    result_melted_df_chr.reset_index(inplace=True)
    fg,axes = plt.subplots(1,1)
    sns.kdeplot(data=result_melted_df_bsl, x = 'c_i',hue = 'celltype',hue_order = ['NI','P'],palette = ['#0f36e2','#cc1b1b'],linewidth = 2.5,ax = axes,cumulative=True,common_norm=False, common_grid=True)     # CDF
    sns.kdeplot(data=result_melted_df_chr,ls = '-.', x = 'c_i',hue = 'celltype',hue_order = ['NI','P'],palette = ['#0f36e2','#cc1b1b'],linewidth = 2.5,ax = axes,cumulative=True,common_norm=False, common_grid=True)     # CDF
    axes.set(xlabel=None, ylabel = None,title = None)
    axes.legend().set_visible(False)
    axes.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right=False,      # ticks along the bottom edge are off
        left=True) # labels along the bottom edge are off
    axes.set_ylim(bottom = 0,top = 0.12)
    axes.set_xticks([-0.1,0,0.2,0.4,0.6])
    axes.set_yticks([0,0.5,1])
    axes.set_ylim(0.0,1.05)
    axes.set_xlim(-0.1,0.6)
    sns.despine()
    fg.set_size_inches(3,2.5)
    filename_l = f'Ci_CDF_celltype.png'
    fg.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    
    result_melted_df_bsl = result_melted_df_bsl.filter(['c_i','celltype'])
    result_melted_df_chr = result_melted_df_chr.filter(['c_i','celltype'])
    
    result_melted_df_bsl_P = result_melted_df_bsl.loc[result_melted_df_bsl['celltype'] == 'P']
    result_melted_df_bsl_NI = result_melted_df_bsl.loc[result_melted_df_bsl['celltype'] == 'NI']
    result_melted_df_chr_P = result_melted_df_chr.loc[result_melted_df_chr['celltype'] == 'P']
    result_melted_df_chr_NI = result_melted_df_chr.loc[result_melted_df_chr['celltype'] == 'NI']
    
    df_stat_test.loc['P','dist'] = sc_st.kstest(result_melted_df_bsl_P['c_i'],result_melted_df_chr_P['c_i'])[1]  # Stat tests KS Test to check if the distribution is the same or not
    df_stat_test.loc['NI','dist'] = sc_st.kstest(result_melted_df_bsl_NI['c_i'],result_melted_df_chr_NI['c_i'])[1]  # Stat tests KS Test to check if the distribution is the same or not
    
    df_stat_test.to_csv(os.path.join(output_folder,'stat_tests.csv'))
    
    # save statistical tests with multi-index
    # df_sig_array.index = ['barplot','CDF']
    # df_sig_array.to_csv(os.path.join(output_folder,f'c_avg_stats_{str_filesave_flag}.csv'))
    # df_sig_matrix.to_csv(os.path.join(output_folder,f'delta_c_stats_{str_filesave_flag}.csv'))
    
    plt.close('all')

if __name__ == '__main__':

    script_name = '_pop_coupling'
    t = time.localtime()
    current_time = time.strftime("%m_%d_%Y_%H_%M", t)
    current_time = current_time + script_name
    output_folder = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results',current_time)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    input_dir_list = ['/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh3/Shank_2/Processed/count_analysis',
                      '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_bc7/Shank_2/Processed/count_analysis',
                      '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_bc7/Shank_3/Processed/count_analysis',
                      '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh8/Shank_1/Processed/count_analysis',
                      '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh8/Shank_3/Processed/count_analysis',
                      '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_0/Processed/count_analysis',
                      '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_1/Processed/count_analysis',
                      '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_3/Processed/count_analysis',
                      '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh7/Shank_3/Processed/count_analysis'
                      
        ]
    
    dict_config =  {}
    dict_config['rh3'] = np.array([-3,-2,14,21,28,49])
    dict_config['bc7'] = np.array([-3,-2,14,21,28,42])
    dict_config['rh8'] = np.array([-3,-2,14,21,28,35,42,49,56])
    dict_config['rh11'] = np.array([-3,-2,14,21,28,35,42,49])
    dict_config['rh7'] = np.array([-3,-2,14,21,28,35,42,49])
    result_df = pd.DataFrame(data=None,columns = [dict_config['rh8'].tolist()])
    result_df.columns = dict_config['rh8'].tolist()
    for shank_id in input_dir_list:
        
        # Get session for the mouse (using regular expressions)
        pattern = re.compile(r'processed_data_.*?/')
        match = pattern.search(shank_id)
        indx_str = match.group()[15:-1]
        # Call main processing function for c_i ie coupling coefficient
        df_c_i_pr = process_single_shank(shank_id,dict_config[indx_str].tolist())
        df_c_i_pr.columns = dict_config[indx_str].tolist()
        result_df = pd.concat([result_df, df_c_i_pr], axis=0)
        
    result_df.drop(columns=[14,35,56],inplace=True)     # keeping only chronic phase of the stroke
    with open(os.path.join(output_folder,'pop_coupling_tracked.pkl'), 'wb') as f:    # saves all spike times 
        pickle.dump(result_df,f)
        
    with open(os.path.join('/home/hyr2-office/Documents/Data/NVC/Results/04_02_2024_10_21_pop_coupling','pop_coupling_tracked.pkl'), 'rb') as f:    # load all spike times 
        result_df = pickle.load(f)
    # this function is used to plot everything for c_i statistics
    output_foldertmp = os.path.join(output_folder,'all') 
    os.makedirs(output_foldertmp)
    full_panel_plot_fig6(result_df,output_foldertmp,'all')
    
    # Importing data from Figure 5 and correlating with Figure 5's PCA clusters
    # Note that the numbers of clustered into each category are slighlty different because this was a 2nd code run from the one used to generate Fig5.
    with open(os.path.join('/home/hyr2-office/Documents/Data/NVC/Results/PAPER_RESULTS/Fig5/Fig5_plots_04_19_2024_10_25','Fig6_dictionary.pkl'), 'rb') as f:
        dict_fig5 = pickle.load(f)
        
    mask_reject = dict_fig5['mask_reject']
    cluster_labels = dict_fig5['cluster_labels']
    
    result_df = result_df.loc[np.logical_not(mask_reject),:]
    result_df_up = result_df.loc[cluster_labels == 0,:]     # potentiate
    result_df_nc = result_df.loc[cluster_labels == 1,:]     # no change 
    result_df_dn = result_df.loc[cluster_labels == 2,:]     # attenuated
    
    output_foldertmp = os.path.join(output_folder,'no_change') 
    os.makedirs(output_foldertmp)
    full_panel_plot_fig6(result_df_nc,output_foldertmp,'nc')
    output_foldertmp = os.path.join(output_folder,'attenuate')
    os.makedirs(output_foldertmp)
    full_panel_plot_fig6(result_df_dn,output_foldertmp,'dn')
    output_foldertmp = os.path.join(output_folder,'potentiate') 
    os.makedirs(output_foldertmp)
    full_panel_plot_fig6(result_df_up,output_foldertmp,'up')
    
    
    # cell type and population coupling (Figure 6F)
    celltype_labels = dict_fig5['celltype']
    result_df_ni = result_df.loc[celltype_labels == 'NI',:]     # fast spiking
    result_df_p = result_df.loc[np.logical_or(celltype_labels == 'P',celltype_labels =='WI'),:]   # regular spiking
    
    output_foldertmp = os.path.join(output_folder,'celltype') 
    os.makedirs(output_foldertmp)
    full_panel_plot_fig6_celltype(result_df_p,result_df_ni,output_foldertmp,'celltype')
    # output_foldertmp = os.path.join(output_folder,'Pyr') 
    # os.makedirs(output_foldertmp)
    # full_panel_plot_fig6(result_df_p,output_foldertmp,'Pyr')
    
    # subset of neurons which changed tuning
    result_df_pos_detuned = result_df.loc[dict_fig5['detuned_pos'],:]
    result_df_detuned = result_df.loc[dict_fig5['detuned_all'],:]
    
    output_foldertmp = os.path.join(output_folder,'detuned_all') 
    os.makedirs(output_foldertmp)
    full_panel_plot_fig6(result_df_detuned,output_foldertmp,'detuned_all')
    
    output_foldertmp = os.path.join(output_folder,'detuned_pos') 
    os.makedirs(output_foldertmp)
    full_panel_plot_fig6(result_df_pos_detuned,output_foldertmp,'detuned_pos')

    # For Lan. Checking if the potentiated cells had stronger Ci than other cells before stroke.
    df_bsl_potentiate = result_df_up.loc[:,(-3,-2)]
    df_bsl_potentiate_n = pd.concat((result_df_nc.loc[:,(-3,-2)],result_df_dn.loc[:,(-3,-2)]) , axis = 0)
    
    df_bsl_potentiate_melt = df_bsl_potentiate.melt(var_name = 'potentiate',value_name = 'c_i') 
    df_bsl_potentiate_melt['potentiate'] = 'Y'
    df_bsl_potentiateN_melt = df_bsl_potentiate_n.melt(var_name = 'potentiate',value_name = 'c_i') 
    df_bsl_potentiateN_melt['potentiate'] = 'N'
    df_bsl_potentiate_melt.reset_index(inplace=True,drop = True)
    df_bsl_potentiateN_melt.reset_index(inplace=True,drop = True)
    df_potentiate_final = pd.concat((df_bsl_potentiate_melt,df_bsl_potentiateN_melt),axis = 0)
    df_potentiate_final.reset_index(inplace = True, drop = True)
    df_potentiate_final.dropna(axis = 0,inplace=True)
    
    
    fig,axes = plt.subplots(1,1)
    sns.barplot(data=df_potentiate_final, x="potentiate", y="c_i",errorbar = 'se',ax = axes,linewidth=3.5,color = '#a9a8a9',alpha=0.9,edgecolor='k',width=0.9)
    axes.set(xlabel=None, ylabel = None,title = None)
    axes.legend().set_visible(False)
    axes.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right=False,      # ticks along the bottom edge are off
        left=True) # labels along the bottom edge are off
    axes.set_ylim(bottom = 0,top = 0.1)
    axes.set_yticks([0,0.05,0.1])
    fig.set_size_inches((4, 5), forward=True)
    sns.despine()
    filename_l = f'Ci_potnetiatedonly_preVSpost.png'
    fig.savefig(os.path.join(output_folder,filename_l),dpi = 300)
    sc_st.ttest_ind(df_potentiate_final.loc[df_potentiate_final['potentiate'] == 'Y','c_i'],df_potentiate_final.loc[df_potentiate_final['potentiate'] == 'N','c_i'])[1]  # stat test for bar plot
