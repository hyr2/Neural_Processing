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
    sns_plot = sns.heatmap(data=df,cmap=cmap,vmax = 0.15,vmin = 0.00,square=True,linewidths= 2,cbar_kws={"shrink": .5})
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

    with open(os.path.join(output_folder,'pop_coupling_tracked.pkl'), 'wb') as f:    # saves all spike times 
        pickle.dump(result_df,f)
        
        
    # significance matrix for baseline vs chronic
    df_sig_matrix = pd.DataFrame(data = None, columns = [0,21,28,42,49],index = [0,21,28,42,49])
    # df_sig_matrix_ks = pd.Series(data=None, index = [21,28,42,49])
    
    result_df.fillna(0,inplace = True)
    result_df.drop(columns=[14,35,56],inplace=True)
    delta_c_bsl = result_df.iloc[:,0] - result_df.iloc[:,1]
    delta_c_bsl = delta_c_bsl.to_numpy(dtype=float)
    bsl_avg = (result_df.iloc[:,0] + result_df.iloc[:,1])/2
    # delta_c_bsl_14 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,2].to_numpy(dtype = float)
    delta_c_bsl_21 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,2].to_numpy(dtype = float)
    delta_c_bsl_28 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,3].to_numpy(dtype = float)
    # delta_c_bsl_35 = bsl_avg.to_numpy(dtype = float) - result_df.iloc[:,5].to_numpy(dtype = float)
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
    ax.scatter(np.nanmean(bsl,axis=1),chrr1,alpha = 0.3,s = 25,c = '#4873b7',linewidth=0)
    fg.set_size_inches(3,3)
    sns.despine()
    fg.savefig(os.path.join(output_folder,'Fig6_scatter_bsl_vs_chr.png'),dpi = 300)
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
    axes.set_ylim(bottom = 0,top = 0.12)
    axes.set_yticks([0,0.05,0.1])
    fig.set_size_inches((2.5, 3), forward=True)
    sns.despine()
    fig.savefig(os.path.join(output_folder,'Ci_barplot_bsl_vs_chr.png'),dpi = 300)
    sc_st.ttest_ind(df_bsl,df_chronic,alternative= 'less')  # stat test for bar plot
    
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
    axes.set_ylim(0.0,1)
    sns.despine()
    fg.set_size_inches(4,2.5)
    fg.savefig(os.path.join(output_folder,'Ci_CDF_bsl_vs_chr.png'),dpi = 300)
    sc_st.kstest(df_bsl,df_chronic,alternative = 'greater')  # Stat tests KS Test to check if the distribution is the same or not

    # prob distribution of the coupling coefficients PDF
    fg,axes = plt.subplots(1,1)
    sns.kdeplot(data=result_df_melted, x = 'c_i',hue = 'session',palette = ['#4f85b0','#060606'],ax = axes,common_norm=False, common_grid=False,linewidth = 3)     # PDF
    axes.set(xlabel=None, ylabel = None,title = None)
    axes.legend().set_visible(False)
    axes.set_xlim(-0.1,0.3)
    axes.set_xticks([-0.1,0,0.1,0.2,0.3])
    axes.set_ylim(0.0,7.5)
    fg.set_size_inches(4,2.5)
    sns.despine()
    fg.savefig(os.path.join(output_folder,'Ci_PDF_bsl_vs_chr.png'),dpi = 300)
    

    # Add marginal plots and how they change over sessions ie pdf of # of simultaneous spikes 
    

    
    
