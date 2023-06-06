def temp_func(clus_id,FR_series,spike_times_by_clus):
	firing_rate_avg = FR_series[clus_id][0]
	t_axis = np.linspace(0,firing_rate_avg.shape[0]*WINDOW_LEN_IN_SEC,firing_rate_avg.shape[0])
	t_1 = np.squeeze(np.where(t_axis >= 1.4))[0]      # stim start set to 2.45 seconds
	t_2 = np.squeeze(np.where(t_axis >= 2.0))[0]      # end of bsl region (actual value is 2.5s but FR starts to increase before. Maybe anticipatory spiking due to training)
	t_3 = np.squeeze(np.where(t_axis <= 2.4))[-1]        # stim end set to 5.15 seconds
	t_4 = np.squeeze(np.where(t_axis <= 4.5))[-1]
	t_5 = np.squeeze(np.where(t_axis <= 10))[-1]        # post stim quiet period
	t_6 = np.squeeze(np.where(t_axis <= 11))[-1]
	t_7 = np.squeeze(np.where(t_axis <= 4.5))[-1]
	#Zscore
	bsl_mean = np.mean(np.hstack((firing_rate_avg[t_1:t_2],firing_rate_avg[t_5:t_6]))) # baseline is pre and post stim
	bsl_std = np.std(np.hstack((firing_rate_avg[t_1:t_2],firing_rate_avg[t_5:t_6])))
	firing_rate_zscore = zscore_bsl(firing_rate_avg, bsl_mean, bsl_std)
	(t_axis_s,firing_rate_zscore_s,normalized,indx,SsD) = FR_classifier_classic_zscore(firing_rate_zscore,t_axis,[t_1,t_7],[t_3,t_4])
	max_zscore_stim = np.amax(np.absolute(firing_rate_zscore[t_3:t_4]))
	t_stat, pval_2t = ttest_ind(
			firing_rate_avg[t_1:t_2], 
			firing_rate_avg[t_3:t_4], 
			equal_var=False)
	max_z = np.amax((firing_rate_zscore[t_3:t_4]))
	min_z = np.amin((firing_rate_zscore[t_3:t_4]))
	clus_property_1 = 0
	N_spikes_local = spike_times_by_clus[clus_id].size

	if max_zscore_stim > 2.5:
		if indx == 0 and N_spikes_local>(Ntrials*3) and (np.abs(max_z) > np.abs(min_z)):
			print('activated neuron')
			clus_property_1 = 1
		elif indx == 1 and N_spikes_local>(Ntrials*3) and (np.abs(max_z) < np.abs(min_z)):
			print('suppressed neuron')
			clus_property_1 = -1
	
	print(min_z,max_z)
	print('SsD: ',SsD)
	print('indx ',indx)
	print('Act/Sup ',clus_property_1)
	print('Thresh ',max_zscore_stim)
	print('T-test ',pval_2t)
	print('# Spikes',N_spikes_local)