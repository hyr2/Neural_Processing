function [spikes, spike_times_all, n_ch, present_unit_ids] = construct_spikes(firings, templates, Fs, ch2shank)
    % input
    % firings : MountainSort4 firings data format (3, n_spikes)
    % templates : MountainSort4 templates data format (n_ch, clip_size, n_clus)
    % Fs : sampling rate in HZ
    % ch2shank : specifies which shank each channel is in. (n_ch,)
    n_ch = size(templates,1);
    n_clus = size(templates, 3);
    % disp(n_clus)
    START_TIME = 0*Fs;
    DURATION = 400000*Fs; % more than ten hours
    spike_times_all = firings(2,:);
    idx = find(spike_times_all>START_TIME & spike_times_all<START_TIME+DURATION);
    fprintf("n_spikes_all=%d\n", length(idx));

    % initialization
    spike_times_all = firings(2,idx) - START_TIME;
    % disp(size(spike_times_all))
    spike_labels = firings(3,idx);
    ch_stamp = firings(1,idx);
    spike_times_by_clus = cell(1, n_clus);
    ts_by_clus = cell(1, n_clus);
    pri_ch_lut = -1*ones(1, n_clus);
    for i=1:n_clus
        spike_times_by_clus{i} = [];
        ts_by_clus{i} = [];
    end


    % count spikes by unit
    for i=1:length(spike_times_all)
        spk_lbl = spike_labels(i);
        pri_ch_lut(spk_lbl) = ch_stamp(i);
        spike_times_by_clus{spk_lbl}(end+1) = spike_times_all(i)/Fs;
        ts_by_clus{spk_lbl}(end+1) = spike_times_all(i);
    end

    % Find and discard absent untis
    absent_unit_ids = find(pri_ch_lut==-1);
    present_unit_ids = find(pri_ch_lut~=-1);
    n_clus_present = length(present_unit_ids);
    map_label2indices = zeros(1, n_clus);
    map_label2indices(present_unit_ids) = 1:n_clus_present;

    if (size(absent_unit_ids, 2)>0)
        fprintf("Info: there are units not firing in this segment. They will be removed without affecting indexing\n");
    end

    cluID_all = 1:n_clus;
    cluID_present = cluID_all(1, present_unit_ids);
    pri_ch_lut = pri_ch_lut(1, present_unit_ids);
    spike_times_by_clus = spike_times_by_clus(1, present_unit_ids);
    ts_by_clus = ts_by_clus(1, present_unit_ids);
    spike_indices = zeros(size(spike_labels));
    for i=1:length(spike_indices)
        spk_lbl = spike_labels(i);
        spike_indices(i) = map_label2indices(spk_lbl);
    end

    % set waveforms
    filtWaveform = cell(1, n_clus_present);
    timeWaveform = cell(1, n_clus_present);
    total_counts = zeros(1,n_clus_present);
    for i=1:n_clus_present
        filtWaveform{i} = templates(pri_ch_lut(i), :, i);
        timeWaveform{i} = [-49:50]/Fs*1000;
        total_counts(i) = length(ts_by_clus{i});
    end

    spikes.ts = ts_by_clus;
    spikes.times=spike_times_by_clus;
    spikes.shankID = ch2shank(pri_ch_lut);
    spikes.cluID = cluID_present;
    spikes.filtWaveform=filtWaveform;
    spikes.timeWaveform=timeWaveform;
    spikes.spindices = [spike_times_all/Fs; spike_indices]';
    spikes.numcells = n_clus_present;
    spikes.total = total_counts;
end