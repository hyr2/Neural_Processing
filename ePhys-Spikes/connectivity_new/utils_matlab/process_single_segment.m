function [] = process_single_segment(foldername, Fs, cellexp_path)
    % calculate connectivity
    % Hanlin's CellExplorer hecking codes
    % Fs : seconds
    % Assumes that the working directory is where this matlab file is
    addpath(genpath(cellexp_path))
    results_tmp_matpath = [foldername '/mono_res.cellinfo.mat'];
    templates = readmda([foldername '/templates_clean.mda']);
    firings = readmda([foldername '/firings_clean.mda']);
    ch2shank = readmatrix([foldername '/ch2shank.csv']);
    [spikes, spike_times_all, n_ch, present_unit_ids] = construct_spikes(firings, templates, Fs, ch2shank);
    tic
    mono_res = ce_MonoSynConvClick(spikes, 'includeInhibitoryConnections', true, 'binSize', 0.001); 
    toc
    save(results_tmp_matpath,'mono_res');

    connecs_exc = present_unit_ids(mono_res.sig_con_excitatory);
    n_exc = size(connecs_exc, 1);
    
    if isfield(mono_res,'sig_con_inhibitory')
        connecs_inh = present_unit_ids(mono_res.sig_con_inhibitory);
    else
        connecs_inh = [];
    end 
    n_inh = size(connecs_inh, 1);

    n_cons = n_exc + n_inh;
    fprintf("n_cons=%d\n", n_cons);
    arr_cons = ones(n_cons, 3);
    arr_cons(1:n_exc, 1:2) = connecs_exc;
    arr_cons((1+n_exc):end, 1:2) = connecs_inh;
    arr_cons((1+n_exc):end, 3) = -1;
    writematrix(arr_cons, [foldername '/connecs.csv']);
end
