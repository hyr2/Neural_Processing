% MATLAB Code
% Using Cell Explorer script: calc_ACG_metrics
% Fig2C
% Standalone code will read sample_data.mat which contains the spikes
% vector and the Fs variable from cell explorer
% pick_session = ['bsl','wk5'];
pick_session = 'wk5'
if pick_session == 'bsl'

    colors_clusters = {'#00008B','#FF2400','#00008B','#FF2400','#FF2400'};  % for bc7 21-12-09
    clusters = [1,3,5,125,44]; % for bc7 21-12-09
    filename_save = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/tmp_clusters_2cbsl/'; % for bc7 21-12-09
    filename_in = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/tmp_bc7/21-12-09/';
    
    load(fullfile(filename_in,'spikes_data.mat'))
    sr = Fs;
    spikes.numcells = length(spikes.times);
    cell_indexes = find(spikes.total>0);
    
    % For ACG: 
    bins_narrow = 50;
    acg_narrow = zeros(bins_narrow*2+1,numel(spikes.times));
    for i = cell_indexes'
	    acg_narrow(:,i) = CCG(spikes.times{i},ones(size(spikes.times{i})),'binSize',0.001,'duration',0.100,'norm','rate','Fs',1/sr);
    end
    time_axis = 1000*linspace(-0.05,0.05,101);
    iter_local = 1;
    for iter_l = clusters
    
	    % Create a new figure with preset properties
	    fig = figure('Name', strcat('ACG of Cluster',string(iter_l)), 'NumberTitle', 'off');
	    bar(time_axis,acg_narrow(:,iter_l),'FaceColor',colors_clusters{iter_local},'EdgeColor','none','BarWidth', 1)
	    ax = gca;  % Get current axis handle
	    ax.XAxis.FontSize = 24;  % Set x-axis tick font size
	    ax.YAxis.FontSize = 24;  % Set y-axis tick font size
        ylim([0,23.5]);
	    box off
        axis off
	    print(fullfile(filename_save,strcat(string(iter_l) ,'.svg')),'-dsvg')
	    close(fig)
        iter_local = iter_local+1;
    end
elseif pick_session == 'wk5'

    colors_clusters = {'#6F8FAF','#F88379','#6F8FAF','#F88379','#F88379'};  % for bc7 21-12-31
    clusters = [1,3,5,66,31]; % for bc7 21-12-31
    filename_save = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/tmp_clusters_2cwk5/'; % for bc7 21-12-31
    filename_in = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/tmp_bc7/21-12-31/';
    
    load(fullfile(filename_in,'spikes_data.mat'))
    sr = Fs;
    spikes.numcells = length(spikes.times);
    cell_indexes = find(spikes.total>0);
    
    % For ACG: 
    bins_narrow = 50;
    acg_narrow = zeros(bins_narrow*2+1,numel(spikes.times));
    for i = cell_indexes'
	    acg_narrow(:,i) = CCG(spikes.times{i},ones(size(spikes.times{i})),'binSize',0.001,'duration',0.100,'norm','rate','Fs',1/sr);
    end
    time_axis = 1000*linspace(-0.05,0.05,101);
    iter_local = 1;
    for iter_l = clusters
        
	    % Create a new figure with preset properties
	    fig = figure('Name', strcat('ACG of Cluster',string(iter_l)), 'NumberTitle', 'off');
	    bar(time_axis,acg_narrow(:,iter_l),'FaceColor',colors_clusters{iter_local},'EdgeColor','none','BarWidth', 1)
	    ax = gca;  % Get current axis handle
	    ax.XAxis.FontSize = 24;  % Set x-axis tick font size
	    ax.YAxis.FontSize = 24;  % Set y-axis tick font size
        ylim([0,23.5]);
	    box off
        axis off
	    print(fullfile(filename_save,strcat(string(iter_l) ,'.svg')),'-dsvg')
	    close(fig)
        iter_local = iter_local+1;
    end
end

% # color = '#f30101' (bsl) and #f69c9c for pyr
% # color = '#0302f6' (bsl) and #aaa9f5 for FS/PV