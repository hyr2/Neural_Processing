clusters = [1,3,5,125,44]; % for bc7 21-12-09
clusters = [1,3,5,66,31]; % for bc7 21-12-31
filename_save = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/tmp_clusters_2cwk5/';
filename_save = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/tmp_clusters_2cbsl/';

% For ACG: 
bins_narrow = 50;
acg_narrow = zeros(bins_narrow*2+1,numel(spikes.times));
for i = cell_indexes'
	acg_narrow(:,i) = CCG(spikes.times{i},ones(size(spikes.times{i})),'binSize',0.001,'duration',0.100,'norm','rate','Fs',1/sr);
end
time_axis = 1000*linspace(-0.05,0.05,101);

for iter_l = clusters

	% Create a new figure with preset properties
	fig = figure('Name', strcat('ACG of Cluster',string(iter_l)), 'NumberTitle', 'off');
	bar(time_axis,acg_narrow(:,iter_l),'FaceColor','#f69c9c','EdgeColor','none','BarWidth', 1)
	ax = gca;  % Get current axis handle
	ax.XAxis.FontSize = 24;  % Set x-axis tick font size
	ax.YAxis.FontSize = 24;  % Set y-axis tick font size
	box off
	print(fullfile(filename_save,strcat(string(iter_l) ,'.svg')),'-dsvg')
	close(fig)
end

% # color = '#f30101' (bsl) and #f69c9c for pyr
% # color = '#0302f6' (bsl) and #aaa9f5 for FS/PV