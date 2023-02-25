clear all
close all
folder = '/home/hyr2/Documents/git/Time-Series/CellExplorer/calc_CellMetrics';
addpath(genpath(folder));

% TODO 
% *****(1) read connectivity sauce code
% *****(2) read trials_times.mat to obtain spikes during stim only
% *****(3) read trials_times.mat to obtain spikes during baseline only
Fs=30e3;        % read from .json file
datafolder = '/home/hyr2-office/Documents/Data/NVC/RH-7/10-27-22-temp/';

GW_BETWEENSHANK = 300;
GH = 25;
datestr = '10-2';
plotfolder = fullfile(datafolder,'Processed','cell_type');
% plotfolder = '/home/hyr2-office/Documents/Data/NVC/RH-3/processed_data_rh3/tmp/Processed/Connectivity/';
mkdir(plotfolder);
templates = readmda([datafolder 'templates.mda']);
firings = readmda([datafolder 'firings.mda']);
% Location = csvread("data/mustang220123/location.csv");
curation_mask = logical(csvread([datafolder 'accept_mask.csv']));
response_mask = csvread(fullfile(datafolder,'Processed','count_analysis','cluster_response_mask.csv'));
clus_locations = csvread([datafolder 'clus_locations.csv']);
% disp(Location(1,:))
n_ch = size(templates,1);
n_clus = size(templates, 3);
START_TIME = 0*Fs;
DURATION = 12500*Fs;
spike_times_all = firings(2,:);
idx = find(spike_times_all>START_TIME & spike_times_all<START_TIME+DURATION);
disp('------')
disp(size(idx))
disp('------')

% initialization : curation
spike_times_all = firings(2,idx) - START_TIME;
disp(size(spike_times_all))
spike_labels = firings(3,idx);
ch_stamp = firings(1,idx);
spike_times_by_clus = cell(1, n_clus);
ts_by_clus = cell(1, n_clus);
pri_ch_lut = -1*ones(1, n_clus);
for i=1:n_clus
    spike_times_by_clus{i} = [];
    ts_by_clus{i} = [];
end


% count spikes by unit : curation
for i=1:length(spike_times_all)
    spk_lbl = spike_labels(i);
    pri_ch_lut(spk_lbl) = ch_stamp(spk_lbl);
    spike_times_by_clus{spk_lbl}(end+1) = spike_times_all(i)/Fs;
    ts_by_clus{spk_lbl}(end+1) = spike_times_all(i);
end


% More curation code
curation = 1;
if curation
    % extract accepted cluster locations & their responsiveness
    response_mask = response_mask(curation_mask,:);
    clus_locations = clus_locations(curation_mask,:);
    spike_times_curated = [];
    spike_labels_curated = [];
    spike_labels_mapping = zeros(n_clus);
    spike_times_by_clus_cur = cell(1, sum(curation_mask));
    ts_by_clus_cur = cell(1, sum(curation_mask));
    for i=1:sum(curation_mask)
        spike_times_by_clus_cur{i}=[];
        ts_by_clus_cur{i}=[];
    end
    % accepted waveforms
    filtWaveform = cell(1, sum(curation_mask));
    timeWaveform = cell(1, sum(curation_mask));
    k = 1;
    for i=1:length(curation_mask)
        if curation_mask(i)==1
            spike_labels_mapping(i) = k;
            k = k + 1;
        end
    end
    for i=1:n_clus
        if curation_mask(i)==1
%           filtWaveform{spike_labels_mapping(i)} = templates(pri_ch_lut(i), [25:75], i);
            filtWaveform{spike_labels_mapping(i)} = templates(pri_ch_lut(i), [2:52], i);    % for 25 KHZ
            timeWaveform{spike_labels_mapping(i)} = [-25:25]/Fs*1000;
            % spike_times_by_clus{i} = spike_times_by_clus{i}';
        end
    end
    % accepted spike timestamps and labels
    for i=1:length(spike_times_all)
        
        if curation_mask(spike_labels(i))==1
            spk_label = spike_labels_mapping(spike_labels(i));
            spike_times_curated(end+1) = spike_times_all(i);
            spike_labels_curated(end+1) = spk_label;
            spike_times_by_clus_cur{spk_label}(end+1) = spike_times_all(i)/Fs;
            ts_by_clus_cur{spk_label}(end+1) = spike_times_all(i);
            %get primary channel for each label; safely assumes each cluster has only one primary channel
        %     pri_ch_lut(i) = firings(1,i);

         end
    end
    % Location = Location(pri_ch_lut, :);
    

    spikes.ts = ts_by_clus_cur;
    spikes.times=spike_times_by_clus_cur;
    spikes.shankID = pri_ch_lut(curation_mask);
    spikes.cluID = [1:sum(curation_mask)];
    spikes.filtWaveform=filtWaveform;
    spikes.timeWaveform=timeWaveform;
    spikes.spindices = [spike_times_curated/Fs; spike_labels_curated]';
else
    spikes.ts = ts_by_clus;
    spikes.times=spike_times_by_clus;
    spikes.shankID = pri_ch_lut;
    spikes.cluID = 1:n_clus;
    spikes.filtWaveform=filtWaveform;
    spikes.timeWaveform=timeWaveform;
    spikes.spindices = [spike_times_all/Fs; spike_labels]';
end
% end of curation
disp(size(spikes.spindices));
tic
mono_res = ce_MonoSynConvClick(spikes,'includeInhibitoryConnections',true); % detects the monosynaptic connections
toc
%% Copied from ProcessCellMetrics ,section Putative MonoSynaptic connections

save('mono_res.cellinfo.mat','mono_res','-v7.3','-nocompression');
spkExclu=1;
cell_metrics.putativeConnections.excitatory = mono_res.sig_con_excitatory; % Vectors with cell pairs
cell_metrics.general.cellCount=size(mono_res.completeIndex,1);
if isfield(mono_res,'sig_con_inhibitory')
    cell_metrics.putativeConnections.inhibitory = mono_res.sig_con_inhibitory;
else
    cell_metrics.putativeConnections.inhibitory = [];
end 

cell_metrics.synapticEffect = repmat({'Unknown'},1,cell_metrics.general.cellCount);
cell_metrics.synapticEffect(cell_metrics.putativeConnections.excitatory(:,1)) = repmat({'Excitatory'},1,size(cell_metrics.putativeConnections.excitatory,1)); % cell_synapticeffect ['Inhibitory','Excitatory','Unknown']
if ~isempty(cell_metrics.putativeConnections.inhibitory)
    cell_metrics.synapticEffect(cell_metrics.putativeConnections.inhibitory(:,1)) = repmat({'Inhibitory'},1,size(cell_metrics.putativeConnections.inhibitory,1));
end
cell_metrics.synapticConnectionsOut = zeros(1,cell_metrics.general.cellCount);
cell_metrics.synapticConnectionsIn = zeros(1,cell_metrics.general.cellCount);

[a,b]=hist(cell_metrics.putativeConnections.excitatory(:,1),unique(cell_metrics.putativeConnections.excitatory(:,1)));
cell_metrics.synapticConnectionsOut(b) = a; 
cell_metrics.synapticConnectionsOut = cell_metrics.synapticConnectionsOut(1:cell_metrics.general.cellCount);
[a,b]=hist(cell_metrics.putativeConnections.excitatory(:,2),unique(cell_metrics.putativeConnections.excitatory(:,2)));
cell_metrics.synapticConnectionsIn(b) = a; 
cell_metrics.synapticConnectionsIn = cell_metrics.synapticConnectionsIn(1:cell_metrics.general.cellCount);

% Connection strength
disp('  Determining transmission probabilities')
ccg2 = mono_res.ccgR;
ccg2(isnan(ccg2)) =  0;

for i = 1:size(cell_metrics.putativeConnections.excitatory,1)
    [trans,prob,prob_uncor,pred] = ce_GetTransProb(ccg2(:,cell_metrics.putativeConnections.excitatory(i,1),cell_metrics.putativeConnections.excitatory(i,2)),  mono_res.n(cell_metrics.putativeConnections.excitatory(i,1)),  mono_res.binSize,  0.020);
    cell_metrics.putativeConnections.excitatoryTransProb(i) = trans;
end

% Inhibitory connections
for i = 1:size(cell_metrics.putativeConnections.inhibitory,1)
    [trans,prob,prob_uncor,pred] = ce_GetTransProb(ccg2(:,cell_metrics.putativeConnections.inhibitory(i,1),cell_metrics.putativeConnections.inhibitory(i,2)), mono_res.n(cell_metrics.putativeConnections.inhibitory(i,1)),  mono_res.binSize,  0.020);
    cell_metrics.putativeConnections.inhibitoryTransProb(i) = trans;
end
%% Copied from Cell type classification 
waveform_metrics = calc_waveform_metrics(spikes,Fs,'showFigures',false);
cell_metrics.troughToPeak = waveform_metrics.troughtoPeak;


spikes.numcells=numel(spikes.cluID);
spikes.total=mono_res.n;
acg_metrics = calc_ACG_metrics(spikes,Fs,'showFigures',false);
fit_params = fit_ACG(acg_metrics.acg_narrow,false);
cell_metrics.acg_tau_rise = fit_params.acg_tau_rise;


preferences.putativeCellType.troughToPeak_boundary=0.425;
preferences.putativeCellType.acg_tau_rise_boundary=6;
cell_metrics.putativeCellType =celltype_classification.standard(cell_metrics,preferences);

% Addition based on excitatory vs inhibitory (not connections but based on
% whisker stimulation)
idx_act = (response_mask == 1);
idx_inhib = (response_mask == -1);
acg_tau_rise_excit = cell_metrics.acg_tau_rise(idx_act);
acg_tau_rise_inhib = cell_metrics.acg_tau_rise(idx_inhib);
troughtopeak_excit = cell_metrics.troughToPeak(idx_act);
troughtopeak_inhib = cell_metrics.troughToPeak(idx_inhib);
type_excit = cell_metrics.putativeCellType(idx_act);
type_inhib = cell_metrics.putativeCellType(idx_inhib);

% scatter plot for cell type
scatter(troughtopeak_excit,acg_tau_rise_excit,28,'r','filled')
hold on;
scatter(troughtopeak_inhib,acg_tau_rise_inhib,28,'b','filled');
xlabel('Trough to Pk (ms)');
ylabel('\tau_{rise} (ms)');
xline(0.425,"--",'Wide Interneuron','Color','y','LineWidth',2.5);
vec_line_x = linspace(0.425,1,10);
vec_line_y = 6*ones(1,10);
plot(vec_line_x,vec_line_y,'--','Color','g','LineWidth',2.5);
text(0.7,5.75,'Pyramidal neuron')
text(0.15,14,'Narrow Interneuron');
axis([0.1,1,-inf,inf]);
filename = fullfile(plotfolder,'cell_type_analysis.png');
print(filename,'-dpng','-r0');
filename = fullfile(plotfolder,'pop_celltypes.mat');
save(filename,'type_excit','type_inhib');
%{
%% Copied from Graph Plotting example script Getting cell type groups (for coloring notes)
cellTypes = unique(cell_metrics.putativeCellType,'stable');
clusClas = ones(1,length(cell_metrics.putativeCellType));
for i = 1:length(cellTypes)
    clusClas(strcmp(cell_metrics.putativeCellType,cellTypes{i}))=i;
end

% Getting connection pairs and cells with connections
putativeConnections = cell_metrics.putativeConnections.excitatory;
putativeConnections_inh = cell_metrics.putativeConnections.inhibitory;
[cellSubset,~,pairsSubset] = unique([putativeConnections;putativeConnections_inh]);
pairsSubset = reshape(pairsSubset,size([putativeConnections;putativeConnections_inh]));

% Generating connectivity matrix (A)
A = zeros(length(cellSubset),length(cellSubset));
for i = 1:size(putativeConnections,1)
    A(pairsSubset(i,1),pairsSubset(i,2)) = 1;
end
for i = size(putativeConnections,1)+1:size(pairsSubset,1)
    A(pairsSubset(i,1),pairsSubset(i,2)) = 2;
end

%% Plotting connectivity matrix
% figure, subplot(1,2,1)
% imagesc(A), title('Connectivity matrix')

% Generating connectivity graph (only for subset of cells with connections)
connectivityGraph = digraph(A);

% count statistics
connectivityCountByShank = zeros(4,1);
for i = 1:size(connectivityGraph.Edges, 1)
    tmp = connectivityGraph.Edges{i, "EndNodes"};
    src_unit = cellSubset(tmp(1));
    src_shanknum = round(clus_locations(src_unit,1)/GW_BETWEENSHANK)+1;
%     disp(src_shanknum)
    snk_unit = cellSubset(tmp(2));
    snk_shanknum = round(clus_locations(snk_unit,1)/GW_BETWEENSHANK)+1;
    if (src_shanknum == snk_shanknum)
        connectivityCountByShank(src_shanknum) = connectivityCountByShank(src_shanknum) + 1;
    end
end


% Plotting connectivity graph
% geom = csvread("map.csv");
% subplot(1,2,2)
% figure()
figure('Renderer', 'painters')%, 'Position', [10 10 100 900])
% connectivityGraph_plot = plot(connectivityGraph,'Layout','force','Iterations',500 ,'UseGravity',true,'MarkerSize',6,'EdgeCData',connectivityGraph.Edges.Weight,'HitTest','off','EdgeColor',[0.2 0.2 0.2],'NodeLabel',num2cellstr(cellSubset));
connectivityGraph_plot = plot(...
    connectivityGraph,...
    'MarkerSize',2,...
    'EdgeColor',[0.2 0.2 0.2],...
    'NodeLabel', {});
connectivityGraph_plot.XData=clus_locations(cellSubset,1)+80*(rand(length(cellSubset),1)-0.5);
connectivityGraph_plot.YData=clus_locations(cellSubset,2);
% layout(connectivityGraph_plot, "layered");
% connectivityGraph_plot.NodeLabel = {};

% for i=1:32
%     geom(i, 1), geom(i, 2)
% end
set(gca, 'YDir','reverse')
axis("on")
% viscircles(geom, 12.5*ones(32,1),'Color','k', 'LineWidth', 0.5)
xlim([-100 3*GW_BETWEENSHANK+100])
ylim([-30 31*GH+30])
% pbaspect([55 655 1])
title(sprintf('%s\nA%d B%d C%d D%d', ...
    datestr,...
    connectivityCountByShank(1),...
    connectivityCountByShank(2),...
    connectivityCountByShank(3),...
    connectivityCountByShank(4) ...
    ))

% Coloring nodes by cell types
classes2plotSubset = [-1 0 1];
colors = [[0,1,0];[.5,.5,.5];[1,0,0]];
for k = 1:length(classes2plotSubset)
    idx3 = find(response_mask(cellSubset)==classes2plotSubset(k));
    if ~isempty(idx3)
        highlight(connectivityGraph_plot,idx3,'NodeColor',colors(k,:))
    end
end
savefig([plotfolder datestr '.fig']);
% savefig('D:\Rice-Courses\neuroeng_lab\codes\stroke_proj_connec\plots\NVC\BC7\01-21-2022.fig')
saveas(gcf, [plotfolder datestr '.png']);
% figure
% gui_MonoSyn(mono_res) % Shows the GUI for manual curation
%}