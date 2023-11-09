function func_CE_BarrelCortex(datafolder,Fs,Chan_map_path)
%FUNC_CE_BARRELCORTEX is used to call cell explorer's core functions to
%perform cell type classification analysis based on firing properties of
%the putative neurons. In addition it also can perform connectivity
%analysis (monosynaptic as well as other metrics)

%   INPUT:
%          datafolder: single session ephys data. both
%          rickshaw_preprocessing_HR.py and rickshaw_postporcessing_full.py
%          should have been successfully run without any errors for this
%          code to work.
%          Fs: the sampling freq used during electrophysiological recording
%          Chan_map_path : full file path to the channel map for recording
%          device used

% Note: This code also requires the use of a 3rd party .npy file reader
% obtained here: https://github.com/kwikteam/npy-matlab. Make sure the
% functions from this repo are in the matlab path 

close all;

folder = '/home/hyr2-office/Documents/git/CellExplorer/';       % cell explorer repo has been cloned locally to call its functions from this script
addpath(genpath(folder));

% Fs = 30e3;
% datafolder = '/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/';
disp(datafolder)
% channel specific code. 2x16 vs 1x32 microelectrode array designs
json_file = fullfile(datafolder,'pre_MS.json');
fid = fopen(json_file); 
raw = fread(fid,inf); 
tmp_str = char(raw'); 
fclose(fid); 
json_struct = jsondecode(tmp_str);
chmap2x16 = json_struct.ELECTRODE_2X16;
if (chmap2x16 == true)     % 2x16 channel map
    GH = 30;
    GW_BWTWEENSHANK = 250;
elseif (chmap2x16 == false)  % 1x32 channel map
    GH = 25;
    GW_BWTWEENSHANK = 250;
end
plotfolder = fullfile(datafolder,'Processed','cell_type');
mkdir(plotfolder);
templates = readmda(fullfile(datafolder, 'templates_clean_merged.mda'));
firings = readmda(fullfile(datafolder, 'firings_clean_merged.mda'));
chmap_mat = load(Chan_map_path,'Ch_Map_new');
chmap_mat = chmap_mat.Ch_Map_new;
Native_orders = readNPY(fullfile(datafolder,'native_ch_order.npy'));    
curation_mask = logical(csvread(fullfile(datafolder, 'accept_mask.csv')));  % deparcated
pos_mask = not(logical(csvread(fullfile(datafolder , 'positive_mask.csv' )))); % deparcated
response_mask = csvread(fullfile(datafolder,'Processed','count_analysis','cluster_response_mask.csv')); 
clus_loc = csvread(fullfile(datafolder,'clus_locations_clean_merged.csv'));
% clus_locations = csvread(fullfile(datafolder, 'clus_locations.csv'));
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
spike_labels_unique = unique(spike_labels);
ch_stamp = firings(1,idx); %primary channel
spike_times_by_clus = cell(1, n_clus);
ts_by_clus = cell(1, n_clus);
pri_ch_lut = -1*ones(1, n_clus);
for i=1:n_clus
    spike_times_by_clus{i} = [];
    ts_by_clus{i} = [];
end

n_pri_ch_known = 0;
% count spikes by unit : curation
for i=1:length(spike_times_all)
    spk_lbl = spike_labels(i);
    if pri_ch_lut(spk_lbl) == -1
        pri_ch_lut(spk_lbl) = ch_stamp(i);
        n_pri_ch_known = n_pri_ch_known + 1;
            if n_pri_ch_known == n_clus
                break;
            end
    end
    spike_times_by_clus{spk_lbl}(end+1) = spike_times_all(i)/Fs;
    ts_by_clus{spk_lbl}(end+1) = spike_times_all(i);
end

% Locating the cluster (shank ID + location)
if min(chmap_mat(:)) == 1
    disp("Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
    chmap_mat = chmap_mat - 1;
end
shank_num = int8 (-1 * ones(1,n_clus));
for i_clus = 1:n_clus   % each cluster in mountainsort output is assigned a shank here
    % prim_ch = pri_ch_lut(i_clus);   % this is the Mountainsort channel ID (starts from 1)]
%     disp(get_shanknum_from_msort_id(prim_ch,Native_orders,chmap_mat,chmap2x16))
    % shank_num(i_clus) = int8(get_shanknum_from_msort_id(prim_ch,Native_orders,chmap_mat,chmap2x16)); % function used to get shank ID  
    shank_num(i_clus) = idivide(int16(clus_loc(i_clus,1)),250) + 1;         % Starts from 1 to 4 
end

% More curation code
curation = 1;
if curation
    curation_mask = and(curation_mask,pos_mask);
    curation_mask = ones(size(spike_labels_unique));
    % extract accepted cluster locations & their responsiveness
    % response_mask = response_mask(curation_mask,:);   % Update to the response mask from python script population_analysis.py
    % clus_locations = clus_locations(curation_mask,:);
    % shank_num = shank_num(curation_mask);
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
          filtWaveform{spike_labels_mapping(i)} = templates(pri_ch_lut(i), [10:90], i);
%             filtWaveform{spike_labels_mapping(i)} = templates(pri_ch_lut(i), [2:52], i);    % for 25 KHZ
          timeWaveform{spike_labels_mapping(i)} = [-40:40]/Fs*1000;
%             figure;plot(timeWaveform{spike_labels_mapping(i)},filtWaveform{spike_labels_mapping(i)})
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
mono_res = ce_MonoSynConvClick(spikes); %,'includeInhibitoryConnections',false); % detects the monosynaptic connections
toc
%% Copied from ProcessCellMetrics ,section Putative MonoSynaptic connections

% save('mono_res.cellinfo.mat','mono_res','-v7.3','-nocompression');
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
cell_metrics.derivative_TroughToPeak = waveform_metrics.derivative_TroughtoPeak;
cell_metrics.assymetry = waveform_metrics.ab_ratio;
troughToPeak = cell_metrics.troughToPeak;
derivative_TroughtoPeak = cell_metrics.derivative_TroughToPeak;
assymetry = cell_metrics.assymetry;

spikes.numcells=numel(spikes.cluID);
spikes.total=mono_res.n;
acg_metrics = calc_ACG_metrics(spikes,Fs,'showFigures',false);
fit_params = fit_ACG(acg_metrics.acg_narrow,false);
cell_metrics.acg_tau_rise = fit_params.acg_tau_rise;

preferences.putativeCellType.troughToPeak_boundary=0.425;       % From CE website (edited for Barrel Cortex)
preferences.putativeCellType.acg_tau_rise_boundary=6;           % From CE website
cell_metrics.putativeCellType = celltype_classification.standard(cell_metrics,preferences);
celltype = cell_metrics.putativeCellType;

% Busrting and theta modulation index
burstIndex_Royer2012 = acg_metrics.burstIndex_Royer2012;
% burstIndex_Doublets = acg_metrics.burstIndex_Doublets;
thetaModulationIndex = acg_metrics.thetaModulationIndex;
% Tau rise time
tau_rise = fit_params.acg_tau_rise;
% Addition based on excitatory vs inhibitory (based on waveform shape which is same as CCG based)
% values modified for Barrel cortex data from the original value of 0.55ms
type_excit = (cell_metrics.troughToPeak > 0.425);    % Buzsaki lab (https://www.cell.com/neuron/pdfExtended/S0896-6273(18)31085-7)
type_inhib = (cell_metrics.troughToPeak <= 0.425);

idx_act = (type_excit == 1);     
idx_inhib = (type_inhib == 1);  
acg_tau_rise_excit = cell_metrics.acg_tau_rise(idx_act);
acg_tau_rise_inhib = cell_metrics.acg_tau_rise(idx_inhib);
troughtopeak_excit = cell_metrics.troughToPeak(idx_act);
troughtopeak_inhib = cell_metrics.troughToPeak(idx_inhib);

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
save(filename,'type_excit','type_inhib','troughToPeak','assymetry','celltype','shank_num','burstIndex_Royer2012','tau_rise','thetaModulationIndex'); % all these quantities are only for the curated clusters
% type_err = (cell_metrics.troughToPeak <= 0.2);

disp(datafolder);
% disp(sum(type_err));

end

