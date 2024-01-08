% Wraper for calc_ACG_metrics used for plotting ACGs of longitudinal
% tracked Single Units
function [acg_out] = wraper_calc_ACG_metrics(spikes,FR)
    % spikes : the spike time stamps in samples
    % FR : The sampling rate used
    close all;
   
    folder = '/home/hyr2-office/Documents/git/CellExplorer/';       % cell explorer repo has been cloned locally to call its functions from this script
    addpath(genpath(folder));
    
    spikes = cast(spikes,'double');
    spikes = spikes/FR;     % in seconds
    acg_out = CCG(spikes,ones(size(spikes)),'binSize',0.0005,'duration',0.100,'norm','rate','Fs',1/FR);
    acg_out = acg_out';
    

end