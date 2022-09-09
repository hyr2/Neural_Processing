% Script for extracting data from CCG

% 1. Synchrony


load('/home/hyr2/Documents/Data/BC7/2021-12-09/ePhys/Processed/Connectivity/changing-sigWindow/4ms/mono_res_stim.mat');

% Set these parameters:
t_window = 15e-3;                              % Time window for synchrony (7.5 ms typical)

% Setting up:
mono_res = mono_res_stim;
binSize = mono_res.binSize;                     % The bin size used in the binning of spikes 
CCGR = mono_res.ccgR;                           % The cross-correlogram histogram
halfBins = round(mono_res.duration/binSize/2);  
t = (-halfBins:halfBins)'*binSize;              % The time axis in seconds
n = mono_res.n;                                 % The number of spikes of each cluster

% Output 2D matrix:
sz_matrix_out = size(CCGR,2);
sz_matrix_out = sz_matrix_out^2 - sz_matrix_out;% Considering only the non-diagonal synchronies (b/c diagonal is ACG)
out_sync = struct;                              % Output struct of all the synchrony values
vec_synch_out = zeros(sz_matrix_out,1);         
ref = zeros(sz_matrix_out,1);
trg = zeros(sz_matrix_out,1);

% CCGR is 3D. 1st dimension is time axis. The other two dimensions form a
% square matrix consisiting of values of the ccg histogram counts
iter_local = 1;
for iterx = 1:size(CCGR,2)
    for itery = 1:size(CCGR,3)
        if (iterx ~= itery)
            local_ccg = CCGR(:,iterx,itery);
            n_local = [n(iterx),n(itery)];
            [synch,p] = synchrony_compute(local_ccg,n,t,t_window);  % Computing synchrony value
            
            vec_synch_out(iter_local) = synch;
            ref(iter_local) = iterx;
            trg(iter_local) = itery;
            
            iter_local = iter_local + 1;
        end
    end
end

out_sync.synchrony = vec_synch_out;
out_sync.ref = ref;
out_sync.trg = trg;
            
            
            