function [Synchrony,p] = synchrony_compute(local_ccg,n,t,t_window)
%SYNCHRONY_COMPUTE Computes the synchrony value between two clusters
%cross-correlogram histogram
%   Synchrony is a measure of connetivity between spiking data and tells us
%   how synchronized was the firing of two neurons in time. For more
%   information about how its is computed, refer to supplemental note 4 of: 
%   https://static-content.springer.com/esm/art%3A10.1038%2Fnn.2670/MediaObjects/41593_2010_BFnn2670_MOESM53_ESM.pdf
%   
%   INPUT:
%
%           local_ccg: 1D cross-correlogram histogram between two clusters
%           n: two element vector containing the number of spikes in each cluster
%           t: 1D time vector (x axis of histogram)
%           t_window: The synchrony time window 

% Numerator:
indx = find(abs(t)<t_window);
numerator_synch = sum(local_ccg(indx),'all');       % Integrate the counts (sum the counts):
% Denominator:
denominator_synch = sqrt(dot(n,n)/2);               

Synchrony = numerator_synch/denominator_synch;
p = 0;
end

