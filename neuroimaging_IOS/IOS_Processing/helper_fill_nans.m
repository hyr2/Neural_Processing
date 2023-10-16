function [area_out] = helper_fill_nans(xaxis_ideal,xaxis_local,area_local)
% Assumes first element is the baseline
% xaxis_local(1) = -3;
area_local([1]) = [];
area_local(1) = 1;

area_out = zeros(size(xaxis_ideal));
area_out(ismember(xaxis_ideal,xaxis_local)) = area_local;
area_out(~ismember(xaxis_ideal,xaxis_local)) = nan;    

end

