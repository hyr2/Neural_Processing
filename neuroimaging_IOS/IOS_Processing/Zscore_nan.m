function [output_arr] = Zscore_nan(input_arr)
%ZSCORE_NAN Summary of this function goes here
%   Detailed explanation goes here

mean_local = mean(input_arr,'all','omitnan');
std_local = std(input_arr,0,'all','omitnan');

output_arr = (input_arr - mean_local)/std_local;

end

