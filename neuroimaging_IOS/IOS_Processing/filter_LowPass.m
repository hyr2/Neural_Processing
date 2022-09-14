function [output_cell_array] = filter_LowPass(Data_cell,lowpass_filter)
%FILTER_LOWPASS Performs low pass filtering on the cell array
%   The cell array has its row index as trial and column index as frame.
%   Each unit of the cell array is a raw (X,Y) intrinsic optical image of one
%   particular wavelength

[num_trials, len_trials] = size(Data_cell);
[X,Y] = size(Data_cell{1,1});
mat_single_trial = zeros(len_trials,X,Y);
mat_all_trials = zeros(num_trials,len_trials,X,Y);

for iter_trial = 1:num_trials
    for iter_seq = 1:len_trials
        mat_single_trial(iter_seq,:,:) = Data_cell{iter_trial,iter_seq};
    end
    % Perform filering
    mat_single_trial_filtered = filtfilt(lowpass_filter,mat_single_trial);
    % Storing back into 4D array 
    mat_all_trials(iter_trial,:,:,:) = mat_single_trial_filtered;
    clear mat_single_trial_filtered mat_single_trial
end

output_cell_array = mat_all_trials;

end

