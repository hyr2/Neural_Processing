function [ROI_time_series_img,names] = Time_Series_extract_single(img_stack,mask,len_trials,X,Y)
%TIME_SERIES_EXTRACT Summary of this function goes here
%   Detailed explanation goes here
% time series analysis on the raw I/I_0 values



% Introduce NaN for computational purposes
ROI_NaN = double(mask);
ROI_NaN(ROI_NaN == 0) = NaN;
% Generate name labels for the ROIs
N_ROI = size(mask,3);
names = generateROILabels(N_ROI);
ROI_time_series_img = zeros(len_trials,N_ROI);

for iter_seq = 1:len_trials
    for iter_ROI = 1:N_ROI
        ROI_time_series_img(iter_seq,iter_ROI) = mean(reshape(img_stack(iter_seq,:,:),[X,Y]).*reshape(ROI_NaN(:,:,iter_ROI),[X,Y]),'all','omitnan');       
    end
end
end

