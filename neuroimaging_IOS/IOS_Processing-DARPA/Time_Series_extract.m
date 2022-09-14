function [ROI_time_series_imgA,ROI_time_series_imgB,ROI_time_series_imgC,names] = Time_Series_extract(img_stackA,img_stackB,img_stackC,mask,len_trials,X,Y)
%TIME_SERIES_EXTRACT Summary of this function goes here
%   Detailed explanation goes here
% time series analysis on the raw I/I_0 values



% Introduce NaN for computational purposes
ROI_NaN = double(mask);
ROI_NaN(ROI_NaN == 0) = NaN;
% Generate name labels for the ROIs
N_ROI = size(mask,3);
names = generateROILabels(N_ROI);
ROI_time_series_imgA = zeros(len_trials,N_ROI);
ROI_time_series_imgB = zeros(len_trials,N_ROI);
ROI_time_series_imgC = zeros(len_trials,N_ROI);

for iter_seq = 1:len_trials
    for iter_ROI = 1:N_ROI
        ROI_time_series_imgA(iter_seq,iter_ROI) = mean(reshape(img_stackA(iter_seq,:,:),[X,Y]).*reshape(ROI_NaN(:,:,iter_ROI),[X,Y]),'all','omitnan');
        ROI_time_series_imgB(iter_seq,iter_ROI) = mean(reshape(img_stackB(iter_seq,:,:),[X,Y]).*reshape(ROI_NaN(:,:,iter_ROI),[X,Y]),'all','omitnan');
        ROI_time_series_imgC(iter_seq,iter_ROI) = mean(reshape(img_stackC(iter_seq,:,:),[X,Y]).*reshape(ROI_NaN(:,:,iter_ROI),[X,Y]),'all','omitnan');
    end
end
end

