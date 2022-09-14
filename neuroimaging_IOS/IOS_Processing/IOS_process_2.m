% This code will perform multiple post-processing on the deltaR/R values
% computed in process_1.m code. 

% 1. Beer Lambert Law and Least Squares to find haemoglobin concentrations
% 2. Temporal response curves after ROI selection
% 3. Spatial response plot after peak response frames selection

% Future additions:
% 1. Time to peak of the response curve
% 2. Width of the response curve
% 3. Time to start of the response curve
% 4. Activation map area from IOS (spatial gaussian filtered image)
% 4. cross-correlation (correlation distance) b/w different ROI-time-series
% 5. Mutual information + joint entropy + relative entropy
% 6. cross-correlation functional connectivity plot - image voxels

clear all;

source_dir = input('Please enter the source directory:\n','s');
% selpath = uigetdir('Select source directory containing scans');
output_dir = fullfile(source_dir,'Processed');
output_plot = fullfile(output_dir,'Hb_Conc');
mat_dir = fullfile(output_dir,'mat_files');
if ~exist(output_plot, 'dir')
   mkdir(output_plot)
end
% whiskerStim_txt = fullfile(source_dir,'whisker_stim.txt');
trigger_Hb = input('Specify whether you want to compute [HbO],[HbR]? (y/N) \n','s');

% Experimental Parameters:
load(fullfile(mat_dir,'ROI.mat'));

% Extracting data from process_1.m
load(fullfile(mat_dir,'data_blue.mat'));
load(fullfile(mat_dir,'data_amber.mat'));
load(fullfile(mat_dir,'data_wv3.mat'));
I_ratio_blue = Avg_blue;
I_ratio_amber = Avg_amber;
I_ratio_wv3 = Avg_wv3;

clear Avg_amber Avg_blue Avg_wv3

%% Hb concentrations : This code is optional

if trigger_Hb == 'y'
    % initializing cell
    delta_c_average = cell(len_trials,1);
    HbO = zeros(len_trials,X,Y);
    HbR = zeros(len_trials,X,Y);
    HbT = zeros(len_trials,X,Y);
    HbO_pk = zeros(X,Y);
    HbR_pk = zeros(X,Y);
    HbT_pk = zeros(X,Y);

    % Compute change in concentrations
    for iter_seq = [1:len_trials]
        I_ratio_blue_temp = reshape(I_ratio_blue(iter_seq,:,:),[X,Y]);
        I_ratio_amber_temp = reshape(I_ratio_amber(iter_seq,:,:),[X,Y]);
        I_ratio_wv3_temp = reshape(I_ratio_wv3(iter_seq,:,:),[X,Y]);
        if wv_3 == 510 
            delta_c_average{iter_seq} = HyperSpectral_haemoglobin_track_BYG(I_ratio_blue_temp,I_ratio_amber_temp,I_ratio_wv3_temp);
        elseif wv_3 == 632
            delta_c_average{iter_seq} = HyperSpectral_haemoglobin_track_BYR(I_ratio_blue_temp,I_ratio_amber_temp,I_ratio_wv3_temp);
        else
            error('incorrect wavelength selected! \n');
        end
        HbO(iter_seq,:,:) = 1e6 * real(delta_c_average{iter_seq}(:,:,1));
        HbR(iter_seq,:,:) = 1e6 * real(delta_c_average{iter_seq}(:,:,2));
        HbT(iter_seq,:,:) = HbO(iter_seq,:,:) + HbR(iter_seq,:,:);

    end

    clear delta_c_average
    delta_c_average = cell(length(pk_indx),1);

    % Changes in conc. for 2D spatially filtered data
    for iter_seq = 1:length(pk_indx)
        I_ratio_blue_temp = reshape(gauss_filtered_blue_pk(iter_seq,:,:),[X,Y]);
        I_ratio_amber_temp = reshape(gauss_filtered_amber_pk(iter_seq,:,:),[X,Y]);
        I_ratio_wv3_temp = reshape(gauss_filtered_wv3_pk(iter_seq,:,:),[X,Y]);
        if wv_3 == 510 
            delta_c_average{iter_seq} = HyperSpectral_haemoglobin_track_BYG(I_ratio_blue_temp,I_ratio_amber_temp,I_ratio_wv3_temp);
        elseif wv_3 == 632
            delta_c_average{iter_seq} = HyperSpectral_haemoglobin_track_BYR(I_ratio_blue_temp,I_ratio_amber_temp,I_ratio_wv3_temp);
        else
            error('incorrect wavelength selected! \n');
        end
    end
    tmp_hbo = zeros(X,Y);
    tmp_hbr = zeros(X,Y);
    for iter_seq = 1:length(pk_indx)
        tmp_hbo = tmp_hbo + delta_c_average{iter_seq}(:,:,1);
        tmp_hbr = tmp_hbr + delta_c_average{iter_seq}(:,:,2);
    end
    HbO_pk = 1e6 * tmp_hbo/length(pk_indx);
    HbR_pk = 1e6 * tmp_hbr/length(pk_indx);
    HbT_pk = HbO_pk + HbR_pk;

    save(fullfile(mat_dir,'delta_c.mat'),'HbO','HbR','HbT','HbO_pk','HbR_pk','HbT_pk');
    
    % Plotting
    plotter_Hb_conc(HbO,output_plot,'HbO',[-12 12]);
    plotter_Hb_conc(HbR,output_plot,'HbR',[-5 5]);
    plotter_Hb_conc(HbT,output_plot,'HbT',[-10 10]);
    fg = imshow(HbO_pk,[0 12]);
    colormap('jet');
    colorbar;
    name_iter = fullfile(output_plot,strcat('pk-response-HbO','.svg'));
    saveas(fg,name_iter,'svg');
    fg = imshow(HbR_pk,[-5 10]);
    colormap('jet');
    colorbar;
    name_iter = fullfile(output_plot,strcat('pk-response-HbR','.svg'));
    saveas(fg,name_iter,'svg');
    fg = imshow(HbT_pk,[-10 10]);
    colormap('jet');
    colorbar;
    name_iter = fullfile(output_plot,strcat('pk-response-HbT','.svg'));
    saveas(fg,name_iter,'svg');

end

% performing spatial average for deltaR/R
SA_480 = reshape(mean(gauss_filtered_blue_pk,1,'omitnan'),[X,Y]);
SA_580 = reshape(mean(gauss_filtered_amber_pk,1,'omitnan'),[X,Y]);
SA_wv3 = reshape(mean(gauss_filtered_wv3_pk,1,'omitnan'),[X,Y]);

outputFileName = fullfile(output_dir_avg,'gauss_avg_480.svg'); 
fg = figure();
imshow(SA_480-1,[-0.03 0.005]);
colorbar
colormap('jet');
saveas(fg,outputFileName);
close(fg);
outputFileName = fullfile(output_dir_avg,'gauss_avg_580.svg'); 
fg = figure();
imshow(SA_580-1,[-0.04 0.005]);
colorbar
colormap('jet');
saveas(fg,outputFileName);
close(fg);

if wv_3 == 510
    outputFileName = fullfile(output_dir_avg,'gauss_avg_510.svg'); 
    fg = figure();
    imshow(SA_wv3-1,[-0.03 0.005]);
    colorbar
    colormap('jet');
    saveas(fg,outputFileName);
    close(fg);
elseif wv_3 == 632
    outputFileName = fullfile(output_dir_avg,'gauss_avg_632.svg'); 
    fg = figure();
    imshow(SA_wv3-1,[-0.005 0.01]);
    colorbar
    colormap('jet');
    saveas(fg,outputFileName);
    close(fg);
else 
    error('incorrect wavelength selected! \n');
end

%% Time-Series analysis and Area analysis

load(fullfile(mat_dir,'ROI.mat'));
load(fullfile(mat_dir,'data_blue.mat'));
load(fullfile(mat_dir,'data_amber.mat'));
load(fullfile(mat_dir,'data_wv3.mat'));

if trigger_Hb == 'y'
    % call time series extract function here
    [TS_HbO,TS_HbR,TS_HbT] = Time_Series_extract(HbO,HbR,HbT,mask,len_trials,X,Y);
    output_file = fullfile(mat_dir,'TimeS_Hb.mat');
    save(output_file,'TS_HbO','TS_HbR','TS_HbT','HbO_pk','HbR_pk','HbT_pk');
end

% time series analysis on the raw I/I_0 values
Avg_blue = Avg_blue-1;
Avg_amber = Avg_amber-1;
Avg_wv3 = Avg_wv3-1;
% call time series extract function here
[TS_480,TS_580,TS_wv3] = Time_Series_extract(Avg_blue,Avg_amber,Avg_wv3,mask,len_trials,X,Y);
output_file = fullfile(mat_dir,'TimeS_dR.mat');
save(output_file,'TS_480','TS_580','TS_wv3','SA_480','SA_580','SA_wv3');



