% The main code that does the following operations:
% 1. Bins the data into wavelengths
% 2. Temporal low pass filtering of data
% 3. Spatial gaussian filtering of data
% 4. Bins the data into trials
% 5. Generate video of the activation
% 6. Selection of 3rd wavelength: 632 nm vs 510 nm
% 7. Skip first 6 trials and other bad trials (read from trials_reject.mat)
% To Do:
% 1. Binarization and thresholding to remove vessel activity to focus on
% the tissue response
% 2. Export mask overlayed showing ROIs
% 3. Export single trial datasets into mat files


%% data dir
% source_dir
% |
% |__Raw/
% |__whisker_stim.txt

clear all;

source_dir = '/home/hyr2/Documents/Data/MW-LSCI1/IOS/';
% selpath = uigetdir('Select source directory containing scans');
output_dir = fullfile(source_dir,'Processed');
output_dir_avg = fullfile(output_dir,'Average');
mat_dir = fullfile(output_dir,'mat_files');
single_trial_dir = fullfile(output_dir,'single_trial');
whiskerStim_txt = fullfile(source_dir,'whisker_stim.txt');
Raw_dir = fullfile(source_dir,'Raw');
if ~exist(output_dir, 'dir')
   mkdir(output_dir)
end
if ~exist(output_dir_avg, 'dir')
   mkdir(output_dir_avg)
end
if ~exist(mat_dir, 'dir')
   mkdir(mat_dir)
end
if ~exist(single_trial_dir, 'dir')
   mkdir(single_trial_dir)
end
files_raw = dir_sorted(fullfile(Raw_dir, '*.tif'));

% wavelength selection
wv_3 = input('The 3rd wavelength was 510nm or 632nm ? \n','s');
wv_3 = int16(str2num(wv_3));

if wv_3 == 510
    disp('Selected wavelengths are 480 nm, 580 nm and 510 nm');
elseif wv_3 == 632
    disp('Selected wavelengths are 480 nm, 580 nm and 632 nm');
else
    error('incorrect wavelength selected! \n')
end

cam_in = input('Is this the Andor Camera? [1,0]\n', 's');
cam_in = int16(str2num(cam_in));

% Experimental Parameters:
number_wavelength = 3;          % Enter Number of wavelengths here
[len_trials,FramesPerSeq, stim_start_time, seq_period, stim_num, num_trials] = read_whiskerStimTxT(whiskerStim_txt);
start_time = stim_start_time*1000; % ms
duration_time = seq_period*1000; % ms

% Checking total scans for all trails
N_scans = int16(num_trials*len_trials*number_wavelength);
n_scans = int16(length(files_raw));
assert(N_scans == n_scans);

% other settings for the code
start_trails = 5;  % disregard the first several trails
% calculate the start time
start_t = start_time/1000; %second
end_t = (start_time + duration_time*stim_num)/1000;
% strat stimulation time
start_stimulate = floor(start_time/duration_time);
baseline_end1 = start_stimulate-2; % baselines before the stimulation


len_trials_all_Lambda = number_wavelength * len_trials;
start_trial = [1:len_trials_all_Lambda:num_trials*len_trials_all_Lambda];

% %% ROI Selection
sample_img = imread(fullfile(Raw_dir,files_raw(2).name));
sample_img = mat2gray(sample_img);
if cam_in == 1
    sample_img = image_transform_B(sample_img);
else
    sample_img = image_transform_A(sample_img);
end
amax = max(sample_img(:));
amin = min(sample_img(:));
[X,Y] = size(sample_img);
% If ROI argument is not provided, prompt user to define
ROI_selector = input('Do you want to use a pre-defined ROI? [1,0] \n');
ROI_selector = logical(ROI_selector);
if ROI_selector
    ROI_loc = input('Path to ROI.mat file\n','s');
    ROI_loc = fullfile(ROI_loc,'ROI.mat');
    load(ROI_loc,'mask','BW');
  if size(mask, 1) ~= X || size(mask, 2) ~= Y
    error('ROI dimensions (%dx%d) do not match intrinsic data (%dx%d)', size(ROI,1), size(ROI, 2), X, Y);
  end
else
    mask = drawROIs(sample_img,[amin,amax]);
    % Define a craniotomy
    BW = defineCraniotomy(sample_img,[amin amax]);
end
N_ROI = size(mask,3);

% *** Code has been changed **** 
% important processing parameters
sigma_2d = 5;   % spatial filtering width of 2d gaussian
order = 4;      % Order of the IIR filter
pk_indx = [int16(start_stimulate+1.8/seq_period):stim_num + start_stimulate];   % peak time extraction
bsl_indx = [order+1:start_stimulate-1];                                         % baseline time extraction
bsl_indx = [50:150]
pk_indx = [170:180]
trials_reject = [1:7,9];             % must be loaded from the file
% trials_reject = horzcat([1:start_trails],trials_reject);

% save parameters and ROI masks to mat_dir  ********************************
eff_num_trials = num_trials - length(trials_reject);
eff_num_trials = eff_num_trials; % This is extra
save(fullfile(mat_dir,'ROI.mat'));
global_t_stack = zeros(eff_num_trials,len_trials,number_wavelength);

% Dry Run switch
dry = input('Quick run for trial summary? [0/1]\n');
dry = logical(dry);

%% Blue 480 nm
output_video = fullfile(output_dir_avg,'480nm');
if ~exist(output_video, 'dir')
   mkdir(output_video)
end
Data_all_blue = cell(eff_num_trials,len_trials);
% TS_480 = zeros(eff_num_trials,len_trials,N_ROI);  % array for single trial data storage
% Loop over trials
iter_trial_local = 0;
for iter_trial = 1:num_trials
    temp_start = start_trial(iter_trial); 
    if any(trials_reject == iter_trial)
        continue;
    else
        iter_trial_local = iter_trial_local + 1;
    end
    % Reading wavelength 1 data: blue 480 nm
    for iter_seq = 1:len_trials
        index = temp_start + (iter_seq-1)*number_wavelength;
        file = files_raw(index).name;
        filePath = fullfile(Raw_dir,file);
        Data = imread(filePath);
        if cam_in == 1
            rawData = image_transform_B(Data);
        else
            rawData = image_transform_A(Data);
        end
        rawData(BW == false) = 0;           % applying craniotomy
        global_t_stack(iter_trial_local,iter_seq,1) = mean(rawData(:));
        Data_all_blue{iter_trial_local,iter_seq}=rawData;
    end
end

if ~dry
    % Spatial Gaussian Filtering (sigma = 40 um)
    d2_spatial_avg_mat = spatial_gauss_filt(Data_all_blue(:,pk_indx),sigma_2d);
    d2_spatial_avg_bsl = spatial_gauss_filt(Data_all_blue(:,bsl_indx),sigma_2d);
    d2_spatial_avg_bsl = reshape(mean(d2_spatial_avg_bsl,2,'omitnan'),[eff_num_trials,X,Y]);
    % compute deltaR/R
    for iter_trial = 1:eff_num_trials
        for iter_seq = 1:length(pk_indx)
            bsl_blue_local = d2_spatial_avg_bsl(iter_trial,:,:);
            bsl_blue_local = reshape(bsl_blue_local,[X,Y]);
            d2_spatial_avg_mat(iter_trial,iter_seq,:,:) = (reshape(d2_spatial_avg_mat(iter_trial,iter_seq,:,:),[X,Y]))./bsl_blue_local;
        end
    end
    % averaging over trials
    gauss_filtered_blue_pk = reshape(mean(d2_spatial_avg_mat,1,'omitnan'),[length(pk_indx),X,Y]);

    % GSR (Global Signal Regression)
    % Due to memory constraints, the loops are run separately.
    % for iter_trial=1:num_trials
    %     R_data_blue{1,iter_trial} = GSR_main(Data_all_blue(iter_trial,:),global_t_stack(iter_trial,:,1));   % 480 nm
    % end
    % clear Data_all_blue

    % Low Pass filtering 
    fcutlow = 2;    % cutoff frequency
    Fs = 1/seq_period;        % Sampling frequency
    lowpass_filter = designfilt('lowpassiir', 'FilterOrder', order, ...
                                'HalfPowerFrequency', fcutlow, ...
                                'SampleRate', Fs, 'DesignMethod', ...
                                'butter');
    R_data_blue = filter_LowPass(Data_all_blue,lowpass_filter);
    Avg_blue = zeros(len_trials,X,Y);
    % Trial Averaging and I/I_0
    for iter_trial = 1:eff_num_trials
        bsl_blue_local = R_data_blue(iter_trial,bsl_indx,:,:);
        bsl_blue_local = reshape(mean(bsl_blue_local,2,'omitnan'),X,Y);
        for iter_seq = 1:len_trials
            R_data_blue(iter_trial,iter_seq,:,:) = (reshape(R_data_blue(iter_trial,iter_seq,:,:),X,Y))./bsl_blue_local;
        end
        % Storing individual trials data (but time series extracted due to
        % storage constraints):
%         [TS_480_tmp,~] = Time_Series_extract_single(reshape(R_data_blue(iter_trial,:,:,:),[len_trials,X,Y]),mask,len_trials,X,Y);
%         TS_480(iter_trial,:,:) = TS_480_tmp;
    end
    for iter_seq = 1:len_trials             % taking average over trials
        Avg_blue(iter_seq,:,:) = reshape(mean(R_data_blue(:,iter_seq,:,:),1,'omitnan'),[X,Y]);
        [~] = cal_dr_r_indi_green(reshape(Avg_blue(iter_seq,:,:)-1,[X,Y]),output_video,'Blue-',num2str(iter_seq));
    end
    % clear R_data_blue
    % Generate video
    Avg_blue = permute(Avg_blue,[2,3,1]);
    Generate_video_stack(Avg_blue-1,20,seq_period,output_video,sample_img,'label','\DeltaR_n','sc_range',[amin,amax],'overlay_range',[-0.025 -0.01],'show_scalebar',false);
    Avg_blue = permute(Avg_blue,[3,1,2]);

    % store .mat
    mat_dir = fullfile(output_dir,'mat_files');
    
    if ~exist(mat_dir, 'dir')
       mkdir(mat_dir)
    end
    save(fullfile(mat_dir,'data_blue.mat'),'Avg_blue','gauss_filtered_blue_pk');
%     save(fullfile(single_trial_dir,'TS_480.mat'),'TS_480');
end
%% Amber 580 nm
clearvars -except global_t_stack mat_dir dry;
load(fullfile(mat_dir,'ROI.mat'));
output_video = fullfile(output_dir_avg,'580nm');
if ~exist(output_video, 'dir')
   mkdir(output_video)
end
Data_all_amber = cell(eff_num_trials,len_trials);
% Loop over trials
iter_trial_local = 0;
for iter_trial = 1:num_trials
temp_start = start_trial(iter_trial); 
temp_start_amber = temp_start + 1;
if any(trials_reject == iter_trial)
    continue;
else
    iter_trial_local = iter_trial_local + 1;
end
% Reading wavelength 2 data: amber 580 nm
    for iter_seq = 1:len_trials
        index = temp_start_amber + (iter_seq-1)*number_wavelength;
        file = files_raw(index).name;
        filePath = fullfile(Raw_dir,file);
        Data = imread(filePath);
        if cam_in == 1
            rawData = image_transform_B(Data);
        else
            rawData = image_transform_A(Data);
        end
        rawData(BW == false) = 0;           % applying craniotomy
        global_t_stack(iter_trial_local,iter_seq,2) = mean(rawData(:));
        Data_all_amber{iter_trial_local,iter_seq}=rawData;
    end
end
if ~dry
    % Spatial Gaussian Filtering (sigma = 150 um)
    d2_spatial_avg_mat = spatial_gauss_filt(Data_all_amber(:,pk_indx),sigma_2d);
    d2_spatial_avg_bsl = spatial_gauss_filt(Data_all_amber(:,bsl_indx),sigma_2d);
    d2_spatial_avg_bsl = reshape(mean(d2_spatial_avg_bsl,2,'omitnan'),[eff_num_trials,X,Y]);
    % compute deltaR/R
    for iter_trial = 1:eff_num_trials
        for iter_seq = 1:length(pk_indx)
            bsl_local = d2_spatial_avg_bsl(iter_trial,:,:);
            bsl_local = reshape(bsl_local,[X,Y]);
            d2_spatial_avg_mat(iter_trial,iter_seq,:,:) = (reshape(d2_spatial_avg_mat(iter_trial,iter_seq,:,:),[X,Y]))./bsl_local;
        end
    end
    % averaging over trials
    gauss_filtered_amber_pk = reshape(mean(d2_spatial_avg_mat,1,'omitnan'),[length(pk_indx),X,Y]);

    % GSR (Global Signal Regression)
    % Due to memory constraints, the loops are run separately.
    % for iter_trial=1:num_trials
    %     R_data_amber{1,iter_trial} = GSR_main(Data_all_amber(iter_trial,:),global_t_stack(iter_trial,:,2));   % 580 nm
    % end

    % Low Pass filtering 
    fcutlow = 2;    % cutoff frequency
    Fs = 1/seq_period;        % Sampling frequency
    lowpass_filter = designfilt('lowpassiir', 'FilterOrder', order, ...
                                'HalfPowerFrequency', fcutlow, ...
                                'SampleRate', Fs, 'DesignMethod', ...
                                'butter');
    R_data_amber = filter_LowPass(Data_all_amber,lowpass_filter);
    Avg_amber = zeros(len_trials,X,Y);
    % Trial Averaging and deltaR/R
    for iter_trial = 1:eff_num_trials
        bsl_local = R_data_amber(iter_trial,bsl_indx,:,:);
        bsl_local = reshape(mean(bsl_local,2,'omitnan'),X,Y);
        for iter_seq = 1:len_trials
            R_data_amber(iter_trial,iter_seq,:,:) = (reshape(R_data_amber(iter_trial,iter_seq,:,:),X,Y))./bsl_local;
        end    
    end
    for iter_seq = 1:len_trials             % taking average over trials
        Avg_amber(iter_seq,:,:) = reshape(mean(R_data_amber(:,iter_seq,:,:),1,'omitnan'),[X,Y]);
        [~] = cal_dr_r_indi_amber(reshape(Avg_amber(iter_seq,:,:)-1,[X,Y]),output_video,'Amber-',num2str(iter_seq));
    end
    % Generate video
    Avg_amber = permute(Avg_amber,[2,3,1]);
    Generate_video_stack(Avg_amber-1,20,seq_period,output_video,sample_img,'label','\DeltaR_n','sc_range',[amin,amax],'overlay_range',[-0.035 -0.01],'show_scalebar',false);
    Avg_amber = permute(Avg_amber,[3,1,2]);
    % store .mat
    mat_dir = fullfile(output_dir,'mat_files');
    if ~exist(mat_dir, 'dir')
       mkdir(mat_dir)
    end
    save(fullfile(mat_dir,'data_amber.mat'),'Avg_amber','gauss_filtered_amber_pk');
end

%% Green 510 nm or Red 632 nm
clearvars -except global_t_stack mat_dir dry;
load(fullfile(mat_dir,'ROI.mat'));
if wv_3 == 510
    output_video = fullfile(output_dir_avg,'510nm');
elseif wv_3 == 632 
    output_video = fullfile(output_dir_avg,'632nm');
else
    error('incorrect wavelength selected! \n')
end

if ~exist(output_video, 'dir')
   mkdir(output_video)
end
Data_all_green = cell(eff_num_trials,len_trials);
% Loop over trials
iter_trial_local = 0;
for iter_trial = 1:num_trials
temp_start = start_trial(iter_trial); 
temp_start_amber = temp_start + 2;
if any(trials_reject == iter_trial)
    continue;
else
    iter_trial_local = iter_trial_local + 1;
end
% Reading wavelength 3 data: green 510 nm or red 632 nm
    for iter_seq = 1:len_trials
        index = temp_start_amber + (iter_seq-1)*number_wavelength;
        file = files_raw(index).name;
        filePath = fullfile(Raw_dir,file);
        Data = imread(filePath);
        if cam_in == 1
            rawData = image_transform_B(Data);
        else
            rawData = image_transform_A(Data);
        end
        rawData(BW == false) = 0;           % applying craniotomy
        global_t_stack(iter_trial_local,iter_seq,3) = mean(rawData(:));
        Data_all_green{iter_trial_local,iter_seq}=rawData;
    end
end

if ~dry
    % Spatial Gaussian Filtering (sigma = 150 um)
    d2_spatial_avg_mat = spatial_gauss_filt(Data_all_green(:,pk_indx),sigma_2d);
    d2_spatial_avg_bsl = spatial_gauss_filt(Data_all_green(:,bsl_indx),sigma_2d);
    d2_spatial_avg_bsl = reshape(mean(d2_spatial_avg_bsl,2,'omitnan'),[eff_num_trials,X,Y]);
    % compute deltaR/R
    for iter_trial = 1:eff_num_trials
        for iter_seq = 1:length(pk_indx)
            bsl_local = d2_spatial_avg_bsl(iter_trial,:,:);
            bsl_local = reshape(bsl_local,[X,Y]);
            d2_spatial_avg_mat(iter_trial,iter_seq,:,:) = (reshape(d2_spatial_avg_mat(iter_trial,iter_seq,:,:),[X,Y]))./bsl_local;
        end
    end
    % averaging over trials
    gauss_filtered_wv3_pk = reshape(mean(d2_spatial_avg_mat,1,'omitnan'),[length(pk_indx),X,Y]);

    % GSR (Global Signal Regression)
    % Due to memory constraints, the loops are run separately.
    % for iter_trial=1:num_trials
    %     R_data_amber{1,iter_trial} = GSR_main(Data_all_amber(iter_trial,:),global_t_stack(iter_trial,:,2));   % 580 nm
    % end

    % Low Pass filtering 
    fcutlow = 2;    % cutoff frequency
    Fs = 1/seq_period;        % Sampling frequency
    lowpass_filter = designfilt('lowpassiir', 'FilterOrder', order, ...
                                'HalfPowerFrequency', fcutlow, ...
                                'SampleRate', Fs, 'DesignMethod', ...
                                'butter');
    R_data_green = filter_LowPass(Data_all_green,lowpass_filter);
    Avg_wv3 = zeros(len_trials,X,Y);
    % Trial Averaging and deltaR/R
    for iter_trial = 1:eff_num_trials
        bsl_local = R_data_green(iter_trial,bsl_indx,:,:);
        bsl_local = reshape(mean(bsl_local,2,'omitnan'),X,Y);
        for iter_seq = 1:len_trials
            R_data_green(iter_trial,iter_seq,:,:) = (reshape(R_data_green(iter_trial,iter_seq,:,:),X,Y))./bsl_local;
        end    
    end
    for iter_seq = 1:len_trials             % taking average over trials
        Avg_wv3(iter_seq,:,:) = reshape(mean(R_data_green(:,iter_seq,:,:),1,'omitnan'),[X,Y]);
        if wv_3 == 510
            [~] = cal_dr_r_indi_green(reshape(Avg_wv3(iter_seq,:,:)-1,[X,Y]),output_video,'Green-',num2str(iter_seq));
        elseif wv_3 == 632
            [~] = cal_dr_r_indi_red(reshape(Avg_wv3(iter_seq,:,:)-1,[X,Y]),output_video,'Red-',num2str(iter_seq));
        else
            error('incorrect wavelength selected! \n');
        end
    end
    % Generate video
    Avg_wv3 = permute(Avg_wv3,[2,3,1]);
    if wv_3 == 510
        Generate_video_stack(Avg_wv3-1,20,seq_period,output_video,sample_img,'label','\DeltaR_n','sc_range',[amin,amax],'overlay_range',[-0.025 -0.01],'show_scalebar',false);
    elseif wv_3 == 632
        Generate_video_stack(Avg_wv3-1,20,seq_period,output_video,sample_img,'label','\DeltaR_n','sc_range',[amin,amax],'overlay_range',[0.004 0.01],'show_scalebar',false);
    else
        error('incorrect wavelength selected! \n');
    end
    Avg_wv3 = permute(Avg_wv3,[3,1,2]);
    % store .mat
    mat_dir = fullfile(output_dir,'mat_files');
    if ~exist(mat_dir, 'dir')
       mkdir(mat_dir)
    end
    save(fullfile(mat_dir,'data_wv3.mat'),'Avg_wv3','gauss_filtered_wv3_pk');
end

%% Plot g(t) for all trials
fig_gs_plot = global_signal_plot(global_t_stack, len_trials, eff_num_trials);
fig_gs_plot_filename = fullfile(output_dir,'gs_plot.fig');
savefig(fig_gs_plot,fig_gs_plot_filename,'compact');
fig_gs_plot_filename = fullfile(output_dir,'gs_plot.svg');
saveas(fig_gs_plot,fig_gs_plot_filename,'svg')
close(fig_gs_plot);
