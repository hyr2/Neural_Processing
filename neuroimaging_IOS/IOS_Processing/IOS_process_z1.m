% Compute I/I_0 and save
% Apply Craniatomy mask
% Ask user to select reference coordinates (ie origin of the image)

% ROI 1-4: shanks A to D
% ROI 5  : Far Away (can be used for GSR)
% ROI 6  : Artery (close by feeding)
% ROI 7  : Vein   (close by draining)

%% data dir
% source_dir
% |
% |__Raw/
% |__whisker_stim.txt

% clear all;
% input parameters:
wv_3 = '510';   % The 3rd wavelength was 510nm or 632nm
cam_in = '0';   % Is this the Andor Camera? [1,0] 0: Hamamatsu Camera (installed on 2022-08)
ROI_selector = 1;  % run using Zidong_IOS_Full.m 
sigma_2d = 5;   % spatial filtering width of 2d gaussian
order = 7;      % Order of the IIR filter

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
% wv_3 = input('The 3rd wavelength was 510nm or 632nm ? \n','s');
wv_3 = int16(str2num(wv_3));

if wv_3 == 510
    disp('Selected wavelengths are 480 nm, 580 nm and 510 nm');     % In this sequence
elseif wv_3 == 632
    disp('Selected wavelengths are 480 nm, 580 nm and 632 nm');     % In this sequence
else
    error('incorrect wavelength selected! \n')
end

% cam_in = input('Is this the Andor Camera? [1,0]\n', 's');
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
start_trails = 0;  % disregard the first several trails
% calculate the start time
start_t = start_time/1000; %second
end_t = (start_time + duration_time*stim_num)/1000;
% strat stimulation time
start_stimulate = floor(start_time/duration_time);
baseline_end1 = start_stimulate-5; % baselines before the stimulation

len_trials_all_Lambda = number_wavelength * len_trials;
start_trial = [1:len_trials_all_Lambda:num_trials*len_trials_all_Lambda];
% %% ROI Selection
iter_local_local = 1;       
vec_img = [98:3:131];       % for background image (overlay)
for iter_local = vec_img    % averaging 580nm image for a good underlay image
    thisImage = double(imread(fullfile(Raw_dir,files_raw(iter_local).name)));
    if iter_local_local
        sumImage = thisImage;
    else
        sumImage = sumImage + thisImage;
    end
    iter_local_local = iter_local_local + 1;
end
sumImage = sumImage / length(vec_img);
sample_img = sumImage; sample_img_temp = sumImage;
% sample_img = imread(fullfile(Raw_dir,files_raw(2).name));
sumImage = mat2gray(sumImage);
if cam_in == 1
    sumImage = image_transform_B(sumImage);
else
    sumImage = image_transform_A(sumImage);
end
amax = max(sumImage(:));
amin = min(sumImage(:));
[X,Y] = size(sumImage);
% If ROI argument is not provided, prompt user to define
% ROI_selector = input('Do you want to use a pre-defined ROI? [1,0] \n');
% ROI_selector = logical(ROI_selector);   
if ROI_selector
%     ROI_loc = input('Path to ROI.mat file\n','s');
    ROI_loc = fullfile(source_dir,'Processed','mat_files');
    ROI_loc = fullfile(ROI_loc,'ROI.mat');
    load(ROI_loc,'mask','BW');
  if size(mask, 1) ~= X || size(mask, 2) ~= Y
    error('ROI dimensions (%d x %d) do not match intrinsic data (%d x %d)', size(mask,1), size(mask, 2), X, Y);
  end
else
    mask = drawROIs(sumImage,[amin,amax]);              % Draw ROI masks
    BW = defineCraniotomy(sumImage,[amin amax]);        % Define a craniotomy
end
N_ROI = size(mask,3);

% important processing parameters
pk_indx = [int16(start_stimulate+1.8/seq_period):stim_num + start_stimulate];   % peak time extraction
bsl_indx = [baseline_end1:start_stimulate];         % baseline time extraction

% save parameters and ROI masks to mat_dir  ********************************
eff_num_trials = num_trials;
clearvars sample_img
save(fullfile(mat_dir,'ROI.mat'),'BW','mask');
sample_img = sample_img_temp;
% sample_img = sumImage;
if cam_in == 1
    sample_img = image_transform_B(sample_img);
else
    sample_img = image_transform_A(sample_img);
end
sample_img = double(sample_img .* BW);
tmp_std = std(sample_img(:),'omitnan');
amin = mean(sample_img(:),'omitnan') - tmp_std;
amax = mean(sample_img(:),'omitnan') + 2.25*tmp_std;
F = SpeckleFigure(sample_img, [amin,amax], 'visible', true);
F.showROIs(mask);
F.savePNG(fullfile(output_dir_avg,'ROI_overlay.png'),220);
len_t = eff_num_trials * len_trials;
% global_t_stack = zeros(len_t,3);    % first is the global signal % second is the ROI5 % third is the cement area average

%% Blue 480 nm
output_video = fullfile(output_dir_avg,'480nm');
if ~exist(output_video, 'dir')
   mkdir(output_video)
end
Data_all_blue = cell(eff_num_trials,len_trials);
TS_480 = zeros(eff_num_trials,len_trials,N_ROI);  % array for single trial data storage
% Loop over trials
iter_local = 1;
for iter_trial = 1:num_trials
    temp_start = start_trial(iter_trial); 
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
%         global_t_stack(iter_local,3) = mean(rawData.*not(BW),'all');   % cement area
        rawData(BW == false) = 0;           % applying craniotomy
%         global_t_stack(iter_local,1) = mean(rawData,'all');         % full frame
%         global_t_stack(iter_local,2) = mean(rawData.*mask(:,:,5),'all');    % only far away region
        Data_all_blue{iter_trial,iter_seq} = rawData;
        iter_local = iter_local+1;
    end
    
end
% GSR (Global Signal Regression)
% Data_all_blue = permute(Data_all_blue,[2,1]);           % columns are trials. Rows are frames of a particular trial
% Reg_stack = GSR_z(Data_all_blue,global_t_stack(:,1));   % full frame
% Reg_stack = GSR_z(Data_all_blue,global_t_stack(:,2));   % far away region
% Reg_stack = GSR_z(Data_all_blue,global_t_stack(:,3));   % cement region
% iter_local = 1;
% Data_all_blue_new = cell(eff_num_trials,len_trials);
% for iter_trial = 1:num_trials
%     for iter_seq = 1:len_trials
%         Data_all_blue_new{iter_trial,iter_seq} = reshape(Reg_stack(iter_local,:,:),X,Y);
%         iter_local = iter_local + 1;
%     end
% end
% DeltaR/R computation MAIN PLOTS
% Low Pass filtering 
fcutlow = 1.25;    % cutoff frequency
Fs = 1/seq_period;        % Sampling frequency
lowpass_filter = designfilt('lowpassiir', 'FilterOrder', order, ...
                            'HalfPowerFrequency', fcutlow, ...
                            'SampleRate', Fs, 'DesignMethod', ...
                            'butter');
R_data_blue = filter_LowPass(Data_all_blue,lowpass_filter);
% Avg_blue = zeros(len_trials,X,Y);

% Compute I/I_0
for iter_trial = 1:eff_num_trials
    bsl_blue_local = R_data_blue(iter_trial,bsl_indx,:,:);
    bsl_blue_local = reshape(mean(bsl_blue_local,2,'omitnan'),X,Y);
    for iter_seq = 1:len_trials
        R_data_blue(iter_trial,iter_seq,:,:) = (reshape(R_data_blue(iter_trial,iter_seq,:,:),X,Y))./bsl_blue_local;
    end
end

% Testing Plots
time_vec = linspace(0,len_trials*duration_time,len_trials)/1000;    % seconds
Avg_480 = reshape(mean(R_data_blue-1,1,'omitnan'),len_trials,X,Y);     % avg over trials
artery = zeros(len_trials,1);
roi2 = zeros(len_trials,1);
for iter_l = 1:len_trials
    artery(iter_l) = mean(reshape(Avg_480(iter_l,:,:),X,Y) .* mask(:,:,6),'all','omitnan');
    roi2(iter_l) = mean(reshape(Avg_480(iter_l,:,:),X,Y) .* mask(:,:,2),'all','omitnan');
end
figure;
plot(time_vec,artery);hold on;
plot(time_vec,roi2);
legend('artery','R2');
print(fullfile(output_dir_avg,'lineplot_480.png'),'-dpng');
% Saving image files here
R_data_blue = single(R_data_blue);
save(fullfile(mat_dir,'480nm_data.mat'),'R_data_blue','-v7.3');

%% Amber 580 nm
output_video = fullfile(output_dir_avg,'580nm');
if ~exist(output_video, 'dir')
   mkdir(output_video)
end
Data_all_blue = cell(eff_num_trials,len_trials);
TS_480 = zeros(eff_num_trials,len_trials,N_ROI);  % array for single trial data storage
% Loop over trials
iter_local = 1;
for iter_trial = 1:num_trials
    temp_start = start_trial(iter_trial)+1; 
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
%         global_t_stack(iter_local,3) = mean(rawData.*not(BW),'all');   % cement area
        rawData(BW == false) = 0;           % applying craniotomy
%         global_t_stack(iter_local,1) = mean(rawData,'all');         % full frame
%         global_t_stack(iter_local,2) = mean(rawData.*mask(:,:,5),'all');    % only far away region
        Data_all_blue{iter_trial,iter_seq} = rawData;
        iter_local = iter_local+1;
    end
    
end
% GSR (Global Signal Regression)
% Data_all_blue = permute(Data_all_blue,[2,1]);           % columns are trials. Rows are frames of a particular trial
% Reg_stack = GSR_z(Data_all_blue,global_t_stack(:,1));   % full frame
% Reg_stack = GSR_z(Data_all_blue,global_t_stack(:,2));   % far away region
% Reg_stack = GSR_z(Data_all_blue,global_t_stack(:,3));   % cement region
% iter_local = 1;
% Data_all_blue_new = cell(eff_num_trials,len_trials);
% for iter_trial = 1:num_trials
%     for iter_seq = 1:len_trials
%         Data_all_blue_new{iter_trial,iter_seq} = reshape(Reg_stack(iter_local,:,:),X,Y);
%         iter_local = iter_local + 1;
%     end
% end
% DeltaR/R computation MAIN PLOTS
% Low Pass filtering 
fcutlow = 1.25;    % cutoff frequency
Fs = 1/seq_period;        % Sampling frequency
lowpass_filter = designfilt('lowpassiir', 'FilterOrder', order, ...
                            'HalfPowerFrequency', fcutlow, ...
                            'SampleRate', Fs, 'DesignMethod', ...
                            'butter');
R_data_blue = filter_LowPass(Data_all_blue,lowpass_filter);
% Avg_blue = zeros(len_trials,X,Y);

% Compute I/I_0
for iter_trial = 1:eff_num_trials
    bsl_blue_local = R_data_blue(iter_trial,bsl_indx,:,:);
    bsl_blue_local = reshape(mean(bsl_blue_local,2,'omitnan'),X,Y);
    for iter_seq = 1:len_trials
        R_data_blue(iter_trial,iter_seq,:,:) = (reshape(R_data_blue(iter_trial,iter_seq,:,:),X,Y))./bsl_blue_local;
    end
end

% Testing Plots
time_vec = linspace(0,len_trials*duration_time,len_trials)/1000;    % seconds
Avg_480 = reshape(mean(R_data_blue-1,1,'omitnan'),len_trials,X,Y);     % avg over trials
artery = zeros(len_trials,1);
roi2 = zeros(len_trials,1);
for iter_l = 1:len_trials
    artery(iter_l) = mean(reshape(Avg_480(iter_l,:,:),X,Y) .* mask(:,:,6),'all','omitnan');
    roi2(iter_l) = mean(reshape(Avg_480(iter_l,:,:),X,Y) .* mask(:,:,2),'all','omitnan');
end
figure;
plot(time_vec,artery);hold on;
plot(time_vec,roi2);
legend('artery','R2');
print(fullfile(output_dir_avg,'lineplot_580.png'),'-dpng');
%Saving image files here
R_data_amber = single(R_data_blue);
save(fullfile(mat_dir,'580nm_data.mat'),'R_data_amber','-v7.3');


%% Wv3 510 nm or 632 nm
output_video = fullfile(output_dir_avg,'wv3');
if ~exist(output_video, 'dir')
   mkdir(output_video)
end
Data_all_blue = cell(eff_num_trials,len_trials);
TS_480 = zeros(eff_num_trials,len_trials,N_ROI);  % array for single trial data storage
% Loop over trials
iter_local = 1;
for iter_trial = 1:num_trials
    temp_start = start_trial(iter_trial)+2; 
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
%         global_t_stack(iter_local,3) = mean(rawData.*not(BW),'all');   % cement area
        rawData(BW == false) = 0;           % applying craniotomy
%         global_t_stack(iter_local,1) = mean(rawData,'all');         % full frame
%         global_t_stack(iter_local,2) = mean(rawData.*mask(:,:,5),'all');    % only far away region
        Data_all_blue{iter_trial,iter_seq} = rawData;
        iter_local = iter_local+1;
    end
    
end
% GSR (Global Signal Regression)
% Data_all_blue = permute(Data_all_blue,[2,1]);           % columns are trials. Rows are frames of a particular trial
% Reg_stack = GSR_z(Data_all_blue,global_t_stack(:,1));   % full frame
% Reg_stack = GSR_z(Data_all_blue,global_t_stack(:,2));   % far away region
% Reg_stack = GSR_z(Data_all_blue,global_t_stack(:,3));   % cement region
% iter_local = 1;
% Data_all_blue_new = cell(eff_num_trials,len_trials);
% for iter_trial = 1:num_trials
%     for iter_seq = 1:len_trials
%         Data_all_blue_new{iter_trial,iter_seq} = reshape(Reg_stack(iter_local,:,:),X,Y);
%         iter_local = iter_local + 1;
%     end
% end
% DeltaR/R computation MAIN PLOTS
% Low Pass filtering 
fcutlow = 1.25;    % cutoff frequency
Fs = 1/seq_period;        % Sampling frequency
lowpass_filter = designfilt('lowpassiir', 'FilterOrder', order, ...
                            'HalfPowerFrequency', fcutlow, ...
                            'SampleRate', Fs, 'DesignMethod', ...
                            'butter');
R_data_blue = filter_LowPass(Data_all_blue,lowpass_filter);
% Avg_blue = zeros(len_trials,X,Y);

% Compute I/I_0
for iter_trial = 1:eff_num_trials
    bsl_blue_local = R_data_blue(iter_trial,bsl_indx,:,:);
    bsl_blue_local = reshape(mean(bsl_blue_local,2,'omitnan'),X,Y);
    for iter_seq = 1:len_trials
        R_data_blue(iter_trial,iter_seq,:,:) = (reshape(R_data_blue(iter_trial,iter_seq,:,:),X,Y))./bsl_blue_local;
    end
end

% Testing Plots
time_vec = linspace(0,len_trials*duration_time,len_trials)/1000;    % seconds
Avg_480 = reshape(mean(R_data_blue-1,1,'omitnan'),len_trials,X,Y);     % avg over trials
artery = zeros(len_trials,1);
roi2 = zeros(len_trials,1);
for iter_l = 1:len_trials
    artery(iter_l) = mean(reshape(Avg_480(iter_l,:,:),X,Y) .* mask(:,:,6),'all','omitnan');
    roi2(iter_l) = mean(reshape(Avg_480(iter_l,:,:),X,Y) .* mask(:,:,2),'all','omitnan');
end
figure;
plot(time_vec,artery);hold on;
plot(time_vec,roi2);
legend('artery','R2');
print(fullfile(output_dir_avg,'lineplot_wv3.png'),'-dpng');
% Saving image files here
R_data_wv3 = single(R_data_blue);
save(fullfile(mat_dir,'wv3nm_data.mat'),'R_data_wv3','-v7.3');
