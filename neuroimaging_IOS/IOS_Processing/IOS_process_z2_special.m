% Assumes IOS_process_z1.m has already been run. 
% AnimalID/
% │
% ├── SessionID1/
% │   │
% │   └── Processed/
% │       │
% │       ├── Average/
% │       │
% │       ├── mat_files/
% │       │
% │       └── single_trial/
% │
% └── SessionID2/
%     │
%     └── Processed/
%         │
%         ├── Average/
%         │
%         ├── mat_files/
%         │
%         └── single_trial/
%
% This script needs the trial_mask.csv file and will apply the trial mask.

% A special case of plotting deltaR/R only with and without spatial
% filtering for all the wavelenvgths

% Some important parameters
sigma_2d = 3;
z_thresh = -2.5;            % -2.5 or -2.75
thresh_480 = -0.02;
thresh_580 = -0.025;
dist_thresh=140; %in pixels
c_limits_480 = [-0.03,-0.01]; % Should be dynamic?
c_limits_580 = [-0.04,-0.01]; % Should be dynamic?
c_limits_510 = [-0.03,-0.01]; % Should be dynamic?
frames_load = [19:32];

% Loading Trial mask file
trial_mask = fullfile(source_dir,'trial_mask.csv');
trial_mask = logical(readmatrix(trial_mask));
fprintf('Number of accepted trials in current session %d \n ',sum(trial_mask));

%% Loading ROI data
mat_folder = fullfile(source_dir,'Processed','mat_files');
load(fullfile(mat_folder,'ROI.mat'));
len_trials = 75;            % Hard-coded
[X,Y] = size(mask,[1,2]);
output_folder = fullfile(source_dir,'Processed','Average');
load(fullfile(output_folder,'sample_image.mat'));
output_folder = fullfile(source_dir,'Processed','Average_special');
mkdir(output_folder);
mkdir(fullfile(output_folder,'480nm'));
mkdir(fullfile(output_folder,'580nm'));
mkdir(fullfile(output_folder,'wv3'));
sample_img = double(sample_img .* BW);
tmp_std = std(sample_img(:),'omitnan');
amin = mean(sample_img(:),'omitnan') - 1.5*tmp_std;
amax = mean(sample_img(:),'omitnan') + 2.75*tmp_std;

%% Adding Vessel segmentation using adaptive thresholding
T = adaptthresh(sample_img,0.795);
BW_vessel = imbinarize(sample_img,T);
BW_vessel = double(BW_vessel);
BW_vessel(BW_vessel == 0) = NaN;
%% User draws activation area region (where do you expect the activation to be located?)
% [activation_region,x_act,y_act] = defineArea(sample_img,[amin amax]);  %
% uncomment this line to draw define an activation area. I did this for RH9
% x_act = [1;1;Y-140;Y-dist_thresh;1];
% y_act = [1;X;X;1;1];
%% Working on 480nm
load(fullfile(mat_folder,'480nm_data.mat'))
try 
    % Check for error
    if ~isequal(size(trial_mask,1), size(R_data_blue,1))
        error('The dimensions of the trial mask does not match that of the raw data. Check IOS_process_z1.m');
    end
    R_data_blue = R_data_blue(trial_mask,:,:,:);
    img_stack_local = reshape(mean(R_data_blue,1),[75,X,Y]);    % mean over all accepted trials
    img_stack_local = permute(img_stack_local,[2,3,1]);
    img_stack_local = img_stack_local .* BW_vessel;      % mask        
    % img_stack_local(img_stack_local == 0) = NaN;         % mask is set to NaN to remove it from Z score calculations
    img_stack_local = img_stack_local - 1;      % deltaR/R
    img_stack_local = permute(img_stack_local,[3,1,2]);

    % Saving averaged activation after performing 2D spatial gaussian filtering
    % loop for nice overlay images
    img_stack_local_filtered = twoD_gauss(img_stack_local,sigma_2d/2); % 3D matrix (trial averaged image time stack)
    Area_activationZ = zeros(length(frames_load),1);
    CentroidsZ = zeros(length(frames_load),2);
    iter_local_local = 0;
    for iter_seq = frames_load
        iter_local_local = iter_local_local + 1;
        name_pic = strcat('480nm-',num2str(iter_seq));
        delRR = squeeze(img_stack_local(iter_seq,:,:)); 
        delRR_z = Zscore_nan(delRR);
        % ActMask_local = (delRR <= c_limits_480(2)) & (delRR >= c_limits_480(1));                % Binary mask absolute
        ActMask_local_z = delRR_z < z_thresh;     % Following threshold set by Zeiger et al (2021)    % Binary mask using Z score
        plot_overlay_auto_zscored(sample_img,ActMask_local_z,fullfile(output_folder,'480nm'),name_pic);
        % plot_overlay()
        
    end

catch exception     % Handle errors
    fprintf('Error in session ID: %s \n Error message: %s ',source_dir, exception.message);
    stackInfo = exception.stack;
    fprintf('Error occurred in file: %s, line: %d\n', stackInfo(1).file, stackInfo(1).line);
end


%% Working on 580nm
load(fullfile(mat_folder,'580nm_data.mat'))
try 
    % Check for error
    if ~isequal(size(trial_mask,1), size(R_data_amber,1))
        error('The dimensions of the trial mask does not match that of the raw data. Check IOS_process_z1.m');
    end
    R_data_amber = R_data_amber(trial_mask,:,:,:);
    img_stack_local = reshape(mean(R_data_amber,1),[75,X,Y]);    % mean over all accepted trials
    img_stack_local = permute(img_stack_local,[2,3,1]);
    img_stack_local = img_stack_local .* BW_vessel;      % mask        
    % img_stack_local(img_stack_local == 0) = NaN;         % mask is set to NaN to remove it from Z score calculations
    img_stack_local = img_stack_local - 1;      % deltaR/R
    img_stack_local = permute(img_stack_local,[3,1,2]);

    % Saving averaged activation after performing 2D spatial gaussian filtering
    % loop for nice overlay images
    img_stack_local_filtered = twoD_gauss(img_stack_local,sigma_2d/2); % 3D matrix (trial averaged image time stack)
    Area_activationZ = zeros(length(frames_load),1);
    CentroidsZ = zeros(length(frames_load),2);
    iter_local_local = 0;
    for iter_seq = frames_load
        iter_local_local = iter_local_local + 1;
        name_pic = strcat('580nm-',num2str(iter_seq));
        delRR = squeeze(img_stack_local(iter_seq,:,:)); 
        delRR_z = Zscore_nan(delRR);
        % ActMask_local = (delRR <= c_limits_480(2)) & (delRR >= c_limits_480(1));                % Binary mask absolute
        ActMask_local_z = delRR_z < z_thresh;     % Following threshold set by Zeiger et al (2021)    % Binary mask using Z score
        plot_overlay_auto_zscored(sample_img,ActMask_local_z,fullfile(output_folder,'580nm'),name_pic);
        % plot_overlay()
        
    end

catch exception     % Handle errors
    fprintf('Error in session ID: %s \n Error message: %s ',source_dir, exception.message);
    stackInfo = exception.stack;
    fprintf('Error occurred in file: %s, line: %d\n', stackInfo(1).file, stackInfo(1).line);
end

%% Working on wv3 (510 nm or 632 nm)
load(fullfile(mat_folder,'wv3nm_data.mat'))
try 
    % Check for error
    if ~isequal(size(trial_mask,1), size(R_data_wv3,1))
        error('The dimensions of the trial mask does not match that of the raw data. Check IOS_process_z1.m');
    end
    R_data_wv3 = R_data_wv3(trial_mask,:,:,:);
    img_stack_local = reshape(mean(R_data_wv3,1),[75,X,Y]);    % mean over all accepted trials
    img_stack_local = permute(img_stack_local,[2,3,1]);
    img_stack_local = img_stack_local .* BW_vessel;      % mask        
    % img_stack_local(img_stack_local == 0) = NaN;         % mask is set to NaN to remove it from Z score calculations
    img_stack_local = img_stack_local - 1;      % deltaR/R
    img_stack_local = permute(img_stack_local,[3,1,2]);

    % Saving averaged activation after performing 2D spatial gaussian filtering
    % loop for nice overlay images
    img_stack_local_filtered = twoD_gauss(img_stack_local,sigma_2d/2); % 3D matrix (trial averaged image time stack)
    Area_activationZ = zeros(length(frames_load),1);
    CentroidsZ = zeros(length(frames_load),2);
    iter_local_local = 0;
    for iter_seq = frames_load
        iter_local_local = iter_local_local + 1;
        name_pic = strcat('wv3-',num2str(iter_seq));
        delRR = squeeze(img_stack_local(iter_seq,:,:)); 
        delRR_z = Zscore_nan(delRR);
        % ActMask_local = (delRR <= c_limits_480(2)) & (delRR >= c_limits_480(1));                % Binary mask absolute
        ActMask_local_z = delRR_z < z_thresh;     % Following threshold set by Zeiger et al (2021)    % Binary mask using Z score
        plot_overlay_auto_zscored(sample_img,ActMask_local_z,fullfile(output_folder,'wv3'),name_pic);
        % plot_overlay()
        
    end

catch exception     % Handle errors
    fprintf('Error in session ID: %s \n Error message: %s ',source_dir, exception.message);
    stackInfo = exception.stack;
    fprintf('Error occurred in file: %s, line: %d\n', stackInfo(1).file, stackInfo(1).line);
end

%{
%% Working on 510nm
load(fullfile(mat_folder,'wv3nm_data.mat'))
try 
    % Check for error
    if ~isequal(size(trial_mask,1), size(R_data_wv3,1))
        error('The dimensions of the trial mask does not match that of the raw data. Check IOS_process_z1.m');
    end
    R_data_wv3 = R_data_wv3(trial_mask,:,:,:);
    img_stack_local = reshape(mean(R_data_wv3,1),[75,X,Y]) - 1;  % mean over all accepted trials 
    [ROI_time_series_img,names] = Time_Series_extract_single(img_stack_local,mask,len_trials,X,Y);      % Time series for all ROIs
    % Check for error
    if ~isequal(size(ROI_time_series_img,2), 7)
        error('ROIs found in the ROI.mat file !=7')
    end 
    % Saving averaged activation after performing 2D spatial gaussian filtering
    % loop for nice overlay images
    img_stack_local_filtered = twoD_gauss(img_stack_local,sigma_2d/2); % 3D matrix (trial averaged image time stack)
    Area_activationZ = zeros(length(frames_load),1);
    CentroidsZ = zeros(length(frames_load),2);
    iter_local_local = 0;
    for iter_seq = frames_load
        iter_local_local = iter_local_local + 1;
        name_pic = strcat('wv3nm-',num2str(iter_seq));
        delRR = squeeze(img_stack_local_filtered(iter_seq,:,:)); 
        % plot_overlay(sample_img,delRR,c_limits,[amin,amax],fullfile(output_folder,'480nm'),name_pic);
        % plot_overlay_auto(sample_img,delRR,c_limits_480,fullfile(output_folder,'480nm'),name_pic);
        % Tracking activation centroid and Area for current session
        % thresholdRange = [-0.01,-0.03]; % Example threshold range
        delRR_z = Zscore_nan(delRR);
        % ActMask_local = (delRR <= c_limits_480(2)) & (delRR >= c_limits_480(1));                % Binary mask absolute
        ActMask_local_z = delRR_z < -3;     % Following threshold set by Zeiger et al (2021)    % Binary mask using Z score
        % Area_local_raw = sum(ActMask_local(:));
        % Area_local_Z = sum(ActMask_local_z);
        props = regionprops(ActMask_local_z,'Centroid','FilledArea');               % Find properties of the binary mask
        if isempty(props)
            Area_activationZ(iter_local_local) = 0;
            CentroidsZ(iter_local_local,:) = NaN(1,2);
            % plot_overlay_auto_zscored(sample_img,ActMask_local_z,fullfile(output_folder,'480nm'),name_pic,);
            continue;
        elseif length(props) == 1
            Area_activationZ_local = props.FilledArea;
            CentroidsZ_local = props.Centroid;
            Area_activationZ(iter_local_local) = Area_activationZ_local;
            CentroidsZ(iter_local_local,:) = CentroidsZ_local;
            wv_wv3_local.Area = Area_activationZ_local;
            wv_wv3_local.Coord_c = CentroidsZ_local;
            plot_overlay_auto_zscored_centroid(sample_img,ActMask_local_z,fullfile(output_folder,'wv3'),name_pic,wv_wv3_local);
            continue;
        end
        tmp_a = [props.FilledArea];
        [~,tmpindex] = max(tmp_a);
        act_site_local = props(tmpindex);
        Area_activationZ(iter_local_local) = act_site_local.FilledArea;
        CentroidsZ(iter_local_local,:) = act_site_local.Centroid;
        wv_wv3_local.Area = act_site_local.FilledArea;
        wv_wv3_local.Coord_c = act_site_local.Centroid;
        plot_overlay_auto_zscored_centroid(sample_img,ActMask_local_z,fullfile(output_folder,'wv3'),name_pic,wv_wv3_local);
    end
    [maxA,max_I] = max(Area_activationZ);
    coord_c = CentroidsZ(max_I,:);
    wv_wv3.Area = maxA;
    wv_wv3.Coord_c = coord_c;
    % Save the highest activated frame
    iter_seq = max_I + frames_load(1) - 1;
    name_pic = strcat('wv3nm-Max_Area',num2str(iter_seq));
    delRR = squeeze(img_stack_local_filtered(iter_seq,:,:)); 
    delRR_z = Zscore_nan(delRR);
    ActMask_local_z = delRR_z < -3;  
    plot_overlay_auto_zscored(sample_img,ActMask_local_z,fullfile(output_folder,'wv3'),name_pic);
    vec_loc = coord_c - coord_r;        
    wv_wv3.vec_rel = vec_loc;
    wv_wv3.delRR = delRR;
    wv_wv3.sample_img = sample_img;
    wv_wv3.activation_mask = ActMask_local_z;
    save(fullfile(source_dir,'Processed','mat_files','wv3nm_processed.mat'),'ROI_time_series_img','names','wv_wv3');
    close all;
catch exception     % Handle errors
    fprintf('Error in session ID: %s \n Error message: %s ',source_dir, exception.message);
end
%}