% The main code that does the following operations:
% 1. Bins the data into wavelengths
% 2. Temporal low pass filtering of data
% 3. Spatial gaussian filtering of data
% 4. Bins the data into trials
% 5. Beer Lambert Law and Least Squares to find haemoglobin concentrations
% 6. Temporal response curves after ROI selection
% 7. Spatial response plot after peak response frames selection

% Optional:
% 1. Generate video of the activation
% 2. Binarization and thresholding to remove vessel activity to focus on
% the tissue response
% 3. 

% Future additions:
% 1. Time to peak of the response curve
% 2. Width of the response curve
% 3. Time to start of the response curve

%% data dir
% source_dir
% |
% |__Raw/
% |__whisker_stim.txt


selpath = uigetdir('Select source directory containing scans');
selpathOut = fullfile(selpath,'Processed');
whiskerStim_txt = fullfile(selpath,'whisker_stim.txt');
Raw_dir = fullfile(selpath,'Raw');
if ~exist(selpathOut, 'dir')
   mkdir(selpathOut)
end

% Experimental Parameters:
number_wavelength = 3;          % Enter Number of wavelengths here
[len_trials,FramesPerSeq, stim_start_time, seq_period, stim_num, num_trials] = read_whiskerStimTxT(whiskerStim_txt);
number_scans = len_trials;
number_trails = num_trials;
start_time = stim_start_time*1000; % ms
duration_time = seq_period*1000; % ms
stimulated_scans = stim_num;

% Checking total scans for all trails
segListing = dir(Raw_dir);   
segListing(1:2) = []; % remove directory entries
N_scans = int16(num_trials*len_trials*number_wavelength);
n_scans = int16(length(segListing));
assert(N_scans == n_scans);

% other settings for the algorithm
start_trails = 14;  % disregard the first several trails
% calculate the start time
start_t = start_time/1000; %second
end_t = (start_time + duration_time*stimulated_scans)/1000;
% strat stimulation time
start_stimulate = floor(start_time/duration_time);
baseline_end1 = start_stimulate-2; % baselines before the stimulation
% order all the trails 
files_raw = dir_sorted(fullfile(Raw_dir, '*.tif'));
segListing = files_raw; 
len_trials = number_wavelength * number_scans;
start_trial = [1:len_trials:number_trails*len_trials];

% %% ROI Selection
sample_img = imread(fullfile(Raw_dir,files_raw(2).name));
sample_img = mat2gray(sample_img);
sample_img=rot90(sample_img);
sample_img=fliplr(sample_img);
amax = max(sample_img(:));
amin = min(sample_img(:));
mask = drawROIs(sample_img,[amin,amax]);
% save roi masks to mat_dir  ********************************

% variables
% store the intensity changes across scans
Intensity_amber = zeros(number_trails*number_scans,1);
Intensity_red = zeros(number_trails*number_scans,1);
Intensity_blue = zeros(number_trails*number_scans,1);
% store the data across all slices
Data_all_blue = cell(number_trails,number_scans);
Data_all_amber = cell(number_trails,number_scans);
Data_all_red = cell(number_trails,number_scans);

for i = 1:number_trails
    temp_start = start_trial(i);
    temp_start_amber = temp_start + 1;
    temp_start_red = temp_start + 2;  
    for j = 1:number_scans
        %blue amber red
        index = temp_start + (j-1)*number_wavelength;
        file = segListing(index).name;
        filePath = fullfile(Raw_dir,file);
        Data = imread(filePath);
        Data=rot90(Data);
        Data=fliplr(Data);
        rawData=double(Data);
        [x y] = size(rawData);
        Intensity_blue((i-1)*number_scans+j) = sum(rawData(:));
        Data_all_blue{i,j}=rawData;
        
        %read amber
        index_amber = temp_start_amber + (j-1)*number_wavelength;
        file = segListing(index_amber).name;
        filePath = fullfile(Raw_dir,file);
        Data = imread(filePath);
        Data=rot90(Data);
        Data=fliplr(Data);
        rawData=double(Data);
        [x y] = size(rawData);
        Intensity_amber((i-1)*number_scans+j) = sum(rawData(:));
        Data_all_amber{i,j}=rawData;
        
%         % read red
        index_red = temp_start_red + (j-1)*number_wavelength;
        file = segListing(index_red).name;
        filePath = fullfile(Raw_dir,file);
        Data = imread(filePath);
        Data=rot90(Data);
        Data=fliplr(Data);
        rawData=double(Data);
        Intensity_red((i-1)*number_scans+j) = sum(rawData(:));
        Data_all_red{i,j}=rawData;
%         disp(j);
    end
%     disp(i);
end

curfig = figure();
intensity_x = 1:number_trails*number_scans;
plot(intensity_x,Intensity_blue)
hold on
plot(intensity_x,Intensity_amber)
hold on
plot(intensity_x,Intensity_red)
legend({"Blue","Amber","Red"})
title('Photo Value Change across time')
xlabel('Seq')
ylabel('Value')
outputFileName = fullfile(selpathOut,strcat('Value_change.jpg'));
saveas(curfig,outputFileName);
% 
curfig = figure();
Intensity_blue = Intensity_blue/Intensity_blue(1);
Intensity_amber = Intensity_amber/Intensity_amber(1);
Intensity_red = Intensity_red/Intensity_red(1);
plot(intensity_x,Intensity_blue)
hold on
plot(intensity_x,Intensity_amber)
hold on
plot(intensity_x,Intensity_red)
legend({"Green","Amber","Red"})
title('Photo Value Change across time Normalized')
xlabel('Seq')
ylabel('Value')
outputFileName = fullfile(selpathOut,strcat('Value_change_normalized.jpg'));
saveas(curfig,outputFileName);

%set up baselines
%baseline setup
baseline_result_blue = cell(number_trails,1);
baseline_result_amber = cell(number_trails,1);
baseline_result_red = cell(number_trails,1);
intensity_bs_blue = zeros(number_trails,1);
intensity_bs_amber = zeros(number_trails,1);
intensity_bs_red = zeros(number_trails,1);
baseline_index = [1:baseline_end1];
% baseline_result_amber = cell(2,1);
for m = 1:number_trails
    temp_index = baseline_index;
    for j = 1:length(temp_index)
        if j == 1
            temp_baseline_amber = Data_all_amber{m,temp_index(j)};
            temp_baseline_red = Data_all_red{m,temp_index(j)};
            temp_baseline_blue = Data_all_blue{m,temp_index(j)};
        else
            temp_baseline_amber = temp_baseline_amber + Data_all_amber{m,temp_index(j)};
            temp_baseline_red = temp_baseline_red + Data_all_red{m,temp_index(j)};
            temp_baseline_blue = temp_baseline_blue + Data_all_blue{m,temp_index(j)};
        end
    end
    temp_baseline_amber = temp_baseline_amber/length(temp_index);
    temp_baseline_red = temp_baseline_red/length(temp_index);
    temp_baseline_blue = temp_baseline_blue/length(temp_index);

    baseline_result_amber{m} = temp_baseline_amber;
    baseline_result_red{m} = temp_baseline_red;
    baseline_result_blue{m} = temp_baseline_blue;
    intensity_bs_amber(m) = sum(temp_baseline_amber(:));
    intensity_bs_red(m) = sum(temp_baseline_red(:));
    intensity_bs_blue(m) = sum(temp_baseline_blue(:));

%     outputFileName = fullfile(selpathOut,strcat('Baseline_trail_',num2str(m),'_blue.tif'));
%     imwrite(temp_baseline_blue(:, :), outputFileName, 'WriteMode', 'append',  'Compression','none');
%     outputFileName = fullfile(selpathOut,strcat('Baseline_trail_',num2str(m),'_amber.tif'));
%     imwrite(temp_baseline_amber(:, :), outputFileName, 'WriteMode', 'append',  'Compression','none');
%     outputFileName = fullfile(selpathOut,strcat('Baseline_trail_',num2str(m),'_red.tif'));
%     imwrite(temp_baseline_red(:, :), outputFileName, 'WriteMode', 'append',  'Compression','none');
end
% % plot the section baseline distributation
% curfig = figure();
% intensity_x = 1:number_trails;
% plot(intensity_x,intensity_bs_amber(:))
% % legend({"Blue","Amber","Red"})
% title('Baseline Intensity across trails - Amber')
% xlabel('Trails')
% ylabel('Intensity')
% outputFileName = fullfile(selpathOut,strcat('Baseline_Distributation_amber.jpg'));
% saveas(curfig,outputFileName);
% 
% curfig = figure();
% intensity_x = 1:number_trails;
% plot(intensity_x,intensity_bs_red(:))
% % legend({"Blue","Amber","Red"})
% title('Baseline Intensity across trails - Red')
% xlabel('Trails')
% ylabel('Intensity')
% outputFileName = fullfile(selpathOut,strcat('Baseline_Distributation_red.jpg'));
% saveas(curfig,outputFileName);
% 
% curfig = figure();
% intensity_x = 1:number_trails;
% plot(intensity_x,intensity_bs_blue(:))
% title('Baseline Intensity across trails - Blue')
% xlabel('Trails')
% ylabel('Intensity')
% outputFileName = fullfile(selpathOut,strcat('Baseline_Distributation_blue.jpg'))
% saveas(curfig,outputFileName);

%% Computing deltaR/R
% roi for three different result
length_mask = length(mask(1,1,:));
ROI_blue = zeros(length_mask,number_trails,number_scans);
ROI_amber = zeros(length_mask,number_trails,number_scans);
ROI_red = zeros(length_mask,number_trails,number_scans);

% indi_sub_amber_trails = cell(number_scans,number_trails); %section X - baseline
% indi_sub_red_trails = cell(number_scans,number_trails);
% indi_sub_blue_trails = cell(number_scans,number_trails);
indi_sub_amber = cell(number_scans,1); %section X - baseline
indi_sub_red = cell(number_scans,1);
indi_sub_blue = cell(number_scans,1);
% indi_sub_blue_neg = cell(number_scans,1);
% indi_sub_amber_neg = cell(number_scans,1);
% indi_sub_red_pos = cell(number_scans,1);
%for m = 1:1%2
flag = 0;
for i = 1:number_trails
    baseline_amber = baseline_result_amber{i};
    baseline_red = baseline_result_red{i};
    baseline_blue = baseline_result_blue{i};
    for j = 1:number_scans
        %blue/green
        temp_data_blue = Data_all_blue{i,j};
        Nsub = temp_data_blue - baseline_blue;
%         title_blue = ['Individual blue:' num2str(j) ' Trail:' num2str(i)];
%         filename_blue = ['Individual_blue' num2str(j) '_Trail' num2str(i)];
        baseline_blue(baseline_blue==0) = 100000000;
%             sub_amber{j} = Nsub./temp_data_amber;
        temp_temp = Nsub./baseline_blue;
%         temp_nsub = temp_temp;
%         temp_nsub(temp_temp>0)=0;
%         indi_sub_blue_trails{j,i}=temp_temp;
        for p = 1:length_mask
            ROI_blue(p,i,j) = aver_roi(squeeze(mask(:,:,p)),temp_temp);
        end
        if i>10&&i<13 % just plot two figures, otherwise there will be too much figures
%             result = cal_dr_r_indi_amber(temp_temp,selpathOut,title_blue,filename_blue);
        end
        if i > start_trails
            if flag == 0
                indi_sub_blue{j} = temp_temp;
%                 indi_sub_blue_neg{j} = temp_nsub;
            else
                indi_sub_blue{j} = indi_sub_blue{j} + temp_temp;
%                 indi_sub_blue_neg{j} = indi_sub_blue_neg{j}+ temp_nsub;
            end
        end
        %amber
        temp_data_amber = Data_all_amber{i,j};
        Nsub = temp_data_amber - baseline_amber;
        title_amber = ['Individual Amber:' num2str(j) ' Trail:' num2str(i)];
        filename_amber = ['Individual_Amber' num2str(j) '_Trail' num2str(i) ];
        baseline_amber(baseline_amber==0) = 100000000;
%             sub_amber{j} = Nsub./temp_data_amber;
        temp_temp = Nsub./baseline_amber;
%         temp_nsub = temp_temp;
%         temp_nsub(temp_temp>0)=0;
%         indi_sub_amber_trails{j,i}=temp_temp;
        for p = 1:length_mask
            ROI_amber(p,i,j)= aver_roi(squeeze(mask(:,:,p)),temp_temp);
        end
        if i>10&&i<13
%             result = cal_dr_r_indi_amber(temp_temp,selpathOut,title_amber,filename_amber);
        end
        if i > start_trails
            if flag == 0
                indi_sub_amber{j} = temp_temp;
%                 indi_sub_amber_neg{j} = temp_nsub;
            else
                indi_sub_amber{j} = indi_sub_amber{j} + temp_temp;
%                 indi_sub_amber_neg{j} = indi_sub_amber_neg{j}+ temp_nsub;
            end
        end
        % red
        temp_data_red = Data_all_red{i,j};
        Nsub = temp_data_red - baseline_red;
        title_red = ['Individual Red:' num2str(j) ' Trail:' num2str(i)];
        filename_red = ['Individual_Red' num2str(j) '_Trail' num2str(i)];
        baseline_red(baseline_red==0) = 100000000;
%             sub_red{j} = Nsub./temp_data_red;
        temp_temp = Nsub./baseline_red;
%         temp_nsub = temp_temp;
%         temp_nsub(temp_temp<0)=0;
%         indi_sub_red_trails{j,i}=temp_temp;
        for p = 1:length_mask
            ROI_red(p,i,j) = aver_roi(squeeze(mask(:,:,p)),temp_temp);
        end
        if i>10&&i<13
%             result = cal_dr_r_indi_red(temp_temp,selpathOut,title_red,filename_red);
        end
        if i > start_trails
            if flag == 0
                indi_sub_red{j} = temp_temp;
%                 indi_sub_red_pos{j} = temp_nsub;
            else
                indi_sub_red{j} = indi_sub_red{j} + temp_temp;
%                 indi_sub_red_pos{j} = indi_sub_red_pos{j} + temp_nsub;
            end
        end
    end
    if i > start_trails
        flag = flag + 1;
    end
end

dir_avg = fullfile(selpathOut,'Average');
if ~exist(dir_avg, 'dir')
   mkdir(dir_avg)
end

% Division by len_trials
for j =1:number_scans
    indi_sub_blue{j} = indi_sub_blue{j}/(number_trails-start_trails);
    title_blue = ['Individual blue:' num2str(j) ' averageTrails'];
    file_blue = ['Individual_aver_blue' num2str(j) '_averageTrails'];
    result = cal_dr_r_indi_amber(indi_sub_blue{j},dir_avg,title_blue,file_blue);
    
    indi_sub_amber{j} = indi_sub_amber{j}/(number_trails-start_trails);
    title_amber = ['Individual Amber:' num2str(j) ' averageTrails'];
    file_amber = ['Individual_aver_Amber' num2str(j) '_averageTrails'];
    result = cal_dr_r_indi_amber(indi_sub_amber{j},dir_avg,title_amber,file_amber);

    indi_sub_red{j} = indi_sub_red{j}/(number_trails-start_trails);
    title_red = ['Individual Red:' num2str(j) ' averageTrails'];
    file_red = ['Individual_aver_Red' num2str(j) '_averageTrails'];
%     result = cal_dr_r_indi_red(indi_sub_red{j},dir_avg,title_red,file_red);
    result = cal_dr_r_indi_green(indi_sub_red{j},dir_avg,title_red,file_red);
    
% 
%     indi_sub_blue_neg{j} = indi_sub_blue_neg{j}/(number_trails-start_trails);
%     title_blue = ['IndividualNEG blue:' num2str(j) ' averageTrails'];
%     file_blue = ['IndividualNEG_aver_blue_' num2str(j) '_averageTrails'];
%     result = cal_dr_r_indi_amber(indi_sub_blue_neg{j},dir_avg,title_blue,file_blue);

%     indi_sub_amber_neg{j} = indi_sub_amber_neg{j}/(number_trails-start_trails);
%     title_amber = ['IndividualNEG Amber:' num2str(j) ' averageTrails'];
%     file_amber = ['IndividualNEG_aver_Amber_' num2str(j) '_averageTrails'];
%     result = cal_dr_r_indi_amber(indi_sub_amber_neg{j},dir_avg,title_amber,file_amber);
% 
%     indi_sub_red_pos{j} = indi_sub_red_pos{j}/(number_trails-start_trails);
%     title_red = ['IndividualPOS Red:' num2str(j) ' averageTrails'];
%     file_red = ['IndividualPOS_aver_Red_' num2str(j) '_averageTrails'];
%     result = cal_dr_r_indi_red(indi_sub_red_pos{j},dir_avg,title_red,file_red);
end

% Make a sepearate thresholding function that uses deltaR/R values from the
% mat file

%{

% thresholding
% indi_sub_amber = cell(number_scans,2); %section X - baseline
% indi_sub_red = cell(number_scans,2);
% indi_sub_blue = cell(number_scans,2);
% indi_sub_blue_neg = cell(number_scans,2);
% indi_sub_amber_neg = cell(number_scans,2);
% indi_sub_red_pos = cell(number_scans,2);
thr_amber = -4:0.2:-1;
thr_blue = -4:0.2:-1;
thr_red = 0.04:0.04:0.64;
for i = 1:number_scans
    for j =1:length(thr_amber)
        temp_amber = thr_amber(j);
        temp_blue = thr_blue(j);
        temp_red = thr_red(j);
        %amber
        tempdata_amber = indi_sub_amber{i};
        tempdata_amber = tempdata_amber*100;
        tempdata_amber(tempdata_amber>temp_amber)=0;
        curfig = figure();
        imshow(tempdata_amber,'InitialMagnification','fit');
        c = jet;
        c = flipud(c);
        colormap(c);      
%         colormap('jet');
        colorbar
        caxis([-3 0])
        title(['Amber - ' num2str(i) ' -thr-' num2str(temp_amber)])
        outputFileName = [selpathOut '\Individual_thresholding_amber_' num2str(i) '_' num2str(temp_amber) '.jpg'];
        saveas(curfig,outputFileName);

        tempdata_amber = indi_sub_amber_neg{i};
        tempdata_amber = tempdata_amber*100;
        tempdata_amber(tempdata_amber>temp_amber)=0;
        curfig = figure();
        imshow(tempdata_amber,'InitialMagnification','fit');
        c = jet;
        c = flipud(c);
        colormap(c);      
%         colormap('jet');
        colorbar
        caxis([-3 0])
        title(['Amber - ' num2str(i) ' -thr-' num2str(temp_amber)])
        outputFileName = [selpathOut '\Individual_thresholding_amber_NEG_' num2str(i) '_' num2str(temp_amber) '.jpg'];
        saveas(curfig,outputFileName);

        %blue
        tempdata_blue = indi_sub_blue{i};
        tempdata_blue = tempdata_blue*100;
        tempdata_blue(tempdata_blue>temp_blue)=0;
        curfig = figure();
        imshow(tempdata_blue,'InitialMagnification','fit');
        c = jet;
        c = flipud(c);
        colormap(c);      
%         colormap('jet');
        colorbar
        caxis([-3 0])
        title(['Blue - ' num2str(i) ' -thr-' num2str(temp_blue)])
        outputFileName = [selpathOut '\Individual_thresholding_blue_' num2str(i) '_' num2str(temp_blue) '.jpg'];
        saveas(curfig,outputFileName);

        tempdata_blue = indi_sub_blue_neg{i};
        tempdata_blue = tempdata_blue*100;
        tempdata_blue(tempdata_blue>temp_blue)=0;
        curfig = figure();
        imshow(tempdata_blue,'InitialMagnification','fit');
        c = jet;
        c = flipud(c);
        colormap(c);      
%         colormap('jet');
        colorbar
        caxis([-3 0])
        title(['Blue - ' num2str(i) ' -thr-' num2str(temp_blue)])
        outputFileName = [selpathOut '\Individual_thresholding_blue_NEG_' num2str(i) '_' num2str(temp_blue) '.jpg'];
        saveas(curfig,outputFileName);
        %red
        tempdata_red = indi_sub_red{i};
        tempdata_red = tempdata_red*100;
        tempdata_red(tempdata_red<temp_red)=0;
        curfig = figure();
        imshow(tempdata_red,'InitialMagnification','fit');
        c = jet;
        %c = flipud(c);
        colormap(c);      
%         colormap('jet');
        colorbar
        caxis([0 0.6])
        title(['Red - ' num2str(i) ' -thr-' num2str(temp_red)])
        outputFileName = [selpathOut '\Individual_thresholding_red_' num2str(i) '_' num2str(temp_amber) '.jpg'];
        saveas(curfig,outputFileName);

        tempdata_red = indi_sub_red_pos{i};
        tempdata_red = tempdata_red*100;
        tempdata_red(tempdata_red<temp_red)=0;
        curfig = figure();
        imshow(tempdata_red,'InitialMagnification','fit');
        c = jet;
        %c = flipud(c);
        colormap(c);      
%         colormap('jet');
        colorbar
        caxis([0 0.6])
        title(['Red - ' num2str(i) ' -thr-' num2str(temp_red)])
        outputFileName = [selpathOut '\Individual_thresholding_red_POS_' num2str(i) '_' num2str(temp_amber) '.jpg'];
        saveas(curfig,outputFileName);
        close all
    end
end

%}


%% ROI stimulation map
% ROI_blue = zeros(length_mask,number_trails,number_scans,2);
% ROI_amber = zeros(length_mask,number_trails,number_scans,2);
% ROI_red = zeros(length_mask,number_trails,number_scans,2);


for i = 1:length_mask
    % blue/green
    temp_fig = squeeze(ROI_blue(i,:,:));
    curfig = figure();
    imshow(temp_fig,'InitialMagnification','fit');
    c = jet;
    c = flipud(c);
    colormap(c);      
%         colormap('jet');
    colorbar
    caxis([-5 0])
    title(['Blue - dr/r inside ROI ' num2str(i)])
    xlabel('Time / s')
    ylabel('Trails')
    axis('on');
    %axis_x = (duration_time/1000)* [1:number_scans];
%     axis_x = (duration_time/1000)*20*[1:3];
%     set(gca,'XTickLabel',axis_x);
    hold on
    start_x = ceil(1000*start_t/duration_time);
    end_x = ceil(1000*end_t/duration_time);
    line([start_x start_x],[0 100],'linestyle','-','Color','r','LineWidth',1);
    line([end_x end_x],[0 100],'linestyle','-','Color','r','LineWidth',1);
    outputFileName = fullfile(selpathOut,strcat('Individual_ROIMAP_blue_ROI_',num2str(i),'.jpg'));
    saveas(curfig,outputFileName);

    % amber
    temp_fig = squeeze(ROI_amber(i,:,:));
    curfig = figure();
    imshow(temp_fig,'InitialMagnification','fit');
    c = jet;
    c = flipud(c);
    colormap(c);      
%         colormap('jet');
    colorbar
    caxis([-5 0])
    title(['Amber - dr/r inside ROI ' num2str(i)])
    xlabel('Time / s')
    ylabel('Trails')
    axis('on');
    %set(gca,'XTickLabel',axis_x);
    hold on
    line([start_x start_x],[0 100],'linestyle','-','Color','r','LineWidth',1);
    line([end_x end_x],[0 100],'linestyle','-','Color','r','LineWidth',1);
    outputFileName = fullfile(selpathOut,strcat('Individual_ROIMAP_amber_ROI_',num2str(i),'.jpg'));
    saveas(curfig,outputFileName);

    % red
    temp_fig = squeeze(ROI_red(i,:,:));
    curfig = figure();
    imshow(temp_fig,'InitialMagnification','fit');
%         c = jet;
%         c = flipud(c);
%         colormap(c);      
    colormap('jet');
    colorbar
    caxis([0 1])
    title(['Red - dr/r inside ROI ' num2str(i)])
    xlabel('Time / s')
    ylabel('Trails')
    axis('on');
    %set(gca,'XTickLabel',axis_x);
    hold on
    line([start_x start_x],[0 100],'linestyle','-','Color','r','LineWidth',1);
    line([end_x end_x],[0 100],'linestyle','-','Color','r','LineWidth',1);
    outputFileName = fullfile(selpathOut,strcat('Individual_ROIMAP_red_ROI_',num2str(i),'.jpg'));
    saveas(curfig,outputFileName);
end

%% store .mat
% Data_store_blue = cell(number_trails,number_scans); %blue may refer to green, sometimes
% Data_store_amber = cell(number_trails,number_scans);
% Data_store_red = cell(number_trails,number_scans);

mat_dir = fullfile(selpathOut,'mat_files');
if ~exist(mat_dir, 'dir')
   mkdir(mat_dir)
end

% save(fullfile(mat_dir,'data_blue.mat'),'Data_store_blue');
% save(fullfile(mat_dir,'data_amber.mat'),'Data_store_amber');
% save(fullfile(mat_dir,'data_red.mat'),'Data_store_red');
% 
% Data_aver_blue = cell(number_sections,1);
% Data_aver_amber = cell(number_sections,1);
% Data_aver_red = cell(number_sections,1);
% for j =1:number_sections
%     for i = start_trails:number_trails
%         if i == start_trails
%             Data_aver_blue{j}=Data_store_blue{i,j};
%             Data_aver_amber{j}=Data_store_amber{i,j};
%             Data_aver_red{j}=Data_store_red{i,j};
%         else
%             Data_aver_blue{j}=Data_aver_blue{j}+Data_store_blue{i,j};
%             Data_aver_amber{j}=Data_aver_amber{j}+Data_store_amber{i,j};
%             Data_aver_red{j}=Data_aver_red{j}+Data_store_red{i,j};
%         end
%     end
%     Data_aver_blue{j}=Data_aver_blue{j}/(number_trails-start_trails);
%     Data_aver_amber{j}=Data_aver_amber{j}/(number_trails-start_trails);
%     Data_aver_red{j}=Data_aver_red{j}/(number_trails-start_trails);
% end
% save(fullfile(mat_dir,'data_blue_mod.mat'),'indi_sub_blue_neg');
% save(fullfile(mat_dir,'data_amber_mod.mat'),'indi_sub_amber_neg');
% save(fullfile(mat_dir,'data_red_mod.mat'),'indi_sub_red_pos');
save(fullfile(mat_dir,'data_blue.mat'),'indi_sub_blue');
save(fullfile(mat_dir,'data_amber.mat'),'indi_sub_amber');
save(fullfile(mat_dir,'data_red.mat'),'indi_sub_red');
close all;
