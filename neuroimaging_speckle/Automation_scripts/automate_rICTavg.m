%% Read txt files
source_dir = pwd;

% source_dir = 'H:\Data\10-27-2021\10-27-2021\data-b';
file_sc = fullfile(source_dir, 'data' , 'CompleteTrials_SC');
matlabTxT = fullfile(source_dir, 'extras' , 'whisker_stim.txt');
output_dir = fullfile(source_dir,'Processed');
mkdir(output_dir);

[len_trials,FramesPerSeq, stim_start_time, seq_period, stim_num, num_trials] = read_whiskerStimTxT(matlabTxT);

delta_t = seq_period; 
trials = num_trials;
bsl = floor(stim_start_time / seq_period);
bsl = bsl - 5;

%% 6-8-2021 (Awake-stim)

% Splitting analysis (20 trial buckets)
trials_vec = [0:20:trials];
for iter = [1:length(trials_vec)-1]
    output_part = fullfile(output_dir,[num2str(trials_vec(iter)+1),'-',num2str(trials_vec(iter+1))]);
    rICT_avg_fltr(file_sc,len_trials,1,[trials_vec(iter)+1,trials_vec(iter+1)],[0.018,0.35],'file_dst',output_part,'delta_t',delta_t,'baseline','F','num_BSL',bsl);
end
% rICT_avg_fltr(file_sc,len_trials,1,[trials_vec(1)+1 trials_vec(2)],[0.018 0.35],'file_dst','H:\Data\7-8-2021\CBF\awake\data\Processed\1-20\rICT-avg-fltr','delta_t',delta_t,'baseline','F','num_BSL',bsl);
% rICT_avg_fltr(file_sc,len_trials,1,[trials_vec(2)+1 trials_vec(3)],[0.018 0.35],'file_dst','H:\Data\7-8-2021\CBF\awake\data\Processed\21-40\rICT-avg-fltr','delta_t',delta_t,'baseline','F','num_BSL',bsl);
% rICT_avg_fltr(file_sc,len_trials,1,[trials_vec(3)+1 trials_vec(4)],[0.018 0.35],'file_dst','H:\Data\7-8-2021\CBF\awake\data\Processed\41-60\rICT-avg-fltr','delta_t',delta_t,'baseline','F','num_BSL',bsl);
% rICT_avg_fltr(file_sc,len_trials,1,[trials_vec(4)+1 trials_vec(5)],[0.018 0.35],'file_dst','H:\Data\7-8-2021\CBF\awake\data\Processed\61-80\rICT-avg-fltr','delta_t',delta_t,'baseline','F','num_BSL',bsl);

% Combining averages (20 trial buckets)
list = dir_sorted(output_dir);
list([1,2]) = [];
cellarray_folders = cell(1,length(list));
for iter = 1:length(list)
    if (list(iter).isdir)
        tmp = fullfile(list(iter).folder,list(iter).name);
%         tmp = fullfile(tmp,'rICT-avg-fltr');
        tmp = dir_sorted(fullfile(tmp, '*.mat'));
        cellarray_folders{iter} = tmp;
    end
end

% Get dimensions of the image:
load(fullfile(cellarray_folders{1}(1).folder, cellarray_folders{iter}(1).name));
dim_CT = size(avg);
% Performing averaging and computing standard deviation
output_dir_alpha = fullfile(output_dir,['1-',num2str(trials)]);
mkdir(output_dir_alpha);
for iter_seq = 1:len_trials
    filename = fullfile(output_dir_alpha,strcat('CT-',sprintf('%03d',iter_seq),'.mat'));
    arr = double.empty(dim_CT(1),dim_CT(2),0);
    arr_S = double.empty(dim_CT(1),dim_CT(2),0);
    N_trials_tmp = 0;
    for iter = 1:length(list)
        load(fullfile(cellarray_folders{iter}(1).folder, cellarray_folders{iter}(iter_seq).name));
        arr = cat(3,arr,avg);
        S = S.^2;
        N_trials_tmp = N_trials + N_trials_tmp;
        arr_S = cat(3,arr_S,S);
        clear avg S N_trials
    end
    avg = mean(arr,3,'omitnan');
    S = sqrt(sum(arr_S,3,'omitnan')/(length(list))^2);
    N_trials = N_trials_tmp;
    save(filename, 'avg' , 'S' , 'N_trials');
end

% Deleting subfolders of trials
for iter = 1:length(list)
    if (list(iter).isdir)
        tmp = fullfile(list(iter).folder,list(iter).name);
        rmdir(tmp,'s');
    end
end