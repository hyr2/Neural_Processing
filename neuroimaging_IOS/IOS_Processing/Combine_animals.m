% Combining animals
% Assumes IOS_process_z2.m has already been run using the Zidong_IOS_Full.m automation script

output_folder = '/home/hyr2-office/Documents/Data/IOS_imaging';

mouse_bc7 = '/home/hyr2-office/Documents/Data/IOS_imaging/bc7/';
mouse_bc8 = '/home/hyr2-office/Documents/Data/IOS_imaging/bc8/';
mouse_rh3 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh3/';
mouse_rh7 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh7/';
mouse_rh8 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh8/';
mouse_rh9 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh9/';
mouse_rh11 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh11/';
% mouse_bbc5 = '/home/hyr2-office/Documents/Data/IOS_imaging/bbc5/';
mouse_bhc7 = '/home/hyr2-office/Documents/Data/IOS_imaging/bhc7/';

num_animals = 7;    % number of animals     (only for area normalization)
output_dir = '/home/hyr2-office/Documents/Data/IOS_imaging/';

% BC8
shank_max_id_i = 1;   % Shank A
shank_max_id_f = 1;   % Shank A or Shank B (both have heightened spiking activity)
file_X = dir_sorted(mouse_bc8);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_bc8 = {};
struct_bc8_580 = {};
struct_bc8_shank = {};
iter_local = 0; 
xaxis_bc8 = [-5,-3,2,7,14,21,42];  
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_bc8,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','480nm_processed.mat');
        struct_bc8{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_bc8_580{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        tmp_load = load(File_load_local);
        struct_bc8_shank{iter_local}.I = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_i));
        struct_bc8_shank{iter_local}.F = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_f));
    end
end

% BHC7
shank_max_id_i = 2;   % Shank B
shank_max_id_f = 2;   % Shank B
file_X = dir_sorted(mouse_bhc7);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_bhc7 = {};
struct_bhc7_580 = {};
iter_local = 0; 
xaxis_bhc7 = [-5,-3,7,14];  
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_bhc7,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','480nm_processed.mat');
        struct_bhc7{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_bhc7_580{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        tmp_load = load(File_load_local);
        struct_bhc7_shank{iter_local}.I = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_i));
        struct_bhc7_shank{iter_local}.F = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_f));
    end
end
struct_bhc7 = [struct_bhc7(1) , struct_bhc7];   % copy the baseline since only one baseline session exists
struct_bhc7_580 = [struct_bhc7_580(1) , struct_bhc7_580];   % copy the baseline since only one baseline session exists
struct_bhc7_shank = [struct_bhc7_shank(1) , struct_bhc7_shank];   % copy the baseline since only one baseline session exists
% BC7
shank_max_id_i = 1;   % Shank A
shank_max_id_f = 4;   % Shank D
file_X = dir_sorted(mouse_bc7);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_bc7 = {};
struct_bc7_580 = {};
struct_bc7_shank = {};
iter_local = 0; 
xaxis_bc7 = [-5,-3,2,7,14,21,28,42];  %(baselines: 10-24,10-27)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_bc7,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','480nm_processed.mat');
        struct_bc7{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_bc7_580{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        tmp_load = load(File_load_local);
        struct_bc7_shank{iter_local}.I = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_i));
        struct_bc7_shank{iter_local}.F = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_f));
    end
end
% RH3
shank_max_id_i = 1;   % Shank A
shank_max_id_f = 2;   % Shank B
file_X = dir_sorted(mouse_rh3);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh3 = {};
struct_rh3_580 = {};
struct_rh3_shank = {};
iter_local = 0; 
xaxis_rh3 = [-5,-3,2,7,14,21,28];  %(baselines: 10-24,10-27)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh3,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','480nm_processed.mat');
        struct_rh3{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh3_580{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        tmp_load = load(File_load_local);
        struct_rh3_shank{iter_local}.I = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_i));
        struct_rh3_shank{iter_local}.F = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_f));
    end
end
% BBC5
% file_X = dir_sorted(mouse_bbc5);
% file_X = {file_X.name};
% file_X =  file_X(~ismember(file_X,{'.','..'}));
% struct_bbc5 = {};
% iter_local = 0; 
% xaxis_bbc5 = [-5,-3,2,7,14,21];  %(baselines: 10-24,10-27)
% for iter_filename = file_X
%     iter_local = iter_local + 1;
%     iter_filename = string(iter_filename);
%     source_dir = fullfile(mouse_bbc5,iter_filename);
%     disp(source_dir);
%     if isdir(source_dir)
%         File_load_local = fullfile(source_dir,'Processed','mat_files','480nm_processed.mat');
%         struct_bbc5{iter_local} = load(File_load_local);
%     end
% end
% RH11
shank_max_id_i = 3;   % Shank C
shank_max_id_f = 3;   % Shank C
file_X = dir_sorted(mouse_rh11);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh11 = {};
struct_rh11_580 = {};
struct_rh11_shank = {};
iter_local = 0; 
xaxis_rh11 = [-5,-3,2,7,14,21,28,35,42,49,56];  %(baselines: 10-24,10-27)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh11,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','480nm_processed.mat');
        struct_rh11{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh11_580{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        tmp_load = load(File_load_local);
        struct_rh11_shank{iter_local}.I = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_i));
        struct_rh11_shank{iter_local}.F = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_f));
    end
end
% RH7
shank_max_id_i = 2;   % Shank B (Shank C few working electrodes)
shank_max_id_f = 4;   % Shank D
file_X = dir_sorted(mouse_rh7);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh7 = {};
struct_rh7_580 = {};
struct_rh7_shank = {};
iter_local = 0; 
xaxis_rh7 = [-5,-3,2,7,14,21,28,35,42,49];  %(baselines: 10-24,10-27)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh7,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','480nm_processed.mat');
        struct_rh7{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh7_580{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        tmp_load = load(File_load_local);
        struct_rh7_shank{iter_local}.I = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_i));
        struct_rh7_shank{iter_local}.F = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_f));
    end
end
% RH8
shank_max_id_i = 1;   % Shank A 
shank_max_id_f = 3;   % Shank C
file_X = dir_sorted(mouse_rh8);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh8 = {};
struct_rh8_580 = {};
struct_rh8_shank = {};
iter_local = 0;
xaxis_rh8 = [-5,-3,2,7,14,21,28,35,42,49,56];   %(baselines: 12-3,12-5)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh8,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','480nm_processed.mat');
        struct_rh8{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh8_580{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        tmp_load = load(File_load_local);
        struct_rh8_shank{iter_local}.I = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_i));
        struct_rh8_shank{iter_local}.F = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_f));
    end
end
% RH9
shank_max_id_i = 1;   % Shank A  same as Shank B
shank_max_id_f = 3;   % Shank C
file_X = dir_sorted(mouse_rh9);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh9 = {};
struct_rh9_580 = {};
struct_rh9_shank = {};
iter_local = 0; 
xaxis_rh9 = [-5,-3,2,7,14,21,28,35,42,49]; %(baselines: 12-16,12-17)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh9,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','480nm_processed.mat');
        struct_rh9{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh9_580{iter_local} = load(File_load_local);
        File_load_local = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        tmp_load = load(File_load_local);
        struct_rh9_shank{iter_local}.I = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_i));
        struct_rh9_shank{iter_local}.F = extract_centroid_from_mask(tmp_load.mask(:,:,shank_max_id_f));
    end
end
bsl_indx = [1,2];
filename_fig = fullfile(output_dir,'bc7_out');
out_bc7 = func_plot3d(struct_bc7,struct_bc7_580,xaxis_bc7,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh3_out');
out_rh3 = func_plot3d(struct_rh3,struct_rh3_580,xaxis_rh3,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh7_out');
out_rh7 = func_plot3d(struct_rh7,struct_rh7_580,xaxis_rh7,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh8_out');
out_rh8 = func_plot3d(struct_rh8,struct_rh8_580,xaxis_rh8,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh9_out');
out_rh9 = func_plot3d(struct_rh9,struct_rh9_580,xaxis_rh9,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh11_out');
out_rh11 = func_plot3d(struct_rh11,struct_rh11_580,xaxis_rh11,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'bhc7_out');
out_bhc7 = func_plot3d(struct_bhc7,struct_bhc7_580,xaxis_bhc7,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'bc8_out');
out_bc8 = func_plot3d(struct_bc8,struct_bc8_580,xaxis_bc8,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'bhc7_out');
out_bhc7 = func_plot3d(struct_bhc7,struct_bhc7_580,xaxis_bhc7,filename_fig,bsl_indx);
% filename_fig = fullfile(output_dir,'bbc5_out');
% out_bbc5 = func_plot3d(struct_bbc5,xaxis_bbc5,filename_fig);

% Distance between shanks and cortical activation center
bsl_distances = zeros(1,2*8);
bsl_distances([1,2]) = [norm(struct_bc7_580{1,1}.wv_580.Coord_c - struct_bc7_shank{1,1}.I),norm(struct_bc7_580{1,2}.wv_580.Coord_c - struct_bc7_shank{1,2}.I)];
bsl_distances([3,4]) = [norm(struct_bc8_580{1,1}.wv_580.Coord_c - struct_bc8_shank{1,1}.I),norm(struct_bc8_580{1,2}.wv_580.Coord_c - struct_bc8_shank{1,2}.I)];
bsl_distances([5,6]) = [norm(struct_rh3_580{1,1}.wv_580.Coord_c - struct_rh3_shank{1,1}.I),norm(struct_rh3_580{1,2}.wv_580.Coord_c - struct_rh3_shank{1,2}.I)];
bsl_distances([7,8]) = [norm(struct_rh7_580{1,1}.wv_580.Coord_c - struct_rh7_shank{1,1}.I),norm(struct_rh7_580{1,2}.wv_580.Coord_c - struct_rh7_shank{1,2}.I)];
bsl_distances([9,10]) = [norm(struct_rh8{1,1}.wv_480.Coord_c - struct_rh8_shank{1,1}.I),norm(struct_rh8{1,2}.wv_480.Coord_c - struct_rh8_shank{1,2}.I)];
bsl_distances([11,12]) = [norm(struct_rh9_580{1,1}.wv_580.Coord_c - struct_rh9_shank{1,1}.I),norm(struct_rh9_580{1,2}.wv_580.Coord_c - struct_rh9_shank{1,2}.I)];
bsl_distances([13,14]) = [norm(struct_rh11_580{1,1}.wv_580.Coord_c - struct_rh11_shank{1,1}.I),norm(struct_rh11_580{1,2}.wv_580.Coord_c - struct_rh11_shank{1,2}.I)];
bsl_distances([15,16]) = [norm(struct_bhc7_580{1,1}.wv_580.Coord_c - struct_bhc7_shank{1,1}.I),norm(struct_bhc7_580{1,2}.wv_580.Coord_c - struct_bhc7_shank{1,2}.I)];
bsl_distances = 8e-3 * bsl_distances;
chr_distances = zeros(1,2*7);
chr_distances([1,2]) = [norm(struct_bc7_580{1,7}.wv_580.Coord_c - struct_bc7_shank{1,7}.F),norm(struct_bc7_580{1,8}.wv_580.Coord_c - struct_bc7_shank{1,8}.F)];
chr_distances([3,4]) = [norm(struct_bc8_580{1,6}.wv_580.Coord_c - struct_bc8_shank{1,6}.F),norm(struct_bc8_580{1,7}.wv_580.Coord_c - struct_bc8_shank{1,7}.F)];
chr_distances([5,6]) = [norm(struct_rh3_580{1,6}.wv_580.Coord_c - struct_rh3_shank{1,6}.F),norm(struct_rh3_580{1,7}.wv_580.Coord_c - struct_rh3_shank{1,7}.F)];
chr_distances([7,8]) = [norm(struct_rh7_580{1,9}.wv_580.Coord_c - struct_rh7_shank{1,9}.F),norm(struct_rh7_580{1,10}.wv_580.Coord_c - struct_rh7_shank{1,10}.F)];
chr_distances([9,10]) = [norm(struct_rh8{1,8}.wv_480.Coord_c - struct_rh8_shank{1,8}.F),norm(struct_rh8{1,9}.wv_480.Coord_c - struct_rh8_shank{1,9}.F)];
chr_distances([11,12]) = [norm(struct_rh9_580{1,8}.wv_580.Coord_c - struct_rh9_shank{1,8}.F),norm(struct_rh9_580{1,9}.wv_580.Coord_c - struct_rh9_shank{1,9}.F)];
chr_distances([13,14]) = [norm(struct_rh11_580{1,8}.wv_580.Coord_c - struct_rh11_shank{1,8}.F),norm(struct_rh11_580{1,9}.wv_580.Coord_c - struct_rh11_shank{1,9}.F)];
chr_distances = 8e-3 * chr_distances;
[h,p] = ttest2(bsl_distances,chr_distances,'Tail','left');
fprintf('Quote in Fig1 caption. Ephys vs IOS shift has a p-value: %0.3f', p);


% Statistics on average shifts post stroke
X_vec = abs([out_rh3{2},out_rh7{2},out_rh8{2},out_rh9{2},out_rh11{2},out_bc8{2},out_bc7{2},out_bhc7{2}]);
avg_X = nanmean(X_vec);
sem_X = nanstd(X_vec)/sqrt(sum(~isnan(X_vec)));

% Statistics on average shifts post stroke
Y_vec = abs([out_rh3{3},out_rh7{3},out_rh8{3},out_rh9{3},out_rh11{3},out_bc8{3},out_bc7{3},out_bhc7{3}]);
avg_Y = nanmean(Y_vec);
sem_Y = nanstd(Y_vec)/sqrt(sum(~isnan(Y_vec)));

% Statistics on average relative area post stroke
area_vec = [out_rh3{1},out_rh7{1},out_rh8{1},out_rh9{1},out_rh11{1},out_bc8{1},out_bc7{1},out_bhc7{1}];   % bhc7,bbc5, bc7 not included. Bad animals
avg_area = mean(area_vec);
sem_area = std(area_vec)/sqrt(sum(~isnan(area_vec)));

% Statistics on average distance shifts post stroke
% shift_vec_bsl = [out_rh3{5}(find(out_rh3{4} <= 2))',out_rh7{5}(find(out_rh7{4} <= 2))',...
%     out_rh8{5}(find(out_rh8{4} <= 2))',out_rh9{5}(find(out_rh9{4} <= 2))',...
%     out_rh11{5}(find(out_rh11{4} <= 2))',out_bhc7{5}(find(out_bhc7{4} <= 2))',...
%     out_bc8{5}(find(out_bc8{4} <= 2))',out_bc7{5}(find(out_bc7{4} <= 2))'];
shift_vec_early = [out_rh3{5}(find(out_rh3{4} < 21))',out_rh7{5}(find(out_rh7{4} < 21))',...
    out_rh8{5}(find(out_rh8{4} < 21))',out_rh9{5}(find(out_rh9{4} < 21))',...
    out_rh11{5}(find(out_rh11{4} < 21))',out_bhc7{5}(find(out_bhc7{4} < 21))',...
    out_bc8{5}(find(out_bc8{4} < 21))', out_bc7{5}(find(out_bc7{4} < 21))' ];
shift_vec_chronic = [out_rh3{5}(find(out_rh3{4} >= 21))',out_rh7{5}(find(out_rh7{4} >= 21))',...
    out_rh8{5}(find(out_rh8{4} >= 21))',out_rh9{5}(find(out_rh9{4} >= 21))',...
    out_rh11{5}(find(out_rh11{4} >= 21))',out_bc8{5}(find(out_bc8{4} >= 21))', ...
    out_bc7{5}(find(out_bc7{4} >= 21))'];

% For quoting value of shift during baseline sessions (averaged all mice)
vec_bsl_shift = [out_rh3{6},out_rh7{6},out_rh8{6},out_rh9{6},out_rh11{6},out_bc8{6},out_bhc7{6},out_bc7{6}]; % I include bhc7 which has good baselines
fprintf('Quote in Fig1 Text. Inter-session shift during pre-stroke sessions: %0.3f +/- %0.3f\n', mean(vec_bsl_shift),std(vec_bsl_shift)/sqrt(length(vec_bsl_shift)));

% For Figure 1H (new)
day_local = 2;
shift_vec_2 = [out_rh3{5}(find(out_rh3{4} == day_local))',out_rh7{5}(find(out_rh7{4} == day_local))',...
    out_rh8{5}(find(out_rh8{4} == day_local))',out_rh9{5}(find(out_rh9{4} == day_local))',...
    out_rh11{5}(find(out_rh11{4} == day_local))',out_bhc7{5}(find(out_bhc7{4} == day_local))',...
    out_bc8{5}(find(out_bc8{4} == day_local))', out_bc7{5}(find(out_bc7{4} == day_local))'];
day_local = 7;
shift_vec_7 = [out_rh3{5}(find(out_rh3{4} == day_local))',out_rh7{5}(find(out_rh7{4} == day_local))',...
    out_rh8{5}(find(out_rh8{4} == day_local))',out_rh9{5}(find(out_rh9{4} == day_local))',...
    out_rh11{5}(find(out_rh11{4} == day_local))',out_bhc7{5}(find(out_bhc7{4} == day_local))',...
    out_bc8{5}(find(out_bc8{4} == day_local))', out_bc7{5}(find(out_bc7{4} == day_local))'];
day_local = 14;
shift_vec_14 = [out_rh3{5}(find(out_rh3{4} == day_local))',out_rh7{5}(find(out_rh7{4} == day_local))',...
    out_rh8{5}(find(out_rh8{4} == day_local))',out_rh9{5}(find(out_rh9{4} == day_local))',...
    out_rh11{5}(find(out_rh11{4} == day_local))',out_bhc7{5}(find(out_bhc7{4} == day_local))',...
    out_bc8{5}(find(out_bc8{4} == day_local))', out_bc7{5}(find(out_bc7{4} == day_local))'];
day_local = 21;
shift_vec_21 = [out_rh3{5}(find(out_rh3{4} == day_local))',out_rh7{5}(find(out_rh7{4} == day_local))',...
    out_rh8{5}(find(out_rh8{4} == day_local))',out_rh9{5}(find(out_rh9{4} == day_local))',...
    out_rh11{5}(find(out_rh11{4} == day_local))',out_bhc7{5}(find(out_bhc7{4} == day_local))',...
    out_bc8{5}(find(out_bc8{4} == day_local))', out_bc7{5}(find(out_bc7{4} == day_local))'];
day_local = 35;
shift_vec_35 = [out_rh3{5}(find(out_rh3{4} == day_local))',out_rh7{5}(find(out_rh7{4} == day_local))',...
    out_rh8{5}(find(out_rh8{4} == day_local))',out_rh9{5}(find(out_rh9{4} == day_local))',...
    out_rh11{5}(find(out_rh11{4} == day_local))',out_bhc7{5}(find(out_bhc7{4} == day_local))',...
    out_bc8{5}(find(out_bc8{4} == day_local))', out_bc7{5}(find(out_bc7{4} == day_local))'];
day_local = 42;
shift_vec_42 = [out_rh3{5}(find(out_rh3{4} == day_local))',out_rh7{5}(find(out_rh7{4} == day_local))',...
    out_rh8{5}(find(out_rh8{4} == day_local))',out_rh9{5}(find(out_rh9{4} == day_local))',...
    out_rh11{5}(find(out_rh11{4} == day_local))',out_bhc7{5}(find(out_bhc7{4} == day_local))',...
    out_bc8{5}(find(out_bc8{4} == day_local))', out_bc7{5}(find(out_bc7{4} == day_local))'];
day_local = 49;
shift_vec_49 = [out_rh3{5}(find(out_rh3{4} == day_local))',out_rh7{5}(find(out_rh7{4} == day_local))',...
    out_rh8{5}(find(out_rh8{4} == day_local))',out_rh9{5}(find(out_rh9{4} == day_local))',...
    out_rh11{5}(find(out_rh11{4} == day_local))',out_bhc7{5}(find(out_bhc7{4} == day_local))',...
    out_bc8{5}(find(out_bc8{4} == day_local))', out_bc7{5}(find(out_bc7{4} == day_local))'];
day_local = 56;
shift_vec_56 = [out_rh3{5}(find(out_rh3{4} == day_local))',out_rh7{5}(find(out_rh7{4} == day_local))',...
    out_rh8{5}(find(out_rh8{4} == day_local))',out_rh9{5}(find(out_rh9{4} == day_local))',...
    out_rh11{5}(find(out_rh11{4} == day_local))',out_bhc7{5}(find(out_bhc7{4} == day_local))',...
    out_bc8{5}(find(out_bc8{4} == day_local))', out_bc7{5}(find(out_bc7{4} == day_local))'];

sem_shift_by_day = [std(shift_vec_2)/sqrt(length(shift_vec_2)), std(shift_vec_7)/sqrt(length(shift_vec_7)), std(shift_vec_14)/sqrt(length(shift_vec_14)), ...
    std(shift_vec_21)/sqrt(length(shift_vec_21)), std(shift_vec_35)/sqrt(length(shift_vec_35)), std(shift_vec_42)/sqrt(length(shift_vec_42)), ...
    std(shift_vec_49)/sqrt(length(shift_vec_49)), std(shift_vec_56)/sqrt(length(shift_vec_56))];
avg_shift_by_day = [mean(shift_vec_2),mean(shift_vec_7),mean(shift_vec_14),mean(shift_vec_21),mean(shift_vec_35),mean(shift_vec_42),mean(shift_vec_49),mean(shift_vec_56)];
day_axis = [2,7,14,21,35,42,49,56];

avg_shift_early = mean(shift_vec_early);
avg_shift_chronic = mean(shift_vec_chronic);
sem_shift_early = std(shift_vec_early)/sqrt(length(shift_vec_early));
sem_shift_chronic = std(shift_vec_chronic)/sqrt(length(shift_vec_chronic));


string_out = sprintf('Avg shift in X : %.3f +/- %.3f mm', avg_X, sem_X);
disp(string_out);
string_out = sprintf('Avg shift in Y : %.3f +/- %.3f mm', avg_Y, sem_Y);
disp(string_out);
string_out = sprintf('Avg ratio of area : %.3f +/- %.3f', avg_area, sem_area);
disp(string_out);
string_out = sprintf('Avg shift of cortical activation map (early) : %.3f +/- %.3f mm', avg_shift_early, sem_shift_early);
disp(string_out);
string_out = sprintf('Avg shift of cortical activation map (chronic) : %.3f +/- %.3f mm', avg_shift_chronic, sem_shift_chronic);
disp(string_out);

%% Statistical tests
[h,p] = ttest2(shift_vec_early,shift_vec_chronic,'Tail','right');
string_out = sprintf('The shift in cortical activity map has a p-value of: %.4f', p);
disp(string_out);

%% Plots for Figure1G (inter-day shift):
histogram(shift_vec_early, 'BinWidth', 0.25,'FaceColor', [0.1, 0.9, 0.9],'EdgeColor',[0.08, 0.65, 0.9]);
hold on;
histogram(-shift_vec_chronic, 'BinWidth', 0.25,'FaceColor', [0.9, 0.1, 0.2],'EdgeColor',[1.0, 0.05, 0.1]);
xlim([-2,2.1])
xlabel('Inter-day Shift (mm)');
ylabel('Count');

%% Plots for Figure1F (area):
% No bbc5 and no bhc7
xaxis_ideal = [-3, 2, 7, 14,21,28,35,42,49,56];
num_animals = 6;
arr_area_all = zeros(num_animals,size(xaxis_ideal,2));
arr_area_all(1,:) = out_rh3{6};
arr_area_all(2,:) = out_rh7{6};
arr_area_all(3,:) = out_rh8{6};
arr_area_all(4,:) = out_rh9{6};     
arr_area_all(5,:) = out_rh11{6};
% arr_area_all(6,:) = out_bc7{6};     % day 7 is messed up
arr_area_all(6,:) = out_bc8{6};
area_avg = nanmean(arr_area_all,1);
figure;
plot(xaxis_ideal,area_avg);

%% Saving data for future use
save(fullfile(output_folder,'output_data.mat'),'xaxis_ideal','arr_area_all','shift_vec_early','shift_vec_chronic','chr_distances','bsl_distances','avg_shift_by_day','sem_shift_by_day','day_axis');


