% Combining animals
% Assumes IOS_process_z2.m has already been run using the Zidong_IOS_Full.m automation script

mouse_bc7 = '/home/hyr2-office/Documents/Data/IOS_imaging/bc7/';
mouse_rh3 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh3/';
mouse_rh7 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh7/';
mouse_rh8 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh8/';
mouse_rh9 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh9/';
mouse_rh11 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh11/';
% mouse_bbc5 = '/home/hyr2-office/Documents/Data/IOS_imaging/bbc5/';

output_dir = '/home/hyr2-office/Documents/Data/IOS_imaging/';

% BC7
file_X = dir_sorted(mouse_bc7);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_bc7 = {};
iter_local = 0; 
xaxis_bc7 = [-5,-3,2,7,14,21,28,42];  %(baselines: 10-24,10-27)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_bc7,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_bc7{iter_local} = load(File_load_local);
    end
end
% RH3
file_X = dir_sorted(mouse_rh3);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh3 = {};
iter_local = 0; 
xaxis_rh3 = [-5,-3,2,7,14,21,28];  %(baselines: 10-24,10-27)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh3,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh3{iter_local} = load(File_load_local);
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
%         File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
%         struct_bbc5{iter_local} = load(File_load_local);
%     end
% end
% RH11
file_X = dir_sorted(mouse_rh11);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh11 = {};
iter_local = 0; 
xaxis_rh11 = [-5,-3,2,7,14,21,28,35,42,49];  %(baselines: 10-24,10-27)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh11,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh11{iter_local} = load(File_load_local);
    end
end
% RH7
file_X = dir_sorted(mouse_rh7);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh7 = {};
iter_local = 0; 
xaxis_rh7 = [-5,-3,2,7,14,21,28,35,42,49,56];  %(baselines: 10-24,10-27)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh7,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh7{iter_local} = load(File_load_local);
    end
end
% RH8
file_X = dir_sorted(mouse_rh8);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh8 = {};
iter_local = 0;
xaxis_rh8 = [-3,-5,2,7,14,21,28,35,42,49,56];   %(baselines: 12-3,12-5)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh8,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh8{iter_local} = load(File_load_local);
    end
end
% RH9
file_X = dir_sorted(mouse_rh9);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh9 = {};
iter_local = 0; 
xaxis_rh9 = [-5,-3,2,7,14,21,28,35,42,49]; %(baselines: 12-16,12-17)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_rh9,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_rh9{iter_local} = load(File_load_local);
    end
end
bsl_indx = [1,2];
filename_fig = fullfile(output_dir,'bc7_out');
out_bc7 = func_plot3d(struct_bc7,xaxis_bc7,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh3_out');
out_rh3 = func_plot3d(struct_rh3,xaxis_rh3,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh7_out');
out_rh7 = func_plot3d(struct_rh7,xaxis_rh7,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh8_out');
out_rh8 = func_plot3d(struct_rh8,xaxis_rh8,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh9_out');
out_rh9 = func_plot3d(struct_rh9,xaxis_rh9,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh11_out');
out_rh11 = func_plot3d(struct_rh11,xaxis_rh11,filename_fig,bsl_indx);
% filename_fig = fullfile(output_dir,'bbc5_out');
% out_bbc5 = func_plot3d(struct_bbc5,xaxis_bbc5,filename_fig);

% Statistics on average shifts post stroke
X_vec = abs([out_rh3{2},out_rh7{2},out_rh8{2},out_rh9{2},out_rh11{2}]);
avg_X = mean(X_vec);
sem_X = std(X_vec)/sqrt(length(X_vec));

% Statistics on average shifts post stroke
Y_vec = abs([out_rh3{3},out_rh7{3},out_rh8{3},out_rh9{3},out_rh11{3}]);
avg_Y = mean(Y_vec);
sem_Y = std(Y_vec)/sqrt(length(Y_vec));

% Statistics on average relative area post stroke
area_vec = [out_rh3{1},out_rh7{1},out_rh8{1},out_rh9{1},out_rh11{1}];
avg_area = mean(area_vec);
sem_area = std(area_vec)/sqrt(length(area_vec));

string_out = sprintf('Avg shift in X : %.3f +/- %.3f mm', avg_X, sem_X);
disp(string_out);
string_out = sprintf('Avg shift in Y : %.3f +/- %.3f mm', avg_Y, sem_Y);
disp(string_out);
string_out = sprintf('Avg ratio of area : %.3f +/- %.3f', avg_area, sem_area);
disp(string_out);

% Combining X axes for Rh7-9
% filename_fig = fullfile(pwd,'combined_animals');
% func_plot2d_multiX(struct_rh7,struct_rh8,struct_rh9,xaxis_rh7,xaxis_rh8,xaxis_rh9,filename_fig);
% Combinng Y axes for Rh7-9
% func_plot2d_multiY(struct_rh7,struct_rh8,struct_rh9,xaxis_rh7,xaxis_rh8,xaxis_rh9,filename_fig);


% To add: 
% 1. average change in centroid due to stroke in X
% 2. average change in centroid due to stroke in Y