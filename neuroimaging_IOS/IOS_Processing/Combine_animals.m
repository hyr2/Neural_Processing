% Combining animals

mouse_rh3 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh3/';
mouse_rh7 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh7/';
mouse_rh8 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh8/';
mouse_rh9 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh9/';
mouse_rh11 = '/home/hyr2-office/Documents/Data/IOS_imaging/rh11/';
mouse_bbc5 = '/home/hyr2-office/Documents/Data/IOS_imaging/bbc5/';

output_dir = '/home/hyr2-office/Documents/Data/IOS_imaging/';

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
file_X = dir_sorted(mouse_bbc5);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_bbc5 = {};
iter_local = 0; 
xaxis_bbc5 = [-5,-3,2,7,14,21];  %(baselines: 10-24,10-27)
for iter_filename = file_X
    iter_local = iter_local + 1;
    iter_filename = string(iter_filename);
    source_dir = fullfile(mouse_bbc5,iter_filename);
    disp(source_dir);
    if isdir(source_dir)
        File_load_local = fullfile(source_dir,'Processed','mat_files','580nm_processed.mat');
        struct_bbc5{iter_local} = load(File_load_local);
    end
end
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
filename_fig = fullfile(output_dir,'rh3_out');
func_plot3d(struct_rh3,xaxis_rh3,filename_fig,bsl_indx);
filename_fig = fullfile(output_dir,'rh7_out');
func_plot3d(struct_rh7,xaxis_rh7,filename_fig);
filename_fig = fullfile(output_dir,'rh8_out');
func_plot3d(struct_rh8,xaxis_rh8,filename_fig);
filename_fig = fullfile(output_dir,'rh9_out');
func_plot3d(struct_rh9,xaxis_rh9,filename_fig);
filename_fig = fullfile(output_dir,'rh11_out');
func_plot3d(struct_rh11,xaxis_rh11,filename_fig);
filename_fig = fullfile(output_dir,'bbc5_out');
func_plot3d(struct_bbc5,xaxis_bbc5,filename_fig);

% Combining X axes for Rh7-9
% filename_fig = fullfile(pwd,'combined_animals');
% func_plot2d_multiX(struct_rh7,struct_rh8,struct_rh9,xaxis_rh7,xaxis_rh8,xaxis_rh9,filename_fig);
% Combinng Y axes for Rh7-9
% func_plot2d_multiY(struct_rh7,struct_rh8,struct_rh9,xaxis_rh7,xaxis_rh8,xaxis_rh9,filename_fig);
