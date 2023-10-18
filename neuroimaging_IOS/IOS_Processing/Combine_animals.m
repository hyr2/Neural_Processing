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
file_X = dir_sorted(mouse_bc8);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_bc8 = {};
struct_bc8_580 = {};
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
    end
end

% BHC7
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
    end
end
struct_bhc7 = [struct_bhc7(1) , struct_bhc7];   % copy the baseline since only one baseline session exists
% BC7
file_X = dir_sorted(mouse_bc7);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_bc7 = {};
struct_bc7_580 = {};
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
    end
end
% RH3
file_X = dir_sorted(mouse_rh3);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh3 = {};
struct_rh3_580 = {};
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
file_X = dir_sorted(mouse_rh11);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh11 = {};
struct_rh11_580 = {};
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
    end
end
% RH7
file_X = dir_sorted(mouse_rh7);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh7 = {};
struct_rh7_580 = {};
iter_local = 0; 
xaxis_rh7 = [-5,-3,2,7,14,21,28,35,42,49,56];  %(baselines: 10-24,10-27)
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
    end
end
% RH8
file_X = dir_sorted(mouse_rh8);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh8 = {};
struct_rh8_580 = {};
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
    end
end
% RH9
file_X = dir_sorted(mouse_rh9);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
struct_rh9 = {};
struct_rh9_580 = {};
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
% filename_fig = fullfile(output_dir,'bbc5_out');
% out_bbc5 = func_plot3d(struct_bbc5,xaxis_bbc5,filename_fig);

% Statistics on average shifts post stroke
X_vec = abs([out_rh3{2},out_rh7{2},out_rh8{2},out_rh9{2},out_rh11{2},out_bc8{2}]);
avg_X = mean(X_vec);
sem_X = std(X_vec)/sqrt(length(X_vec));

% Statistics on average shifts post stroke
Y_vec = abs([out_rh3{3},out_rh7{3},out_rh8{3},out_rh9{3},out_rh11{3},out_bc8{3}]);
avg_Y = mean(Y_vec);
sem_Y = std(Y_vec)/sqrt(length(Y_vec));

% Statistics on average relative area post stroke
area_vec = [out_rh3{1},out_rh7{1},out_rh8{1},out_rh9{1},out_rh11{1},out_bc8{1}];   % bhc7,bbc5 not included. Bad animals
avg_area = mean(area_vec);
sem_area = std(area_vec)/sqrt(length(area_vec));

% Statistics on average distance shifts post stroke
shift_vec_early = [out_rh3{5}(find(out_rh3{4} < 21))',out_rh7{5}(find(out_rh7{4} < 21))',...
    out_rh8{5}(find(out_rh8{4} < 21))',out_rh9{5}(find(out_rh9{4} < 21))',...
    out_rh11{5}(find(out_rh11{4} < 21))',out_bhc7{5}(find(out_bhc7{4} < 21))',...
    out_bc8{5}(find(out_bc8{4} < 21))'];
shift_vec_chronic = [out_rh3{5}(find(out_rh3{4} >= 21))',out_rh7{5}(find(out_rh7{4} >= 21))',...
    out_rh8{5}(find(out_rh8{4} >= 21))',out_rh9{5}(find(out_rh9{4} >= 21))',...
    out_rh11{5}(find(out_rh11{4} >= 21))',out_bc8{5}(find(out_bc8{4} >= 21))'];

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
histogram(shift_vec_early, 'BinWidth', 0.28,'FaceColor', [0.1, 0.9, 0.9],'EdgeColor',[0.08, 0.65, 0.9]);
hold on;
histogram(-shift_vec_chronic, 'BinWidth', 0.28,'FaceColor', [0.9, 0.1, 0.2],'EdgeColor',[1.0, 0.05, 0.1]);
xlim([-2,2.1])
xlabel('Inter-day Shift (mm)');
ylabel('Count');

%% Plots for Figure1G (area):
% No bbc5 and no bhc7
xaxis_ideal = [-3, 2, 7, 14,21,28,35,42,49,56];

arr_area_all = zeros(num_animals,size(xaxis_ideal,2));
arr_area_all(1,:) = out_rh3{6};
arr_area_all(2,:) = out_rh7{6};
arr_area_all(3,:) = out_rh8{6};
arr_area_all(4,:) = out_rh9{6};     
arr_area_all(5,:) = out_rh11{6};
arr_area_all(6,:) = out_bc7{6};     % day 7 is messed up
arr_area_all(7,:) = out_bc8{6};
area_avg = nanmean(arr_area_all,1);
figure;
plot(xaxis_ideal,area_avg);

%% Saving data for future use
save(fullfile(output_folder,'output_data.mat'),'xaxis_ideal','arr_area_all','shift_vec_early','shift_vec_chronic');


