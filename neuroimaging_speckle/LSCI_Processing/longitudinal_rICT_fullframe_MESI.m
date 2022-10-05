% rICT script (full frame mesi fit)
% Needs properly compiled binary file for the correct OS 
% We have the compiled file for linux and MacOS

% Source_dir:
%           |
%           |
%           |__09-02-aged
%           |__09-04B-aged
%           |__09-05-aged

% Each mouse ID folder must have only two datasets: 1. baseline 2. Post
% stroke

% The goal of this code is to perform image registration and apply the
% transformation to rICT MESI full frame fit (also computed in this script)

%% Folder management
input_folder = input('Enter source dir:\n','s');     % ...
% /home/hyr2-office/Documents/Data/Yifu_rICT/
parameter_file = input('Enter location of parameter file for image registration:\n','s'); % ...
% /home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/neuroimaging_speckle/Image_Registration/Parameters_48.txt
dry_run = 0;
% cd(input_folder);
folder_datasets_mouse = dir_sorted(input_folder);
folderNames = {folder_datasets_mouse([folder_datasets_mouse.isdir]).name};
folderNames = folderNames(~ismember(folderNames ,{'.','..'}));
folder_datasets_mouse = folderNames;
% Assumes 200 sequences and 15 MESI exposures
Nexp = 15;
Nseq = 200;
NFrames = Nexp * Nseq; 
idx = 95*Nexp+1:105*Nexp;      % middle 10 seq
iter_num = 1;
filename_local_sc_bl = {};
filename_local_sc_st = {};

%% Main processing for loop
for iter_local = folder_datasets_mouse
    folder_local = fullfile(input_folder,iter_local);
    folder_local = folder_local{1};
    folder_datasets = dir_sorted(folder_local);
    folderNames = {folder_datasets([folder_datasets.isdir]).name};
    folderNames = folderNames(~ismember(folderNames ,{'.','..'}));
    folder_datasets = folderNames;
    output_dir = fullfile(input_folder,folder_datasets_mouse{1,iter_num});
    % Baseline
    folder_datasets_bl = fullfile(folder_local,folder_datasets{1});
    folder_datasets_bl = dir_sorted(fullfile(folder_datasets_bl,'*.sc'));
    sc_bl = read_subimage(folder_datasets_bl,-1,-1,idx);
    sc_bl = reshape(sc_bl,size(sc_bl,1),size(sc_bl,2),Nexp,[]);
    sc_bl = permute(sc_bl,[2,1,3,4]);
    sc_bl = mean(sc_bl,4);
    filename_local_sc_bl{iter_num} = fullfile(folder_local,[folder_datasets{1},'.sc']);
    if ~dry_run
        mask_bl = defineCraniotomy(sc_bl(:,:,9),[0.03,0.4]);
        sc_bl = sc_bl .* mask_bl;
        write_sc(sc_bl, filename_local_sc_bl{iter_num});
    end
    % Post-Stroke
    folder_datasets_st = fullfile(folder_local,folder_datasets{2});
    folder_datasets_st = dir_sorted(fullfile(folder_datasets_st,'*.sc'));
    sc_st = read_subimage(folder_datasets_st,-1,-1,idx);
    sc_st = reshape(sc_st,size(sc_st,1),size(sc_st,2),Nexp,[]);
    sc_st = permute(sc_st,[2,1,3,4]);
    sc_st = mean(sc_st,4);
    filename_local_sc_st{iter_num} = fullfile(folder_local,[folder_datasets{2},'.sc']);
    if ~dry_run
        mask_st = defineCraniotomy(sc_st(:,:,9),[0.03,0.4]);
        sc_st = sc_st .* mask_st;
        write_sc(sc_st, filename_local_sc_st{iter_num});
    end
    % Transformation (Image registration file)
    r = [0.05 0.36];
    fixed = mat2gray(sc_bl(:,:,9), r);
    moving = mat2gray(sc_st(:,:,9), r);
    d = elastix(fixed,moving,parameter_file);
    if ~exist(fullfile(output_dir,'transform_param'))
        mkdir(fullfile(output_dir,'transform_param'));
        folder_datasets = dir_sorted(folder_local);
        folderNames = {folder_datasets([folder_datasets.isdir]).name};
        folderNames = folderNames(~ismember(folderNames ,{'.','..'}));
        folder_datasets = folderNames;
    end
    copyfile(d,fullfile(output_dir,'transform_param'));
    [~,~] = transformix(d,moving);
    folder_datasets_reg{iter_num} = fullfile(folder_local,folder_datasets{contains(folder_datasets,'transform')});
    iter_num = iter_num + 1;
end
% Prepare cell array for parallel proc
cl = {};
for it = 1:size(filename_local_sc_st,2)
    cl{end+1} = filename_local_sc_st{it};
    cl{end+1} = filename_local_sc_bl{it};
end
if ~dry_run
    % Computing Tau (non-linear least squares fit)
    code_dir = matlab.desktop.editor.getActiveFilename;
    idx = strfind(code_dir,'/');
    code_dir = code_dir(1:idx(end)-1);
    c_compiled_mesi = fullfile(code_dir,'mesi.linux');
    parpool;
    parfor i = 1:size(cl,2)
        CMD = sprintf('%s %s -n 1',c_compiled_mesi,cl{1,i});
        system(CMD);
    end
    delete(gcp('nocreate'));
end

%% Applying transformations
j_iter = 1;
for j_iter = 1:size(filename_local_sc_st,2)
    moving_local_st = strcat(filename_local_sc_st,'.mesi'); 
    moving_image_st_local = read_subimage(dir_sorted(moving_local_st{1,j_iter}),-1,-1,1);
    [SC_REG,mask] = transformix_perm(folder_datasets_reg{1,j_iter},moving_image_st_local');
end
%% Compute relative CT

%% Plotting
