% rICT script (full frame mesi fit)
% Needs properly compiled binary file for the correct OS 
% We have the compiled file for linux and MacOS

% Source_dir:  09-02-aged
%                       |
%                       |___2022-01-06(baseline)
%                       |___2022-01-09(post stroke)


% Each mouse ID folder must have only two datasets: 1. baseline 2. Post
% stroke

% The goal of this code is to perform image registration and apply the
% transformation to rICT MESI full frame fit (also computed in this script)

%% Folder management
input_folder = input('Enter source dir:\n','s');     % ...
% /home/hyr2-office/Documents/Data/Yifu_rICT/09-02-aged
parameter_file = input('Enter location of parameter file for image registration:\n','s'); % ...
% /home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/neuroimaging_speckle/Image_Registration/Parameters_48.txt
dry_run = 0;            % Nothing is saved or changed
vessel_flag = 1;        % Remove vessels 

% cd(input_folder);
folder_datasets_mouse = dir_sorted(input_folder);
folderNames = {folder_datasets_mouse([folder_datasets_mouse.isdir]).name};
folderNames = folderNames(~ismember(folderNames ,{'.','..'}));
folder_datasets_mouse = folderNames;
% Assumes 200 sequences and 15 MESI exposures
r = [0.05 0.36];
Nexp = 15;
Nseq = 200;
NFrames = Nexp * Nseq; 
idx = 95*Nexp+1:105*Nexp;      % middle 10 seq
iter_num = 1;
filename_local_sc = {};
% filename_local_sc = {};
moving = {};

%% Main processing for loop (load datasets all)
for iter_local = folder_datasets_mouse
    folder_local = fullfile(input_folder,iter_local);
    folder_local = folder_local{1};
    % Baseline (first dataset)
    folder_datasets = dir_sorted(fullfile(folder_local,'*.sc'));
    sc = read_subimage(folder_datasets,-1,-1,idx);
    sc = reshape(sc,size(sc,1),size(sc,2),Nexp,[]);
    sc = permute(sc,[2,1,3,4]);
    sc = mean(sc,4);
    filename_local_sc{iter_num} = fullfile(input_folder,[iter_local{1},'.sc']);
    if ~dry_run
        mask = defineCraniotomy(sc(:,:,9),[0.03,0.4]);
%         mask = ones(size(sc(:,:,9)));
        sc = sc .* mask;
        moving{iter_num} = mat2gray(sc(:,:,9),r);
        T = adaptthresh(moving{iter_num},0.8);         % remove vessels using locally adapative threshold
        BW = imbinarize(moving{iter_num},T);
        sc = sc .* BW;
        write_sc(sc, filename_local_sc{iter_num});
    end
    iter_num = iter_num + 1;
end
%% Registration
fixed_bl = moving{1};
for iter_num = 2:size(moving,2)     % skipping baseline in the loop
    % Transformation (Image registration file)
    d = elastix(fixed_bl,moving{iter_num},parameter_file);
    tmp_foldername = fullfile(input_folder,'transform_param',folder_datasets_mouse{iter_num});
    if ~exist(tmp_foldername)
        mkdir(tmp_foldername);
    end
    copyfile(d,tmp_foldername);
    [~,~] = transformix(d,moving{iter_num});      % This will delete the temp directory
    folder_datasets_reg{iter_num} = tmp_foldername;
end    
%% Prepare for Full Frame MESI non-linear least squares fit 
if ~dry_run
    % Computing Tau (non-linear least squares fit)
    code_dir = matlab.desktop.editor.getActiveFilename;
    idx = strfind(code_dir,'/');
    code_dir = code_dir(1:idx(end)-1);
    c_compiled_mesi = fullfile(code_dir,'mesi.linux');
    parpool;
    parfor i = 1:size(filename_local_sc,2)
        CMD = sprintf('%s %s -n 1',c_compiled_mesi,filename_local_sc{1,i});
        system(CMD);
    end
    delete(gcp('nocreate'));
end

%% Applying transformations and Compute relative ICT
fixed_local_bl = strcat(filename_local_sc{1,2},'.mesi'); 
fixed_image_bl_local = read_subimage(dir_sorted(fixed_local_bl),-1,-1,1);
for j_iter = 3:size(filename_local_sc,2)    % skipping baseline in the loop
    moving_local_st = strcat(filename_local_sc,'.mesi'); 
    moving_image_st_local = read_subimage(dir_sorted(moving_local_st{1,j_iter}),-1,-1,1);
    [REG_img_st,mask] = transformix_perm(folder_datasets_reg{1,j_iter},moving_image_st_local);
    rICT{1,j_iter} = REG_img_st./fixed_image_bl_local;  % compute rICT
    rICT_filt{1,j_iter} = imgaussfilt(rICT{1,j_iter},6);     % compute rICT
end

%% Plotting and overlay
close all;
overlay_range = [0.72,0.82];

for j_iter = 3:size(filename_local_sc,2)    % skipping baseline in the loop

    
    F = SpeckleFigure(moving{j_iter}, [0,1] , 'visible', true);
    F.showOverlay(zeros(size(moving{j_iter})), overlay_range , zeros(size(moving{j_iter})),'use_divergent_cmap', false);
    F.showScalebar();

    % Generate the overlay alpha mask excluding values outside overlay range
    alpha = 0.5*ones(size(moving{j_iter})); % transparency
    alpha(rICT{1,j_iter} < overlay_range(1)) = false;
    alpha(rICT{1,j_iter} > overlay_range(2)) = false;
    alpha = alpha;
    % Update the figure
    F.updateOverlay(rICT_filt{1,j_iter}, alpha);
    F.savePNG(fullfile(input_folder,folder_datasets_mouse{j_iter}));
end


