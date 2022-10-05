% rICT script (full frame mesi fit)
% Needs properly compiled binary file for the correct OS 
% We have the compiled file for linux and MacOS

% Folder management
input_folder = '/home/hyr2-office/Documents/Data/Yifu_rICT/';
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
% Main processing for loop
for iter_local = folder_datasets_mouse
    folder_local = fullfile(input_folder,iter_local);
    folder_local = folder_local{1};
    folder_datasets = dir_sorted(folder_local);
    folderNames = {folder_datasets([folder_datasets.isdir]).name};
    folderNames = folderNames(~ismember(folderNames ,{'.','..'}));
    folder_datasets = folderNames;
    % Transformation (Image registration file)
    folder_datasets_reg = fullfile(folder_local,folder_datasets{3});
    % Baseline
    folder_datasets_bl = fullfile(folder_local,folder_datasets{1});
    folder_datasets_bl = dir_sorted(fullfile(folder_datasets_bl,'*.sc'));
    sc_bl = read_subimage(folder_datasets_bl,-1,-1,idx);
    sc_bl = reshape(sc_bl,size(sc_bl,1),size(sc_bl,2),Nexp,[]);
    sc_bl = mean(sc_bl,4);
    filename_local_sc_bl{iter_num} = fullfile(folder_local,[folder_datasets{1},'.sc']);
    mask_bl = defineCraniotomy(sc_bl(:,:,9),[0.03,0.4]);
    sc_bl = sc_bl .* mask_bl;
    write_sc(sc_bl, filename_local_sc_bl{iter_num});
    % Post-Stroke
    folder_datasets_st = fullfile(folder_local,folder_datasets{2});
    folder_datasets_st = dir_sorted(fullfile(folder_datasets_st,'*.sc'));
    sc_st = read_subimage(folder_datasets_st,-1,-1,idx);
    sc_st = reshape(sc_st,size(sc_st,1),size(sc_st,2),Nexp,[]);
    sc_st = mean(sc_st,4);
    filename_local_sc_st{iter_num} = fullfile(folder_local,[folder_datasets{2},'.sc']);
    mask_st = defineCraniotomy(sc_st(:,:,9),[0.03,0.4]);
    sc_st = sc_st .* mask_st;
    write_sc(sc_st, filename_local_sc_st{iter_num});
    iter_num = iter_num + 1;
end
% Prepare cell array for parallel proc
cl = {};
for it = 1:size(filename_local_sc_st,2)
    cl{end+1} = filename_local_sc_st{it};
    cl{end+1} = filename_local_sc_bl{it};
end

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

%   