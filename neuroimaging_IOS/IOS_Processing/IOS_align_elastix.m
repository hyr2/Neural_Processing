function IOS_align_elastix(file_sc,n_avg,varargin)
%IOS_align_elastix Averages and aligns IOS raw data
% IOS_align_elastix(file_sc,n_avg) loads sequence of .tif files and aligns them
% using Elastix. The first n frames are used as the alignment reference.
% Each aligned frame is saved into a .mat file.
%
%   file_sc = Path to directory of speckle contrast files
%   n_avg = Number of speckle contrast frames to average per video frame
%
% speckle_align_elastix(file_sc,n_avg,parameter_file) performs registration
% with the user-defined Elastix parameter file. 
% 
%   parameter_file = Full path to Elastix parameter file (Default = 'Speckle/elastix_parameters.txt')
%

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'n_avg', @isscalar);
addOptional(p, 'parameter_file', which('Elastix/Parameters_BSpline.txt'), @isfile);
addOptional(p, 'out_dir', @(x) exist(x, 'dir'));
parse(p, file_sc, n_avg, varargin{:});

file_sc = p.Results.file_sc;
n_avg = p.Results.n_avg;
parameter_file = p.Results.parameter_file;
out_dir = p.Results.out_dir;

% Get list of .sc files to be aligned
files = dir_sorted(fullfile(file_sc, '*.tif'));
file_sc_temp = files(1).folder;
N_total = length(files);              
N = floor(N_total/n_avg);
idx = reshape(1:N*n_avg,n_avg,N);

% Define reference frame from first frame
filePath = fullfile(file_sc,files(1).name);
SC_REF = imread(filePath);

% Save first frame since it won't be aligned
SC_REG = SC_REF;
mask = true(size(SC_REG));
imwrite(SC_REG,fullfile(out_dir,files(1).name));
clear mask SC_REG;

% Rescale image display range for use with Elastix
SC_REF = mat2gray(SC_REF, [0 65535]);

% Initialize parallelization
if isempty(gcp('nocreate')); parpool; end
fprintf('Aligning %d frames and saving to %s\n', N - 1, out_dir);
tic;

% p = ParforProgressbar(N - 1, 'title', 'Aligning Speckle Images');
p = ParforProgMon('Aligning IOS Images - ', N-1);

parfor i = 2:N
    % Load current speckle frame
    filePath = fullfile(file_sc,files(i).name);
    SC = imread(filePath);
    SC = double(SC);
    % Rescale image display range
    SC_scaled = mat2gray(SC, [0 65535]);

    % Use Elastix + Transformix register the images
    d = elastix(SC_REF,SC_scaled,parameter_file);    
    [SC_REG, mask] = transformix(d, SC);

    % Zero pixels outside of the mask
    SC_REG(~mask) = 0;

    % Save data to .mat file
    out_file = fullfile(out_dir,files(i).name);
    SC_REG = uint16(SC_REG);
    parsave_tif(out_file,SC_REG);
%     parsave(out_file,SC_REG);
%     parsave(sprintf('%s/X_%05d.tif',out_dir,i),SC_REG);

    
    p.increment(); 

end

delete(p);

fprintf('Elapsed Time: %.2fs (%.2fs per frame)\n',toc,toc/(N));
end

