function speckle_align_elastix(file_sc,n_avg,varargin)
%speckle_align_elastix Averages and aligns speckle contrast data
% speckle_align_elastix(file_sc,n_avg) loads sequence of .sc files and aligns them
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
parse(p, file_sc, n_avg, varargin{:});

file_sc = p.Results.file_sc;
n_avg = p.Results.n_avg;
parameter_file = p.Results.parameter_file;

% Get list of .sc files to be aligned
files = dir_sorted(fullfile(file_sc, '*.sc'));
file_sc_temp = files(1).folder;
% N_total = totalFrameCount(files);   % use if you did not average in FOIL
N_total = length(files);              % use if you averaged in FOIL
N = floor(N_total/n_avg);
idx = reshape(1:N*n_avg,n_avg,N);

% Get timing information
% t = loadSpeckleTiming(fullfile(file_sc, '*.timing'));
% t = reshape(t(1:floor(numel(t)/N_total)*n_avg*N),[],N);
% t = mean(t, 1);
% t_start = getSpeckleStartTime(fullfile(file_sc, '*.log')) + seconds(t(1));
% t = t - t(1);

% Create output directory for aligned data
d_parts = strsplit(file_sc_temp, filesep);
output = fullfile(pwd, [d_parts{end-2:end} '_Aligned_Bspline_10avg']);
if ~exist(output,'dir')
  mkdir(output);
end

% Define reference frame from first frame
SC_REF = read_subimage(files, -1, -1, idx(:,1));
SC_REF = mean(SC_REF,3)';

% Save first frame since it won't be aligned
SC_REG = SC_REF;
mask = true(size(SC_REG));
write_sc(SC_REG',fullfile(output,'Speckle_0001.sc'));
% save(fullfile(output,'Speckle_0001.mat'),'SC_REG','mask');
ROI = drawROIs(SC_REG, [0.02 0.37]);
clear mask SC_REG;



% Rescale image display range for use with Elastix
SC_REF = mat2gray(SC_REF, [0.018 0.37]);

% Initialize parallelization
if isempty(gcp('nocreate')); parpool; end
fprintf('Aligning %d frames and saving to %s\n', N - 1, output);
tic;

%p = ParforProgressbar(N - 1, 'title', 'Aligning Speckle Images');
p = ParforProgMon('Aligning Speckle Images - ', N-1);

parfor i = 2:N

    % Load current speckle frame
    SC = read_subimage(files, -1, -1, idx(:,i));
    SC = mean(SC, 3)';
    
    % Rescale image display range
%     SC_scaled = mat2gray(SC, [0 0.4]);
    SC_scaled = mat2gray(SC, [0.018 0.4]);

    % Use Elastix + Transformix register the images
    d = elastix_ROI(SC_REF,SC_scaled,parameter_file,ROI);    
    [SC_REG, mask] = transformix(d, SC);

    % Zero pixels outside of the mask
    SC_REG(~mask) = 0;

    % Save data to .mat file
%     parsave(sprintf('%s/Speckle_%04d.mat',output,i),SC_REG,mask);
    psave(sprintf('%s/Speckle_%04d.sc',output,i),SC_REG');
    p.increment(); 

end

delete(p);

fprintf('Elapsed Time: %.2fs (%.2fs per frame)\n',toc,toc/(N));

end

 %#ok<*NASGU>
 %#ok<*PFOUS>
 %#ok<*PFBNS>
