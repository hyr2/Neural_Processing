function speckle_align_norm(d,n)
%speckle_align_norm Averages, normalizes, and aligns sequence of .sc data files
%   speckle_align_norm(d,n) loads sequence of .sc files and aligns them using 
%   normalized intensity-based image registration. The first n frames are 
%   used as the alignment reference. Each aligned frame is saved into a .mat file
%
%   d = Directory of .sc files to align
%   n = Number of .sc files to average per frame
%

if(d(end) == '/'); d = d(1:end-1); end;

% Create output directory
workingDir = pwd;
output = sprintf('%s/%s_Aligned',pwd,d);
if ~exist(output,'dir')
    mkdir(output);
end

% Get list of .sc files to be aligned
cd(d);
files = dir_sorted('*sc');
N = floor(length(files)/n);

% Get timing information
f = dir('*.timing');
f = fopen(f(1).name,'rb');
t = fread(f,'float');
fclose(f);
t = mean(reshape(t,[],length(files)))/1000; % Frames -> Sequence
t = reshape(t(1:N*n),[],N); % Reshape based on n-frame averaging
t = mean(t,1) - mean(t(:,1)); % Calculate time vector from first frame

% Get absolute time for first frame
t_start = files(n).datenum;

% Optimize data structure for parfor
files = reshape(files(1:n*N),[n,N]);

% Define reference frame from first frame
SC_REF = mean(read_subimage(files(:,1)),3)';
SC_REF_r = [min(SC_REF(:)) max(SC_REF(:))];
SC_REF_norm = mat2gray(SC_REF,SC_REF_r);

% Setup image registration optimizer
[optimizer, metric] = imregconfig('multimodal');
optimizer.MaximumIterations = 200;

% Initialize parallelization
if isempty(gcp('nocreate')); parpool; end
fprintf('Aligning %d frames\n', N);
tic; 

p = ParforProgMon('Aligning Speckle Images - ', N);
parfor i = 1:N

    % Load current speckle frame
    SC = mean(read_subimage(files(:,i)),3)';
    SC_r = [min(SC(:)) max(SC(:))];
    SC_norm = mat2gray(SC,SC_r);

    % Align using first frame as reference
    SC_REG_norm = imregister(SC_norm,SC_REF_norm,'rigid',optimizer,metric);
    
    % Denormalize data
    SC_REG = SC_REG_norm*(SC_r(2) - SC_r(1)) + SC_r(1);

    % Create hybrid image for background of image sequence
    alpha = imdilate(SC_REG == 0, strel('disk',2));
    SC_REG = imadd(SC_REG.*~alpha, SC_REF.*alpha);

    % Save data to .mat file
    parsave(sprintf('%s/Speckle_%04d.mat',output,i),SC_REG);

    p.increment(); %#ok<*PFBNS>
end

cd(workingDir);

save(sprintf('%s/timing.mat',output),'t','t_start');
fprintf('Elapsed Time: %.2fs (%.2fs per frame)\n',toc,toc/(N));
    
end