function speckle_frame_sequence(d_ref,d,r,t_start,t_end,dt)
%speckle_frame_sequence Generates aligned SC sequence with custom timing
%   speckle_frame_sequence(d_ref,d,r,t_start,t_end,dt) uses a directory 
%   of reference .sc files to align a user-defined selection of .sc files 
%   from another directory to produce a sequence of speckle contrast images
%
%   d_ref = Directory of .sc files used for baseline
%   d = Directory of .sc files to process
%   r = Speckle display range [min max]
%   t_start = Time to start sequence (seconds)
%   t_end = Time to stop sequence (seconds)
%   dt = Time interval between frames (seconds)
%

% Suppress image display size warnings
warning('off', 'Images:initSize:adjustingMag');

% Standardize directory strings
if(d(end) == '/'); d = d(1:end-1); end;
if(d_ref(end) == '/'); d_ref = d_ref(1:end-1); end;

% Load in the baseline speckle data
workingDir = pwd; cd(d_ref);
files = dir_sorted('*.sc');
SC_REF = mean(read_subimage(files),3)';

% Load in the main speckle data
cd(sprintf('%s/%s',workingDir,d));
files = dir_sorted('*.sc');
N = length(files);

% Get timing information
f = dir('*.timing');
f = fopen(f(1).name,'rb');
ts = fread(f,'float');
fclose(f);
ts = mean(reshape(ts,[],N))/1000;
T = t_start:dt:t_end;
n = length(T);

% Prepare folder for data output
output = sprintf('%s/%s_SC',workingDir,d);
if ~exist(output,'dir')
    mkdir(output);
end

if isempty(gcp('nocreate')); parpool; end
fprintf('Processing %d frames:\n',n);
p = ParforProgMon('Generating Speckle Frames - ', N);
tic;

for i = T

    % Load speckle files between where i < t < i + dt
    SC = mean(read_subimage(files(ts >= i & ts < i + dt)),3)';

    % Convert to grayscale for registration (Mapped to [min max])
    r1 = [min(SC_REF(:)) max(SC_REF(:))];
    r2 = [min(SC(:)) max(SC(:))];
    SC_REF_norm = mat2gray(SC_REF,r1);
    SC_norm = mat2gray(SC,r2);

    % Register SC with SC_REF
    opts.transform = 'euclidean';
    opts.levels = 3;
    opts.iterations = 50;
    MX = 1:size(SC_REF,2);
    MY = 1:size(SC_REF,1);
    
    [warp, ~] = iat_ecc(SC_norm, SC_REF_norm, opts);
    [SC_REG, ~] = iat_inverse_warping(SC, warp, opts.transform, MX, MY);
    
    % Create figure
    F = figure('visible','off'); % Supress figure display
    imshow(SC_REG,r);

    % Save image to file
    imagePrint(size(SC,1), size(SC,2), 600, '-dpng', sprintf('%s/SC_%04d',output,i));

    close(F);
    p.increment();

end

fprintf('Elapsed time: %.2fs (%.2fs per frame)\n',toc,toc/n);
cd(workingDir);
