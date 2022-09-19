function speckle_align_dft(d,n)
%speckle_align_dft Fast DFT image alignment of .sc files
%   speckle_align_dft(d,n) loads sequence of .sc files and aligns them using 
%   cross-correlation image registration. The first n frames are used as
%   the alignment reference. Each aligned frame is saved into a .mat file.
%
%   WARNING: THIS ONLY HANDLES TRANSLATION MOVEMENT
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
cd(sprintf('%s/%s',workingDir,d));
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
t_start = files(n).datenum;

% Optimize data structure for parfor
files = reshape(files(1:n*N),[n,N]);

% Define first n frames as reference
SC_REF = mean(read_subimage(files(:,1)),3)';

% Initialize parallelization
if isempty(gcp('nocreate')); parpool; end
fprintf('Aligning %d frames\n', N);
tic;
p = ParforProgMon('Aligning Speckle Images - ', N);

parfor i = 1:N

    % Load current speckle frame
    SC = mean(read_subimage(files(:,i)),3)';

    % Align using first frame as reference
    [~, SC_REG_fft] = dftregistration(fft2(SC_REF),fft2(SC),10);
    SC_REG = abs(ifft2(SC_REG_fft));

    % Save data to .mat file
    parsave(sprintf('%s/Speckle_%04d.mat',output,i),SC_REG);

    p.increment(); %#ok<*PFBNS>
end

cd(workingDir);

save(sprintf('%s/timing.mat',output),'t','t_start');
fprintf('Elapsed Time: %.2fs (%.2fs per frame)\n',toc,toc/(N));

end
