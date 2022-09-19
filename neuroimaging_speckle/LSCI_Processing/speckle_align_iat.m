function speckle_align_iat(d,n)
% speckle_align_iat Averages and aligns sequence of .sc data files
%   speckle_align_iat(d,n) loads sequence of .sc files and aligns them using
%   the Image Alignment Toolbox (IAT) with the forwards-addiitve version of
%   the ECC image alignment algorithm. The first n frames are used as the
%   alignment reference. Each aligned frame is saved into a .mat file
%
%   d = Directory of .sc files to align
%   n = Number of .sc files to average per frame
%
%   IAT -> http://iatool.net/
%

r = [0 0.2]; % Speckle contrast range to use during image registration

% Suppress image display size warnings
warning('off', 'Images:initSize:adjustingMag');

if(d(end) == '/'); d = d(1:end-1); end;

if(exist(sprintf('%s/tform.mat',d),'file') == 2)
    button = questdlg('This data has been previously registered using this function. Would you like to apply that transform?',...
        'Apply Previous Transform?','Yes','No','No');
    if(strcmp(button,'Yes'))
        sprintf('Using previous transform to align data');
        speckle_realign_iat(d);
        return;
    end
end

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
t = mean(t,1) - t(1,1); % Calculate time vector from first file

% Get absolute time for first frame
t_start = files(1).datenum;

% Optimize data structure for parfor
files = reshape(files(1:n*N),[n,N]);

% Define reference frame from first frame
SC_REF = mean(read_subimage(files(:,1)),3)';
SC_END = mean(read_subimage(files(:,end)),3)';

% Prompt user to define and verify region used for alignment
button = 'No';
SC_REF_crop = [];
rect = [];
while strcmp(button,'No')

    % Prompt user to draw rectangular ROI
    [SC_REF_crop, rect] = imcrop(mat2gray(SC_REF,r));

    % Compare to final frame
    set(gcf, 'Position', get(0, 'Screensize'));
    subplot(1,2,1);
    imshow(SC_REF,r);
    hold all;
    rectangle('Position',rect,'EdgeColor','r','LineWidth',3);
    subplot(1,2,2);
    imshow(SC_END,r);
    hold all;
    rectangle('Position',rect,'EdgeColor','r','LineWidth',3);

    % Prompt to add another ROI
    button = questdlg('Would you like to use this ROI for the image registration?',...
        'Redraw ROI?','Yes','No','Yes');

    close(gcf);
    pause(0.1);
end

% Setup image registration paramters
opts.transform = 'euclidean';
opts.levels = 3;
opts.iterations = 50;

% Initialize parallelization
if isempty(gcp('nocreate')); parpool; end
tform = zeros(2,3,N);
SC_REF_X = 1:size(SC_REF,2);
SC_REF_Y = 1:size(SC_REF,1);
fprintf('Aligning %d frames\n', N);
tic;

p = ParforProgMon('Aligning Speckle Images - ', N);
parfor i = 1:N

    % Load current speckle frame
    SC = mean(read_subimage(files(:,i)),3)';
    SC_crop = imcrop(mat2gray(SC,r), rect);

    % Align using first frame as reference
    [warp, ~] = iat_ecc(SC_crop, SC_REF_crop, opts);
    tform(:,:,i) = warp;
    [SC_REG, mask] = iat_inverse_warping(SC, warp, opts.transform, SC_REF_X, SC_REF_Y);

    % Create hybrid image using reference image to eliminate alignment gap
    SC_REG = imadd(SC_REG.*mask, SC_REF.*~mask);

    % Save data to .mat file
    parsave(sprintf('%s/Speckle_%04d.mat',output,i),SC_REG,mask);

    p.increment();
end

cd(workingDir);

% Save timing.mat in aligned data folder
save(sprintf('%s/timing.mat',output),'t','t_start');

% Save transform in original data folder
save(sprintf('%s/%s/tform.mat',pwd,d),'n','opts','tform','SC_REF_X','SC_REF_Y');

fprintf('Elapsed Time: %.2fs (%.2fs per frame)\n',toc,toc/(N));

end

 %#ok<*NASGU>
 %#ok<*PFOUS>
 %#ok<*PFBNS>
