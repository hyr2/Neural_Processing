function rICT_frame_sequence(d,r,cutoff,t_frames,dt,varargin)
%rICT_frame_sequence Generates sequence of rICT frames from directory of .sc files
% rICT_frame_sequence(d,r,cutoff,t_frames,dt) generates a sequence of 
% relative inverse correlation time (rICT) images at user-defined times 
% from a directory of speckle contrast files. The rICT baseline is 
% calculated over a user-defined timespan.
%
% ASSUMES 5MS EXPOSURE TIME FOR INVERSE CORRELATION TIME CALCULATIONS
%
% rICT_frame_sequence(d,r,cutoff,t_frames,dt,d_ref) loads and aligns a 
% separate directory of .sc files as the rICT baseline.
%
%
% Input Arguments:
%   d = Path to directory of .sc files
%   r = Speckle contrast display range [min max]
%   cutoff = Range of rICT values to show [0 0.5]
%   t_frames = Array of timestamps to generate frames at
%   dt = For each timestamp, average data over the following dt seconds
%
% Optional Input Arguments:
%   d_ref = Path to directory of .sc files to use as rICT baseline
%

warning('off', 'Images:initSize:adjustingMag'); % Suppress image size warnings

% Standardize directory string
if(d(end) == '/'); d = d(1:end-1); end

% Get .sc files
files = dir_sorted(fullfile(d, '*.sc'));
N_total = totalFrameCount(files);

% Get timing information
t = loadSpeckleTiming(fullfile(d, '*.timing'));
t = mean(reshape(t,[],N_total), 1);
t = t - t(1);

fprintf('Identified %d frames in %s/ collected over %.2f seconds\n', N_total, d, range(t));

% If no reference is provided, prompt user to define baseline. Otherwise,
% load all the files in the reference directory.
if(isempty(varargin))
  idx = getBaselineIndex(t);
  SC_REF = read_subimage(files, -1, -1, idx);
  SC_REF = mean(SC_REF, 3)';
else
  d_ref = varargin{1};
  SC_REF = loadReferenceSC(d_ref);
end

% Compute correlation time for baseline
CT_REF = get_tc_band(SC_REF,5e-3,1);

% Have user define craniotomy area
BW = defineCraniotomy(SC_REF, r);

% Prepare folder for data output
d_parts = strsplit(fileparts(fullfile(d, filesep)), filesep);
output = fullfile(pwd, [d_parts{end} '_rICT_Sequence']);
if ~exist(output, 'dir')
  mkdir(output);
end

% Create the figure
F = SpeckleFigure(SC_REF, r, 'visible', false);
F.showOverlay(CT_REF, cutoff, BW);

% Generate the frames by iterating over the LSCI indices
tic;
N_frames = length(t_frames);
fprintf('Generating %d frames in %s/\n', N_frames, output);
WB = waitbar(0, '', 'Name', 'Generating rICT Frames');
for i = 1:N_frames
  
  % Update the progressbar
  waitbar(i/N_frames, WB, sprintf('Frame %d of %d', i, N_frames));
  
  % Load speckle files where t_frame < t < t_frame + dt
  idx_1 = find(t > t_frames(i), 1);
  idx_2 = find(t > t_frames(i) + dt, 1) - 1;
  SC = read_subimage(files, -1, -1, idx_1:idx_2);
  SC = mean(SC, 3)';
  
  % Align frame with reference frame
  p = which('Elastix/Parameters_BSpline.txt');
  [SC_REG, mask] = alignTwoSpeckle(SC, SC_REF, p);
  
  % Calculate rICT
  CT = get_tc_band(SC_REG, 5e-3, 1);
  rICT = CT_REF./CT;
  
  % Generate the overlay alpha mask excluding values outside cutoff
  alpha = true(size(rICT));
  alpha(rICT < cutoff(1)) = false;
  alpha(rICT > cutoff(2)) = false;
  alpha = alpha & BW & mask;
  
  % Update the figure and write the frame
  F.updateOverlay(rICT, alpha);
  F.saveBMP(fullfile(output, sprintf('rICT_%04d', t_frames(i))));
  
end

delete(WB);

fprintf('Elapsed time: %.2fs (%.1f fps)\n', toc, N/toc);

end