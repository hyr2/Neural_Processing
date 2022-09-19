function rICT_video(file_sc,n_avg,sc_range,fps,varargin)
%rICT_video Generates rICT video from speckle contrast data
% rICT_video(file_sc,n_avg,sc_range,fps) generates a video overlaying 
% relative inverse correlation time (rICT) on speckle contrast data. The 
% rICT is calculated with a user-define timespan as the baseline.
%
%   file_sc = Path to directory of speckle contrast files
%   n_avg = Number of speckle contrast frames to load and average
%   sc_range = Display range for speckle contrast data [min max]
%   fps = Framerate of the rendered video
%
% ASSUMES 5MS EXPOSURE TIME FOR INVERSE CORRELATION TIME CALCULATIONS
%
% rICT_video(file_sc,n_avg,sc_range,fps,file_sc_baseline) loads separate 
% directory of speckle contrast data for the rICT baseline.
%
%   file_sc_baseline = Path to directory of speckle contrast files
%
% rICT_video(___,Name,Value) modifies the overlay appearance using one or 
% more name-value pair arguments.
%
%   'overlay_range': Set display range for rICT overlay (Default = [0 0.5])
%
%   'show_scalebar': Toggle display of the scalebar (Default = true)
%       Scale is hard-coded in SpeckleFigure to that of the multimodal system
%
%   'show_timestamp': Toggle display of timestamp overlay (Default = true)
%

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'n_avg', @isscalar);
validateFnc = @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'});
addRequired(p, 'sc_range', validateFnc);
addRequired(p, 'fps', @isscalar);
addOptional(p, 'file_sc_baseline', '', @(x) exist(x, 'dir'));
addParameter(p, 'overlay_range', [0 0.5], validateFnc);
addParameter(p, 'show_scalebar', true, @islogical);
addParameter(p, 'show_timestamp', true, @islogical);
parse(p, file_sc, n_avg, sc_range, fps, varargin{:});

file_sc = p.Results.file_sc;
n_avg = p.Results.n_avg;
sc_range = p.Results.sc_range;
fps = p.Results.fps;
file_sc_baseline = p.Results.file_sc_baseline;
overlay_range = p.Results.overlay_range;
show_scalebar = p.Results.show_scalebar;
show_timestamp = p.Results.show_timestamp;

% Get list of speckle contrast files
files = dir_sorted(fullfile(file_sc, '*.sc'));
file_sc = files(1).folder;
[N_total, dims] = totalFrameCount(files);

% Get timing information
t = loadSpeckleTiming(fullfile(file_sc, '*.timing'));
t = mean(reshape(t,[],N_total), 1);

fprintf('Identified %d frames in %s collected over %.2f seconds\n', N_total, file_sc, range(t));

% If no reference is provided, prompt user to define baseline.
% Otherwise, load all the files in the reference directory and register
% to the first n frames of the primary data.
if isempty(file_sc_baseline)
  idx = getBaselineIndex(t);
  SC_REF = mean(read_subimage(files, -1, -1, idx), 3)';
  mask = true(size(SC_REF));
else
  SC_FIRST = mean(read_subimage(files, -1, -1, 1:n_avg), 3)';
  [SC_REF, mask] = loadReferenceSC(file_sc_baseline, SC_FIRST);
end

% Compute correlation time for baseline
CT_REF = get_tc_band(SC_REF,5e-3,0.3); % original value for beta is 1

% Reshape data for parallelization
N = floor(N_total/n_avg);
idx = reshape(1:N*n_avg,n_avg,N);
t = reshape(t(1:N*n_avg),[],N);
t = mean(t,1);
t = t - t(1);

% Have user define craniotomy area
BW = defineCraniotomy(SC_REF,sc_range);
BW = BW & mask; % Mask output from reference image registration
BW(SC_REF > sc_range(2)) = false;

% Create the figure
F = SpeckleFigure(SC_REF, sc_range, 'visible', true);
F.showOverlay(zeros(dims), overlay_range, zeros(dims));

if show_scalebar
  F.showScalebar();
end

if show_timestamp
  F.showTimestamp(0);
end

% Open the VideoWriter
d_parts = strsplit(file_sc, filesep);
output = fullfile(pwd, [d_parts{end} '_rICT.mp4']);
v = VideoWriter(output, 'MPEG-4');
v.FrameRate = fps;
open(v);

% Generate the frames by iterating over the LSCI indices 
tic;
fprintf('Rendering %d frames to %s\n', N, output);
WB = waitbar(0, '', 'Name', 'Generating Video');
for i = 1:N
  
  % Update the progressbar
  waitbar(i/N, WB, sprintf('Frame %d of %d', i, N));
  
  % Load the speckle contrast frames and average
  SC = read_subimage(files, -1, -1, idx(:,i));
  SC = mean(SC, 3)';
  
  % Calculate correlation time and relative ICT
  CT = get_tc_band(SC, 5e-3, 0.3); % original value 1
  rICT = CT_REF./CT;
  
  % Generate the overlay alpha mask excluding values outside overlay range
  alpha = true(size(rICT));
  alpha(rICT < overlay_range(1)) = false;
  alpha(rICT > overlay_range(2)) = false;
  alpha = alpha & BW;
  
  % Update the figure
  F.updateOverlay(rICT, alpha);
  if show_timestamp
    F.updateTimestamp(t(i));
  end
  
  % Write the frame
  writeVideo(v, F.getFrame());
  
end

close(v);
delete(WB);

fprintf('Elapsed time: %.2fs (%.1f fps)\n', toc, N/toc);
    
end