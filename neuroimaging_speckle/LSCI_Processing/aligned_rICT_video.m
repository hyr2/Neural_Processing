function aligned_rICT_video(file_aligned,sc_range,fps,varargin)
%aligned_rICT_video Generates rICT video from aligned speckle contrast data
% aligned_rICT_video(file_aligned,sc_range,fps) generates a video 
% overlaying relative inverse correlation time (rICT) on aligned speckle 
% contrast data. The rICT is calculated with a user-define timespan as the
% baseline.
%
%   file_aligned = Path to directory of aligned speckle contrast files
%   sc_range = Display range for speckle contrast data [min max]
%   fps = Framerate of the rendered video
%
% ASSUMES 5MS EXPOSURE TIME FOR INVERSE CORRELATION TIME CALCULATIONS
%
% aligned_rICT_video(file_aligned,sc_range,fps,file_sc_baseline) loads 
% separate directory of speckle contrast data for the rICT baseline.
%
%   file_sc_baseline = Path to directory of speckle contrast files
%
% aligned_rICT_video(___,Name,Value) modifies the overlay appearance using
% one or more name-value pair arguments.
%
%   'overlay_range': Set display range for rICT overlay (Default = [0 0.5])
%
%   'show_scalebar': Toggle display of the scalebar (Default = true)
%       Scale is hard-coded in SpeckleFigure to that of the multimodal system
%
%   'show_timestamp': Toggle display of timestamp overlay (Default = true)
%

p = inputParser;
addRequired(p, 'file_aligned', @(x) exist(x, 'dir'));
validateFnc = @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'});
addRequired(p, 'sc_range', validateFnc);
addRequired(p, 'fps', @isscalar);
addOptional(p, 'file_sc_baseline', '', @(x) exist(x, 'dir'));
addParameter(p, 'overlay_range', [0 0.5], validateFnc);
addParameter(p, 'show_scalebar', true, @islogical);
addParameter(p, 'show_timestamp', true, @islogical);
parse(p, file_aligned, sc_range, fps, varargin{:});

file_aligned = p.Results.file_aligned;
sc_range = p.Results.sc_range;
fps = p.Results.fps;
file_sc_baseline = p.Results.file_sc_baseline;
overlay_range = p.Results.overlay_range;
show_scalebar = p.Results.show_scalebar;
show_timestamp = p.Results.show_timestamp;

% Get list of aligned data files and timing
files = dir_sorted(fullfile(file_aligned, 'Speckle*.mat'));
N = length(files);
t = load(fullfile(file_aligned, 'timing.mat'));
t = t.t;

fprintf('Identified %d aligned frames in %s collected over %.2f seconds\n', N, files(1).folder, range(t));

% If no reference is provided, prompt user to define baseline.
% Otherwise, load all the files in the reference directory and register
% to the first frame of the primary data.
if isempty(file_sc_baseline)
  idx = getBaselineIndex(t);
  SC_REF = loadAlignedSC(files(idx));
  mask = true(size(SC_REF));
else
  SC_FIRST = loadAlignedSC(files(1));
  [SC_REF, mask] = loadReferenceSC(file_sc_baseline, SC_FIRST);
end

% Compute correlation time for baseline
CT_REF = get_tc_band(SC_REF,5e-3,1);

% Have user define craniotomy area
BW = defineCraniotomy(SC_REF, sc_range);
BW = BW & mask; % Mask output from reference image registration
BW(SC_REF > sc_range(2)) = false;

% Create the figure
F = SpeckleFigure(SC_REF, sc_range, 'visible', false);
F.showOverlay(CT_REF ./ CT_REF, overlay_range, BW);

if show_scalebar
  F.showScalebar();
end

if show_timestamp
  F.showTimestamp(0);
end

% Open the VideoWriter
d_parts = strsplit(files(1).folder, filesep);
output = fullfile(pwd, [d_parts{end} '_A39_rICT4_s_25.mp4']);
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
  
  % Read in aligned speckle data and mask
  [SC, alpha] = loadAlignedSC(files(i));
  
  % Calculate correlation time and relative ICT image
  CT = get_tc_band(SC, 5e-3, 1);
  rICT = CT_REF./CT;
  
  % Generate the overlay alpha mask excluding values outside overlay range
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