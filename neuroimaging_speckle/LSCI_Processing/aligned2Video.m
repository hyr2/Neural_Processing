function aligned2Video(file_aligned,sc_range,fps,varargin)
%aligned2Video Creates video from aligned speckle contrast data
% aligned2Video(file_sc,sc_range,fps) generate a video from aligned speckle
% contrast data.
%
%   file_aligned = Path to directory of aligned speckle contrast files
%   sc_range = Display range for speckle contrast data [min max]
%   fps = Framerate of the rendered video
%
% aligned2Video(___,Name,Value) modifies the video appearance using one or 
% more name-value pair arguments.
%
%   'show_scalebar': Toggle display of the scalebar (Default = true)
%       Scale is hard-coded in SpeckleFigure to that of the multimodal system
%
%   'show_timestamp': Toggle display of timestamp overlay (Default = true)
%

p = inputParser;
addRequired(p, 'file_aligned', @(x) exist(x, 'dir'));
addRequired(p, 'sc_range', @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'}));
addRequired(p, 'fps', @isscalar);
addParameter(p, 'show_scalebar', true, @islogical);
addParameter(p, 'show_timestamp', true, @islogical);
parse(p, file_aligned, sc_range, fps, varargin{:});

file_aligned = p.Results.file_aligned;
sc_range = p.Results.sc_range;
fps = p.Results.fps;
show_scalebar = p.Results.show_scalebar;
show_timestamp = p.Results.show_timestamp;

% Get aligned data files and load first frame for figure initialization
%files = dir_sorted(fullfile(file_aligned, 'Speckle*.mat'));
files = dir_sorted(fullfile(file_aligned, 'Speckle*.sc'));
file_aligned = files(1).folder;
N = length(files);
SC = loadAlignedSC(files(1));

% Load timing data if showing timestamp
if show_timestamp
  t = load(fullfile(file_aligned, 'timing.mat'));
  t = t.t;
end

% Create the figure
F = SpeckleFigure(SC, sc_range, 'visible', false);

if show_scalebar
  F.showScalebar();
end

if show_timestamp
  F.showTimestamp(0);
end

% Open the VideoWriter
d_parts = strsplit(file_aligned, filesep);
output = fullfile(pwd, [d_parts{end} '.mp4']);
v = VideoWriter(output, 'MPEG-4');
v.FrameRate = fps;
open(v);

% Generate the frames by iterating over the file indices 
tic;
fprintf('Rendering %d frames to %s\n', N, output);
WB = waitbar(0, '', 'Name', 'Generating Video');
for i = 1:N
    
  % Update the progressbar
  waitbar(i/N, WB, sprintf('Frame %d of %d', i, N));
  
  % Read in aligned speckle data
  SC = loadAlignedSC(files(i));
    
  % Update the figure
  F.update(SC);
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