function speckle2Video(file_sc,n_avg,sc_range,fps,varargin)
%speckle2Video Creates video from speckle contrast data
% speckle2Video(file_sc,n_avg_sc_range,fps) generate a video from averaged
% speckle contrast data.
%
%   file_sc = Path to directory of speckle contrast files
%   n_avg = Number of speckle contrast frames to average per video frame
%   sc_range = Display range for speckle contrast data [min max]
%   fps = Framerate of the rendered video
%
% speckle2Video(___,Name,Value) modifies the video appearance using one or 
% more name-value pair arguments.
%
%   'show_scalebar': Toggle display of the scalebar (Default = true)
%       Scale is hard-coded in SpeckleFigure to that of the multimodal system

%   'show_timestamp': Toggle display of timestamp overlay (Default = true)
%

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'n_avg', @isscalar);
addRequired(p, 'sc_range', @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'}));
addRequired(p, 'fps', @isscalar);
addParameter(p, 'show_scalebar', true, @islogical);
addParameter(p, 'show_timestamp', true, @islogical);
parse(p, file_sc, n_avg, sc_range, fps, varargin{:});

file_sc = p.Results.file_sc;
n_avg = p.Results.n_avg;
sc_range = p.Results.sc_range;
fps = p.Results.fps;
show_scalebar = p.Results.show_scalebar;
show_timestamp = p.Results.show_timestamp;
    
% Get list of speckle contrast files and index for averaging
files = dir_sorted(fullfile(file_sc, '*.sc'));
[N_total, dims] = totalFrameCount(files);
N = floor(N_total/n_avg);
idx = reshape(1:N*n_avg, n_avg, N);

% If overlaying timestamp, load the .timing file 
if show_timestamp
  t = loadSpeckleTiming(fullfile(file_sc, '*.timing'));
  t = reshape(t(1:floor(numel(t)/N_total)*n_avg*N), [], N);
  t = mean(t, 1);
  t = t - t(1); % Relative to the first frame
end

% Create the figure
F = SpeckleFigure(zeros(dims), sc_range, 'visible', false);

if show_scalebar
  F.showScalebar();
end

if show_timestamp
  F.showTimestamp(0);
end

% Open the VideoWriter
d_parts = strsplit(files(1).folder, filesep);
output = fullfile(pwd, [d_parts{end} '.mp4']);
v = VideoWriter(output, 'MPEG-4');
v.FrameRate = fps;
open(v);

% Generate the frames by iterating over the LSCI indices
tic;
fprintf('Rendering %d frames (%d-frame averaging) to %s\n', N, n_avg, output);
WB = waitbar(0, '', 'Name', 'Generating Video');
for i = 1:N
  
  % Update the progressbar
  waitbar(i/N, WB, sprintf('Frame %d of %d', i, N));

  % Load the speckle contrast frames and average
  sc = read_subimage(files, -1, -1, idx(:, i));
  sc = mean(sc, 3)';

  % Update the figure
  F.update(sc);
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