function rICT_binned_video(file_mat,file_avgSC,fps,delta_t,varargin)
%RICT_BINNED_VIDEO Summary of this function goes here

%
%   file_mat = Path to directory of speckle contrast files
%   delta_t = Time resolution of each frame in the file_mat folder
%   sc_range = Display range for speckle contrast data [min max]
%   fps = Framerate of the rendered video
%
%   'overlay_range': Set display range for rICT overlay (Default = [0 0.5])
%
%   'show_scalebar': Toggle display of the scalebar (Default = true)
%       Scale is hard-coded in SpeckleFigure to that of the multimodal system
%
%   'show_timestamp': Toggle display of timestamp overlay (Default = true)
%

p = inputParser;
addRequired(p, 'file_mat', @(x) exist(x, 'dir'));
addRequired(p, 'file_avgSC', @(x) exist(x, 'dir'));
addRequired(p, 'fps', @isscalar);
addRequired(p, 'delta_t', @isscalar);
validateFnc = @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'});
addParameter(p, 'sc_range',[0.018 0.34], validateFnc);
addParameter(p, 'overlay_range', [1.0 1.09], validateFnc);    % for rICT
addParameter(p, 'overlay_range1', [-0.01 0.05], validateFnc);    % for ndCT
addParameter(p, 'show_scalebar', true, @islogical);
addParameter(p, 'show_timestamp', true, @islogical);

parse(p, file_mat, file_avgSC, fps, delta_t,varargin{:});

file_mat = p.Results.file_mat;
file_avgSC = p.Results.file_avgSC;
fps = p.Results.fps;
delta_t = p.Results.delta_t;
sc_range = p.Results.sc_range;
overlay_range = p.Results.overlay_range;
overlay_range1 = p.Results.overlay_range1;
show_scalebar = p.Results.show_scalebar;
show_timestamp = p.Results.show_timestamp;


% Average SC for overlay
files = dir_sorted(fullfile(file_avgSC,'*.sc'));
SC = read_subimage(files,-1,-1,1);
SC = SC;
% Reading .mat files 
files = dir_sorted(fullfile(file_mat, '*.mat'));
folder = files(1).folder;
num_bins = length(files);
% Timestamps
time_vec = [0:delta_t:delta_t*(num_bins-1)];    

% Open the VideoWriter
d_parts = strsplit(file_mat, filesep);
output = fullfile(pwd, [d_parts{end} '_rICT.mp4']);
output1 = fullfile(pwd, [d_parts{end} '_ndCT.mp4']);
v = VideoWriter(output, 'MPEG-4');
v.FrameRate = fps;
open(v);
v1 = VideoWriter(output1, 'MPEG-4');
v1.FrameRate = fps;
open(v1);
tic;
fprintf('Rendering %d frames to %s\n', num_bins, output);
WB = waitbar(0, '', 'Name', 'Generating Video');

% Create figure and overlay
F = SpeckleFigure(SC, sc_range, 'visible', false);
G = SpeckleFigure(SC,sc_range,'visible',false);
F.showOverlay(zeros(size(SC)), overlay_range, zeros(size(SC)),'use_divergent_cmap', true);
G.showOverlay(zeros(size(SC)), overlay_range1, zeros(size(SC)),'use_divergent_cmap', true,'label','\DeltaCT_n');
if show_scalebar
  F.showScalebar();
  G.showScalebar();
end
if show_timestamp
  F.showTimestamp(0);
  G.showTimestamp(0);
end
alpha = 0.3*ones(size(SC));

% writing to video files

for iter = 1:num_bins
    
    % Update the progressbar
    waitbar(iter/num_bins, WB, sprintf('Frame %d of %d', iter, num_bins));
    
    str_file = fullfile(folder, files(iter).name);
    load(str_file,'avg','avg_ndeltaCT');
    
    % Update the figure
    F.updateOverlay(avg, alpha);
    if show_timestamp
        F.updateTimestamp(time_vec(iter));
    end
  
    % Write the frame
    writeVideo(v, F.getFrame());
    
end

close(v);
delete(WB);

WB = waitbar(0, '', 'Name', 'Generating Video');
for iter = 1:num_bins
    
    % Update the progressbar
    waitbar(iter/num_bins, WB, sprintf('Frame %d of %d', iter, num_bins));
    
    str_file = fullfile(folder, files(iter).name);
    load(str_file,'avg','avg_ndeltaCT');
    
    % Update the figure
    G.updateOverlay(avg_ndeltaCT, alpha);
    if show_timestamp
        G.updateTimestamp(time_vec(iter));
    end
  
    % Write the frame
    writeVideo(v1, G.getFrame());
    
end

close(v1);






delete(WB);

fprintf('Elapsed time: %.2fs (%.1f fps)\n', toc, num_bins/toc);

end



