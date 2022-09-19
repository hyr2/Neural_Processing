function speckle2Video(file_sc,n_avg,sc_range,fps,delta_t,varargin)
%speckle2Video Creates video from speckle contrast data
% speckle2Video(file_sc,n_avg,sc_range,fps) generate a video from averaged
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
% [N_total, dims] = totalFrameCount(files);
N_total = length(files);
N = floor(N_total/n_avg);
idx = reshape(1:N*n_avg, n_avg, N);
sc = read_subimage(files, -1, -1, idx(1));
sc = mean(sc,3)';
dims = size(sc,[1,2]);
t = [0:delta_t:delta_t*(N_total-1)];


% Design filter
order = 10;      % Order of the IIR filter
fcutlow = 3.5;    % cutoff frequency
Fs = 1/delta_t;        % Sampling frequency
lowpass_filter = designfilt('lowpassiir','FilterOrder',order, ...
    'StopbandFrequency',fcutlow,'SampleRate',Fs,'DesignMethod','cheby2');

% Video writer
% Create the figure
F = SpeckleFigure(zeros(dims), sc_range, 'visible', false);

if show_scalebar
  F.showScalebar();
end

if show_timestamp
  F.showTimestamp(0);
end

% Image stack
stack_size = 500;
iter_local = [0:stack_size:N_total];
temp_diff = N_total - iter_local(end);
if temp_diff > (stack_size-100)
    iter_local = [iter_local,N_total];
end

sc_final = zeros(dims(1),dims(2),stack_size);        % Empty matrix of all sequences

for Jt = [1:length(iter_local)-1]
    
    % Open the VideoWriter
    d_parts = strsplit(files(1).folder, filesep);
    output = fullfile(pwd, [d_parts{3} num2str(Jt) '.mp4']);
    v = VideoWriter(output, 'MPEG-4');
    v.FrameRate = fps;
    open(v);

    % Generate the frames by iterating over the LSCI indices
    tic;
    fprintf('Rendering %d frames (%d-frame averaging) to %s\n', N, n_avg, output);
    WB = waitbar(0, '', 'Name', 'Generating Video');
    
    iter = 1;
    for i = [iter_local(Jt)+1:iter_local(Jt+1)]    % loop over sequences
        % Load the speckle contrast frames and average
        sc = read_subimage(files, -1, -1, idx(:, i));
        sc = mean(sc, 3)';
        % Storing in a 3D array
        sc_final(:,:,iter) = sc; 
        iter = iter + 1;
    end
    % Filtering
    sc_final(isnan(sc_final)) = -1;                 % replacing NaN with zeros
    sc_final = permute(sc_final,[3 2 1]);           % Row index is time
    sc_final = filtfilt(lowpass_filter,sc_final);   % Filtering on row index
    sc_final = permute(sc_final,[3 2 1]);           % Z index is time
    sc_final(sc_final == -1) = NaN;                 % Replacing NaNs back
    
    for i = 1:stack_size
  
      % Update the progressbar
      waitbar(i/N, WB, sprintf('Frame %d of %d', i, N));

      sc = sc_final(:,:,i);

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



end