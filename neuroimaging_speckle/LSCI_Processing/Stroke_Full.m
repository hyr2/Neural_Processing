file_v = 'H:\Data\6-3-2021\stroke\Processed';
files = dir_sorted(fullfile(file_v, '*.mp4'));
output_video = fullfile(file_v,'stroke-video.mp4');

vid = VideoReader(fullfile(file_v,files(1).name));
NumFrames = vid.NumFrames;
fps = vid.FrameRate;
N = NumFrames * length(files);


v = VideoWriter(output_video, 'MPEG-4');
v.FrameRate = fps;
open(v);
WB = waitbar(0, '', 'Name', 'Generating Video');
iter = 1;
for i = [1:length(files)]
    
    vid = VideoReader(fullfile(file_v,files(i).name));
    NumFrames = vid.NumFrames;
    frame = read(vid,[1 NumFrames]);
    frame = frame(:,:,1,:);
    for ii = 1:NumFrames
      iter  = iter+1;  
      % Update the progressbar
      waitbar(iter/N, WB, sprintf('Frame %d of %d', iter, N));

      img = frame(:,:,ii);

      % Write the frame
      writeVideo(v, img);

    end
    
end

close(v);
delete(WB);