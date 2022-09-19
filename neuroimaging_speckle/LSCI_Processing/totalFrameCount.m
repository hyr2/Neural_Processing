function [N,dims] = totalFrameCount(d)
%totalFrameCount Returns the total number of frames in a directory listing
%   N = totalFrameCount(d) takes a MATLAB dir() listing and counts the 
%   total number of frames present in the corresponding files. This is 
%   performed quickly by reading the header on each raw or .sc file.
% 
%   [N,dims] = totalFrameCount(d) also returns the dimensions [height, width]
%   of the frames.
%

  N = 0;
  for i = 1:length(d)
    f = fopen(fullfile(d(i).folder, d(i).name),'rb');
    h = fread(f,4,'ushort');
    N = N + h(3);
    fclose(f);
    
    if i == 1
      dims = [h(1) h(2)];
    end
  end
end