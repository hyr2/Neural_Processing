function image2Video(d,fps,varargin) 
%image2Video Assembles directory of images into MP4 video using ffmpeg
% image2Video(d,fps) converts an image sequence into an MP4 video using the
% ffmpeg library. The directory (d) should only contain images named using 
% a standardized convention (e.g. Frame_0001.bmp). The images will be 
% displayed at framerate fps.
%
% image2Video(...,debug) toggle verbose command line output (Default = true).
%

debug = true;
if ~isempty(varargin) && islogical(varargin{1})
  debug = logical(varargin{1});
end

% Verify that directory exists
if ~(exist(d,'dir') == 7)
  error('The directory %s does not exist',d);
end

% Standardize directory strings
if(d(end) == '/'); d = d(1:end-1); end

% Figure out nomenclature
tmp = dir([d filesep '*0001.*']);  % Grab the first file in sequence
tmp = strsplit(tmp(1).name,'_');
prefix = tmp{1};            % Prefix for the files
tmp = strsplit(tmp{2},'.');
index = length(tmp{1});   % Length of index for files
ext = tmp{2};             % Extension
file_string = fullfile(d, sprintf('%s_%%0%dd.%s',prefix,index,ext));

% Run ffmpeg
tic;
CMD = sprintf('ffmpeg -y -r %d -i "%s" -crf 20 -pix_fmt yuv420p "%s.mp4"',fps,file_string,d);
[status, result] = system(CMD);

if debug || status ~= 0
  disp(result);
  fprintf('\nElapsed time: %.2fs\n',toc);
end

end