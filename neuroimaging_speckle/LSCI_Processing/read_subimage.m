function sc = read_subimage(files, varargin)
% read_subimage - reads a portion of a .sc or raw image file(s)
%
% This function reads in entire images or only pixels within 
% the specified region of interest. You can also specify which
% images to read data from. It works with both .sc files and raw
% images (8, 12 or 16-bit).
% The major advantage of this script is that it enables efficient
% loading of portions of data into matlab without any unnecessary
% disk access
%
% syntax: y=read_subimage(files, width_roi, height_roi, file_indices);
%   files is the only required input argument. All others are optional
%   files is a directory structure containing a list of image files and 
%   is created with matlab's dir command (see examples below)
%
%   width_roi and height_roi are ranges of pixels in the horizontal
%   and vertical dimensions to read in. If width_roi and/or height_roi
%   are scalar and equal to -1, all pixels along that dimension are read
%
%   file_indices is a vector containing a list of images to be read. If
%   it is not specified, all images are loaded
%
% Examples:
%   files = dir('experiment1.*.sc'); % create list of files
%   sc = read_subimage(files); % loads all sc images into a single variable
%
%   Load all pixels of every 10th image 
%   sc = read_subimage(files, -1, -1, 1:5:100); 
%
%   Read in the top left 256x256 region of every other image (assumes 1 image/file):
%   sc = read_subimage(files, 1:256, 1:256, 1:2:length(files));
%
%   Read in first 100 raw images
%   I = read_subimage(dir('raw_images.0*'), -1, -1, 1:100);
%

w_idx=-1;
h_idx=-1;
frame_idx=-1;
if(nargin > 1)
  w_idx = varargin{1};
  h_idx = varargin{2};
  frame_idx = varargin{3};
end

% read number of images in first file
folder = files(1).folder;
fp=fopen(fullfile(folder, files(1).name),'rb');
h=fread(fp,4,'ushort');
im_w = h(2);
im_h = h(1);
fclose(fp);

% see if we're reading raw or sc
if(isempty(strfind(files(1).name,'.sc')))
	header_size = 4*2; 
    if((h(1)*h(2)*h(3)+8) == files(1).bytes)
      pix_bytes = 1;
      pix_bytes_str = 'uchar';
    elseif((h(1)*h(2)*h(3)*2+8) == files(1).bytes)
      pix_bytes = 2;
      pix_bytes_str = 'ushort';
    else
      warning('header size and file size do not match. Assuming 8-bits/pixel')
      pix_bytes = 1;
      pix_bytes_str = 'uchar';
    end
	Nimg_per_file=h(3);
else
	header_size = 3*2;
	pix_bytes = 4;
	pix_bytes_str = 'float';
	Nimg_per_file=h(3);
end

% total number of images in these files
tot_img = Nimg_per_file*length(files);
% indices of which frames are in which files
frame_idx_in_files = 1:Nimg_per_file:tot_img;

% check to see what ROI to read in
if(length(w_idx)==1)
  if(w_idx<1)
    w_idx = 1:im_w;
  end
end
if(length(h_idx)==1)
  if(h_idx<1)
    h_idx = 1:im_h;
  end
end

if(frame_idx<0)
  frame_idx = 1:tot_img;
end
   
% set up precision string for matlab's fread command
precision_str = sprintf('%i*%s',length(w_idx),pix_bytes_str);
% for each frame, read appropriate pixels
sc = zeros(length(w_idx), length(h_idx), length(frame_idx));
curr_file=1;
fp=fopen(fullfile(folder, files(curr_file).name), 'rb');
for ii=1:length(frame_idx)
  absolute_frame_index = frame_idx(ii);
  % decide which file the current image is in
  file_idx = find(frame_idx_in_files <= absolute_frame_index,1,'last');
  if(file_idx ~= curr_file)
    fclose(fp);
    curr_file = file_idx;
    fp=fopen(fullfile(folder, files(curr_file).name), 'rb');
  end
  
  % find the index of the image to read within the current file, given
  % the absolute index of the image - note that this is zero offset
  this_frame_index = absolute_frame_index - Nimg_per_file*file_idx + Nimg_per_file - 1;

  % move to start of this image data
  fseek(fp, header_size+this_frame_index*im_w*im_h*pix_bytes, 'bof');
  % move to start of pixel in this file to read
  fseek(fp, (h_idx(1)-1)*im_w*pix_bytes + (w_idx(1)-1)*pix_bytes, 'cof');
  foo = fread(fp, length(w_idx)*length(h_idx), precision_str, ((im_w-w_idx(end))+(w_idx(1)-1))*pix_bytes );
  foo = reshape(foo, [length(w_idx) length(h_idx)]);
  sc(:,:,ii)=foo;
end
fclose(fp);
end