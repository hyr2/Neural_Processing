function [col_min, col_max, row_min, row_max] = drawRectangleROI(img)
% drawROIs Prompts user to select one rectangular ROI from image
% and returns the cropped boundaries

  % Suppress warnings about figure size
  warning('off', 'Images:initSize:adjustingMag');

  r1 = [];
  button = 'No';
  figure('Name', 'Draw a rectangle ROI', 'NumberTitle', 'off');
  imshow(mat2gray(img));
  while strcmp(button,'No')
    if ~isempty(r1)
      delete(r1)
    end
    % Have user define polygonal ROI
    r1 = drawrectangle();
    
    % If not empty, append to the ROI list and overlay on figure
    if ~isempty(r1)
      col_min = ceil(r1.Position(1));
      row_min = ceil(r1.Position(2));
      col_max = ceil(r1.Position(1)+r1.Position(3));
      row_max = ceil(r1.Position(2)+r1.Position(4));
    end

    % Prompt to modify ROI
    button = questdlg('Would you like to confirm this ROI?', 'Confirm ROI', 'Yes', 'No', 'Yes');

  end

  close(gcf);
%   pause(0.1);
%   
%   ROI = logical(ROI);

end