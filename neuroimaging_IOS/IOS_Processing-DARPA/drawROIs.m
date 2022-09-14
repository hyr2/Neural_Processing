function ROI = drawROIs(SC, r)
%drawROIs Prompts user to select ROIs using speckle contrast image
% drawROIs(SC, r) returns an array of binary masks corresponding to
% user-defined regions of interest (ROI) drawn using a speckle contrast
% image for guidance.
%

  % Suppress warnings about figure size
  warning('off', 'Images:initSize:adjustingMag');

  ROI = [];
  button = 'Yes';
  figure('Name', 'Draw an ROI', 'NumberTitle', 'off');
  while strcmp(button,'Yes')

    % Have user define polygonal ROI
    this_ROI = roipoly(SC);
    
    % If not empty, append to the ROI list and overlay on figure
    if ~isempty(this_ROI)
      ROI = cat(3, ROI, this_ROI);
      SC(this_ROI) = 1;
    end

    % Prompt to add another ROI
    button = questdlg('Would you like to add another ROI?', 'Add ROI?', 'Yes', 'No', 'No');

  end

  close(gcf);
  pause(0.1);
  
  ROI = logical(ROI);

end