function pos = drawROIrects(SC, r)
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
  imshow(SC,r);
 while strcmp(button,'Yes')

    % Have user define polygonal ROI
    
    this_ROI = drawrectangle('Rotatable',true,'Color','r');
    wait(this_ROI);
    
    % If not empty, append to the ROI list and overlay on figure
    if ~isempty(this_ROI)
      ROI = [ROI;this_ROI];
      
    end

    % Prompt to add another ROI
    button = questdlg('Would you like to add another ROI?', 'Add ROI?', 'Yes', 'No', 'No');

  end
  

zlength = length(ROI);
pos = zeros(4,2,zlength);

for i = 1:zlength
    pos(:,:,i) = get(ROI(i),'Vertices');
end
close(gcf);
end