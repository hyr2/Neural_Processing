function ROI = loadROIs(d)
  %loadROIs Loads directory of Speckle Software ROIs into a matrix
  % loadROIs(d) takes a directory of ROI masks exported from the Speckle 
  % Software and returns them as a logical matrix. The masks should be binary
  % BMP images and named according to the '*_ccd.bmp' nomenclature.
  %
  
  % Get the list of masks
  files = dir(fullfile(d, '*_ccd.bmp'));
  
  % Iterate over them and load them into a variable
  ROI = [];
  for i = 1:length(files)
    this_ROI = imread(fullfile(files(i).folder, files(i).name));
    ROI = cat(3, ROI, this_ROI);
  end
  
  % Convert to logical
  ROI = logical(ROI);
  
end