function [SC,mask] = loadAlignedSC(files)
%loadAlignedSC Loads and averages aligned speckle contrast data
% SC = loadAlignedSC(files) Loads and averages aligned speckle contrast
% data from a dir() struct.
%
% [SC,mask] = loadAlignedSC(files) also returns the alignment mask if
% available. Should only be used when length(files) == 1.
%

% Iteratively load the aligned speckle contrast data
SC = [];
for i = 1:length(files)
  FRAME = load(fullfile(files(i).folder, files(i).name));
  SC = cat(3, SC, FRAME.SC_REG);
end
SC = mean(SC, 3);

% Extract the mask if available and requested
if nargout == 2 && length(files) == 1
  if isfield(FRAME, 'mask')
    mask = FRAME.mask;
  else
    mask = true(size(SC));
  end
end

end
