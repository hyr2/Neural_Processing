function [SC_REF,varargout] = loadReferenceSC(d,varargin)
%loadReferenceSC Loads reference from directory of .sc files
% SC_REF = loadReferenceSC(d) Loads and averages a directory of .sc files
% for use as a reference during rICT calculations.
%
% [SC_REF,N] = loadReferenceSC(d) Returns the number of .sc files averaged.
%
% SC_REF = loadReferenceSC(d,SC_FIRST) Aligns the reference frame to SC_FIRST.
%
% [SC_REF,mask] = loadReferenceSC(d,SC_FIRST) Return the alignment mask.
%
% Input Arguments:
%   d_ref = Path to directory of .sc files to use as rICT baseline
%
% Optional Input Arguments:
%   SC_FIRST = Single speckle contrast frame for use as alignment target
%

  % Load and average the directory of reference .sc files
  files = dir_sorted(fullfile(d, '*.sc'));
  N = totalFrameCount(files);
  fprintf('Defining baseline from %d frames in %s\n', N, files(1).folder);
  SC_REF = read_subimage(files);
  SC_REF = mean(SC_REF, 3)';
    
  % If the alignment target is provided, then register the reference frame
  if(~isempty(varargin))
    SC_FIRST = varargin{1};
    disp('Performing intensity-based image alignment of reference frame with first frame of sequence');
    p = which('Elastix/Parameters_BSpline.txt');
    [SC_REF, mask] = alignTwoSpeckle(SC_REF, SC_FIRST, p);
    if nargout == 2
      varargout{1} = mask;
    end
  else
    if nargout == 2
      varargout{1} = N;
    end
  end
  
end

