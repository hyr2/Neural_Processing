function zeropad_SC_filename(d)
%zeropad_SC_filename Zero-pads the filenames of a directory of .sc files
% zeropad_SC_filename(d) takes a directory of .sc files and applies zero-
% padding to the filenames to prevent ordering issues in Windows.
%

  % Standardize directory strings
  if(d(end) == '/'); d = d(1:end-1); end

  % Create output directory 
  output = [d '_Padded/'];
  if ~exist(output,'dir')
      mkdir(output);
  end

  % Figure out how much zero-padding to apply
  files = dir(fullfile(d, '/*.sc'));
  N = length(files);
  n = ceil(log(N)./log(10));
  fmt = ['speckle.%0' int2str(n) 'd.sc'];
  
  % Iterate over speckle contrast files and zero-pad the filenames
  for i = 1:N
    idx = regexp(files(i).name, '\d+', 'match');
    new = sprintf(fmt, str2double(cat(1, idx{:})));
    copyfile(fullfile(files(i).folder, files(i).name), fullfile(output, new));
  end

  % Copy over the acquisition and timing files
  files = dir(fullfile(d, '/*.log'));
  copyfile(fullfile(files(1).folder, files(1).name), fullfile(output, files(1).name));
  files = dir(fullfile(d, '/*.timing'));
  copyfile(fullfile(files(1).folder, files(1).name), fullfile(output, files(1).name));

end