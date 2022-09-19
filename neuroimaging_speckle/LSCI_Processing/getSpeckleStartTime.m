function created = getSpeckleStartTime(fname)
%getSpeckleStartTime Returns speckle acquisition start time
% getSpeckleStartTime(log_file) Extracts speckle acquisition start time
% from Speckle Software .log file and returns datetime object.
%
%   log_file = Path to Speckle Software log file
%

  file = dir(fname);

  if isempty(file)
    error('%s does not exist', fname);
  end
  
  fp = fopen(fullfile(file(1).folder, file(1).name), 'r');
  text = fscanf(fp, '%c');
  fclose(fp);
  
  text = strsplit(text, '\n');
  created = datetime(text{2}, 'InputFormat', 'eee MMM d HH:mm:ss y');
  
end