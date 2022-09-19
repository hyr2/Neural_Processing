function t = loadSpeckleTiming(fname)
%loadSpeckleTiming Reads speckle .timing file
% t = loadSpeckleTiming(fname) reads in a speckle .timing file and returns 
% a vector of timestamps with units of seconds.
%

  file = dir(fname);
  
  if isempty(file)
    error('%s does not exist', fname);
  end
  
  fp = fopen(fullfile(file(1).folder, file(1).name), 'rb');
  t = fread(fp, 'float')/1000;
  fclose(fp);

end