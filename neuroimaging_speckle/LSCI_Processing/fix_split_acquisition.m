function fix_split_acquisition(d1, d2)
%fix_split_acquisition merges two Speckle Software acquisitions
% fix_split_acquisition(d1, d2) takes two directories containing speckle 
% contrast files (.sc) and merges them into a single directory with a
% unified nomenclature and .timing file. Used to 'fix' acquisitions that
% are interrupted and result in two datasets.
%

  % Standardize directory strings
  if(d1(end) == '/'); d1 = d1(1:end-1); end;
  if(d2(end) == '/'); d2 = d2(1:end-1); end;

  % Create the output directory
  workingDir = pwd;
  output = sprintf('%s/%s_Merged',workingDir,d1);
  if ~exist(output,'dir')
      mkdir(output);
  end

  % Load details of the first acquisition
  cd(d1);
  files1 = dir('*.sc');
  N1 = length(files1);
  t1 = loadSpeckleTiming('*.timing');
  t1_lastmodified = dir('*.timing');
  t1_lastmodified = t1_lastmodified.datenum;
  t1_start = t1_lastmodified*24*60*60 - t1(end);

  % Check files for corrupted data as indicated by a malformed header and
  % flag for exclusion during the merger process
  d1_ToCopy = true(N1, 1);
  for i = 1:N1
    fp = fopen(files1(i).name, 'rb');
    h = fread(fp, 3, 'ushort');
    fclose(fp);
    
    if h(1) == 0 || h(2) == 0 || h(3) == 0
      fprintf('%s/%s is broken! It will not be copied!\n', d1, files1(i).name);
      d1_ToCopy(i) = false;
    end
  end
  cd(workingDir);
  
  % Exclude files and timestamps corresponding to corrupted data files
  files1(~d1_ToCopy) = [];
  t1 = reshape(t1, 15, []);
  t1(:, ~d1_ToCopy) = [];
  t1 = t1(:);
  N1 = length(files1);
  
  % Load details of the second acquisition
  cd(d2);
  files2 = dir('*.sc');
  N2 = length(files2);
  t2 = loadSpeckleTiming('*.timing');
  t2_lastmodified = dir('*.timing');
  t2_lastmodified = t2_lastmodified.datenum;
  t2_start = t2_lastmodified*24*60*60 - t2(end);
  cd(workingDir);

  % Calculate the offset between the two acquisitions
  t_delta = t2_start - t1_start;

  % Create and write the new .timing file
  t_new = 1000*[t1; t2 + t_delta];
  output_timing = fullfile(output, 'speckle.timing');
  f = fopen(output_timing,'wb');
  fwrite(f,t_new,'float');
  fclose(f);

  % Adjust the last modified date to match the second acquisition
  import java.io.File java.text.SimpleDateFormat
  sdf = SimpleDateFormat('yyyy-MM-dd HH:mm:ss');
  newDate = sdf.parse(datestr(t2_lastmodified, 'yyyy-mm-dd HH:MM:SS'));
  f = File(output_timing);
  f.setLastModified(newDate.getTime);
  
  % Copy and rename the files from the first acquisition to the new folder
  fprintf('\nCopying files from %s to %s\n', fullfile(workingDir, d1), output);
  base = strsplit(files1(1).name, '.');
  for i = 1:N1
    new_name = sprintf('%s.%05d.sc', base{1}, i - 1);
    copyfile(fullfile(d1, files1(i).name), fullfile(output, new_name));    
  end
 
  % Copy and rename the files from the second acquisition to the new folder
  fprintf('Copying files from %s to %s\n', fullfile(workingDir, d2), output);
  for i = 1:N2
    idx = regexp(files2(i).name, '\d+', 'match');
    new_name = sprintf('%s.%05d.sc', base{1}, str2double(idx{:}) + N1);
    copyfile(fullfile(d1, files2(i).name), fullfile(output, new_name));
  end
  
end