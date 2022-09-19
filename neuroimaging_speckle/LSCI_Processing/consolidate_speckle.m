function consolidate_speckle(file_sc,N,varargin)
%consolidate_speckle Consolidates speckle contrast files
% consolidate_speckle(file_sc,N) Merges speckle contrast (.sc) files to 
% reduce the total number of files in a directory. Always outputs alongside
% the input directory with '_New' appended on the directory name.
%
%   file_sc = Path to directory of speckle contrast files
%   N = Number of speckle contrast frames per consolidated file
%
% consolidate_speckle(file_sc,N,delete_original) enable automatic deletion
% of the original directory of speckle contrast files. Always outputs into
% a directory with the same name as the input directory.
%
%   delete_original = Boolean to enable/disable automatic deletion
%
%   Note: There is a risk of data loss when using this feature because of
%   how MATLAB handles files that are currently open or in use by other
%   programs. This function attempts to minimize that risk by stopping the
%   deletion operation such that the entirety of either the original or 
%   consolidated data will ALWAYS remain entact.
%

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'N', @isscalar);
addOptional(p, 'delete_original', false, @islogical);
parse(p, file_sc, N, varargin{:});

file_sc = p.Results.file_sc;
N = p.Results.N;
delete_original = p.Results.delete_original;

% Get list of .sc files
files = dir_sorted(fullfile(file_sc, '*.sc'));
N_frames = totalFrameCount(files);
input = files(1).folder;

% Check that there is enough space on drive to duplicate data
storage_available = java.io.File(input).getFreeSpace();
storage_required = sum([files.bytes]);
if storage_available < storage_required*1.05
    error('Insufficient space on drive to data duplication.');
end

% Extract filename base
base = files(1).name;
base = split(base, '.');
base = strjoin(base(1:end - 2), '.');

% Create directory for consolidated files
output = [input '_New'];
if exist(output, 'dir')
    error('%s already exists. Please rename or delete the directory and try again.', output); 
else
    mkdir(output);
end

% Generate the indicies for each consolidated .sc file
n = ceil(N_frames/N);
idx = reshape(1:N*n, N, n);
idx = num2cell(idx, 1);
idx{n} = ((n-1)*N + 1:N_frames)';

% Consolidate the .sc files
tic;
fprintf(['Consolidating %d .sc files into %d .sc files '...
         '(%d frames per file)\n\t%s\n'], length(files), n, N, output);
WB = waitbar(0, '', 'Name', 'Consolidating .sc files');
for i = 1:n
  waitbar(i/n, WB, sprintf('Files %d-%d of %d', idx{i}(1), idx{i}(end), N_frames));
  sc = read_subimage(files, -1, -1, idx{i});
  write_sc(sc, fullfile(output, sprintf('%s.%05d.sc', base, i)));
end
delete(WB);

% Copy the .timing and .log files
try
  copyfile(fullfile(input, '*.timing'), output);
catch ME
  if (strcmp(ME.identifier, 'MATLAB:COPYFILE:FileDoesNotExist'))
    warning('No .timing file found in %s.', input);
  end
end

try
  copyfile(fullfile(input, '*.log'), output);
catch ME
  if (strcmp(ME.identifier, 'MATLAB:COPYFILE:FileDoesNotExist'))
    warning('No .log file found in %s.', input);
  end
end

% If enabled, replace the original files with the consolidated files
if delete_original
  input_old = [input '_Old'];
  fprintf('Moving old files from %s to %s\n', input, input_old);
  [status, ~] = movefile(input, input_old, 'f');
  if status
    fprintf('Moving new files from %s to %s\n', output, input);
    [status, ~] = movefile(output, input, 'f');
    if status
      fprintf('Deleting old files from %s\n', input_old);
      [status, ~] = rmdir([input '_Old'], 's');
      if ~status
        warning('Error encountered deleting the old files from %s. Deletion operation will not proceed.', input_old);
      end
    else
      warning('Error encountered moving new files from %s to %s. Deletion operation will not proceed.', output, input);
    end
  else
    warning('Error encountered moving old files from %s to %s. Deletion operation will not proceed.', input, input_old);
  end
end

fprintf('Elapsed time: %.2fs\n', toc);

end