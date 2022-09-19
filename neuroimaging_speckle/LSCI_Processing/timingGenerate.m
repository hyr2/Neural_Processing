function timingGenerate(d,n)
%timingGenerate Generates artificial timing file from directory of SC files
% timingGenerate(d,n) generates .timing file from a directory of SC files
% using file timestamps in the event that speckle software failed to
% properly complete an acquisition. This function will prevent the user
% from overwriting an existing .timing file.

    % Check for existing .timing file
    files = dir([d '/*.timing']);
    if(~isempty(files))
        error('Timing file already exists in %s/. Please remove and try again.',d);
    end
    
    % Load list of .sc files
    files = dir_sorted(fullfile(d, 'data.*'));
    N_total = totalFrameCount(files);

    % Generate timevector
    t_start = files(1).datenum;
    t_end = files(end).datenum;
    t_elapsed = (t_end - t_start)*24*60*60*1000;
    N = length(files)*n;
    t = linspace(0,t_elapsed,N);

    % Save to .timing file
    saveSpeckleTiming(t, 'data.timing');

    fprintf('Generated ''%s'' with %d timestamps\n',filename,N);

end