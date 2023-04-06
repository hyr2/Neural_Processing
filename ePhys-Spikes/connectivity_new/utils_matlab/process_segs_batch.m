function process_segs_batch(foldernames, cell_fs)
    % process segments by the batch
    % foldernames : cell array of strings
    % Fs : cell array of floats
    % Assumes that the working directory is where this matlab file is
    for i=1:length(foldernames)
        fprintf("----SEG %d\n----", i);
        process_single_segment(foldernames{i}, cell_fs{i});
    end
end