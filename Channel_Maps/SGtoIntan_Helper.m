% Helper Function for SG to Intan conversion

% Input gridmap which is the channel map to be converted (from Pavlo's excel file)

gridmap_new = zeros(size(gridmap));     % For both 1x32 and 2x16
load('SGIntan.mat');


for iter = 0:127
    replace_local = Intan(find(SG == iter));
    gridmap_new(gridmap == iter) = replace_local;
end