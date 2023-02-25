function [shank_out] = get_shanknum_from_intan_id(i_intan,chmap_mat,chmap2x16)
    % Channel map in the .mat file must start from 0
    if chmap2x16    % 2x16
        [r,c] = find(chmap_mat == i_intan);
        shank_out = ceil(c/2);
    else            % 1x32
        [r,c] = find(chmap_mat == i_intan);
        shank_out = c;
    end
end

