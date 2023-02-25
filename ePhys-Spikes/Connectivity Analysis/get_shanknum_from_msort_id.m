function [shank_out] = get_shanknum_from_msort_id(i_msort,Native_orders,chmap_mat,chmap2x16)
    shank_out = get_shanknum_from_intan_id(uint8(Native_orders(i_msort)),chmap_mat,chmap2x16);
end

