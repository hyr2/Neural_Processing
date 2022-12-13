% empty cell array
% final_out_final = cell(32,2);
% for iter = 65:128
%     [id_r,id_c] = find(tmp == tmp(iter));
%     [id_r_pcb, id_c_pcb] = find(final_out == tmp(iter));
%     final_out_final(id_r_pcb,id_c_pcb) =  tmp1(id_r,id_c);
%     
% end
final_out_maximally_final = zeros(32,4);
for iter = 1:64
    [str_tmp] = strsplit(final_out_final{iter},',');
    shank_ID = str2num(str_tmp(1));
    electrode = str2num(str_tmp(2));
    final_out_maximally_final(electrode,shank_ID) = intan_flipped(iter);
end