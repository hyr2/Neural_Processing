function [result] = cal_roi_dr(sub,mask)
length_sub = length(sub);
result = zeros(length_sub,1);
for i = 1:length_sub
    temp_data = sub{i};
    temp_data(mask == 0) = -10;
    temp_a = temp_data(:);
    temp_a(temp_a == -10) = [];
    result(i)=100*mean(temp_a);
end

