function [result] = cal_roi_dr_individual(sub,mask)
temp_data = sub;
temp_data(mask == 0) = -10;
temp_a = temp_data(:);
temp_a(temp_a == -10) = [];
result=100*mean(temp_a);


