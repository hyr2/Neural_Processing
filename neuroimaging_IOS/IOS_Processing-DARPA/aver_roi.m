function [result] = aver_roi(mask,data)
%mask(mask>0)=1;
fenmu = sum(mask(:));
data(mask==0)=0;
fenzi = sum(data(:));
result = 100*fenzi/fenmu;
end