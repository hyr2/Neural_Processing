function [out_arr] = read_RejectTrialsTxT(file_path)
%Reads the txt file called 'reject.txt' 
%   First line of the text file has to be the trials rejected array
fid = fopen(file_path,'rt');
tline = fgetl(fid);
out_arr = str2num(tline);
end

