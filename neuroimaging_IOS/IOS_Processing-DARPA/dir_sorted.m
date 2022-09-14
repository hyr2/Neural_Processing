function files = dir_sorted(ext)
%dir_sorted List directory sorted by natural order
%   files = dir_sorted('ext') lists files matching 'ext' in current 
%   directory sorted by natural order
%

files = dir(ext);
[~,order] = sort_nat({files.name});
files = files(order);

end