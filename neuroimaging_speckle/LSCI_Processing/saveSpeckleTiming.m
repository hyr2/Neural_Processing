function saveSpeckleTiming(t,fname)
%saveSpeckleTiming Saves speckle .timing file
% saveSpeckleTiming(t,fname) saves a vector of timestamps to file using the
% same format as the Speckle Software.
%
% Input Arguments:
%   t = Time vector in milliseconds
%   fname = Output filename
%

f = fopen(fname, 'wb');
fwrite(f, t, 'float');
fclose(f);

end