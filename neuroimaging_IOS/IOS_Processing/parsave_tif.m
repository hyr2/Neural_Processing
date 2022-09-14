function parsave_tif(fname,img)
%PARSAVE_TIF Summary of this function goes here
%   Detailed explanation goes here
var_name=genvarname(inputname(2));
eval([var_name '=img'])

% try
imwrite(img,fname);
% catch
%     save(fname,var_name)

end

