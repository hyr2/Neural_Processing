function FWHM = calculatedistance_modified(val,zpad)

% INPUTS:
% 
% val = values to be fitted
% zpad = # of zero padding performed at the start and end of the val array

% Outputs:

val = 1 - val;                                           % Invert the function
[max_value, max_index] = max(val);      
[min_value_low, min_index_low] = min(val(1:max_index));         % base level pre peak
[min_value_high, min_index_high] = min(val(max_index:end));   % base level post peak

val = padarray(val,zpad,min_value_low,'pre');          % Pad zeros for a good mathematical fit
val = padarray(val,zpad,min_value_high,'post');         % Pad zeros for a good mathematical fit

x = 1:length(val);                                     % X axis
val = val';

init_guess = [0.08, max_index+zpad, 9, min_value_low, 0.1];     % Assumes 20 zero padding

% Single gaussian fit with a linear line
% X and Y data must be column vectors
[coeff] = mono_gaussfit(x, val,init_guess);
FWHM = sqrt(coeff(3)/2) * 2.35482;

end
