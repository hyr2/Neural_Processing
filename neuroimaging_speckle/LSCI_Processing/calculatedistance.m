function FWHM = calculatedistance(val,zpad)

% INPUTS:
% 
% val = values to be fitted
% zpad = # of zero padding performed at the start and end of the val array

% Outputs:

val = 1 - val;                                           % Invert the function
[max_value, max_index] = max(val);      
[min_value, min_index] = min(val);

val = padarray(val',zpad,min_value);          % Pad zeros for a good mathematical fit
x = 1:length(val);                                     % X axis
val = val';

init_guess = [0.08, max_index+zpad, 16, 0.1, min_value];     % Assumes 20 zero padding

% Single gaussian fit with a linear line
% X and Y data must be column vectors
[coeff] = mono_gaussfit(x, val,init_guess);
FWHM = sqrt(coeff(3)/2) * 2.35482;

end
