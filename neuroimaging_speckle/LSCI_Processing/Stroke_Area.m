function [Area_stroke] = Stroke_Area(Binary_mask,scale_factor)
%STROKE_AREA is used to compute the area of the stroke site.
%   INPUT PARAMETERS:
%       Binary_mask = the mask drawn on FOIL software (on CCD image sensor)
%       scale_factor = scale factor for the imaging system

Area_stroke = bwarea(Binary_mask);
Area_stroke = Area_stroke * (1/(scale_factor))^2;     % in mm^2

end

