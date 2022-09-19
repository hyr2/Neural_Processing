function scaleBar(varargin)
%scaleBar Add a scalebar to the bottom left corner of the current figure
% scaleBar() overlays scale bar onto the bottom left corner of the current 
% figure assuming the resolution of the multimodal system (355 pixels/mm). 
% [Haad Rathore: System magnification set to x2.083 --> 355 pixels/mm]
% The pixel size of the Baseler acA1920-155um is 5.86 um :Haad Rathore]
% 
% scaleBar(scale) overlays a scale bar with user-defined pixel/mm resolution.
%

if length(varargin) >= 1 && ~isempty(varargin{1})
  scale = varargin{1};
else
%   scale = 355;    %OLD VALUE OF SYSTEM
  scale = 426.6213; % pixels/mm - Multimodal System (Basler acA1920-155um) [355 is what the old Microscope Objective mag setting gave]
end

FS = 18;
YLim = get(gca, 'YLim');

% Draw scalebar outlined in black
x = 75;
y = YLim(2) - 100;
rectangle(gca, 'Position', [x - 1, y + 1, scale, 20], 'FaceColor', 'k', 'LineStyle', 'none');
rectangle(gca, 'Position', [x - 1, y - 1, scale, 20], 'FaceColor', 'k', 'LineStyle', 'none');
rectangle(gca, 'Position', [x + 1, y + 1, scale, 20], 'FaceColor', 'k', 'LineStyle', 'none');
rectangle(gca, 'Position', [x + 1, y - 1, scale, 20], 'FaceColor', 'k', 'LineStyle', 'none');
rectangle(gca, 'Position', [x, y, scale, 20], 'FaceColor', 'w', 'LineStyle', 'none');

% Draw white text outlined in black
x = 75 + scale/2;
y = YLim(2) - 55;
s = '1 mm';
text(gca, x - 1, y + 1, s, 'FontUnits', 'pixels', 'FontSize', FS, 'FontWeight', ...
  'bold', 'Color', 'k', 'HorizontalAlignment', 'center');
text(gca, x - 1, y - 1, s, 'FontUnits', 'pixels', 'FontSize', FS, 'FontWeight', ...
  'bold', 'Color', 'k', 'HorizontalAlignment', 'center');
text(gca, x + 1, y + 1, s, 'FontUnits', 'pixels', 'FontSize', FS, 'FontWeight', ...
  'bold', 'Color', 'k', 'HorizontalAlignment', 'center');
text(gca, x + 1, y - 1, s, 'FontUnits', 'pixels', 'FontSize', FS, 'FontWeight', ...
  'bold', 'Color', 'k', 'HorizontalAlignment', 'center');
text(gca, x, y, s, 'FontUnits', 'pixels', 'FontSize', FS, 'FontWeight', ...
  'bold', 'Color', [1 - eps, 1, 1], 'HorizontalAlignment', 'center');

end