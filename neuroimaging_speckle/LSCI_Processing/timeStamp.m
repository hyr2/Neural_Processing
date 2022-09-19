function [H] = timeStamp(t,varargin)
%timeStamp Add a timestamp to the current figure
% timeStamp(t) overlays timestamp on bottom right corner of the current
% figure. Displayed in seconds with two decimal places.
%
% H = timeStamp(t) returns handle to the timestamp text object
%
% timeStamp(t,formatSpec) renders the timestamp with a custom format
%

if length(varargin) >= 1 && ~isempty(varargin{1})
  formatSpec = varargin{1};
else
  formatSpec = '%+06.2fs';
end

FS = 17;
YLim = get(gca,'YLim');
XLim = get(gca,'XLim');

% Draw timestamp background
rectangle(gca, 'Position', [XLim(2) - 300, YLim(2) - 50, 300, 50], ...
  'FaceColor', 'k', 'LineStyle', 'none');

% Draw timestamp
H = text(gca, XLim(2) - 150, YLim(2) - 25, sprintf(formatSpec, t), ...
  'FontUnits', 'pixels', 'FontSize', FS, 'FontName', 'FixedWidth', ...
  'Color', [1 - eps, 1, 1], 'HorizontalAlignment', 'center');

end
