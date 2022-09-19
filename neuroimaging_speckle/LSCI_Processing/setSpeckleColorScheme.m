function c = setSpeckleColorScheme(varargin)
%setSpeckleColorScheme Use Speckle Software colormap
%   setSpeckleColorScheme() creates new figure using Speckle Software
%   color scheme for plot
%
%   setSpeckleColorScheme(offset) circularly shifts the color scheme to
%   start with the nth color
%

    c = [1 0 0;         % Red
        0 1 0;          % Green
        0 0 1;          % Blue
        0 1 1;          % Cyan
        1 0 1;          % Magenta
        1 1 0;          % Yellow
        0.5 0 0;        % Dark Red
        0 0.5 0;        % Dark Green
        0 0 0.5;        % Dark Blue
        0 0.5 0.5;      % Dark Cyan
        0.5 0 0.5;      % Dark Magenta
        0.5 0.5 0;      % Dark Yellow
        0 0 0;          % Black
        0.62 0.62 0.62  % Gray
    ];

    offset = 0;
    if nargin > 0
        offset = varargin{1};
    end

    figure; 
    set(gcf,'DefaultAxesColorOrder',circshift(c,-offset));
    set(gca,'NextPlot','replacechildren');
end

