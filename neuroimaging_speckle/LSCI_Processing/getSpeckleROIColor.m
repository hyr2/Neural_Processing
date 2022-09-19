function c_img = getSpeckleROIColor(n,height,width)
%getSpeckleROIColor returns matrix of nth Speckle Software color
% getSpeckleROIColor(n,height,width) takes integer (n) and image dimensions
% and returns 3-element color image of nth Speckle Software color.
%

    % Speckle Software color order
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

    idx = mod(n - 1,length(c)) + 1;
    c_img = reshape(repmat(c(idx,:),height*width,1),height,width,3);
end