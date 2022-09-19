function aligned_vessel_plot(file_sc,n_avg,varargin)
%aligned_vessel_plot Visualize vessel diameter over time from aligned data
%   aligned_vessel_plot(d,r) loads a directory of aligned speckle data and
%   calculates FWHM across user-defined vessels for each frame.
%


p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'n_avg', @isscalar);
addOptional(p, 'file_sc_baseline', '', @(x) exist(x, 'dir'));
validateFnc = @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'});
addParameter(p, 'sc_range', [0.015 0.40], validateFnc);
parse(p, file_sc, n_avg, varargin{:});


file_sc = p.Results.file_sc;
n_avg = p.Results.n_avg;
file_sc_baseline = p.Results.file_sc_baseline;
sc_range = p.Results.sc_range;

% Suppress image warnings
warning('off', 'Images:initSize:adjustingMag'); % Suppress size warnings



if file_sc(end) == '/'
    file_sc = file_sc(1:end-1);
end

% Get list of aligned data files
% workingDir = pwd; cd(d);
% files = dir('S*.mat');
files = dir_sorted(fullfile(file_sc, '*.sc'));
N_total = totalFrameCount(files);
N = floor(N_total/n_avg);

% Get timing information
% t = loadSpeckleTiming(fullfile(file_sc, '*.timing'));
% t = mean(reshape(t,[],N_total), 1);
% t_start = getSpeckleStartTime(fullfile(file_sc, '*.log'));
t =  % Fill the time vector

% If no reference is provided, prompt user to define baseline.
% Otherwise, load all the files in the reference directory and register
% to the first frame of the primary data.
if isempty(file_sc_baseline)
  idx = getBaselineIndex(t);
  SC_REF = mean(read_subimage(files, -1, -1, idx), 3)';
else
  SC_FIRST = mean(read_subimage(files, -1, -1, 1:n_avg), 3)';
  SC_REF = loadReferenceSC(file_sc_baseline);
end

% Load first frame and prompt user to draw cross sections
% load(files(1).name);
ROI = drawROIs(SC_REF, sc_range);
% need a line segment here ***

L = 5;
FWHM = zeros(N,size(L,3));
RSS = zeros(N,size(L,3));

% Load current speckle frame
SC_frame = load(files(1).name);


pause(0.1);

% Preallocate Storage
FWHM = zeros(N,size(L,3));
RSS = zeros(N,size(L,3));

parfor_progress(N);
for i = 1:N

    % Load current speckle frame
    SC_frame = load(files(i).name);

    % Extract cross sectional profile for each line and calculate FWHM
    for j = 1:size(L,3)

        y = improfile(SC_frame.moving_registered,L(1,:,j),L(2,:,j));
   
        % Calculate distance (in pixels) between coordinates
        l = sqrt((L(1,2,j) - L(1,1,j))^2 + (L(2,2,j) - L(2,1,j))^2);
        
        % Generate spatial distance vector to accompany profile
        x = linspace(-l/2,l/2,length(y))';
        x = x/364; % mm

        % Invert, baseline, and prune data to > 30% max
        y = -y;
        y = y - min(y);
        idx = find(y > 0.3*max(y));
        
        % Fit data to Gaussian
        gaussian = @(f,xdata)f(1)*exp(-(xdata - f(2)).^2/(2*f(3)^2)) + f(4);
        x0 = [0.01 0 -0.05 0];
        opts = optimset('Display','off');
        [F, RSS(i,j)] = lsqcurvefit(gaussian,x0,x(idx),y(idx),[],[],opts);
        FWHM(i,j) = 2*sqrt(2*log(2))*abs(F(3));

%         % Calculate values from fit
%         y_new = F(1)*exp(-(x(idx) - F(2)).^2/(2*F(3)^2)) + F(4);
%         
%         % Visualize
%         plot(x,y,'o');
%         hold all;
%         plot(x(idx),y_new,'-');
 
    end
    
    pause(0.1);
    clf();
    parfor_progress;

end
parfor_progress(0);
close(gcf);
cd(workingDir);

% Prepare for data output
output = sprintf('%s/Vessel',workingDir);
if ~exist(output,'dir')
    mkdir(output);
end       

% Visualize
setSpeckleColorScheme();
plot(t,FWHM);
xlabel('Time (s)'); ylabel('FWHM (mm)'); grid on;
title('Vessel FWHM over Time');
legend(names);
print(sprintf('%s/Profile',output),'-dpng');

setSpeckleColorScheme();
plot(t,RSS);
xlabel('Time (s)'); ylabel('Residual Sum of Squares'); grid on;
title('Residual Sum of Squares over Time');
legend(names);

% Overlay cross sections onto Speckle image
figure;
imshow(SC);
hold all;
for i = 1:size(L,3)
    line(L(1,:,i),L(2,:,i),'LineWidth',5,'Color',getSpeckleROIColor(i,1,1));
end
scaleBar(RES);
imagePrint(size(SC,2),size(SC,1),300,'-dpng',sprintf('%s/ROI_Start',output));

save(sprintf('%s/data.mat',output),'FWHM','RSS','t');
end