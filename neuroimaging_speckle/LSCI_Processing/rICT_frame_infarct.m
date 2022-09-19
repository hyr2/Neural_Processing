function rICT_frame_infarct(d1,varargin)
%rICT_frame_infarct Calculates area with relative ICT <20% of baseline flow
% rICT_frame_infarct(d1) estimates ischemic core area from a directory of 
% .sc files.
%
% rICT_frame_infarct(d1,d2) estimates ischemic core area from two
% directories of .sc files.
%
% ASSUMES 5MS EXPOSURE TIME FOR INVERSE CORRELATION TIME CALCULATIONS
%
%   d1 = Directory of .sc files. If d2 is set, then this is pre-stroke data
%   d2 = Directory of .sc files that are baselined against d1.
%

    % Suppress image display size warnings
    warning('off', 'Images:initSize:adjustingMag');

    % Default values since we don't care about manual image control
    CORE_THRESH = 0.2; % <20% for "ischemic core"
    r = [0 0.2];
    scale = 335; % pixels/mm
    
    % Load in the baseline speckle data
    workingDir = pwd; cd(d1);
    files = dir('*.sc');
    n = length(files);

    % If d2 is set, then load and average all .sc files from reference
    % directory (d1) and then load appropriate files from d2.
    if(nargin == 2)
        
        fprintf('Loading and averaging %d sequences\n',n);
        SC1 = mean(read_subimage(files),3)'; 
        
        % Load in the main speckle data
        cd(sprintf('%s/%s',workingDir,varargin{1}));
        files = dir('*.sc');
        N = length(files);
    
        if(N > n)
            % Get timing information
            f = dir('*.timing');
            f = fopen(f(1).name,'rb');
            ts = fread(f,'float');
            fclose(f);
            ts = mean(reshape(ts,[],N))/1000;

            % Prompt user for baseline frame definition
            fprintf('Identified %d files collected over %.2f seconds\n',N,range(ts));
            t1 = input('Enter desired frame time (s): ');
            idx = find(ts > t1, 1);
            fprintf('Loading and averaging %d sequences\n',n);
            SC2 = mean(read_subimage(files(idx:idx + n)),3)';
        else
            SC2 = mean(read_subimage(files),3)';
        end    
        
    else % If only d1, then prompt user to define desired range
        
        % Get timing information
        f = dir('*.timing');
        f = fopen(f(1).name,'rb');
        ts = fread(f,'float');
        fclose(f);
        ts = mean(reshape(ts,[],length(files)))/1000;

        % Prompt user for baseline frame definition
        fprintf('Identified %d files collected over %.2f seconds\n',length(files),ts(end) - ts(1));
        t1 = input('Enter baseline start time (s): ');
        t2 = input('Enter baseline duration (s): ');
        idx_1 = find(ts > t1, 1);
        idx_2 = find(ts > t1 + t2, 1);
        n = idx_2 - idx_1;
        fprintf('Loading and averaging %d sequences\n',n);
        SC1 = mean(read_subimage(files(idx_1:idx_2)),3)';

        % Prompt user for end frame definition (-1 indicates last frame)
        t1 = input('Enter end frame start time (s): ');
        if(t1 < 0) 
            idx_1 = length(files) - n;
        else    
            idx_1 = find(ts > t1, 1);
        end
        fprintf('Loading and averaging %d sequences\n',n);
        SC2 = mean(read_subimage(files(idx_1:idx_1 + n)),3)';
        
    end
    
    cd(workingDir);

    % Visualize alignment before registration
    figure;
    imshowpair(mat2gray(SC1,r),mat2gray(SC2,r),'Scaling','independent');

    % Have user define craniotomy area
    figure;
    mask = roipoly(mat2gray(SC1,r));
    close(gcf);
    pause(0.1);
        
    % Register SC2 with SC1 
    disp('Performing intensity-based image alignment');
    SC2_REG = alignTwoSpeckle(SC2,SC1);
    
    % Visualize alignment after registration
    figure;
    imshowpair(mat2gray(SC1,r),mat2gray(SC2_REG,r),'Scaling','joint');

    % Convert SC images to CT and calculate relative ICT image
    CT1 = get_tc_band(SC1, 5e-3, 1);
    CT2 = get_tc_band(SC2_REG, 5e-3, 1);
    rICT = CT1./CT2;

    % Overlay relative ICT on speckle frame
    F = figure;
    A1 = axes('Parent',F);
    A2 = axes('Parent',F);
    imshow(SC1,r,'Parent',A1);
    I2 = imshow(rICT,'Parent',A2);
    
    % Calculate transparency of overlay based on cutoff value
    alpha = ones(size(rICT));
    alpha(rICT > CORE_THRESH) = 0; % Drop values above the cutoff
    set(I2, 'AlphaData', alpha.*mask);
    caxis([0 CORE_THRESH]);
    colormap(A2,flipud(pmkmp(256,'CubicYF')));
    colorbar;
        
    % Tally pixels with value <0.2 and convert to area
    pixel_count = sum(alpha(:));
    AREA = pixel_count / (scale^2);
    fprintf('There are %d pixels (%0.2f%%) with relative ICT values <20%%\n',pixel_count,pixel_count/numel(alpha)*100);
    fprintf('With resolution of %d pixels/mm that is a total area of %0.4f mm^2\n',scale,AREA);
     
end