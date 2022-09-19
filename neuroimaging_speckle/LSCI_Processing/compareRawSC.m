function compareRawSC(d,varargin)

    NN = 7; % NxN size
    n = 15;
    r = [0 0.2];

    if(d(end) == '/'); d = d(1:end-1); end;
    
    % Process the raw and speckle data
    workingDir = pwd;
    cd(d);
    files = dir('*.*');
    files = reshape(files,2,[])';
    files(1,:) = []; % Drop the paths
    files(end,:) = []; % Drop the .timing and .log file
    N = length(files);
    
    % Load first frame for reference
    SC_FRAME = mean(read_subimage(files(1,2)),3)';
    [IM_HEIGHT,IM_WIDTH] = size(SC_FRAME);
    
    % If ROI directory is set then load masks (assuming *_ccd.bmp format)
    ROI = [];
    if length(varargin) == 1
        roiDIR = varargin{1}; 
        if roiDIR(end) == '/'
            roiDIR = roiDIR(1:end-1);
        end
        
        roiFiles = dir(sprintf('%s/%s/*_ccd.bmp',workingDir,roiDIR));
        for i = 1:length(roiFiles)
            ROI = cat(3,ROI,imread(sprintf('%s/%s/%s',workingDir,roiDIR,roiFiles(i).name)));
        end
        
    else % Otherwise have user define the ROIs

        % Load first frame and prompt user to draw ROIs
        button = 'Yes';
        while strcmp(button,'Yes')

            % Prompt user to draw ROI and append to ROI list
            ROI = cat(3,ROI,roipoly(mat2gray(SC_FRAME,r)));

            % Prompt to add another ROI
            button = questdlg('Add another ROI?','Add ROI?','Yes','No','No');

        end

        close(gcf);
        pause(0.1);
    
    end
    
    % Introduce NaN for computational purposes
    ROI(ROI == 0) = NaN;
    
    N_ROI = size(ROI,3);
    RAW = zeros(N,N_ROI);
    SC = zeros(N,N_ROI);

    RAW_TRIM_X = (NN-1)/2:(NN-1)/2 + IM_WIDTH - 1;
    RAW_TRIM_Y = (NN-1)/2:(NN-1)/2 + IM_HEIGHT - 1;
    
    mypool = gcp;
    tic
    parfor i = 1:N
        
        files_frame = files(i,:);

        RAW_FRAME = mean(read_subimage(files_frame(1),RAW_TRIM_X,RAW_TRIM_Y,-1),3)';
        SC_FRAME = mean(read_subimage(files_frame(2)),3)';

        for j = 1:N_ROI        
            RAW_MASKED = ROI(:,:,j).*RAW_FRAME;
            RAW(i,j) = mean(RAW_MASKED(~isnan(RAW_MASKED)));

            SC_MASKED = ROI(:,:,j).*SC_FRAME;
            SC(i,j) = mean(SC_MASKED(~isnan(SC_MASKED)));
        end
    end

    fprintf('Total Processing Time: %d seconds\n',toc);
    
    cd(workingDir);
    save([d '.mat'],'RAW','SC','ROI');    
    
    % Generate time vector
    f = dir([d '/*.timing']);
    f = fopen([d '/' f(1).name],'rb');
    t = fread(f,'float');
    fclose(f);
    t = mean(reshape(t,n,[]))/1000;

    % Calculate correlation time
    CT = get_tc_band(SC,5e-6,1);
    
    % Calculate relative ICT (to first 10s) of data
    ICT_REL = bsxfun(@rdivide,CT,mean(CT(t < 10,:)));
    
    % Generate name labels for the ROIs
    names = cell(1,N_ROI);
    for i = 1:N_ROI
        names{i} = char(['R' int2str(i)]);
    end

    setSpeckleColorScheme();
    plot(t,RAW);
    xlabel('Time (s)');
    ylabel('Intensity');
    title(sprintf('%s - Intensity',d));
    legend(names);
    grid on;
    
    setSpeckleColorScheme();
    plot(t,SC);
    xlabel('Time (s)');
    ylabel('Speckle Contrast (K)');
    title(sprintf('%s - Speckle Contrast',d));
    legend(names);
    grid on;
    
    setSpeckleColorScheme();
    plot(t,CT);
    xlabel('Time (s)');
    ylabel('Correlation Time (s)');
    title(sprintf('%s - Correlation Time',d));
    legend(names);
    grid on;
    
    setSpeckleColorScheme();
    plot(t,ICT_REL);
    xlabel('Time (s)');
    ylabel('Relative ICT');
    title(sprintf('%s - Relative ICT',d));
    legend(names);
    grid on;
    
    save([d '.mat'],'t','RAW','SC','CT','ICT_REL','ROI');    
end