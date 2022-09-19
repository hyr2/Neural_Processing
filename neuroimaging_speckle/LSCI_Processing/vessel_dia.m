function [d1,numROI,pos] = vessel_dia(file_sc,len_trials,trials,sc_range,varargin)
%vessel_dia Computes CT for each sequence and then averages along the time
%bins. Trial based averaging
%
%   INPUT ARGUMENTS:
%   file_sc = Path to directory of speckle contrast files
%   len_trials = number of time bins in the experiment (len_trials)
%   num_bin = Size of each time bin (in # of seq)
%   trials = number of trials to include (IMPORTANT : specify as vector [initial trial, final trial])
%   e.g: to study the data of the 51-100 trials you would do: [51,100]
%   file_dst = Path for the destination processed data
%   delta_t = time resolution of experimental aquisition (single time bin)
%   baseline = {'F','B'}. 'F': start of trial. 'B': end of trial
%   num_BSL = number of sequences to be selected as baseline
%
% ASSUMES 5MS EXPOSURE TIME FOR INVERSE CORRELATION TIME CALCULATIONS

warning('off', 'Images:initSize:adjustingMag'); % Suppress image size warnings

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'len_trials', @isscalar);
addRequired(p, 'trials',  @isvector);
addRequired(p, 'sc_range', @isvector);
addOptional(p, 'file_dst', '');
addOptional(p, 'ROI', '');
addOptional(p, 'delta_t', @isscalar);
addParameter(p, 'resolution', get(0, 'screenpixelsperinch'), @isscalar);
parse(p, file_sc, len_trials, trials, sc_range,varargin{:});
file_sc = p.Results.file_sc;
sc_range = p.Results.sc_range;
len_trials = p.Results.len_trials;
trials = p.Results.trials;
file_dst = p.Results.file_dst;
ROI = p.Results.ROI;
resolution = p.Results.resolution;
delta_t = p.Results.delta_t;

% if ~exist(file_dst,'dir')
%   mkdir(file_dst);
% end

trials_initial = trials(1);
trials = trials(2) - trials(1) + 1;

% Get list of speckle contrast files
files = dir_sorted(fullfile(file_sc, '*.sc'));
SC = read_subimage(files,-1,-1,[1:20]);
SC = mean(SC, 3)';
F = SpeckleFigure(SC, sc_range);
filename = fullfile(file_dst,'SCImage');
F.saveBMP(filename, resolution);
dim_SC = size(SC);

%%

start_trial = [1:len_trials:1000*len_trials];
start_trial = start_trial(trials_initial:trials+trials_initial-1);
% Statistics 
avg = zeros(dim_SC(1),dim_SC(2),'double');
S = zeros(dim_SC(1),dim_SC(2),'double');

% Baseline

%% START
% Add code here @ Frank
SCt = read_subimage(files,-1,-1,1:20);
SC1 = SCt(:,:,1);
SC1 = SC1';
%if isempty(pos)
% Instructions: The longer sides of the rectangles should be parallel to the vessel.  

if isempty(ROI)
    pos = drawROIrects(SC1,[0.02 0.4]);
    filename = fullfile(file_dst,'ROI.mat');
    save(filename,'pos');
else
    load(ROI);
end

% load('E:\Xinyu-Li\data_ROI_horizontal.mat');%all four
% load('E:\Xinyu-Li\testROI.mat');% horizontal with little angle
% load('E:\Xinyu-Li\45degree.mat');% 45 degree angle
% load('E:\Xinyu-Li\data.mat');% dataset 1 all four

zlength = size(pos,3);
numROI = zlength;
d1 = zeros(trials,len_trials,zlength);

% Saving Vessel diameter maps
fig = figure(1);
imshow(SC1,[0.02 0.4])
hold on;
for i = 1:numROI
    plot(pos(:,1,i),pos(:,2,i),'LineWidth',1.9);
    hold on;
end
axis equal;
legend;
saveas(fig,fullfile(file_dst,'ROI_vesselDia.png'));
% Main Code to compute vessel diameter (Gaussian Fit model with linear term)
for iter_trial= 1:trials
%      iter_trial = 3;
%     fprintf('Trial # %i\n',iter_trial);
    SCt = read_subimage(files,-1,-1,[start_trial(iter_trial):start_trial(iter_trial)+len_trials-1]);
    for frame = 1:len_trials
%         frame = 105;
        SC2 = SCt(:,:,frame);
        SC2 = SC2';
%        imshow(SC2,[0.02 0.4]);
        fprintf('Trial # %i --- Frame # %i \n',iter_trial, frame);
        %disp(iter_trial)
        %disp("frame")
        %disp(frame)
        for i=1:zlength  % Loop over all the various ROIs
%             i = 3;
            v1 = pos(:,:,i);
            count1 = 0;
            dist = pdist(v1);
            dist = squareform(dist);
            point_a = [];% save the points for cutting lines
            point_b = [];% save the points for cutting lines
            if dist(2,3) <= dist(1,2)
                    vertex1 = v1(2,:);
                    vertex2 = v1(3,:);
                    bin = v1(2:3,:); %store coordinates for polyfit
                    bin2 = [v1(1,:)
                            v1(4,:)];
                    ang = abs((vertex2(2) - vertex1(2))/(vertex1(1) - vertex2(1)));%calculate the tan
                    ang = atand(ang);%arctan for angle
                        if ang <= 45
                            start1 = vertex1(1,1); %use x coordinate for cutting
                            finish1 = vertex2(1,1);
                        else
                            start1 = vertex1(1,2); %use y coordinate for cutting
                            finish1 = vertex2(1,2);
                        end
                        x = [start1 finish1];
                        start1 = min(x);
                        finish1 = max(x);
                        num = length([start1:1:finish1]);
                        t = start1;
                        zz = start1:finish1;
                        p = polyfit(bin(:,1),bin(:,2),1);
                        p1 = polyfit(bin2(:,1),bin2(:,2),1);
                        slope = -1/(p(1)); %slope for perpendicular line
                        if ang <= 45
                            for t = zz
                                count1 = count1 + 1;
                                point_a(count1,:) = [t p(1)*t+p(2)];
                                b = point_a(count1,2)-slope*point_a(count1,1);
                                x1 = (b-p1(2))/(p1(1)-slope);
                                point_b(count1,:) = [x1 p1(1)*x1+p1(2)];
                            end
                        elseif ang == 90
                                for t = zz
                                count1 = count1 + 1;
                                point_a(count1,:) = [bin(1,1) t];
                                point_b(count1,:) = [bin2(1,1) t];
                                end
                        else
                            for t = zz
                                count1 = count1 + 1;
                                point_a(count1,:) = [(t-p(2))/p(1) t];
                                b = point_a(count1,2)-slope*point_a(count1,1);
                                x1 = (b-p1(2))/(p1(1)-slope);
                                point_b(count1,:) = [x1 p1(1)*x1+p1(2)];
                            end
                        end
                %replace by matrix of zeros()
                c1 = zeros();
                %c1 = cell(size(improfile(SC2,[t t+v1(1,1)-v1(2,1)],[v1(2,2)+(t-v1(2,1))/(v1(3,1)-v1(2,1))*(v1(3,2)-v1(2,2)) v1(1,2)+(t-v1(2,1))/(v1(3,1)-v1(2,1))*(v1(3,2)-v1(2,2))])),num);
                L1 = length(improfile(SC2,[point_a(1,1) point_b(1,1)],[point_a(1,2) point_b(1,2)])); %calculate the length of the cutting line
                c1 = zeros(L1,num);
                for t = 1:count1
                    c1(:,t) = improfile(SC2,[point_a(t,1) point_b(t,1)],[point_a(t,2) point_b(t,2)]);%draw the cutting lines
                end
                %disp(c1)
                %disp(mean(c1,2))
                tmp1 = calculatedistance_modified(mean(c1,2),10);
            else    % 1-2 is smaller
                vertex1 = v1(1,:);
                vertex2 = v1(2,:);
                bin = v1(1:2,:);
                bin2 = [v1(3,:)
                        v1(4,:)];
                ang = abs((vertex2(2) - vertex1(2))/(vertex1(1) - vertex2(1)));
                ang = atand(ang);
                if ang <= 45
                    start1 = vertex1(1,1);
                    finish1 = vertex2(1,1);
                else
                    start1 = vertex1(1,2);
                    finish1 = vertex2(1,2);
                end
                x = [start1 finish1];
                start1 = min(x);
                finish1 = max(x);
                num = length([start1:1:finish1]);
                t = start1;
                zz = start1:finish1;
                p = polyfit(bin(:,1),bin(:,2),1);
                p1 = polyfit(bin2(:,1),bin2(:,2),1);
                slope = -1/(p(1));
                if ang <= 45
                    for t = zz
                        count1 = count1 + 1;
                        point_a(count1,:) = [t p(1)*t+p(2)];
                        b = point_a(count1,2)-slope*point_a(count1,1);
                        x1 = (b-p1(2))/(p1(1)-slope);
                        point_b(count1,:) = [x1 p1(1)*x1+p1(2)];
                    end
                elseif ang == 90
                    for t = zz
                        count1 = count1 + 1;
                        point_a(count1,:) = [bin(1,1) t];
                        point_b(count1,:) = [bin2(1,1) t];
                    end
                else 
                    for t = zz
                        count1 = count1 + 1;
                        point_a(count1,:) = [(t-p(2))/p(1) t];
                        b = point_a(count1,2)-slope*point_a(count1,1);
                        x1 = (b-p1(2))/(p1(1)-slope);
                        point_b(count1,:) = [x1 p1(1)*x1+p1(2)];
                    end
                end
                % replace by matrix ..... c1 = cell(size(improfile(SC2,[t t+v1(4,1)-v1(1,1)],[v1(1,2)+(t-v1(1,1))/(v1(2,1)-v1(1,1))*(v1(2,2)-v1(1,2)) v1(4,2)+(t-v1(1,1))/(v1(2,1)-v1(1,1))*(v1(2,2)-v1(1,2))])),num);
                L1 = length(improfile(SC2,[point_a(1,1) point_b(1,1)],[point_a(1,2) point_b(1,2)]));
                c1 = zeros(L1,num);
                for t = 1:count1
                    c1(:,t) = improfile(SC2,[point_a(t,1) point_b(t,1)],[point_a(t,2) point_b(t,2)]);
                end
                %disp(c1)
                %disp(mean(c1,2))
                tmp1 = calculatedistance_modified(mean(c1,2),12);               % zero pad 12 data points to the raw data for ease of fitting
            end
            d1(iter_trial,frame,i)=tmp1;
        end    
    end
end

end
