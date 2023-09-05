clear all;
clc
close all

%% 10-27-2021 (Awake-stim)

source_dir = pwd;

% source_dir = 'H:\Data\10-27-2021\10-27-2021\data-b';
file_sc = fullfile(source_dir, 'data' , 'CompleteTrials_SC');
matlabTxT = fullfile(source_dir, 'extras' , 'whisker_stim.txt');
output_dir = fullfile(pwd, 'Vessel_dia_plots');
mkdir(output_dir);

[len_trials,FramesPerSeq, stim_start_time, seq_period, stim_num, num_trials] = read_whiskerStimTxT(matlabTxT);

delta_t = seq_period; 
trials = [1, num_trials];
bsl = floor(stim_start_time / seq_period);
bsl = bsl - 5;                         % because of the filter artifact, we cant select the full baseline
t_stim_start = stim_start_time;        % stimulation start time in (s)
t_stim_end = stim_num*seq_period + t_stim_start;    % End of stimulation (s)

% Custom axis at the end of code

%% Main Code
[o,numROI,pos] = vessel_dia(file_sc,len_trials,trials,[0.018 0.35],'file_dst',output_dir,'ROI','H:\Data\10-27-2021\10-27-2021\data-b\ROI.mat');
save(fullfile(output_dir,'vessel_data.mat'),'o','numROI','pos');



%% Low pass filtering
order = 6;
Fs = 1/delta_t;
%lowpass_filter = designfilt('lowpassiir','FilterOrder',order, ...
%    'StopbandFrequency',fcutlow,'SampleRate',Fs,'DesignMethod','butter');
lowpass_filter = designfilt('lowpassiir', 'FilterOrder', order, ...
                            'HalfPowerFrequency', 1.5, 'SampleRate', ...
                            6.25, 'DesignMethod', 'butter');
% o = permute(o,[2 1 3]);
% o = filtfilt(lowpass_filter,o);
% [yupper,ylower] = envelope(o,10,'analytic');
% o = permute(o,[2 1 3]);
vec_width1 = mean(o,1);
vec_width1 = reshape(vec_width1,[size(vec_width1,2),size(vec_width1,3)]);
[yupper,ylower] = envelope(vec_width1,20,'analytic');
vec_width1 = filtfilt(lowpass_filter,vec_width1);
%% Plotting code
figure();
x = delta_t * [1:len_trials];
Mag = 2.5;          % Current magnification of the system
px = 5.86;          % in microns
scaling = px / Mag;
o = o * scaling;
vec_width1 = vec_width1 * scaling;
yupper = yupper * scaling;
% vec_width1 = mean(o,1);
% vec_width1 = filtfilt(lowpass_filter,reshape(vec_width1,[size(vec_width1,2),size(vec_width1,3)]));
% vec_width1 = reshape(vec_width1,[1, size(vec_width1,2),size(vec_width1,3)]);
numframe = len_trials;

for i = 1:numROI
    figure();
    y = vec_width1(:,i);
    y1 = yupper(:,i);
    a = plot(x,y,'DisplayName',['Rect_' num2str(i)],'LineWidth',1.9);
    hold on;
    a1 = plot(x,y1,'DisplayName',['Env Rect_' num2str(i)]);
%     legend;
    % Saving figure 
    filename = strcat('plot-',num2str(i),'.png');
    xlabel('Time(s)');ylabel('Vessel Diameter (\mum)');
    title(strcat('ROI-',num2str(i)));
    saveas(gcf,fullfile(output_dir,filename),'png');
    xline(t_stim_start,'r--','LineWidth',1.6);
    xline(t_stim_end,'r--','LineWidth',1.6);

end

vec_width1 = reshape(vec_width1,[len_trials,numROI]);


% %age change of vessel diameters
bsl_indx = round(t_stim_start/delta_t);
dia_bsl = vec_width1(5:bsl_indx-2,:);
dia_bsl = mean(dia_bsl,1);

vec_width1_n = (vec_width1 - dia_bsl)./dia_bsl;
vec_width1_n = vec_width1_n * 100;

figure();
x = x';
for i = 1:numROI
    y = vec_width1_n(:,i);
    a = plot(x,y,'DisplayName',['Rect_' num2str(i)],'LineWidth',1.9);
    hold on;
    
end
xline(t_stim_start,'k--','LineWidth',1.6);
xline(t_stim_end,'k--','LineWidth',1.6);
legend;
xlabel('Time(s)');ylabel(' \Delta D_n (%)');
title('Percentage change in diameter');
axis([x(7) x(end-7) -6 6]);
filename = strcat('plot-ndVesselDia','.png');
saveas(gcf,fullfile(output_dir,filename),'png');


