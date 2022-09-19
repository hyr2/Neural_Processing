clear all;
close all;
clc;

% load files for flows
%folder_name = 'C:\Fei\2018-stroke\2018-06-10-surgery\'; % manually specify
folder_name = pwd; 
userpath([folder_name '\tau_file']);   
flow_files = dir([folder_name '\tau_file\*.mat']);
numROI = 2;
All_flow = cell(size(flow_files,1),1);

flow = zeros(size(All_flow,1),numROI);
flow_error = zeros(size(All_flow,1),numROI);

flow_norm = zeros(size(All_flow,1),numROI);
flow_error_norm = zeros(size(All_flow,1),numROI);

for i = 1: size(All_flow,1)
    All_flow{i} = importdata(flow_files(i).name);
    flow(i,:) = mean(1./All_flow{i}.tau);
    flow_error(i,:) = std(1./All_flow{i}.tau); 
end
fcolor = {'red','blue','green','yellow'};
%load('all_timestamp.mat');
%date_flow = [-7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 10 14 21 28 35]; % manually specify 
date_flow = [-7 -6 -5 -4 -3 -2 -1 0 1 2 3];
B1 = 1:7;
%date_LFP =  [-1 0 1 2 3 4 5 6 10 13 17 24 31 38 45];  % manually specify 

% segment the data into 3 different segmentations [baseline sub_acute(one week after stroke) chronic]
% baseline  = [-1];
% sub_acute = [0 1 2 3 4 5 6];
% chronic   = [ 10 13 17 24 38];
% 
% [C1, A1, B1] = intersect(baseline, date_LFP); 
% [C2, A2, B2] = intersect(sub_acute, date_LFP); 
% [C3, A3, B3] = intersect(chronic, date_LFP); 

%%
figure;
 for i = 1: numROI
       subplot(2,4,i);
       flow_norm(:,i) = flow(:,i)/mean(flow(B1,i));
       flow_error_norm(:,i) = flow_error(:,i)/mean(flow(B1,i));
            
%        lo = flow_norm - flow_error_norm;
%        hi = flow_norm + flow_error_norm;
%        x = date_flow';

%        hp = patch([x; x(end:-1:1); x(1)], [lo(:,i); hi(end:-1:1,i); lo(1,i)],fcolor{i});
%        hold on;
%        hl(i) = line(x,flow_norm(:,i));
% 
%        set(hp, 'facecolor', fcolor{i}, 'FaceAlpha',0.25,'edgecolor', 'none');
%        set(hl(i), 'color', fcolor{i},'linewidth',1.5);
       errorbar(date_flow, flow_norm(:,i),flow_error_norm(:,i))
       %set(gca, 'xlim', [date_flow(1)-1,date_flow(end)+0.1],'ylim', [0.5,1.5]);
       if i ==1
       ylabel('Flow');
       end
       title(['ROI' num2str(i)]);
       xlabel('Days');
      % set(gca,'XTick',date_flow,'YTick',[0 1 2]);
       %set(gca,'XTicklabel',{'Baseline','Stroke','Day1','Day2', 'Day4','Day7','Day11','Day15','Day22','Day29'});
       %set(gca,'XTicklabel',[],'Fontsize',18);
       %h = legend(hl,'ROI1','ROI2','ROI3','ROI4');
       %set(h,'Fontsize',12,'Position',[0.2    0.83    0    0],'box','off','linewidth',0.5);            
 end 