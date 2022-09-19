close all;

%% Experimental Data plot (from FOIL)

% Analysis on FOIL timing data
t_foil = loadSpeckleTiming('D:\Haad\Hefty\2-22-2021\2-22-2021-d\data.timing');
t_foil = t_foil - t_foil(1);
a = diff(t_foil);
b = find(a>0.09);
t_foil_seq = diff(t_foil(b));
subplot(221);
plot(1e3*t_foil_seq(1:4000));
ylim([0 700]);
xlabel('seq #');
ylabel('Acquisition Time (ms)');
title('Dataset-d (FOIL)');

% Analysis on FOIL timing data
t_foil = loadSpeckleTiming('D:\Haad\Hefty\2-22-2021\2-22-2021-e\data.timing');
t_foil = t_foil - t_foil(1);
a = diff(t_foil);
b = find(a>0.09);
t_foil_seq = diff(t_foil(b));
subplot(223);
plot(1e3*t_foil_seq);
ylim([0 700]);
xlabel('seq #');
ylabel('Acquisition Time (ms)');
title('Dataset-e (FOIL)');

% Analysis on FOIL timing data
t_foil = loadSpeckleTiming('D:\Haad\Hefty\2-22-2021\2-22-2021-f\data.timing');
t_foil = t_foil - t_foil(1);
a = diff(t_foil);
b = find(a>0.09);
t_foil_seq = diff(t_foil(b));
subplot(224);
plot(1e3*t_foil_seq);
ylim([0 700]);
xlabel('seq #');
ylabel('Acquisition Time (ms)');
title('Dataset-f (FOIL)');


%% Ideal timing sequence

% load('edge-data.mat');
% t_DAQ = length_seq;
% d = 100;
% t_DAQ = downsample(t_DAQ,d);
% t_DAQ = repmat(t_DAQ,3500,1);
% xaxis = d*[1:length(t_DAQ)]/1e6;
% b = diff(t_DAQ);
% a = find(b>4);
% t_DAQ = xaxis(a); % this is similar to what FOIL generates. (time stamps)
% b = diff(t_DAQ);
% a = find(b>0.0085);
% t_DAQ_seq = t_DAQ(a);
% subplot(236)
% plot(diff(t_DAQ_seq));
% ylim([0 0.5]);

%% Experimental Data plot (from Intan System)
clear all;
load('data_Intan.mat');
b = diff(data_axis);

a = find(b>3);
time_axis = time_axis(a); % this is similar to what FOIL generates. (time stamps)
time_axis = time_axis - time_axis(1);
b = diff(time_axis);
a = find(b>0.1);
t_INTAN_seq = time_axis(a);
subplot(222)
plot(1e3*diff(t_INTAN_seq));
ylim([0 700]);
xlabel('seq #');
ylabel('Acquisition Time (ms)');
title('Dataset-d (Intan ADC)');



%% Plotting Extras

figure;

% Analysis on FOIL timing data
t_foil = loadSpeckleTiming('C:\Lan Luan Lab\Data\Hefty\2-22-2021-a\data.timing');
t_foil = t_foil - t_foil(1);
a = diff(t_foil);
b = find(a>0.09);
t_foil_seq = diff(t_foil(b));
subplot(221);
plot(1e3*t_foil_seq);
ylim([0 700]);
xlabel('seq #');
ylabel('Acquisition Time (ms)');
title('Dataset-a (FOIL)');

% Analysis on FOIL timing data
t_foil = loadSpeckleTiming('C:\Lan Luan Lab\Data\Hefty\2-22-2021-b\data.timing');
t_foil = t_foil - t_foil(1);
a = diff(t_foil);
b = find(a>0.09);
t_foil_seq = diff(t_foil(b));
subplot(222);
plot(1e3*t_foil_seq);
ylim([0 700]);
xlabel('seq #');
ylabel('Acquisition Time (ms)');
title('Dataset-b (FOIL)');

% Analysis on FOIL timing data
t_foil = loadSpeckleTiming('C:\Lan Luan Lab\Data\Hefty\2-22-2021-c\data.timing');
t_foil = t_foil - t_foil(1);
a = diff(t_foil);
b = find(a>0.09);
t_foil_seq = diff(t_foil(b));
subplot(223);
plot(1e3*t_foil_seq);
ylim([0 700]);
xlabel('seq #');
ylabel('Acquisition Time (ms)');
title('Dataset-c (FOIL)');

% Analysis on FOIL timing data
t_foil = loadSpeckleTiming('C:\Lan Luan Lab\Data\Hefty\2-22-2021-i\data.timing');
t_foil = t_foil - t_foil(1);
a = diff(t_foil);
b = find(a>0.09);
t_foil_seq = diff(t_foil(b));
subplot(224);
plot(1e3*t_foil_seq);
ylim([0 700]);
xlabel('seq #');
ylabel('Acquisition Time (ms)');
title('Dataset-i (FOIL)');

