
delta_t = 0.08;
% Digital filter
order = 6;      % Order of the IIR filter
fcutlow = 1.5;    % cutoff frequency
Fs = 1/delta_t;        % Sampling frequency
lowpass_filter = designfilt('lowpassiir','FilterOrder',order, ...
    'StopbandFrequency',fcutlow,'SampleRate',Fs,'DesignMethod','cheby2');

ndCT_final = permute(ndCT_final,[3 2 1]);           % Row index is time
rICT_filt = filtfilt(lowpass_filter,rICT);   % Filtering on row index
ndCT_final = permute(ndCT_final,[3 2 1]);           % Z index is time

plot(t,rICT_filt(:,5)-1.015)
