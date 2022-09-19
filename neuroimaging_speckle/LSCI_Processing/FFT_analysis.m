y1 = fft(rICT(:,1),1024);y2 = fft(rICT(:,2),1024);
y3 = fft(rICT(:,3),1024);y4 = fft(rICT(:,4),1024);

y1 = fft(filtered1',1024);
y2 = fft(filtered2',1024);




y1(1) = 0; y2(1) = 0;
y3(1) = 0; y4(1) = 0;
y1 = fftshift(y1);
y2 = fftshift(y2);
y3 = fftshift(y3);
y4 = fftshift(y4);

N = 1024;
F_s = 1/0.0856;

w_freq = linspace(-pi,pi-(2*pi)/N,N);
freq_proper = F_s*w_freq/(2*pi);        %Spatial frequency+

subplot(221)
stem(freq_proper(N/2+1:end),abs(y1(N/2+1:end)));
ylabel('Power (a.u)')
xlabel('Frequency (Hz)');
% xlim([0 2])

subplot(222)
stem(freq_proper(N/2+1:end),abs(y2(N/2+1:end)));
ylabel('Power (a.u)')
xlabel('Frequency (Hz)');
% xlim([0 2])

subplot(223)
stem(freq_proper(N/2+1:end),abs(y3(N/2+1:end)));
ylabel('Power (a.u)')
xlabel('Frequency (Hz)');

subplot(224)
stem(freq_proper(N/2+1:end),abs(y4(N/2+1:end)));
ylabel('Power (a.u)')
xlabel('Frequency (Hz)');