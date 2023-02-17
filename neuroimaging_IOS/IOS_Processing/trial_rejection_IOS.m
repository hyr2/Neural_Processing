load('/home/hyr2-office/Documents/MATLAB/IOS_processing/IOS_singleTrial/IOS_singleTrial6.mat')
len_trials = size(ios_trials_local,2);
NTrials = size(ios_trials_local,1);
L = 40;
y = ios_trials_local(:,:,2);
v = mean(y,1,'omitnan');
v = v([9:L]);
output_vec = [];


for iter_trial = 1:NTrials
    u = y(iter_trial,[9:L]);
   
    figure;
    plot(u,'g');
    hold on;
    plot(v,'r');
    xline(6);
    xline(21);
    tmp_corr = corrcoef(u,v);
    tmp_std = std(y(iter_trial,[1:14]));
    tmp_std_2 = std(y(iter_trial,[35:45]));
    SSD = sum((u - v).^2);
    output_vec(iter_trial,1) = tmp_corr(1,2);
    output_vec(iter_trial,2) = tmp_std;
    output_vec(iter_trial,3) = tmp_std_2;
    output_vec(iter_trial,4) = SSD;
    title([num2str(tmp_std),'\t',num2str(tmp_corr(1,2))]);
end