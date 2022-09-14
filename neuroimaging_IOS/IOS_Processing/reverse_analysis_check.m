clear all;
source_dir = input('Please enter the source directory:\n','s');
% wavelength selection
wv_3 = input('The 3rd wavelength was 510nm or 632nm ? \n','s');
wv_3 = int16(str2num(wv_3));



% Values obtained from literature (https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frstb.2015.0360&file=rstb20150360supp1.pdf)
e_HbO_blue = 26629.2; e_HbR_blue = 14550; X_blue = 0.064898;
e_HbO_yellow = 50104  ; e_HbR_yellow = 37020; X_yellow = 0.0306261;
e_HbO_green = 20035.2; e_HbR_green = 25773.6; X_green = 0.0712537;
e_HbO_red = 561.2; e_HbR_red = 4930.8; X_red = 0.3924145;


if wv_3 == 510
    e_HbO_wv3 = e_HbO_green; e_HbR_wv3 = e_HbR_green; X_wv3 = X_green;
elseif wv_3 == 632
    e_HbO_wv3 = e_HbO_red; e_HbR_wv3 = e_HbR_red; X_wv3 = X_red;
else
    error('incorrect wavelength selected! \n')
end


output_dir = fullfile(source_dir,'Processed');
mat_dir = fullfile(output_dir,'mat_files');
whiskerStim_txt = fullfile(source_dir,'whisker_stim.txt');
load(fullfile(mat_dir,'delta_c.mat'));



[len_trials,X,Y] = size(HbO);
delta_I_predict_b = zeros(len_trials,X,Y);
delta_I_predict_y = zeros(len_trials,X,Y);
delta_I_predict_wv3 = zeros(len_trials,X,Y);

for iter_seq = 1:len_trials
    delta_I_predict_b(iter_seq,:,:) = reshape(-X_blue * (e_HbO_blue*HbO(iter_seq,:,:)  + e_HbR_blue*HbR(iter_seq,:,:)),[X,Y]);
    delta_I_predict_y(iter_seq,:,:) = reshape(-X_yellow * (e_HbO_yellow*HbO(iter_seq,:,:)  + e_HbR_yellow*HbR(iter_seq,:,:)),[X,Y]);
    delta_I_predict_wv3(iter_seq,:,:) = reshape(-X_wv3 * (e_HbO_wv3*HbO(iter_seq,:,:)  + e_HbR_wv3*HbR(iter_seq,:,:)),[X,Y]);
end

deltaR_predict_b = exp(delta_I_predict_b);
deltaR_predict_y = exp(delta_I_predict_y);
deltaR_predict_wv3 = exp(delta_I_predict_wv3);

load(fullfile(mat_dir,'data_wv3.mat'));
load(fullfile(mat_dir,'data_amber.mat'));
load(fullfile(mat_dir,'data_blue.mat'));

avg_err_b = zeros(1,len_trials);
avg_err_y = zeros(1,len_trials);
avg_err_wv3 = zeros(1,len_trials);

mat_temp = zeros(X,Y);

for iter_seq = 1:len_trials
    
    mat_temp = reshape(deltaR_predict_b(iter_seq,:,:) - Avg_blue(iter_seq,:,:),X,Y);
    avg_err_b(iter_seq) = sum(mat_temp,'all','omitnan');
    
    mat_temp = reshape(deltaR_predict_y(iter_seq,:,:) - Avg_amber(iter_seq,:,:),X,Y);
    avg_err_y(iter_seq) = sum(mat_temp,'all','omitnan');
    
    mat_temp = reshape(deltaR_predict_wv3(iter_seq,:,:) - Avg_wv3(iter_seq,:,:),X,Y);
    avg_err_wv3(iter_seq) = sum(mat_temp,'all','omitnan');
    
end

