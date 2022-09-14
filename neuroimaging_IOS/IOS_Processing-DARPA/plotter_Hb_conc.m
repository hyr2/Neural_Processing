function [] = plotter_Hb_conc(HbO,output_plot,str_filename,rng)
%PLOTTER_HB_CONC Plots the concentrations of Hb
%   INPUT PARAMETERS:
%   HbO : a 3D array. Row is the time stack dimension
%   output_plot : folder location of the output directory
%   str_filename : name of file appended 
%   rng : the range of values for the displayed image [cmin,cmax]

[len_trials,X,Y] = size(HbO);
tmp_mat = reshape(HbO(25,:,:),[X,Y]);
amin = rng(1);
amax = rng(2);

for iter_seq = 1:len_trials
    tmp_mat = reshape(HbO(iter_seq,:,:),[X,Y]);

    fg = imshow(tmp_mat,[amin,amax]);
    colormap('jet');
    colorbar;
    
    name_iter = fullfile(output_plot,strcat(str_filename,'-',num2str(iter_seq),'.svg'));
    saveas(fg,name_iter,'svg');
end

