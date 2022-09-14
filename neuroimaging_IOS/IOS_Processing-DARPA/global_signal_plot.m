function [Fig_hdl] = global_signal_plot(global_t_stack, len_trials, num_trials)
%global_signal_plot Plots the global variation of intensity in the IOS raw image 
%   All trials and all sequences continuous time plot

temp_vec_blue = [];
temp_vec_amber = [];
temp_vec_green = [];
for iter_trial = 1:num_trials
    temp_vec_blue = horzcat(temp_vec_blue, global_t_stack(iter_trial,:,1));
    temp_vec_amber = horzcat(temp_vec_amber, global_t_stack(iter_trial,:,2));
    temp_vec_green = horzcat(temp_vec_green, global_t_stack(iter_trial,:,3));
end
x_axis = linspace(1,num_trials+1,num_trials*len_trials);
Fig_hdl = figure();

hold on;
plot(x_axis,temp_vec_blue);
plot(x_axis,temp_vec_amber);
plot(x_axis,temp_vec_green);
legend({'480 nm','580 nm',"510 nm"});
title('Average intensity inside craniotomy');
xlabel('Trial');
ylabel('Value');

end

