function [out_arr] = func_plot3d(struct_in,struct_in_580,xaxis_in,filename_fig,bsl_indx)

% output array contains the relative area and the relative shift in center
out_arr = cell(1,6);
xaxis_ideal = [-3, 2, 7, 14,21,28,35,42,49,56];
% Plotting the activation area center 
area = zeros(1,length(struct_in));
vec_rel = zeros(length(struct_in),2);
for iter_local = 1:length(struct_in)
    area(iter_local) = struct_in{iter_local}.wv_480.Area;
    vec_rel(iter_local,:) = struct_in{iter_local}.wv_480.vec_rel(1,:);
end

%% performing fit for X axis
z_axis = xaxis_in';
x_axis = 8e-3 * vec_rel(:,1); % in mm
y_axis = 8e-3 * vec_rel(:,2); % in mm
x_bsl = (x_axis(1) + x_axis(2))/2;
x_axis = x_axis - x_bsl;
y_bsl = (y_axis(1) + y_axis(2))/2;
y_axis = y_axis - y_bsl;
area = area/mean(area(bsl_indx));
colors = [0,0,1;0,0,1];
sz_tmp = size(x_axis,1) - size(bsl_indx,2);
s2 = zeros(sz_tmp,3);
colors = vertcat(colors,s2);


ind_z = find(z_axis > 21);
out_arr{1} = mean(area(ind_z));
out_arr{2} = mean(x_axis(ind_z));

out_arr{6} = helper_fill_nans(xaxis_ideal,xaxis_in(2:end),area);    % The area array


z_axis = z_axis(~isnan(y_axis));
area = area(~isnan(y_axis));
colors = colors(~isnan(y_axis),:,:);
x_axis = x_axis(~isnan(x_axis));
y_axis = y_axis(~isnan(y_axis));

scatter3(x_axis,y_axis,z_axis,area*500,colors,'filled');
% Customize the appearance of the plot
ax = gca();
set(gca, 'FontName', 'Arial', 'FontSize', 12);
set(gca, 'Color', [0.95 0.95 0.95]);
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Days (Post Stroke)');
grid on;
lightangle(45, 30); % Adjust lighting angle
lighting gouraud;   % Smooth shading
% ax = gca;
ax.XColor = 'k';
ax.YColor = 'k';
ax.ZColor = 'k';
ax.GridColor = 'k';
ax.GridAlpha = 0.4;
ax.BoxStyle = 'full';
ax.Layer = 'top';
fig = gcf;
fig.Position = [100, 100, 800, 600];

mkdir(filename_fig)
view(0, 0); % Change the view angle
% z_axis([3]) = [];   % ignore day 2
% x_axis([3]) = [];   % ignore day 2
% degree = 4;
% coeef = polyfit(z_axis,x_axis,degree);
% z_range = linspace(min(z_axis),max(z_axis),50);
% x_fit = polyval(coeef,z_range);
% smoother_x_fit = smoothdata(x_fit,'loess',10);
% Plot the original data and the smoothed polynomial fit
% figure;
z_range = z_axis;
y_val = zeros(1,length(z_range));
% hold on;
% plot(gca,x_axis, z_axis, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
hold on;
plot3(ax,x_axis,y_val, z_range, 'Color', '#686869' , 'LineWidth', 2.5);
% plot3(ax,smoother_x_fit,y_val, z_range, 'Color', '#686869' , 'LineWidth', 2.5);
grid off;
xlim([-1.4,1.4]);
zlim([-6,56]);
xticks([-1,0,1]);
set(gca,'ZTickLabel',[]);
set(ax,'FontSize',24)
set(ax,'ZColor','none');
set(fig,'Position',[50 50 400 800])
saveas(fig,fullfile(filename_fig,'X.png'),'png');
close(fig);
%% performing fit for Y axis
z_axis = xaxis_in';
x_axis = 8e-3 * vec_rel(:,1); % in mm
y_axis = 8e-3 * vec_rel(:,2); % in mm
y_bsl = (y_axis(1) + y_axis(2))/2;
y_axis = y_axis - y_bsl;
x_bsl = (x_axis(1) + x_axis(2))/2;
x_axis = x_axis - x_bsl;
% area = area/max(area);
colors = [0,0,1;0,0,1];
sz_tmp = size(x_axis,1) - size(bsl_indx,2);
s2 = zeros(sz_tmp,3);
colors = vertcat(colors,s2);


out_arr{3} = mean(y_axis(ind_z));

z_axis = z_axis(~isnan(y_axis));
% area = area(~isnan(y_axis));
colors = colors(~isnan(y_axis),:,:);
x_axis = x_axis(~isnan(x_axis));
y_axis = y_axis(~isnan(y_axis));
scatter3(x_axis,y_axis,z_axis,area*500,colors,'filled');
% Customize the appearance of the plot
ax = gca();
set(gca, 'FontName', 'Arial', 'FontSize', 12);
set(gca, 'Color', [0.95 0.95 0.95]);
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Days (Post Stroke)');
grid on;
lightangle(45, 30); % Adjust lighting angle
lighting gouraud;   % Smooth shading
ax = gca;
ax.XColor = 'k';
ax.YColor = 'k';
ax.ZColor = 'k';
ax.GridColor = 'k';
ax.GridAlpha = 0.4;
ax.BoxStyle = 'full';
ax.Layer = 'top';
fig = gcf;
fig.Position = [100, 100, 800, 600];
view(90, 0); % Change the view angle
% y_axis([3]) = [];   % ignore day 2
% z_axis([3]) = [];   % ignore day 2
% coeef = polyfit(z_axis,y_axis,degree);
% z_range = linspace(min(z_axis),max(z_axis),50);
% y_fit = polyval(coeef,z_range);
% smoother_y_fit = smoothdata(y_fit,'loess',10);
z_range = z_axis;
x_val = zeros(1,length(z_range));
hold on;
plot3(ax,x_val,y_axis, z_range, 'Color', '#686869' , 'LineWidth', 2.5);
% plot3(ax,x_val,smoother_y_fit, z_range, 'Color', '#686869', 'LineWidth', 2.5);
grid off;
ylim([-1,1]);
zlim([-6,56]);
yticks([-1,0,1]);
set(gca,'ZTickLabel',[]);
set(ax,'ZColor','none');
set(ax,'FontSize',24)
set(fig,'Position',[50 50 400 800])
saveas(fig,fullfile(filename_fig,'Y.png'),'png');
close(fig);

%% Distance (shift)
x_axis([1,2]) = [];
x_axis = vertcat([0],x_axis);
y_axis([1,2]) = [];
y_axis = vertcat([0],y_axis);
z_axis([1,2]) = [];
% z_axis = vertcat([-1],z_axis);
delta_x = diff(x_axis);
delta_y = diff(y_axis);
distance_shifts = sqrt(delta_x.^2 + delta_y.^2);

out_arr{4} = z_axis;        % the day axis for this animal corresponds to distancce_shifts only (not the actual day axis)
out_arr{5} = distance_shifts; % relative shifts between datapoints



end