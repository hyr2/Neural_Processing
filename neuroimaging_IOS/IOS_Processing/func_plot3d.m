function [] = func_plot3d(struct_in,xaxis_in,filename_fig)

% Plotting the activation area center 
area = zeros(1,length(struct_in));
vec_rel = zeros(length(struct_in),2);
for iter_local = 1:length(struct_in)
    area(iter_local) = struct_in{iter_local}.wv_580.Area;
    vec_rel(iter_local,:) = struct_in{iter_local}.wv_580.vec_rel(1,:);
end

%% performing fit for X axis
z_axis = xaxis_in';
x_axis = 8e-3 * vec_rel(:,1); % in mm
y_axis = 8e-3 * vec_rel(:,2); % in mm
area = area/max(area);
colors = [0,0,1;0,0,1];
s2 = zeros(9,3);
colors = vertcat(colors,s2);

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

mkdir(filename_fig)
view(0, 0); % Change the view angle
z_axis([3]) = [];   % ignore day 2
x_axis([3]) = [];   % ignore day 2
degree = 5;
coeef = polyfit(z_axis,x_axis,degree);
z_range = linspace(min(z_axis),max(z_axis),50);
x_fit = polyval(coeef,z_range);
smoother_x_fit = smoothdata(x_fit,'loess',10);
% Plot the original data and the smoothed polynomial fit
% figure;
y_val = zeros(1,length(z_range));
% hold on;
% plot(gca,x_axis, z_axis, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
hold on;
plot3(ax,smoother_x_fit,y_val, z_range, 'Color', '#686869' , 'LineWidth', 2.5);
grid on;
set(ax,'FontSize',24)
saveas(fig,fullfile(filename_fig,'X.png'),'png');
close(fig);
%% performing fit for Y axis
z_axis = xaxis_in';
x_axis = 8e-3 * vec_rel(:,1); % in mm
y_axis = 8e-3 * vec_rel(:,2); % in mm
area = area/max(area);
colors = [0,0,1;0,0,1];
s2 = zeros(9,3);
colors = vertcat(colors,s2);

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
y_axis([3]) = [];   % ignore day 2
z_axis([3]) = [];   % ignore day 2
coeef = polyfit(z_axis,y_axis,degree);
% z_range = linspace(min(z_axis),max(z_axis),50);
y_fit = polyval(coeef,z_range);
smoother_y_fit = smoothdata(y_fit,'loess',10);
x_val = zeros(1,length(z_range));
hold on;
plot3(ax,x_val,smoother_y_fit, z_range, 'Color', '#686869', 'LineWidth', 2.5);
grid on;
set(ax,'FontSize',24)
saveas(fig,fullfile(filename_fig,'Y.png'),'png');
close(fig);

end