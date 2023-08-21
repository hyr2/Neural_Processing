function [] = func_plot2d_multiY(struct1,struct2,struct3,xaxis_1,xaxis_2,xaxis_3,filename_fig)

%% Struct 1
area = zeros(1,length(struct1));
vec_rel = zeros(length(struct1),2);
for iter_local = 1:length(struct1)
    area(iter_local) = struct1{iter_local}.wv_580.Area;
    vec_rel(iter_local,:) = struct1{iter_local}.wv_580.vec_rel(1,:);
end
% performing fit for X axis
z_axis = xaxis_1';
x_axis = 8e-3 * vec_rel(:,1); % in mm
y_axis = 8e-3 * vec_rel(:,2)-1.215+1; % in mm
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
view(90, 0); % Change the view angle
y_axis([3]) = [];   % ignore day 2
z_axis([3]) = [];   % ignore day 2
degree = 5;
coeef = polyfit(z_axis,y_axis,degree);
z_range = linspace(min(z_axis),max(z_axis),50);
y_fit = polyval(coeef,z_range);
smoother_y_fit = smoothdata(y_fit,'loess',10);
x_val = zeros(1,length(z_range));
hold on;
plot3(ax,x_val,smoother_y_fit, z_range, 'Color', '#686869', 'LineWidth', 2.5);

set(ax,'FontSize',24)
% saveas(fig,fullfile(filename_fig,'X.png'),'png');
% close(fig);

%% Struct 2
area = zeros(1,length(struct2));
vec_rel = zeros(length(struct2),2);
for iter_local = 1:length(struct2)
    area(iter_local) = struct2{iter_local}.wv_580.Area;
    vec_rel(iter_local,:) = struct2{iter_local}.wv_580.vec_rel(1,:);
end
% performing fit for X axis
z_axis = xaxis_2';
x_axis = 8e-3 * vec_rel(:,1); % in mm
y_axis = 8e-3 * vec_rel(:,2)-0.32 + 3; % in mm
area = area/max(area);
colors = [0,0,1;0,0,1];
s2 = zeros(9,3);
colors = vertcat(colors,s2);

scatter3(x_axis,y_axis,z_axis,area*500,colors,'filled');

y_axis([3]) = [];   % ignore day 2
z_axis([3]) = [];   % ignore day 2
coeef = polyfit(z_axis,y_axis,degree);
z_range = linspace(min(z_axis),max(z_axis),50);
y_fit = polyval(coeef,z_range);
smoother_y_fit = smoothdata(y_fit,'loess',10);
x_val = zeros(1,length(z_range));
hold on;
plot3(ax,x_val,smoother_y_fit, z_range, 'Color', '#686869', 'LineWidth', 2.5);

set(ax,'FontSize',24)


%% Struct 3
area = zeros(1,length(struct3));
vec_rel = zeros(length(struct3),2);
for iter_local = 1:length(struct3)
    area(iter_local) = struct3{iter_local}.wv_580.Area;
    vec_rel(iter_local,:) = struct3{iter_local}.wv_580.vec_rel(1,:);
end
% performing fit for X axis
z_axis = xaxis_3';
x_axis = 8e-3 * vec_rel(:,1); % in mm
y_axis = 8e-3 * vec_rel(:,2)-0.34 + 5; % in mm
area = area/max(area);
colors = [0,0,1;0,0,1];
s2 = zeros(9,3);
colors = vertcat(colors,s2);

scatter3(x_axis,y_axis,z_axis,area*500,colors,'filled');

y_axis([3]) = [];   % ignore day 2
z_axis([3]) = [];   % ignore day 2
coeef = polyfit(z_axis,y_axis,degree);
z_range = linspace(min(z_axis),max(z_axis),50);
y_fit = polyval(coeef,z_range);
smoother_y_fit = smoothdata(y_fit,'loess',10);
x_val = zeros(1,length(z_range));
hold on;
plot3(ax,x_val,smoother_y_fit, z_range, 'Color', '#686869', 'LineWidth', 2.5);
set(ax,'FontSize',24)
yGridLines = [0:9];
grid on;
set(ax, 'YGrid', 'on', 'YTick', yGridLines);
zlim([-10,60]);
set(gcf, 'PaperUnits', 'inches');
x_width=7.25 ;y_width=7.25;
set(gcf, 'PaperPosition', [0 0 x_width y_width]); %
ylim([-0.5,6]);
% set(fig, 'Position', fig_position,'PaperUnits', 'Inches');

saveas(fig,fullfile(filename_fig,'Y.png'),'png');
close(fig);

end