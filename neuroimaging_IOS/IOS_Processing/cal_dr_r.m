function [result] = cal_dr_r(Nsub,selpathOut,title_pic)
% Nsub = AverScan_red - bs_red;
low_limit = -0.04;
up_limit = 0.04;
dx = 6.667; %um per pixel
length_dx = round(400/dx);
x_location = 50;
y_location = 50;
% Nsub = Nsub*(-1);
p2 = [530,40];
p1 = [530,40+length_dx];
Infi = 10000000;
% [x y]= size(baseline);
% baseline(baseline==0) = Infi;
% result = zeros(x,y);
% result = (Nsub./baseline);
result = Nsub;
% result(10,10)= -1;
% test
% mymap = othercolor('RdYlBu11',60);
% mymap = othercolor('BuOr_8');
curfig = figure()
imshow(result)
colormap('jet');
% colormap(mymap);
% colormap('colorcube');
colorbar
% caxis([-0.1 0.1])
caxis([low_limit up_limit])
hold on
plot([p1(2),p2(2)],[p1(1),p2(1)],'Color','k','LineWidth',2)
hold off
text(40,510,'400 um')
% quiver(x_location,y_location,dx,10,'w','ShowArrowHead','off')
title(title_pic)
outputFileName = [selpathOut '\' title_pic '.jpg'];
saveas(curfig,outputFileName);
outputFileName = [selpathOut '\' title_pic '.tif'];
imwrite(Nsub(:, :), outputFileName, 'WriteMode', 'append',  'Compression','none');

result2 = result;
result2(result>0)=0;
curfig = figure()
imshow(result2)
% mymap = othercolor('Bu_10');
colormap('jet');
% colormap(mymap);
% colormap('colorcube');
colorbar
caxis([low_limit 0])
hold on
plot([p1(2),p2(2)],[p1(1),p2(1)],'Color','k','LineWidth',2)
hold off
text(40,510,'400 um')
% quiver(x_location,y_location,dx,10,'w','ShowArrowHead','off')
title(title_pic)
outputFileName = [selpathOut '\' title_pic '_neg.jpg'];
saveas(curfig,outputFileName);

result2 = result;
result2(result<0)=0;
curfig = figure()
imshow(result2)
% mymap = othercolor('Bu_10');
colormap('jet');
% colormap(mymap);
% colormap('colorcube');
colorbar
caxis([0 up_limit])
hold on
plot([p1(2),p2(2)],[p1(1),p2(1)],'Color','k','LineWidth',2)
hold off
text(40,510,'400 um')
% quiver(x_location,y_location,dx,10,'w','ShowArrowHead','off')
title(title_pic)
outputFileName = [selpathOut '\' title_pic '_pos.jpg'];
saveas(curfig,outputFileName);

close all
% 
% curfig = figure()
% imshow(result)
% colormap('jet');
% % colormap('colorcube');
% colorbar
% caxis([-0.04 0.04])
% title(title_pic)
% outputFileName = [selpathOut '\' title_pic '_2.jpg'];
% saveas(curfig,outputFileName);
% 
% figure
% colorscheme = othercolor('Set16',6);
% set(gcf,'DefaultAxesColorOrder',colorscheme);
% plot(rand(6,20));
% legend(num2str((1:6)'))