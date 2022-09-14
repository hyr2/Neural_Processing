function [Nsub] = cal_dr_r_indi_green(Nsub,selpathOut,title_pic,name_pic)
% Nsub = AverScan_red - bs_red;
low_limit = -0.06;
up_limit = 0.01;
dx = 6.667; %um per pixel
length_dx = round(400/dx);
x_location = 50;
y_location = 50;
% Nsub = Nsub*(-1);
p2 = [530,40];
p1 = [530,40+length_dx];
Infi = 10000000;


Nsub(Nsub>0)=0;
curfig = figure('visible','off');
imshow(Nsub);
% mymap = othercolor('Bu_10');
colormap('jet');
% colormap(mymap);
% colormap('colorcube');
colorbar
caxis([low_limit up_limit])
hold on
plot([p1(2),p2(2)],[p1(1),p2(1)],'Color','k','LineWidth',2)
hold off

% quiver(x_location,y_location,dx,10,'w','ShowArrowHead','off')
title_pic = strcat(title_pic, name_pic);
title(title_pic);
name_pic = strcat(title_pic ,'_full.svg');
outputFileName = fullfile(selpathOut,name_pic);
saveas(curfig,outputFileName);
close(curfig);

end