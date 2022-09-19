load('C:\Experimental-Data\3-30-2021\ON\2-bin average\rICT_ROI_CTavg\data.mat');
s1 = size(rICT);
t = [t,t(end)+0.510];
s2 = s1(1);
s1 = s1(2);
for i = 1:s1
   rICT_temp = [1;rICT(:,i)]; 
   
   [p,S,mu] = polyfit(t,rICT_temp,15);
   y_fit = polyval(p,t,S,mu);
   plot(t,y_fit);
   hold on;
end