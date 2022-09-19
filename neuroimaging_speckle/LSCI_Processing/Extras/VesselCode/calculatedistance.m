function d = calculatedistance(val)
x = 1:length(val);
x = x';
val = -val;
f = fit(x,val,'gauss2');

[max1 indx_max]  = max(f(x));

c = coeffvalues(f);
% syms y;
y = [1:indx_max];
f1 = c(1)*exp(-((y-c(2))/c(3)).^2)+c(4).*exp(-((y-c(5))/c(6)).^2);
% c(7).*exp(-((y-c(8))/c(9)).^2);

[min1 indx_min] = min(f1);


half = max1-(max1-min1)/2;

c = coeffvalues(f);
% syms y;
y = [indx_min:indx_max];

f1 = c(1)*exp(-((y-c(2))/c(3)).^2)+c(4).*exp(-((y-c(5))/c(6)).^2);%+...
%c(7).*exp(-((y-c(8))/c(9)).^2);

%plot(f(x))

indx = find(abs(f1 - half)<2e-2);

if(length(indx)>1)
    f1  = f1(indx);
    [~, indx2] = min(abs(f1-half));
    indx = indx(1) + indx2 - 1;
end
% x1 = vpasolve(f1==half,y,1);
% disp(1)
% disp(x1)
% x2 = vpasolve(f1==half,y,length(val));
% disp(2)
% disp(x2)
d = (indx_max - indx)*2;
% d = abs(x1-x2);
end
