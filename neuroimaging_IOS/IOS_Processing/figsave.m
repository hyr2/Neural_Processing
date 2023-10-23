function figsave(fig,file,rez,txt,bak)
%Save figure as image, with custom resolutions and text scaling.
% figsave(fig,file,rez,txt,bak)
%
%Example:
% clf,text(0.1,0.5,{'This text should be';'50 pixels high';'and the image';'900W x 600H pix'},'FontSize',50)
% figsave(gcf,'Figure.jpg',[900 600])
if nargin<1 || isempty(fig),  fig  = gcf; end          %figure handle
if nargin<2 || isempty(file), file = 'Figure.jpg'; end %output file name
if nargin<3 || isempty(rez),  rez  = [900 600]; end    %resolution [horizontal vertical]
if nargin<4 || isempty(txt),  txt  = 1; end            %text scale factor
if nargin<5 || isempty(bak),  bak  = 1; end            %preserve background colour
set(fig,'PaperPosition',[0 0 rez/(100*txt)],'PaperUnits','inches'); %set paper size (does not affect display)
if bak
    set(fig,'InvertHardcopy','off'); %preserve background colour
end
imwrite(print(gcf,'-RGBImage',['-r' num2str(100*txt,'%f')]),file) %print RGB image to file (slow)
end