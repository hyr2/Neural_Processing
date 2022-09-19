function [sc mean_raw std_raw] = temporal_speckle_contrast2(Iraw,N)
% Computes a temporal speckle contrast image of a series of raw images
% Shams' version

if (size(Iraw,3)-N)+1 < 0
   disp('N chosen exceeds the number of images sent in');
   temp_sc = -1;
   return;
end
rows=size(Iraw,1);
cols=size(Iraw,2);
imgs=size(Iraw,3);

Iraw_cols=squeeze(reshape(Iraw,[rows*cols 1 imgs]));
temp_sc=zeros(rows*cols,imgs/N);


for blocks=1:imgs/N
     std_raw(:,blocks) = std(Iraw_cols(:,1+(blocks-1)*N:N*blocks),0,2);
     mean_raw(:,blocks) = mean(Iraw_cols(:,1+(blocks-1)*N:N*blocks),2);
     temp_sc(:,blocks) = std(Iraw_cols(:,1+(blocks-1)*N:N*blocks),0,2)./mean(Iraw_cols(:,1+(blocks-1)*N:N*blocks),2);
end
   
sc=reshape(temp_sc,[rows,cols,imgs/N]);
mean_raw=reshape(mean_raw,[rows,cols,imgs/N]);
std_raw=reshape(std_raw,[rows,cols,imgs/N]);