function sc=speckle_contrast(Iraw, NxN)
  
sc=zeros(size(Iraw));
N=NxN*NxN;
h=ones(NxN,NxN);

for i=1:size(Iraw,3)
    Im=conv2(Iraw(:,:,i),h,'same');
    Im2=conv2(Iraw(:,:,i).^2,h,'same');
    sc(:,:,i)=sqrt((N*Im2-Im.^2)/(N*(N-1)))./(Im/N);
end

    






