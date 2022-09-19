function SC = speckle_contrast_gpu(I, NxN)

    I_GPU = gpuArray(I); % Convert to GPU array
    SC_GPU = gpuArray(zeros(size(I)));
    N = NxN*NxN;
    h = ones(NxN,NxN);

    for i = 1:size(I_GPU,3)
        Im = conv2(I_GPU(:,:,i),h,'same');
        Im2 = conv2(I_GPU(:,:,i).^2,h,'same');
        SC_GPU(:,:,i) = sqrt((N*Im2-Im.^2)/(N*(N-1)))./(Im/N);
    end
    
    SC = gather(SC_GPU);

end