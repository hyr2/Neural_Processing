obj = VideoReader('C:\Users\Luan-speckle\Desktop\awake\ICT.mp4');
vid = read(obj);
frames = obj.NumberOfFrames;
for x = 1 : frames
    filename = strcat('frame-',num2str(x),'.png');
    filename = fullfile(pwd,filename);
    imwrite(vid(:,:,:,x),filename);
end