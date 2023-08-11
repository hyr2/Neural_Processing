function [out_im_stack] = twoD_gauss(im_stack,sig)
%TWOD_GAUSS Summary of this function goes here
%   Detailed explanation goes here

[len_trials,X,Y] = size(im_stack);
out_im_stack = zeros(len_trials,X,Y);

for iter_seq = 1:len_trials
    out_im_stack(iter_seq,:,:) = imgaussfilt(reshape(im_stack(iter_seq,:,:),X,Y),sig);    %2D Gaussian filtering
end
