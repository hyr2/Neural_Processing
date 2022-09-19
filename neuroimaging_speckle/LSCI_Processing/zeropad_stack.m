function [out_stack] = zeropad_stack(in_stack,N)
%ZEROPAD_STACK This function will zero pad matrices before and after the
% image stack uniformly
%   Will add N 2D matrices, all zeros, front and back of the input image
%   stack
dim_SC = size(in_stack);
dim_SC(3) = [];
out_stack = in_stack;
for iter = 1:N
    out_stack = cat(3,out_stack,zeros(dim_SC(1),dim_SC(2)));
    out_stack = cat(3,zeros(dim_SC(1),dim_SC(2)),out_stack);
end

end

