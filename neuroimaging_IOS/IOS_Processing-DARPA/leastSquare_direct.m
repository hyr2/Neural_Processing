function [b] = leastSquare_direct(X,y)
% leastSquare_direct: Solves a system of linear equations (commonly over
% fitted) using least squares method. The normal equation is implemented.
% Trying to solve the matrix equation: y = A . b
%   INPUT PARAMETERS:
%       X : The matrix (2D)
%       y : Known column vector
%   OUTPUT PARAMETERS:
%       b : unknown column vector

M1 = X.' * X;
V1 = X.' * y;
b = M1\V1;
% b = inv(A.' * A) * A.' * y;

end

