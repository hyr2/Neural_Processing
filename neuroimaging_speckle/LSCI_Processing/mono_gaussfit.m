function [coefficients] = mono_gaussfit(x, y, init_guess)
%CREATEFIT(X,C2)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      x : X axis 
%      y: Y axis
%      init_guess: [b1 b2 b3 b4 b5] (SEE FIT MODEL)
%  Output:
%      coefficients: Fitted values of b vector (SEE FIT MODEL)
%      
%
%  See also FIT, CFIT, SFIT.

% Data
% Convert X and Y into a table, which is the form fitnlm() likes the input data to be in.

tbl = table(x', y');

%% Fit model
modelfun = @(b,x) b(1) * exp(-(x(:, 1) - b(2)).^2/b(3)) + b(4) + b(5) * x(:, 1);
% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
mdl = fitnlm(tbl, modelfun, init_guess);

% Making model more robust by checking MSE (mean square error)
if (mdl.SSE > 0.0170)
    init_guess(3) = 13;
    mdl = fitnlm(tbl, modelfun, init_guess);
end
if (mdl.SSE > 0.0170)
    init_guess(3) = 17;
    mdl = fitnlm(tbl, modelfun, init_guess);
end

coefficients = mdl.Coefficients{:, 'Estimate'};

% % PLOTTING
% X = linspace(min(x), max(x), 1920); % Let's use 1920 points, which will fit across an HDTV screen about one sample per pixel.
% % Create smoothed/regressed data using the model:
% yFitted = coefficients(1) * exp(-(X - coefficients(2)).^2 / coefficients(3)) + coefficients(4) + coefficients(5) * X;
% hold on;
% plot(X, yFitted, 'r-', 'LineWidth', 2);
% plot(x,y,'b.', 'LineWidth', 2, 'MarkerSize', 15);
% grid on;
% title('Exponential Regression with fitnlm()', 'FontSize', 16);
% xlabel('X', 'FontSize', 16);
% ylabel('Y', 'FontSize', 16);

end

