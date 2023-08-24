function [shankID,y_depth] = findShankID(inputArg1,flag)
% Example of input argument is 3.19 which translates to shank C and 19th
% electrode from bottom
    
% For 1x32 only    
if flag == '1x32'
    shankID = floor(inputArg1);

    electrodeID = (inputArg1 - floor(inputArg1))*100;
    
    y_depth = 33 - electrodeID;  % depth from the top of the shank
end

end

