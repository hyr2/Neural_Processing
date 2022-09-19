function labelCentroid(mask,label)
%labelCentroid Add text label centered at centroid of binary mask
%   labelCentroid(mask,label) adds a text label to the current figure
%   centered at the centroid of given binary mask.
%

    % Calculate the centroid of the mask
    cen = regionprops(true(size(mask)), mask,  'WeightedCentroid');
    cen = cen.WeightedCentroid;

    % Create black text outline with four offset text objects
    o = [0 -1; -1 0; 0 1; 1 0]; % offset
    for k = 1:4
        text(cen(1) + 1 +o(k,1), cen(2) + 3 + o(k,2),label,'Color','k',...
            'FontUnits','Pixels','FontWeight','Bold','FontSize',25,...
            'HorizontalAlignment','center','VerticalAlignment','middle');
    end

    % Add main white label text
    text(cen(1) + 1,cen(2) + 3,label,'Color',[1-eps 1 1],...
        'FontUnits','Pixels','FontWeight','Bold','FontSize',25,...
        'HorizontalAlignment','center','VerticalAlignment','middle');
end