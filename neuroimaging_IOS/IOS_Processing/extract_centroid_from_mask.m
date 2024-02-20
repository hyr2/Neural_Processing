function [output_centroid] = extract_centroid_from_mask(input_mask)
%EXTRACT_CENTROID_FROM_MASK Summary of this function goes here

labeledImage = bwlabel(input_mask);    % label each discontinuous area to a unique value
tmp_vee = unique(labeledImage);
tmp_vee = tmp_vee(2:end);
labeledImage(ismember(labeledImage,tmp_vee)) = 1;       % Setting all activated regions as one single region
props = regionprops(labeledImage,'Centroid','FilledArea');               % Find properties of the binary mask
output_centroid = props.Centroid;

end

