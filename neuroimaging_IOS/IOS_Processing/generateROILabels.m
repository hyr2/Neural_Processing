function labels = generateROILabels(N)
%generateROILabels returns a cell array of numbered ROI labels

labels = cell(1,N);
for i = 1:N
  labels{i} = char(['R' int2str(i)]);
end

end