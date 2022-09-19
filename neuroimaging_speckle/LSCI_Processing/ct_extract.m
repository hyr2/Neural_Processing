function ct_extract(d,d_roi)
%sc_extract Calculates CT from speckle data and folder of ROI masks
% sc_extract(d,d_roi) takes directory of .sc files and calculates average
% CT values within ROIs loaded from a directory.
%

if(d(end) == '/'); d = d(1:end-1); end;
if(d_roi(end) == '/'); d_roi = d_roi(1:end-1); end;

% Load speckle contrast data
workingDir = pwd; cd(d);
files = dir('*.sc');
SC = mean(read_subimage(files),3)';
CT = get_tc_band(SC,5e-3,1);
cd(workingDir);

% Get list of ROIS
files = dir(sprintf('%s/*_ccd.bmp',d_roi));
N_ROI = length(files);

% Calculate SC and CT within each ROI
CT_ROI = zeros(N_ROI,1);
names = cell(N_ROI,1);
for i = 1:N_ROI
    ROI = imread(sprintf('%s/%s',d_roi,files(i).name));
    CT_ROI(i) = mean(CT(ROI == 1));
    names{i} = char(['R' int2str(i)]);            
end

fprintf([repmat('%e\t',1,size(CT_ROI,1)) '\n'],CT_ROI);