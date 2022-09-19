function speckle2bin(file_sc,n_repeat,sc_range,averaged,bin_size,varargin)
%speckle2bin Generates image from speckle contrast data and bins them into
%.sc files. Trial based averaging
% speckle2bin(file_sc,n_repeat,sc_range,averaged,bin_size) load, average, and render speckle 
% contrast data to an image file.
%
%   file_sc = Path to directory of speckle contrast files
%   n_repeat = Number of sequences in each trial (Length of trial)
%   sc_range = Display range for speckle contrast data [min max]
%   averaged = Binary indicator. True if you averaged the 15 frames in FOIL
%   bin_size = Size of each time bin in # of sequences

% speckle2Image(___,Name,Value) modifies the image appearance using one or 
% more name-value pair arguments.
%
%   'show_scalebar': Toggle display of the scalebar (Default = true)
%       Scale is hard-coded in SpeckleFigure to that of the multimodal system
%
%   'formattype': Specify output file format (Default = 'png')
%       Select either lossy 'png' or lossless 'bmp'
%
%   'resolution': Specify output file resolution (Default = System Resolution)
%       Output image resolution in pixels-per-inch
%   'trials': Specify the number of trials performed for stimulation. ie
%   the number of stimulation experiments performed

close all;

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'n_repeat', @isscalar);
addRequired(p, 'sc_range', @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'}));
addRequired(p, 'averaged', @isscalar);
addRequired(p, 'bin_size', @isscalar);
addParameter(p, 'show_scalebar', true, @islogical);
addParameter(p, 'formattype', 'png', @(x) any(validatestring(x, {'png', 'bmp'})));
addParameter(p, 'resolution', get(0, 'screenpixelsperinch'), @isscalar);
addParameter(p, 'trials', @isscalar);
addOptional(p,'trials_initial','');
parse(p, file_sc, n_repeat, sc_range,averaged,bin_size,varargin{:});

file_sc = p.Results.file_sc;
n_repeat = p.Results.n_repeat;
sc_range = p.Results.sc_range;
averaged = p.Results.averaged;
bin_size = p.Results.bin_size;
show_scalebar = p.Results.show_scalebar;
formattype = p.Results.formattype;
resolution = p.Results.resolution;
trials = p.Results.trials;
trials_initial = p.Results.trials_initial;



if (~isscalar(trials_initial))
    trials_initial = 1;
end

% Read and average the first n speckle contrast frames
files = dir_sorted(fullfile(file_sc, '*.sc'));

if (~exist(fullfile(pwd,'results'),'dir'))
    mkdir('results');   % create directory to store processed data
end


% My code to create 'n_repeat' bins
if averaged == 1
    x = floor(n_repeat/bin_size);   % number of bins
    fprintf('Total # of bins created: %d\n',x);
    start_trial = [1:n_repeat:1000*n_repeat];
    start_trial = start_trial(trials_initial:trials+trials_initial-1);
    for j = [1:x]
        arr = int32.empty(0,bin_size*trials);
%         v1 = bin_size*[j:bin_size:trials*bin_size];
        for i = 1:trials
            add_arr = [start_trial(i)+(j-1)*bin_size:start_trial(i)+(bin_size-1)+(j-1)*bin_size];
%             add_arr = [v1(i)-bin_size+1:v1(i)];
            arr = [arr,add_arr];
        end
        fprintf('Bin #%d created\n',j);
        sc = read_subimage(files, -1 , -1 ,arr);
        sc = mean(sc, 3)';

        % Create the figure
        F = SpeckleFigure(sc, sc_range);

        if show_scalebar
          F.showScalebar();
        end

        % Write the image to file
        im_ext = '.sc';
        name_sc = ['Bin', num2str(j,2), im_ext];
        name_png = ['Bin_fig', num2str(j,2)];
        filename_sc = fullfile(pwd,'results',name_sc);
        filename_png = fullfile(pwd,'results',name_png);
        write_sc(sc',filename_sc);

        if strcmp(formattype, 'png')
          F.savePNG(filename_png, resolution);
        elseif strcmp(formattype, 'bmp')
          F.saveBMP(filename_png, resolution);
        end
        clear arr;
    end
else
    frame_seq = 15;
    % Need to add code here
end
end