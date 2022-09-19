# Laser Speckle Contrast Imaging Scripts
MATLAB scripts for processing and analyzing laser speckle contrast imaging (LSCI) data.

## Usage

[`SpeckleFigure`](SpeckleFigure.m): Class to facilitate the visualization of speckle contrast data

```
files = dir('*.sc');
sc = read_subimage(files, -1, -1, 1)';
sc_range = [0 0.3];

F = SpeckleFigure(sc, sc_range);        % Construct SpeckleFigure object

F.hide();                               % Hide figure
F.show();                               % Show figure

F.update(sc);                           % Update the data
F.setDisplayRange(sc_range);            % Change the display range

F.showColorbar();                       % Display colorbar
F.hideColorbar();                       % Hide colorbar

F.showScalebar();                       % Overlay 1 mm scalebar
F.hideScalebar();                       % Hide colorbar

F.showTimestamp(t);                     % Overlay timestamp
F.updateTimestamp(t);                   % Update timestamp text
F.hideTimestamp();                      % Hide timestamp

F.showROIs(ROI);                        % Overlay ROIs
F.hideROIs();                           % Hide ROIs

F.showOverlay(data, data_range, alpha); % Overlay data on figure
F.updateOverlay(data, alpha);           % Update overlay data
F.hideOverlay();                        % Hide overlay

F.saveBMP(filename);                    % Save figure to BMP file
F.savePNG(filename);                    % Save figure to PNG file
F.getFrame();                           % Return figure as RGB image array
```


### Basic Visualization

* [`speckle2Image`](speckle2Image.m): Save speckle contrast data as an image
* [`speckle2Video`](speckle2Video.m): Render speckle contrast data to video
* [`roiOverlay`](roiOverlay.m): Overlay ROIs onto speckle contrast image
* [`strokeTargetOverlay`](strokeTargetOverlay.m): Overlay stroke target masks onto speckle contrast image

### Relative Flow Change Analysis

* [`rICT_frame`](rICT_frame.m): Generate rICT overlay image from speckle contrast data
* [`rICT_frame_sequence`](rICT_frame_sequence.m): Generate rICT overlay images at specific timepoints from speckle contrast data
* [`rICT_video`](rICT_video.m): Generate rICT overlay video from speckle contrast data
* [`rICT_plot`](rICT_plot.m): Plot SC, CT, and rICT timecourses from speckle contrast data

### Image Registration

* [`speckle_align_elastix`](speckle_align_elastix.m): Use Elastix to register speckle contrast data
* [`alignTwoSpeckle`](alignTwoSpeckle.m): Use Elastix to align two speckle contrast frames
* [`aligned2Video`](aligned2Video.m): Render aligned data to video
* [`aligned_rICT_video`](aligned_rICT_video.m): Generate rICT overlay video from aligned data
* [`aligned_rICT_plot`](aligned_rICT_plot.m): Plot SC, CT, and rICT timecourses from aligned data
* [`aligned_vessel_plot`](aligned_vessel_plot.m): Estimate vessel diameter from aligned data

### Utility

* [`read_subimage`](read_subimage.m): Read raw and speckle contrast data files from Speckle Software
* [`write_sc`](write_sc.m): Write array to .sc file
* [`zeropad_SC_filename`](zeropad_SC_filename.m): Zeropads .sc filenames to avoid ordering issues in Windows
* [`loadSpeckleTiming`](loadSpeckleTiming.m): Loads .timing file as a vector
* [`saveSpeckleTiming`](saveSpeckleTiming.m): Saves vector of times to .timing file
* [`get_tc_band`](get_tc_band.m): Calculates correlation time from speckle contrast using [Bandyopadhyay, _et al_](https://doi.org/10.1063/1.2037987)
* [`setSpeckleColorScheme`](setSpeckleColorScheme.m): Creates new figure using Speckle Software color scheme
* [`scaleBar`](scaleBar.m): Overlays 1 mm scalebar on bottom left corner of current figure
* [`timeStamp`](timeStamp.m): Overlays timestamp on bottom right corner of current figure
* [`timingGenerate`](timingGenerate.m): Generate artificial .timing file for directory of .sc files
* [`totalFrameCount`](totalFrameCount.m): Quickly count the total number of frames in a .sc file directory listing
* [`image2Video`](image2Video.m): Render sequence of images to video


