classdef SpeckleFigure < handle
%SpeckleFigure Class to facilitate the visualization of speckle contrast data

  properties (Access = private)
    FigureHandle
    AxesHandle
    ImageHandle
    ColorbarHandle
    TimestampHandle
    ScalebarHandle
    ROIHandle
    Overlay
    
    data_width
    data_height
    
    YLimMax
    XLimMax
  end
  
  properties (Constant, Access = private)
    SCALE = 2.5; % Figure size scaling factor (2.5 is theoretical value)
    RESOLUTION = 426.62; % pixels/mm (Multimodal System w/ Basler acA1920-155um)  (426.62 px/mm is theoretical value)
  end
  
  methods
    
    function obj = SpeckleFigure(sc, sc_range, varargin)
      %SpeckleFigure Construct an instance of the SpeckleFigure class
      % SpeckleFigure(sc, sc_range) Create SpeckleFigure object and display
      % an array of speckle contrast data using imshow() with the specified
      % display range.
      %
      %   sc = MxN array of speckle contrast data
      %   sc_range = Speckle contrast display range [min max]
      %
      % SpeckleFigure(___,Name,Value) modifies construction using one or 
      % more name-value pair arguments.
      %
      %   'visible': Toggle visibility of the figure (Default = true)
      %
      
      p = inputParser;
      addRequired(p, 'sc');
      addRequired(p, 'sc_range', @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'}));
      addParameter(p, 'visible', true, @islogical);
      parse(p, sc, sc_range, varargin{:});
      
      sc = p.Results.sc;
      sc_range = p.Results.sc_range;
      visible = p.Results.visible;

      [obj.data_height, obj.data_width, ~] = size(sc);

      % Create the figure, axes, and image objects. In order to avoid 
      % window size issues with Windows, the figure is downsampled by a 
      % factor of two for display purposes. Saving the figure to file using 
      % saveBMP() or savePNG() or rendering the RGB image with getFrame()
      % will upscale to maintain the correct dimensions.
      obj.FigureHandle = figure('Name', 'SpeckleFigure', 'MenuBar', 'none', ...
        'ToolBar', 'none', 'Visible', 'off', 'HandleVisibility', 'off');
      obj.AxesHandle = axes('Parent', obj.FigureHandle, 'Position', [0, 0, 1, 1]);
      obj.ImageHandle = imshow(sc, sc_range, 'Parent', obj.AxesHandle);
      obj.setDefaultPosition();
      
      XLim = get(obj.AxesHandle, 'XLim');
      YLim = get(obj.AxesHandle, 'YLim');
      obj.XLimMax = XLim(2);
      obj.YLimMax = YLim(2);
      
      if visible
        obj.show();
      end
    end
        
    function delete(obj)
      %delete Deleting the SpeckleFigure destroys the figure handle
      % If figure is currently visible, then do not automatically close it
      if ishandle(obj.FigureHandle) && strcmp(obj.FigureHandle.Visible, 'off')
          close(obj.FigureHandle);
      end
    end
    
    function update(obj, sc)
      %update Update the speckle contrast data displayed in the figure
      % update(sc) Replaces the data shown in the figure with the passed
      % array of speckle contast data.
      %
      if obj.isValidSize(sc, 'sc')
        obj.ImageHandle.CData = sc;
      end
    end
    
    function setDisplayRange(obj, sc_range)
      %setDisplayRange Changes the speckle contrast display range
      % setDisplayRange(sc_range) changes the speckle contrast display range.
      %
      %   sc_range = Speckle contrast display range [min max]
      %
      set(obj.AxesHandle, 'Clim', sc_range);
    end
    
    function showColorbar(obj, varargin)
      %showColorbar Add colorbar on the right edge of the figure
      % showColorbar() Add colorbar on the right edge of the figure labeled
      % as 'Speckle Contrast'. 
      %
      % showColorbar(label) use custom label
      %
      
      % If overlay is currently enabled, just resize the figure
      if ~isempty(obj.Overlay) && isvalid(obj.Overlay.AxesHandle)
        if ~obj.Overlay.ColorbarStatus
          obj.setExtendedPosition();
          obj.Overlay.ColorbarStatus = true;
        end
        return;
      end

      if length(varargin) >= 1 && ~isempty(varargin{1})
        label = varargin{1};
      else
        label = 'Speckle Contrast';
      end
            
      % Clear existing colorbar
      if ~isempty(obj.ColorbarHandle) && isvalid(obj.ColorbarHandle)
        obj.hideColorbar();
      end
      
      obj.setExtendedPosition();
      P = obj.FigureHandle.Position;
      fontsize = 12 * 72 / get(0, 'screenpixelsperinch');
      obj.ColorbarHandle = colorbar(obj.AxesHandle, 'Units', 'Pixels', ...
        'Position', [P(3) - 85, 0.125 * P(4), 30, 0.75 * P(4)], ...
        'Fontsize', fontsize);
      ylabel(obj.ColorbarHandle, label, 'FontUnits', 'pixels', 'Fontsize', 14);
    end
    
    function hideColorbar(obj)
      %hideColorbar Remove colorbar from the figure
      delete(obj.ColorbarHandle);
      obj.setDefaultPosition();     
      
      if ~isempty(obj.Overlay) && isvalid(obj.Overlay.AxesHandle)
        obj.Overlay.ColorbarStatus = false;
      end
    end
    
    function showScalebar(obj)
      %showScalebar Add a scalebar to the bottom left corner of the figure
      % showScalebar() overlays 1 mm scale bar onto the bottom left corner 
      % of the figure assuming the 426.62 pixels/mm resolution of the 
      % multimodal system.
      %
      % Note: showScalebar() should be called AFTER addOverlay() to avoid
      % layer ordering issues that could result in overlay content appearing
      % on top of the scalebar.
      %
      
      AX = obj.FigureHandle.CurrentAxes; % Force onto the top-most axes
      
      % Draw scalebar outlined in black
      x = 75;
      y = obj.YLimMax - 100;
      obj.ScalebarHandle(1) = rectangle(AX, 'Position', [x - 1, y + 1, obj.RESOLUTION, 20], ...
        'FaceColor', 'k', 'LineStyle', 'none');
      obj.ScalebarHandle(2) = rectangle(AX, 'Position', [x - 1, y - 1, obj.RESOLUTION, 20], ...
        'FaceColor', 'k', 'LineStyle', 'none');
      obj.ScalebarHandle(3) = rectangle(AX, 'Position', [x + 1, y + 1, obj.RESOLUTION, 20], ...
        'FaceColor', 'k', 'LineStyle', 'none');
      obj.ScalebarHandle(4) = rectangle(AX, 'Position', [x + 1, y - 1, obj.RESOLUTION, 20], ...
        'FaceColor', 'k', 'LineStyle', 'none');
      obj.ScalebarHandle(5) = rectangle(AX, 'Position', [x, y, obj.RESOLUTION, 20], ...
        'FaceColor', 'w', 'LineStyle', 'none');
      
      % Draw white text outlined in black
%       x = 75 + obj.RESOLUTION/2;
%       y = obj.YLimMax - 55;
%       h = obj.renderOutlinedText(AX, x, y, '1 mm', 18);
%       obj.ScalebarHandle = [obj.ScalebarHandle h];
    end

    function hideScalebar(obj)
      %hideScalebar Remove scalebar from the figure
      delete(obj.ScalebarHandle);
    end
    
    function showTimestamp(obj, t)
      %showTimestamp Add a timestamp to the bottom right corner of the figure
      % showTimeStamp(t) overlays timestamp on bottom right corner of the
      % figure. Displayed in seconds with two decimal places.
      % 
      %   t = Timestamp in seconds
      %
      % Note: showTimestamp() should be called AFTER addOverlay() to avoid
      % layer ordering issues that could result in overlay content appearing
      % on top of the timestamp.
      %

      AX = obj.FigureHandle.CurrentAxes; % Force onto the top-most axes
      
      % Draw timestamp background
      p = [obj.XLimMax - 300, obj.YLimMax - 75, 300, 75];
      obj.TimestampHandle.BG = rectangle(AX, 'Position', p, 'FaceColor', 'b', 'LineStyle', 'none');
      
      % Draw timestamp
      x = obj.XLimMax - 150;
      y = obj.YLimMax - 25;
      txt = sprintf('%+06.2fs', t);
      obj.TimestampHandle.TEXT = text(AX, x, y, txt, 'FontUnits', 'pixels', ...
        'FontSize', 22, 'FontName', 'FixedWidth', 'Color', ...
        [1 - eps, 1, 1], 'HorizontalAlignment', 'center');
    end
    
    function updateTimestamp(obj, t)
      %updateTimestamp Update the timestamp displayed on the figure
      % updateTimestamp(t) Replaces timestamp shown in the figure with the
      % passed value.
      %
      %   t = Timestamp in seconds
      %
      obj.TimestampHandle.TEXT.String = sprintf('%+06.2fs', t);
    end
    
    function hideTimestamp(obj)
      %hideTimestamp Remove timestamp from the figure
      delete(obj.TimestampHandle.BG);
      delete(obj.TimestampHandle.TEXT);
    end
    
    function showROIs(obj, ROI, varargin)
      %showROIs Overlay ROIs onto the figure
      % showROIs(ROI) Overlays ROIs onto the figure using the Speckle
      % Software colormap. The ROIs can either be a directory of binary 
      % .bmp files, a .mat file containing an array of binary masks, or an
      % array of logical masks.
      %
      %   ROI = a) Path to directory of masks output by Speckle Software
      %         b) Path to .mat file containing an `ROI` variable
      %         c) Array of logical masks
      %
      % showROIs(___,Name,Value) modifies the overlay appearance using one 
      % or more name-value pair arguments.
      %
      %   'alpha': Define alpha level of ROI overlays (Default = 0.5)
      %
      %   'show_labels': Toggle display of the ROI labels (Default = true)
      %
      %   'stroke_mode': Enable stroke target overlay mode (Default = false)
      %       Enables stroke target overlay mode where all ROIs are colored
      %       green and labels are disabled.
      %
      
      p = inputParser;
      addRequired(p, 'ROI');
      addParameter(p, 'alpha', 0.5, @isscalar);
      addParameter(p, 'show_labels', true, @islogical);
      addParameter(p, 'stroke_mode', false, @islogical);
      parse(p, ROI, varargin{:});

      ROI = p.Results.ROI;
      alpha = p.Results.alpha;
      show_labels = p.Results.show_labels;
      stroke_mode = p.Results.stroke_mode;
      
      if ~islogical(ROI)
        if ~exist(ROI, 'dir') && ~isfile(ROI)
          warning("Overlay not applied because '%s' does not exist.", ROI);
          return;
        elseif exist(ROI, 'dir') % Load ROIs from directory of BMP masks
          files = dir(fullfile(ROI, '*_ccd.bmp'));
          if length(files) >= 1
            ROI = [];
            for i = 1:length(files)
              ROI = cat(3, ROI, imread(fullfile(files(i).folder, files(i).name)));
            end
            ROI = logical(ROI);
          else
            warning("Overlay not applied because '%s' does not contain any ROI masks.", ROI);
            return;
          end
        elseif isfile(ROI) % Load ROIs from .mat file with variable `ROI`
          data = load(ROI);
          if isfield(data, 'ROI')
            ROI = data.ROI;
          else
            warning("Overlay not applied because %s does not contain a variable named `ROI`.", ROI);
            return;
          end
        elseif ~islogical(ROI)
          warning("Overlay not applied because `ROI` is of type '%s' instead of 'logical.'", class(ROI));
          return;
        end
      end
      
      if ~obj.isValidSize(ROI, 'ROIs')
        return;
      end
      
      % Clear existing ROIs
      if ~isempty(obj.ROIHandle)
        obj.hideROIs();
      end
      
      % Generate the ROIs
      obj.ROIHandle = [];
      for i = 1:size(ROI, 3)
        
        % Create the color overlay
        if stroke_mode
          c_img = obj.getSpeckleColor(2, obj.data_height, obj.data_width);
        else
          c_img = obj.getSpeckleColor(i, obj.data_height, obj.data_width);
        end 
        this_axes = axes('Parent', obj.FigureHandle, 'Position', [0, 0, 1, 1]);
        this_image = imshow(c_img, 'Parent', this_axes);
        set(this_image, 'AlphaData', ROI(:,:,i) * alpha);
        obj.ROIHandle = [obj.ROIHandle this_axes this_image];
        
        % If enabled, overlay label centered at the mask centroid
        if show_labels && ~stroke_mode
          cen = regionprops(true(size(ROI(:,:,i))), ROI(:,:,i), 'WeightedCentroid');
          cen = cen.WeightedCentroid;
          h = obj.renderOutlinedText(this_axes, cen(1), cen(2), sprintf('R%d', i), 14);
          obj.ROIHandle = [obj.ROIHandle h];
        end
      end
    end
    
    function hideROIs(obj)
      %hideROIs Remove ROIs from the figure
      delete(obj.ROIHandle);
    end
    
    function showOverlay(obj, data, data_range, alpha, varargin)
      %showOverlay Add an overlay layer onto the figure
      % showOverlay(data, data_range, alpha) Overlay `data` onto the figure
      % using imshow() with the specified display range and pixel-wise 
      % `alpha` transparency. Both `data` and `alpha` must be the same 
      % dimensions as the base figure data. Adds colorbar on the right edge 
      % of the figure labeled as 'Relative ICT'.
      %
      %   data = MxN array of data to be overlaid on the figure
      %   data_range = Data display range [min max]
      %   alpha = MxN array describing pixelwise alpha transparency
      %       
      % showOverlay(___,Name,Value) modifies the overlay using one or more 
      % name-value pair arguments.
      %
      %   'label': Change colorbar label text (Default = 'Relative ICT')
      %
      %   'use_divergent_cmap': Toggle use of divergent colormap (Default = false)
      %       Use divergent red/blue colormap instead of default sequential
      %       pmkmp CubicYF colormap to visualize rICT.
      %

      p = inputParser;
      addRequired(p, 'data');
      addRequired(p, 'data_range');
      addRequired(p, 'alpha');
      addParameter(p, 'label', 'Relative ICT', @ischar);
      addParameter(p, 'use_divergent_cmap', false, @islogical);
      parse(p, data, data_range, alpha, varargin{:});

      data = p.Results.data;
      data_range = p.Results.data_range;
      alpha = p.Results.alpha;
      label = p.Results.label;
      use_divergent_cmap = p.Results.use_divergent_cmap;
      
      if ~obj.isValidSize(data, 'data') || ~obj.isValidSize(alpha, 'alpha')
        return;
      end

      % Clear existing colorbar
      if (~isempty(obj.ColorbarHandle) && isvalid(obj.ColorbarHandle)) || ...
            (~isempty(obj.Overlay) && ~isempty(obj.Overlay.ColorbarHandle) && ...
            isvalid(obj.Overlay.ColorbarHandle))
        obj.hideColorbar();
      end
      
      % Clear existing overlays
      if ~isempty(obj.Overlay) && isvalid(obj.Overlay.AxesHandle)
        obj.hideOverlay();
      end
      
      if use_divergent_cmap
        cmap = colormap(diverging_map(linspace(0,1,256), [0.230, 0.299, 0.754], [0.706, 0.016, 0.150]));
        YTick = data_range(1):0.25:data_range(2);
      else
        cmap = flipud(pmkmp(256, 'CubicYF'));
%         cmap = pmkmp(256,'CubicYF');
        YTick = data_range(1):0.1:data_range(2);
      end
      
      % Create the overlay
      obj.setExtendedPosition();
      obj.Overlay.AxesHandle = axes('Parent', obj.FigureHandle, ...
        'Position', obj.AxesHandle.Position);
      obj.Overlay.ImageHandle = imshow(data, data_range, 'Colormap', ...
        cmap, 'Parent', obj.Overlay.AxesHandle);
      obj.Overlay.ImageHandle.AlphaData = alpha;
      
      % Create the colorbar
      P = obj.FigureHandle.Position;
      fontsize = 13 * 72 / get(0, 'screenpixelsperinch');
      obj.Overlay.ColorbarHandle = colorbar(obj.Overlay.AxesHandle, 'Units', ...
        'Pixels', 'Fontsize', fontsize, 'YTick', YTick, 'Position', ...
        [P(3) - 85, 0.125 * P(4), 30, 0.75 * P(4)],'FontWeight','bold');
      ylabel(obj.Overlay.ColorbarHandle, label, 'FontUnits', 'pixels', 'Fontsize', 18, 'FontWeight','bold');
      obj.Overlay.ColorbarStatus = true;
    end
    
    function updateOverlay(obj, data, alpha)
      %updateOverlay Update the overlay displayed in the figure
      % updateOverlay(data, alpha) Replaces the overlay shown in the figure 
      % with the passed array of data and alpha tranparency.
      %
      if obj.isValidSize(data, 'data') && obj.isValidSize(alpha, 'alpha')
        obj.Overlay.ImageHandle.CData = data;
        obj.Overlay.ImageHandle.AlphaData = alpha;
      end
    end
    
    function hideOverlay(obj)
      %hideOverlay Remove overlay from the figure
      delete(obj.Overlay.AxesHandle);
      delete(obj.Overlay.ImageHandle);
      delete(obj.Overlay.ColorbarHandle);
      obj.Overlay.ColorbarStatus = false;
      obj.setDefaultPosition();
    end
    
    function saveBMP(obj, filename, varargin)
      %saveBMP Print to BMP file
      % saveBMP(filename) Writes the current figure to a BMP file at the
      % current system resolution.
      %
      % saveBMP(filename, ppi) Specify a custom resolution.
      %
      %   filename = Output file
      %   ppi = Resolution to render image at in pixels-per-inch
      %
      
      if length(varargin) >= 1 && ~isempty(varargin{1})
        ppi = varargin{1};
      else
        ppi = get(0, 'screenpixelsperinch');
      end
      
      ppi = sprintf('-r%d', ppi * obj.SCALE);
      print(obj.FigureHandle, '-dbmp', ppi, filename, '-opengl');
    end
    
    function savePNG(obj, filename, varargin)
      %savePNG Print to PNG file
      % savePNG(filename) Writes the current figure to a PNG file at the
      % current system resolution.
      %
      % savePNG(filename, ppi) Specify a custom resolution.
      %
      %   filename = Output file
      %   ppi = Resolution to render image at in pixels-per-inch
      %
      
      if length(varargin) >= 1 && ~isempty(varargin{1})
        ppi = varargin{1};
      else
        ppi = get(0, 'screenpixelsperinch');
      end
      
      ppi = sprintf('-r%d', ppi * obj.SCALE);
      print(obj.FigureHandle, '-dpng', ppi, filename, '-opengl');
    end
    
    function frame = getFrame(obj)
      %getFrame Return current figure contents as an RGB image array
      RES = get(0, 'screenpixelsperinch');
      PRINT_RES = sprintf('-r%d', RES * obj.SCALE);
      frame = print(obj.FigureHandle, '-RGBImage', PRINT_RES);
    end
    
    function show(obj)
      %show Show the figure
      obj.FigureHandle.Visible = 'on';
      obj.FigureHandle.HandleVisibility = 'on';
    end
    
    function hide(obj)
      %hide Hide the figure
      obj.FigureHandle.Visible = 'off';
      obj.FigureHandle.HandleVisibility = 'off';
    end
    
  end
  
  methods (Access = private)
    
    function setDefaultPosition(obj)
      %setDefaultPosition Set default figure position and aspect ratio
      width = obj.data_width/obj.SCALE;
      height = obj.data_height/obj.SCALE;
      s = get(0, 'screensize');
      obj.FigureHandle.Position = [(s(3) - width)/2, (s(4) - height)/2, width, height];
      obj.AxesHandle.Position = [0, 0, 1, 1];
      
      if ~isempty(obj.Overlay) && isvalid(obj.Overlay.AxesHandle)
        obj.Overlay.AxesHandle.Position = [0, 0, 1, 1];
      end

    end
    
    function setExtendedPosition(obj)
      %setExtendedPosition Set figure position and aspect ratio for colorbar display
      width = obj.FigureHandle.Position(3);
      obj.FigureHandle.Position(3) = width + 100;
      obj.AxesHandle.Position(3) = width / obj.FigureHandle.Position(3);
      
      if ~isempty(obj.Overlay) && isvalid(obj.Overlay.AxesHandle)
        obj.Overlay.AxesHandle.Position = obj.AxesHandle.Position;
      end

    end
    
    function status = isValidSize(obj, data, var)
      %isValidSize Validate that data size matches initialized figure size
      status = true;
      [height, width, ~] = size(data);
      if width ~= obj.data_width || height ~= obj.data_height
        msg = 'Figure not updated because `%s` has wrong dimensions (%dx%d). Expected (%dx%d).';
        warning(msg, var, width, height, obj.data_width, obj.data_height);
        status = false;
      end
    end
    
  end
  
  methods (Static)
    
    function h_return = renderOutlinedText(h,x,y,txt,fontsize)
      %renderOutlinedText Render white text outlined in black
      % h_return = renderOutlinedText(h,x,y,txt,fontsize) add white text 
      % outlined in black on the given axes handle and return handle to
      % overlay objects
      %
      %   h = Axes handle
      %   x,y = Coordinates for text
      %   txt = Text to display
      %   size = Font size
      %
      
      h_return = zeros(1, 5);
      
      % Draw the black outline
      offset = [-1 1; -1 -1; 1 1; 1 -1];      
      for i = 1:4
        h_return(i) = text(h, x + offset(i, 1), y + offset(i, 2), txt, ...
          'FontUnits', 'pixels', 'FontSize', fontsize, 'FontWeight', ...
          'bold', 'Color', 'k', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
      end
      
      % Draw the white text
      h_return(5) = text(h, x, y, txt, 'FontUnits', 'pixels', 'FontSize', fontsize, ...
        'FontWeight', 'bold', 'Color', [1 - eps, 1, 1], ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
    
    function c_img = getSpeckleColor(n, height, width)
      %getSpeckleColor Returns array of the nth Speckle Software color
      % c_img = getSpeckleColor(n, height, width) returns an array of the
      % nth Speckle Software color in an MxNx3 RGB format.
      %
      
      % Speckle Software Colors
      c = [
        1 0 0;           % Red
        0 1 0;           % Green
        0 0 1;           % Blue
        0 1 1;           % Cyan
        1 0 1;           % Magenta
        1 1 0;           % Yellow
        0.5 0 0;         % Dark Red
        0 0.5 0;         % Dark Green
        0 0 0.5;         % Dark Blue
        0 0.5 0.5;       % Dark Cyan
        0.5 0 0.5;       % Dark Magenta
        0.5 0.5 0;       % Dark Yellow
        0 0 0;           % Black
        0.62 0.62 0.62   % Gray
      ];
      
      idx = mod(n - 1,length(c)) + 1;
      c_img = zeros(height, width, 3);
      c_img(:,:,1) = c(idx, 1);
      c_img(:,:,2) = c(idx, 2);
      c_img(:,:,3) = c(idx, 3); 
    end
    
  end
end