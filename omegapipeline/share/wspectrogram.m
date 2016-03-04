function handles = wspectrogram(transforms, tiling, outputDirectory,uniqueID,startTime, referenceTime, ...
                                timeRange, frequencyRange, qRange, ...
                                normalizedEnergyRange, horizontalResolution)
% WSPECTROGRAM Display time-frequency Q transform spectrograms
%
% WSPECTROGRAM displays multi-resolution time-frequency spectrograms of
% normalized tile energy produced by WTRANSFORM.  A separate figure is produced
% for each channel and for each time-frequency plane within the requested range
% of Q values.  The resulting spectrograms are logarithmic in frequency and
% linear in time, with the tile color denoting normalized energy.
%
% usage:
%
%   handles = wspectrogram(transforms, tiling, startTime, referenceTime, ...
%                          timeRange, frequencyRange, qRange, ...
%                          normalizedEnergyRange, horizontalResolution);
%
%     transforms              cell array of q transform structures
%     tiling                  q transform tiling structure
%     startTime               start time of transformed data
%     referenceTime           reference time of plot
%     timeRange               vector range of relative times to plot
%     frequencyRange          vector range of frequencies to plot
%     qRange                  scalar Q or vector range of Qs to plot
%     normalizedEnergyRange   vector range of normalized energies for colormap
%     horizontalResolution    number of data points across image
%
%     handles                 matrix of axis handles for each spectrogram
%
% The user can focus on a subset of the times and frequencies available in the
% original transform data by specifying a desired time and frequency range.
% Ranges should be specified as a two component vector, consisting of the
% minimum and maximum value.  Additionally, WSPECTROGRAM can be restricted to
% plot only a subset of the available Q planes by specifying a single Q or a
% range of Qs.  If a single Q is specified, WSPECTROGRAM displays the
% time-frequency plane which has the nearest value of Q in a logarithmic sense
% to the requested value.  If a range of Qs is specified, WSPECTROGRAM displays
% all time-frequency planes with Q values within the requested range.  By
% default all available channels, times, frequencies, and Qs are plotted.  The
% default values can be obtained for any argument by passing the empty matrix
% [].
%
% To determine the range of times to plot, WSPECTROGRAM requires the start time
% of the transformed data, a reference time, and relative time range.  Both the
% start time and reference time should be specified as absolute quantities, but
% the range of times to plot should be specified relative to the requested
% reference time.  The specified reference time is used as the time origin in
% the resulting spectrograms and is also reported in the title of each plot.
%
% If only one time-frequency plane is requested, it is plotted in the current
% figure window.  If more than one spectrogram is requested, they are plotted
% in separate figure windows starting with figure 1.
%
% The optional normalizedEnergyRange specifies the range of values to encode
% using the colormap.  By default, the lower bound is zero and the upper bound
% is autoscaled to the maximum normalized energy encountered in the specified
% range of time, frequency, and Q.
%
% The optional cell array of channel names are used to label the resulting
% figures.
%
% The optional horizontal resolution argument specifies the number data points
% in each frequency row of the displayed image.  The vector of normalized
% energies in each frequency row is then interpolated to this resolution in
% order to produce a rectangular array of data for displaying.  The vertical
% resolution is directly determined by the number of frequency rows available in
% the transform data.  By default, a horizontal resolution of 2048 data points
% is assumed, but a higher value may be necessary if the zoom feature will be
% used to magnify the image.  For aesthetic purposes, the resulting image is
% also displayed with interpolated color shading enabled.
%
% WSPECTROGRAM returns a matrix of axis handles for each spectrogram with
% each channel in a separate row and each Q plane in a separate column.
%
% See also WTILE, WCONDITION, WTRANSFORM, WEVENTGRAM, and WEXAMPLE.

% Shourov K. Chatterji <shourov@ligo.mit.edu>

% $Id: wspectrogram.m 2753 2010-02-26 21:33:24Z jrollins $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            hard coded parameters                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number of horizontal pixels in image
defaultHorizontalResolution = 2048;

% spectrogram boundary
spectrogramLeft = 0.14;
spectrogramWidth = 0.80;
spectrogramBottom = 0.28;
spectrogramHeight = 0.62;
spectrogramPosition = [spectrogramLeft spectrogramBottom ...
                       spectrogramWidth spectrogramHeight];

% colorbar position
colorbarLeft = spectrogramLeft;
colorbarWidth = spectrogramWidth;
colorbarBottom = 0.12;
colorbarHeight = 0.02;
colorbarPosition = [colorbarLeft colorbarBottom ...
                    colorbarWidth colorbarHeight];

% time scales for labelling
millisecondThreshold = 0.5;
secondThreshold = 3 * 60;
minuteThreshold = 3 * 60 * 60;
hourThreshold = 3 * 24 * 60 * 60;
dayThreshold = 365.25 * 24 * 60 * 60;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify number of input arguments
narginchk(2, 11);

% apply default arguments
if (nargin < 5) || isempty(startTime),
  startTime = 0;
end
if (nargin < 6) || isempty(referenceTime),
  referenceTime = 0;
end
if (nargin < 7) || isempty(timeRange),
  timeRange = [-Inf Inf];
end
if (nargin < 8) || isempty(frequencyRange),
  frequencyRange = [-Inf Inf];
end
if (nargin < 9) || isempty(qRange),
  qRange = [-Inf Inf];
end
if (nargin < 10) || isempty(normalizedEnergyRange),
  normalizedEnergyRange = [];
end
if (nargin < 11) || isempty(horizontalResolution),
  horizontalResolution = defaultHorizontalResolution;
end

% force cell arrays
transforms = wmat2cell(transforms);

% force one dimensional cell arrays
transforms = transforms(:);

% determine number of channels
numberOfChannels = length(transforms);

% make channelNames array
for channelNumber = 1 : numberOfChannels,
  if isfield(transforms{channelNumber},'channelName'),
    channelNames{channelNumber} = transforms{channelNumber}.channelName;
  else
    channelNames{channelNumber} = ['Channel ' int2str(channelNumber)];
  end
end

% force ranges to be monotonically increasing column vectors
timeRange = unique(timeRange(:));
frequencyRange = unique(frequencyRange(:));
qRange = unique(qRange(:));
if ~isempty(normalizedEnergyRange),
  normalizedEnergyRange = unique(normalizedEnergyRange(:));
end

% store requested frequency range
requestedFrequencyRange = frequencyRange;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% validate channel names
if ~isempty(channelNames) && (length(channelNames) ~= numberOfChannels),
  error('channel names is inconsistent with number of transform channels');
end

% check for valid discrete Q-transform tile structure
if ~strcmp(tiling.id, 'Discrete Q-transform tile structure'),
  error('The first argument is not a valid Q-transform tiling.');
end

% check for valid discrete Q-transform transform structure
for channelNumber = 1 : numberOfChannels,
  if ~strcmp(transforms{channelNumber}.id, ...
             'Discrete Q-transform transform structure'),
    error('The second argument is not a valid Q-transform result.');
  end
end

% Check for two component range vectors
if length(timeRange) ~= 2,
  error('Time range must be two component vector [tmin tmax].');
end
if length(frequencyRange) ~= 2,
  error('Frequency range must be two component vector [fmin fmax].');
end
if length(qRange) > 2,
  error('Q range must be scalar or two component vector [Qmin Qmax].');
end
if ~isempty(normalizedEnergyRange) && length(normalizedEnergyRange) ~= 2,
  error('Normalized energy range must be two component vector [Zmin Zmax].');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         identify q planes to display                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if non-zero Q range is requested,
if length(qRange) == 2,

  % find planes within requested Q range
  planeIndices = find((tiling.qs >= min(qRange)) & ...
                      (tiling.qs <= max(qRange)));

% otherwise, if only a single Q is requested,
else

  % find plane with Q nearest the requested value
  [~, planeIndices] = min(abs(log(tiling.qs / qRange)));

% continue
end

% number of planes to display
numberOfPlanes = length(planeIndices);

% initialize handle vector
handles = zeros(numberOfChannels, numberOfPlanes);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          identify times to display                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% default start time for display is start time of available data
if timeRange(1) == -Inf,
  timeRange(1) = startTime - referenceTime;
end

% default stop time for display is stop time of available data
if timeRange(2) == +Inf,
  timeRange(2) = startTime - referenceTime + tiling.duration;
end

% validate requested time range
if (timeRange(1) < startTime - referenceTime) || ...
   (timeRange(2) > startTime - referenceTime + tiling.duration),
  error('requested time range exceeds available data');
end

% vector of times to display
times = linspace(min(timeRange), max(timeRange), horizontalResolution);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           begin loop over channels                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% loop over channels
for channelNumber = 1 : numberOfChannels,

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                          begin loop over q planes                          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % loop over planes to display
  for plane = 1 : numberOfPlanes,

    % index of plane in tiling structure
    planeIndex = planeIndices(plane);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                   identify frequency rows to display                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % default minimum frequency is minimum frequency of available data
    if requestedFrequencyRange(1) == -Inf,
      frequencyRange(1) = tiling.planes{planeIndex}.minimumFrequency;
    else
      frequencyRange(1) = requestedFrequencyRange(1);
    end

    % default maximum frequency is maximum frequency of available data
    if requestedFrequencyRange(2) == +Inf,
      frequencyRange(2) = tiling.planes{planeIndex}.maximumFrequency;
    else
      frequencyRange(2) = requestedFrequencyRange(2);
    end

    % validate selected frequency range
    if (frequencyRange(1) < tiling.planes{planeIndex}.minimumFrequency) || ...
       (frequencyRange(2) > tiling.planes{planeIndex}.maximumFrequency),
      error('requested frequency range exceeds available data');
    end

    % vector of frequencies in plane
    frequencies = tiling.planes{planeIndex}.frequencies;

    % find rows within requested frequency range
    rowIndices = find((frequencies >= min(frequencyRange)) & ...
                      (frequencies <= max(frequencyRange)));

    % pad by one row if possible
    if rowIndices(1) > 1,
      rowIndices = [rowIndices(1) - 1 rowIndices];
    end
    if rowIndices(end) < length(frequencies),
      rowIndices = [rowIndices rowIndices(end) + 1];
    end

    % vector of frequencies to display
    frequencies = frequencies(rowIndices);

    % number of rows to display
    numberOfRows = length(rowIndices);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                       initialize display matrix                          %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % initialize matrix of normalized energies for display
    normalizedEnergies = zeros(numberOfRows, horizontalResolution);

    % initialize maximum normalized energy
    maximumNormalizedEnergy = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                     begin loop over frequency rows                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % loop over rows
    for row = 1 : numberOfRows,

      % index of row in tiling structure
      rowIndex = rowIndices(row);

      % vector of times in plane
      rowTimes = ...
          (0 :  tiling.planes{planeIndex}.rows{rowIndex}.numberOfTiles - 1) ... 
	  * tiling.planes{planeIndex}.rows{rowIndex}.timeStep + ...
                 (startTime - referenceTime);

      % find tiles within requested time range
      padTime = 1.5 * tiling.planes{planeIndex}.rows{rowIndex}.timeStep;
      tileIndices = find((rowTimes >= min(timeRange) - padTime) & ...
                         (rowTimes <= max(timeRange) + padTime));

      % vector of times to display
      rowTimes = rowTimes(tileIndices);

      % corresponding tile normalized energies
      rowNormalizedEnergies = transforms{channelNumber}.planes{planeIndex} ...
                                .rows{rowIndex}.normalizedEnergies(tileIndices);

      % update maximum normalized energy
      maximumNormalizedEnergy = max(max(rowNormalizedEnergies), ...
                                    maximumNormalizedEnergy);
      
      % interpolate to desired horizontal resolution
      rowNormalizedEnergies = interp1(rowTimes, rowNormalizedEnergies, times, ...
                                      'pchip');

      % insert into display matrix
      normalizedEnergies(row, :) = rowNormalizedEnergies;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                      end loop over frequency rows                        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % end loop over rows
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                         set colormap scaling                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % if normalized energy range is not specified
    if isempty(normalizedEnergyRange),

      % normalized energy range to code on colormap
      colormapScale = [0 maximumNormalizedEnergy];

    % or if autoscaling of upper limit is requested,
    elseif normalizedEnergyRange(2) == Inf,

      % normalized energy range to code on colormap
      colormapScale = [normalizedEnergyRange(1) maximumNormalizedEnergy];

    % otherwise,
    else

      % use specified range
      colormapScale = normalizedEnergyRange(:).';

    % continue
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                            plot spectrogram                              %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % if plotting more than one figure
    if numberOfChannels * numberOfPlanes > 1,

      % select figure to plot in
      figure((channelNumber - 1) * numberOfPlanes + plane);

    % continue
    end
    frequencies1 = linspace(min(frequencyRange), max(frequencyRange), horizontalResolution*8);
    [X,Y] = meshgrid(times,frequencies1);
    D=repmat(times,[length(frequencies),1]);
    B= repmat(frequencies',[1,length(times)]);
    normalizedEnergies = interp2(D,B,normalizedEnergies,X,Y);
    clear D B X Y
    
    save([outputDirectory '/' uniqueID '_spectrogram_' num2str(abs(diff(timeRange))) '.mat'],...
        'times','frequencies','normalizedEnergies')
    
    % plot imagesc no frequency interp
    if abs(diff(timeRange)) < millisecondThreshold,
      surf(times * 1e3, frequencies, normalizedEnergies,normalizedEnergies);
    elseif abs(diff(timeRange)) < secondThreshold,
      %surf(times * 1, frequencies, normalizedEnergies,normalizedEnergies);
      surf(times * 1,frequencies1, normalizedEnergies);
    elseif abs(diff(timeRange)) < minuteThreshold,
      surf(times / 60, frequencies, normalizedEnergies,normalizedEnergies);
    elseif abs(diff(timeRange)) < hourThreshold,
      surf(times / 3600, frequencies, normalizedEnergies,normalizedEnergies);
    elseif abs(diff(timeRange)) < dayThreshold,
      surf(times / 86400, frequencies, normalizedEnergies,normalizedEnergies);
    else
      surf(times / 31557600, frequencies, normalizedEnergies,normalizedEnergies);
    end

    % apply colormap scaling
    C = load('parulynomials.mat');

    x = linspace(0,1,256)';

    r = polyval(C.R,x);
    g = polyval(C.G,x);
    b = polyval(C.B,x);

    % Clamp:
    r(r<0)=0;
    g(g<0)=0;
    b(b<0)=0;
    r(r>1)=1;
    g(g>1)=1;
    b(b>1)=1;

    map = [r g b];

    colormap(map);
    caxis(colormapScale);

    % set axis position
    set(gca, 'Position', spectrogramPosition);


    % set axis range
    if abs(diff(timeRange)) < millisecondThreshold,
      axis([min(timeRange) * 1e3 max(timeRange) * 1e3 ...
            min(frequencyRange) max(frequencyRange)]);
    elseif abs(diff(timeRange)) < secondThreshold,
      axis([min(timeRange) * 1 max(timeRange) * 1 ...
            min(frequencyRange) max(frequencyRange)]);
    elseif abs(diff(timeRange)) < minuteThreshold,
      axis([min(timeRange) / 60 max(timeRange) / 60 ...
            min(frequencyRange) max(frequencyRange)]);
    elseif abs(diff(timeRange)) < hourThreshold,
      axis([min(timeRange) / 3600 max(timeRange) / 3600 ...
            min(frequencyRange) max(frequencyRange)]);
    elseif abs(diff(timeRange)) < dayThreshold,
      axis([min(timeRange) / 86400 max(timeRange) / 86400 ...
            min(frequencyRange) max(frequencyRange)]);
    else
      axis([min(timeRange) / 31557600 max(timeRange) / 31557600 ...
            min(frequencyRange) max(frequencyRange)]);
    end

    % set view angle
    view([0 0 1]);

    % disable coordinate grid
    grid off;

    % enable interpolated shading
    shading interp;

    % set y axis properties
    ylabel('Frequency [Hz]');
    set(gca, 'YScale', 'log');
    set(gca, 'TickDir', 'out');
    set(gca, 'TickLength', [0.01 0.025]);
    if min(frequencyRange) >= 0.0625,
      set(gca, 'YMinorTick', 'off');
      set(gca, 'YTick', 2.^(ceil(log2(min(frequencyRange))) : 1 : ...
                            floor(log2(max(frequencyRange)))));
    end

    % % set vertical height based on frequency range
    % frequencyOctaves = log2(max(frequencies) / min(frequencies));
    % frequencyOctaves = max(frequencyOctaves, 6);
    % frequencyOctaves = min(frequencyOctaves, 10.5);
    % set(gcf, 'PaperPosition', [0.25 0.25 8.0 frequencyOctaves]);
    % % the axis position property should also be adjusted

    % set x axis properties
    if abs(diff(timeRange)) < millisecondThreshold,
      xlabel('Time [milliseconds]');
    elseif abs(diff(timeRange)) < secondThreshold,
      xlabel('Time [seconds]');
    elseif abs(diff(timeRange)) < minuteThreshold,
      xlabel('Time [minutes]');
    elseif abs(diff(timeRange)) < hourThreshold,
      xlabel('Time [hours]');
    elseif abs(diff(timeRange)) < dayThreshold,
      xlabel('Time [days]');
    else
      xlabel('Time [years]');
    end

    % set figure background color
    set(gca, 'Color', [1 1 1]);
    set(gcf, 'Color', [1 1 1]);
    set(gcf, 'InvertHardCopy', 'off');
    

    % append current axis handle to list of handles
    handles(channelNumber, plane) = gca;

    % display colorbar
    subplot('position', colorbarPosition);
    set(gca, 'FontSize', 16);
    colorbarmap = linspace(min(colormapScale), max(colormapScale), 100);
    imagesc(colorbarmap, 1, colorbarmap, colormapScale);
    set(gca, 'YTick',[])
    set(gca, 'TickDir', 'out')
    xlabel('Normalized tile energy');

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                           end loop over q planes                           %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % end loop over planes
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            end loop over channels                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end loop over channels
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          return to calling function                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
