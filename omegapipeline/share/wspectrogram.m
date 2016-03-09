function wspectrogram(transforms, tiling, outputDirectory,uniqueID,startTime, referenceTime, ...
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
%                         identify q plane to display                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find plane with Q nearest the requested value
[~, planeIndices] = min(abs(log(tiling.qs / qRange)));


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

% index of plane in tiling structure
planeIndex = planeIndices(1);

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
for iN = 1:length(timeRange);
normalizedEnergies{iN} = zeros(numberOfRows, horizontalResolution);
end

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
  for iN = 1:length(timeRange);
    % vector of times to display
    timeRange1{iN} = timeRange(iN) * [-1 +1]*0.5;
    times{iN} = linspace(min(timeRange1{iN}), max(timeRange1{iN}), horizontalResolution);
    padTime = 1.5 * tiling.planes{planeIndex}.rows{rowIndex}.timeStep;
    tileIndices = find((rowTimes >= min(timeRange1{iN}) - padTime) & ...
                     (rowTimes <= max(timeRange1{iN}) + padTime));

    % vector of times to display
    rowTimestemp = rowTimes(tileIndices);

    % corresponding tile normalized energies
    rowNormalizedEnergies = transforms{channelNumber}.planes{planeIndex} ...
                            .rows{rowIndex}.normalizedEnergies(tileIndices);

    % interpolate to desired horizontal resolution
    rowNormalizedEnergies = interp1(rowTimestemp, rowNormalizedEnergies, times{iN}, ...
                                  'pchip');

    % insert into display matrix
    normalizedEnergies{iN}(row, :) = rowNormalizedEnergies;
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      end loop over frequency rows                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end loop over rows
end
    
for iN = 1:length(timeRange);

        surf(times{iN}, frequencies, normalizedEnergies{iN});
        colormapScale = normalizedEnergyRange(:).';

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
        axis([min(timeRange1{iN}) max(timeRange1{iN}) ...
              min(frequencyRange) max(frequencyRange)]);
          
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
        xlabel('Time [seconds]');
        % set figure background color
        set(gca, 'Color', [1 1 1]);
        set(gcf, 'Color', [1 1 1]);
        set(gcf, 'InvertHardCopy', 'off');

        % display colorbar
        subplot('position', colorbarPosition);
        colorbarmap = linspace(min(colormapScale), max(colormapScale), 100);
        imagesc(colorbarmap, 1, colorbarmap, colormapScale);
        set(gca, 'YTick',[])
        set(gca, 'TickDir', 'out')
        xlabel('Normalized tile energy');
        set(findall(gcf,'-property','FontSize'),'FontSize',10)
        N = getframe(gcf);
        clf
        figName = sprintf('%s_%.2f.png', ...
                  uniqueID, abs(diff(timeRange1{iN})));
        figBasePath = [outputDirectory '/' figName];
        savepng(N.cdata,figBasePath);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            plot spectrogram                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iN = 1:length(timeRange);
    frequencies1 = linspace(10, 2048, 4096);
    [X,Y] = meshgrid(times{iN},frequencies1);
    D=repmat(times{iN},[length(frequencies),1]);
    B= repmat(frequencies',[1,length(times{iN})]);
    normalizedEnergies1 = interp2(D,B,normalizedEnergies{iN},X,Y);
    normalizedEnergies1(isnan(normalizedEnergies1)) = 0;
    times1 = times{iN};

    save([outputDirectory '/' uniqueID '_spectrogram_' num2str(abs(diff(timeRange1{iN}))) '.mat'],'-v6',...
        'times1','frequencies1','normalizedEnergies1')
    clear D B X Y normalizedEnergies1 frequencies1 times1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          return to calling function                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
