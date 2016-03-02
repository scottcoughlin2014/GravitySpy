function handles = weventgram(events, outputDirectory,uniqueID,tiling, startTime, referenceTime, ...
                              timeRange, frequencyRange, ...
                              durationInflation, bandwidthInflation, ...
                              normalizedEnergyRange)
% WEVENTGRAM Display statistically significant time-frequency events
%
% WEVENTGRAM Displays filled boxes corresponding to the time-frequency boundary
% of statistically significant events.  WEVENTGRAM takes as input a cell array
% of event matrices, one per channel. A separate figure is produced for each
% channel.  The resulting spectrograms are logarithmic in frequency and linear
% in time, with the tile color denoting normalized energy of tiles or clusters.
%
% usage:
%
%   handles = weventgram(events, tiling, startTime, referenceTime, ...
%                        timeRange, frequencyRange, ...
%                        durationInflation, bandwidthInflation, ...
%                        normalizedEnergyRange);
%
%     events                  cell array of time-frequency event matrices
%     tiling                  q transform tiling structure
%     startTime               start time of transformed data
%     referenceTime           reference time of plot
%     timeRange               vector range of relative times to plot
%     frequencyRange          vector range of frequencies to plot
%     durationInflation       multiplicative factor for tile durations
%     bandwidthInflation      multiplicative factor for tile bandwidths
%     normalizedEnergyRange   vector range of normalized energies for colormap
%
%     handles                 vector of axis handles for each eventgram
%
% WEVENTGRAM expects a cell array of Q transform event structures with one cell
% per channel.  The event structures contain the following fields, which
% describe the properties of statistically significant tiles.
%
%   time                 center time of tile [gps seconds]
%   frequency            center frequency of tile [Hz]
%   q                    quality factor of tile []
%   normalizedEnergy     normalized energy of tile []
%   amplitude            signal amplitude of tile [Hz^-1/2]
%   phase                phase of tile [radians]
%
% If the following optional field is present, the color of tiles in the event
% gram correspond to total cluster normalized energy rather than single tile
% normalized energy.
%
%   clusterNormalizedEnergy
%
% The user can focus on a subset of the times and frequencies available in the
% original transform data by specifying a desired time and frequency range.
% Ranges should be specified as a two component vector, consisting of the
% minimum and maximum value.  By default, the full time and frequency range
% specified in the tiling is displayed.  The default values can be obtained for
% any argument by passing the empty matrix [].
%
% To determine the range of times to plot, WEVENTGRAM also requires a reference
% time in addition to the specified time range.  This reference time should be
% specified as an absolute quantity, while the range of times to plot should be
% specified relative to the requested reference time.  The specified reference
% time is used as the time origin in the resulting eventgrams and is also
% reported in the title of each plot.  A reference time of zero is assumed by
% default.
%
% If only one channel is requested, its eventgram is plotted in the current
% figure window.  If more than one eventgram is requested, they are plotted in
% separate figure windows starting with figure 1.
%
% The optional durationInflation and bandwidthInflation arguments are
% multiplicative scale factors that are applied to the duration and bandwidth of
% displayed events.  If not specified, these parameters both default to unity
% such that the resulting events have unity time-frequency area.
%
% The optional normalizedEnergyRange argument specifies the range of values to
% encode using the colormap.  By default, the lower bound is zero and the upper
% bound is autoscaled to the maximum normalized energy encountered in the
% specified range of time and frequency.
%
% The optional cell array of channel names are used to label the resulting
% figures.
%
% WEVENTGRAM returns a vector of axis handles to the eventgram produced for each
% channel.
%
% See also WTILE, WCONDITION, WTRANSFORM, WTHRESHOLD, WSELECT, WCLUSTER,
% WSPECTROGRAM, and WEXAMPLE.

% Shourov K. Chatterji <shourov@ligo.mit.edu>
% Jameson Rollins <jrollins@phys.columbia.edu>

% $Id: weventgram.m 3415 2012-06-02 17:39:47Z michal.was@LIGO.ORG $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            hard coded parameters                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% eventgram boundary
eventgramLeft = 0.14;
eventgramWidth = 0.80;
eventgramBottom = 0.28;
eventgramHeight = 0.62;
eventgramPosition = [eventgramLeft eventgramBottom ...
                     eventgramWidth eventgramHeight];

% colorbar position
colorbarLeft = eventgramLeft;
colorbarWidth = eventgramWidth;
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

% verify correct number of input arguments
error(nargchk(1, 11, nargin, 'struct'));

% apply default arguments
if (nargin < 4) || isempty(tiling),
  tiling = [];
end
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
if (nargin < 9) || isempty(durationInflation),
  durationInflation = 1.0;
end
if (nargin < 10) || isempty(bandwidthInflation),
  bandwidthInflation = 1.0;
end
if (nargin < 11) || isempty(normalizedEnergyRange),
  normalizedEnergyRange = [];
end

% force cell arrays
events = wmat2cell(events);

% force one dimensional cell arrays
events = events(:);

% determine number of channels
numberOfChannels = length(events);

% make channelNames array
for channelNumber = 1 : numberOfChannels,
  if isfield(events{channelNumber},'channelName'),
    channelNames{channelNumber} = events{channelNumber}.channelName;
  else
    channelNames{channelNumber} = ['Channel ' int2str(channelNumber)];
  end
end

% force ranges to be monotonically increasing column vectors
timeRange = unique(timeRange(:));
frequencyRange = unique(frequencyRange(:));
if ~isempty(normalizedEnergyRange),
  normalizedEnergyRange = unique(normalizedEnergyRange(:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% validate channel names
if ~isempty(channelNames) && (length(channelNames) ~= numberOfChannels),
  error('channel names is inconsistent with number of transform channels');
end

% validate event structures
for channelNumber = 1 : numberOfChannels,
  if ~strcmp(events{channelNumber}.id, ...
             'Discrete Q-transform event structure'),
    error('input argument is not a discrete Q transform event structure');
  end
end

% validate tiling structure
if ~isempty(tiling),
  if ~strcmp(tiling.id, 'Discrete Q-transform tile structure'),
    error('input argument is not a discrete Q transform tiling structure');
  end
end

% Check for two component range vectors
if length(timeRange) ~= 2,
  error('Time range must be two component vector [tmin tmax].');
end
if length(frequencyRange) ~= 2,
  error('Frequency range must be two component vector [fmin fmax].');
end
if ~isempty(normalizedEnergyRange) && length(normalizedEnergyRange) ~= 2,
  error('Normalized energy range must be two component vector [Zmin Zmax].');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           initialize result vector                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize handle vector
handles = zeros(numberOfChannels, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           begin loop over channels                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% begin loop over channels
for channelNumber = 1 : numberOfChannels,

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                         extract event properties                           %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % bandwidth of events
  bandwidths = 2 * sqrt(pi) * events{channelNumber}.frequency ./ ...
      events{channelNumber}.q;

  % duration of events
  durations = 1 ./ bandwidths;

  % apply bandwidth inflation factor
  bandwidths = bandwidthInflation * bandwidths;

  % apply duration inflation factor
  durations = durationInflation * durations;

  % start time of events
  startTimes = events{channelNumber}.time - referenceTime - durations / 2;

  % stop time of events
  stopTimes = events{channelNumber}.time - referenceTime + durations / 2;

  % low frequency of events
  lowFrequencies = events{channelNumber}.frequency - bandwidths / 2;

  % high frequency boundary of events
  highFrequencies = events{channelNumber}.frequency + bandwidths / 2;

  % normalized energy of events or clusters
  if isfield(events{channelNumber}, 'clusterNormalizedEnergy'),
    normalizedEnergies = events{channelNumber}.clusterNormalizedEnergy;
  else
    normalizedEnergies = events{channelNumber}.normalizedEnergy;
  end  

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                        identify events to display                          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % default start time for
  if timeRange(1) == -Inf,
    if isempty(tiling),
      timeRange(1) = floor(min(startTimes));
    else
      timeRange(1) = startTime - referenceTime;
    end
  end

  % default stop time
  if timeRange(2) == +Inf,
    if isempty(tiling),
      timeRange(2) = ceil(max(stopTimes));
    else
      timeRange(2) = startTime - referenceTime + tiling.duration;
    end
  end

  % default minimum frequency
  if frequencyRange(1) == -Inf,
    if isempty(tiling),
      frequencyRange(1) = 2.^floor(log2(min(lowFrequencies)));
    else
      frequencyRange(1) = tiling.planes{1}.minimumFrequency;
    end
  end

  % default maximum frequency
  if frequencyRange(2) == +Inf,
    if isempty(tiling),
      frequencyRange(2) = 2.^ceil(log2(max(highFrequencies)));
    else
      frequencyRange(2) = tiling.planes{end}.maximumFrequency;
    end
  end

  % find events overlapping specified time-frequency ranges
  displayIndices = find((startTimes < max(timeRange)) & ...
                        (stopTimes > min(timeRange)) & ...
                        (lowFrequencies < max(frequencyRange)) & ...
                        (highFrequencies > min(frequencyRange)));

  % number of events
  numberOfEvents = length(displayIndices);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                        sort by normalized energy                           %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % sort in order of increasing normalized energy
  [ignore, sortedIndices] = sort(normalizedEnergies(displayIndices));

  % sorted indices of events to display
  sortedDisplayIndices = displayIndices(sortedIndices);

  % sorted properties of events to display
  startTimes = startTimes(sortedDisplayIndices);
  stopTimes = stopTimes(sortedDisplayIndices);
  lowFrequencies = lowFrequencies(sortedDisplayIndices);
  highFrequencies = highFrequencies(sortedDisplayIndices);
  normalizedEnergies = normalizedEnergies(sortedDisplayIndices);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                         define event boundaries                            %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % time coordinates of event bounding box
  times = [startTimes; stopTimes; stopTimes; startTimes; startTimes;];

  % frequency coordinates of event bounding box
  frequencies = [lowFrequencies; lowFrequencies; highFrequencies; ...
                 highFrequencies; lowFrequencies;];

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                          set colormap scaling                              %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % if normalized energy range is not specified
  if isempty(normalizedEnergyRange),

    % normalized energy range to code on colormap
    colormapScale = [0 max(max(normalizedEnergies))];

    % choose a default range if there are no events
    if numberOfEvents == 0,
      colormapScale = [0 10];
    end

  % otherwise,
  else

    % use specified range
    colormapScale = normalizedEnergyRange(:).';

  % continue
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                                 plot events                                %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % if plotting more than one figure
  if numberOfChannels  > 1,

    % select figure to plot in
    figure(channelNumber);

  % continue
  end


  % reset figure
  clf;
  set(gca, 'FontSize', 16);

  % avoid bug in fill function
  if numberOfEvents == 3,
    times = [times zeros(5, 1)];
    frequencies = [frequencies zeros(5, 1)];
    normalizedEnergies = [normalizedEnergies 0];
  end

  % plot eventgram
  if abs(diff(timeRange)) < millisecondThreshold,
    boxHandles = fill(times * 1e3, frequencies, normalizedEnergies);
  elseif abs(diff(timeRange)) < secondThreshold,
    boxHandles = fill(times * 1, frequencies, normalizedEnergies);
    save([outputDirectory '/' uniqueID '_eventgram_' num2str(abs(diff(timeRange))) '.mat'],'times','frequencies','normalizedEnergies');    
  elseif abs(diff(timeRange)) < minuteThreshold,
    boxHandles = fill(times / 60, frequencies, normalizedEnergies);
  elseif abs(diff(timeRange)) < hourThreshold,
    boxHandles = fill(times / 3600, frequencies, normalizedEnergies);
  elseif abs(diff(timeRange)) < dayThreshold,
    boxHandles = fill(times / 86400, frequencies, normalizedEnergies);
  else
    boxHandles = fill(times / 31557600, frequencies, normalizedEnergies);
  end
  set(boxHandles, 'LineStyle', 'none');

  % apply colormap scaling
  colormap('default');
  colormapMatrix = colormap;
  set(gca, 'Color', colormapMatrix(1, :));
  set(gcf, 'Color', [1 1 1]);
  set(gcf, 'InvertHardCopy', 'off');
  caxis(colormapScale);

  % set axis position
  set(gca, 'Position', eventgramPosition);

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

  % disable coordinate grid
  grid off;

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
  % frequencyOctaves = log2(max(frequencyRange) / min(frequencyRange));
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

  % set title properties
  %titleString = sprintf('%s at %.3f', ...
  %                      channelNames{channelNumber}, referenceTime);
  %title(strrep(titleString, '_', '\_'));

  % append current axis handle to list of handles
  handles(channelNumber) = gca;

  % display colorbar
  subplot('position', colorbarPosition);
  set(gca, 'FontSize', 16);
  colorbarmap = linspace(min(colormapScale), max(colormapScale), 100);
  imagesc(colorbarmap, 1, colorbarmap, colormapScale);
  set(gca, 'YTick',[])
  set(gca, 'TickDir', 'out')
  if isfield(events{channelNumber}, 'clusterNormalizedEnergy'),
    xlabel('Normalized cluster energy');
  else
    xlabel('Normalized tile energy');
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
