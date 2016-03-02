function events = wselect(significants, durationInflation, ...
                          bandwidthInflation, maximumEvents, debugLevel)
% WSELECT Identify statistically significant events in Discrete Q transforms
%
% WSELECT selects statistically significant events from the set of statitically
% significant Q transform tiles.  Events are defined by the properties of their
% most significant tile and are identified by exluding the less significant of
% any overlapping tiles.  The input significant tiles are first sorted by
% decreasing normalized energy.  Starting with the most significant tile, tiles
% are discarded if they overlap with a more significant tile.  The remaining set
% of tiles comprises a minimal set of tiles that describes an event.
%
% WSELECT returns a cell array of significant event properties
%
% usage: events = wselect(significants, durationInflation, ...
%                         bandwidthInflation, maximumEvents, ...
%                         debugLevel);
%
%   significants         cell array of significant tiles properties
%   durationInflation    multiplicative scale factor for duration
%   bandwidthInflation   multiplicative scale factor for bandwidth
%   maximumEvents        maximum allowable number of events
%   debugLevel           verboseness of debug output
%
%   events               cell array of significant event properties
%
% The optional durationInflation and bandwidthInflation arguments are
% multiplicative scale factors that are applied to the duration and bandwidth of
% significant tiles prior to testing for overlap.  If not specified, these
% parameters both default to unity such that the resulting tiles have unity
% time-frequency area.  The normalized energy of the resulting tiles are scaled
% by the product of the duration and bandwidth inflation factors to avoid over
% counting the total energy of clusters of tiles.  Likewise, the amplitude of
% the resulting tiles is scaled by the square root of the product of the
% duration and bandwidth inflation factors.
%
% The optional maximumEvents argument provides a safety mechanism to limit the
% total number of events returned by WSELECT.  If this maximum number of events
% is exceeded, an overflow flag is set, only the maximumEvents most significant
% events are returned, and a warning is issued if debugLevel is set to unity or
% higher.  By default, maximumEvents is set to infinity and debugLevel is set to
% unity.
%
% WSELECT both expects and returns a cell array of Q transform event structures
% with one cell per channel.  The event structures contain the following
% required fields, which describe the properties of statistically significant
% tiles.  Additional fields such as amplitude, phase, or coherent transform
% properties are optional and are retained along with the required fields.
%
%   time                 center time of tile [gps seconds]
%   frequency            center frequency of tile [Hz]
%   duration             duration of tile [seconds]
%   bandwidth            bandwidth of tile [Hz]
%   normalizedEnergy     normalized energy of tile []
%
% The event structures also contain the following flag, which indicates if the
% maximum number of significant tiles or significant events was exceeded.
%
%   overflowFlag         boolean overflow flag
%
% See also WTILE, WCONDITION, WTRANSFORM, WTHRESHOLD, WEXAMPLE, and WSEARCH.

% Shourov K. Chatterji <shourov@ligo.caltech.edu>
% Leo C. Stein <lstein@ligo.mit.edu>

% $Id: wselect.m 3412 2012-05-06 10:19:03Z michal.was@LIGO.ORG $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify correct number of input arguments
error(nargchk(1, 5, nargin, 'struct'));

% apply default arguments
if (nargin < 2) || isempty(durationInflation),
  durationInflation = 1.0;
end
if (nargin < 3) || isempty(bandwidthInflation),
  bandwidthInflation = 1.0;
end
if (nargin < 4) || isempty(maximumEvents),
  maximumEvents = Inf;
end
if (nargin < 5) || isempty(debugLevel),
  debugLevel = 1;
end

% force cell arrays
significants = wmat2cell(significants);

% force one dimensional cell arrays
significants = significants(:);

% determine number of channels
numberOfChannels = length(significants);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% validate significant event structures
for channelNumber = 1 : numberOfChannels,
  if ~strcmp(significants{channelNumber}.id, ...
             'Discrete Q-transform event structure'),
    error('input argument is not a discrete Q transform event structure');
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            initialize statistically significant events structures            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create empty array of significant event indices
eventIndices = cell(size(significants));

% create empty cell array of significant event structures
events = cell(size(significants));

% begin loop over channels
for channelNumber = 1 : numberOfChannels

  % insert structure identification string
  events{channelNumber}.id = 'Discrete Q-transform event structure';
  
  % propogate overflow flag
  events{channelNumber}.overflowFlag = ...
      significants{channelNumber}.overflowFlag;

% end loop over channels
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           begin loop over channels                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% begin loop over channels
for channelNumber = 1 : numberOfChannels,

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                   sort by decreasing normalized energy                     %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % sort tile indices by normalized energy
  [ignore, sortedIndices] = ...
      sort(significants{channelNumber}.normalizedEnergy);

  % sort by decreasing normalized energy
  sortedIndices = fliplr(sortedIndices);

  % reorder significant tile properties by decreasing normalized energy
  significants{channelNumber} = ...
      wcopyevents(significants{channelNumber}, sortedIndices);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                           find tile boundaries                             %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % vector of significant tile start times
  minimumTimes = significants{channelNumber}.time - ...
      durationInflation * ...
      significants{channelNumber}.duration / 2;

  % vector of significant tile stop times
  maximumTimes = significants{channelNumber}.time + ...
      durationInflation * ...
      significants{channelNumber}.duration / 2;

  % vector of significant tile lower frequencies
  minimumFrequencies = significants{channelNumber}.frequency - ...
      bandwidthInflation * ...
      significants{channelNumber}.bandwidth / 2;

  % vector of significant tile upper frequencies
  maximumFrequencies = significants{channelNumber}.frequency + ...
      bandwidthInflation * ...
      significants{channelNumber}.bandwidth / 2;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                      compress significant tile list                        %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % number of significant tiles in list
  numberOfTiles = length(significants{channelNumber}.time);

  % if input significant tile list is empty,
  if numberOfTiles == 0,

    % set empty event list
    events{channelNumber} = wcopyevents(significants{channelNumber}, []);

    % skip to next channel
    continue;

  % otherwise, continue
  end

  % initialize event list
  eventIndices{channelNumber} = [];

  % begin loop over significant tiles
  for tileIndex = 1 : numberOfTiles,

    % determine if current tile overlaps any events
    overlap = (minimumTimes(tileIndex) < ...
               maximumTimes(eventIndices{channelNumber})) & ...
              (maximumTimes(tileIndex) > ...
               minimumTimes(eventIndices{channelNumber})) & ...
              (minimumFrequencies(tileIndex) < ...
               maximumFrequencies(eventIndices{channelNumber})) & ...
              (maximumFrequencies(tileIndex) > ...
               minimumFrequencies(eventIndices{channelNumber}));

    % if tile does not overlap with any event,
    if ~any(overlap),
      % append it to the list of events
      eventIndices{channelNumber} = [eventIndices{channelNumber} tileIndex];
      % initilaize averages over tiles in the event
      significants{channelNumber}.av_frequency(tileIndex) = ...
          significants{channelNumber}.frequency(tileIndex);
      significants{channelNumber}.av_bandwidth(tileIndex) = ...
          significants{channelNumber}.bandwidth(tileIndex);
      significants{channelNumber}.err_frequency(tileIndex) = ...
          significants{channelNumber}.bandwidth(tileIndex)./...
          significants{channelNumber}.frequency(tileIndex);
      significants{channelNumber}.tot_normalizedEnergy(tileIndex) = ...
          significants{channelNumber}.normalizedEnergy(tileIndex);

    % otherwise, add to averages over tiles in the event
    else
      tmp_av_frequency = ...
          significants{channelNumber}.av_frequency(overlap).*...
          significants{channelNumber}.tot_normalizedEnergy(overlap) +...
          significants{channelNumber}.frequency(tileIndex).*...
          significants{channelNumber}.normalizedEnergy(tileIndex);
      tmp_av_bandwidth = ...
          significants{channelNumber}.av_bandwidth(overlap).*...
          significants{channelNumber}.tot_normalizedEnergy(overlap) +...
          significants{channelNumber}.bandwidth(tileIndex).*...
          significants{channelNumber}.normalizedEnergy(tileIndex);
      tmp_2nd_moment_frequency = ...
          significants{channelNumber}.tot_normalizedEnergy(overlap).*...
          (significants{channelNumber}.err_frequency(overlap).^2 + 1).*...
          significants{channelNumber}.av_frequency(overlap).^2 +...
          significants{channelNumber}.normalizedEnergy(tileIndex).*...
          significants{channelNumber}.frequency(tileIndex).^2;
      significants{channelNumber}.tot_normalizedEnergy(overlap) = ...
          significants{channelNumber}.tot_normalizedEnergy(overlap) + ...
          significants{channelNumber}.normalizedEnergy(tileIndex);
      significants{channelNumber}.av_frequency(overlap) = ...
          tmp_av_frequency./ ...
          significants{channelNumber}.tot_normalizedEnergy(overlap);
      significants{channelNumber}.av_bandwidth(overlap) = ...
          tmp_av_bandwidth./ ...
          significants{channelNumber}.tot_normalizedEnergy(overlap);
      significants{channelNumber}.err_frequency(overlap) = ...
          sqrt( (tmp_2nd_moment_frequency - tmp_av_frequency.*significants{channelNumber}.av_frequency(overlap))./ ...
          significants{channelNumber}.tot_normalizedEnergy(overlap) )./...
          significants{channelNumber}.av_frequency(overlap);
    end

  % end loop over significant tiles
  end

  % extract events from significant tiles
  events{channelNumber} = ...
      wcopyevents(significants{channelNumber}, eventIndices{channelNumber});

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                    check for excessive number of events                    %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % determine number of significant tiles in channel
  numberOfEvents = length(events{channelNumber}.time);

  % if maximum allowable number of significant tiles is exceeded
  if numberOfEvents > maximumEvents,

    % issue warning
    wlog(debugLevel, 1, 'WARNING: %s: maximum number of events exceeded.\n', ...
         events{channelNumber}.channelName);

    % set overflow flag
    events{channelNumber}.overflowFlag = 1;

    % indices of most significant tiles
    maximumIndices = 1 : maximumEvents;

    % truncate lists of significant event properties
    events{channelNumber} = ...
        wcopyevents(events{channelNumber}, maximumIndices);

  % otherwise continue
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            end loop over channels                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end loop over channels
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   return statistically significant events                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return
