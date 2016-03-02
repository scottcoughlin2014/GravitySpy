function significants = ...
    wthreshold(transforms, tiling, startTime, falseEventRate, ...
               referenceTime, timeRange, frequencyRange, qRange, ...
               maximumSignificants, analysisMode, falseVetoRate, ...
               uncertaintyFactor, correlationFactor, debugLevel, PSD)
% WTHRESHOLD Identify statistically significant tiles in Discrete Q transforms
%
% WTHRESHOLD identifies discrete Q transform coefficients whose magnitudes
% exceed the threshold that approximately yields the specified single channel
% false rate assuming ideal white noise.
%
% usage: significants = ...
%          wthreshold(transforms, tiling, startTime, falseEventRate, ...
%                     referenceTime, timeRange, frequencyRange, qRange, ...
%                     maximumSignificants, analysisMode, falseVetoRate, ...
%                     uncertaintyFactor, correlationFactor, debugLevel);
%
%   transforms           cell array of input Q transform structures
%   tiling               discrete Q transform tiling structure from WTILE
%   startTime            GPS start time of Q transformed data
%   falseEventRate       desired white noise false event rate [Hz]
%   referenceTime        reference time for time range to threshold on
%   timeRange            vector range of relative times to threshold on
%   frequencyRange       vector range of frequencies to threshold on
%   qRange               scalar Q or vector range of Qs to threshold on
%   maximumSignificants  maximum allowable number of significant tiles
%   analysisMode         string name of analysis mode to implement
%   falseVetoRate        desired white noise veto rate [Hz]
%   uncertaintyFactor    squared calibration uncertainty factor
%   correlationFactor    fractional correlated energy threshold
%   debugLevel           verboseness of debug output
%   PSD                  measured PSD as outputted by wcondition 
%
%   significants         cell array of Q transform event structures
%
% WTHRESHOLD returns a cell array of Q transform event structures that
% contain the properties of the identified statistically significant tiles
% for each channel.  The event structure contains the following fields.
%
%   time                 center time of tile [gps seconds]
%   frequency            center frequency of tile [Hz]
%   q                    quality factor of tile []
%   duration             duration of tile [seconds]
%   bandwidth            bandwidth of tile [Hz]
%   normalizedEnergy     normalized energy of tile []
%   amplitude            signal amplitude of tile [Hz^-1/2]
%   overflowFlag         boolean overflow flag
%
% For coherent transform data, the following field is also returned.
%
%   incoherentEnergy     incoherent energy of tile []
%
% The user can focus on a subset of the times and frequencies available in
% the transform data by specifying a desired range of central times,
% central frequencies, and Qs to threshold on.  Ranges should be specified
% as a two component vector, consisting of a minimum and maximum value.
% Alternatively, if only a single Q is specified, WTHRESHOLD is only
% applied to the time-frequency plane which has the nearest value of Q in a
% logarithmic sense to the requested value.
%
% To determine the range of central times to threshold on, WTHRESHOLD
% requires the start time of the transformed data in addition to a
% reference time and a relative time range.  Both the start time and
% reference time should be specified as absolute quantities, while the
% range of times to analyze should be specified relative to the requested
% reference time.
%
% By default, WTHRESHOLD is applied to all available frequencies and Qs,
% and the reference time and relative time range arguments are set to
% exclude data potentially corrupted by filter transients as identified by
% the transient duration field of the tiling structure.  The default value
% can be obtained for any argument by passing the empty matrix [].
%
% The threshold is set to yield the specified false event rate when applied
% to all available frequencies and Qs, and is not modified to account for
% restricted ranges.  It is also only a rough estimate, and the result
% false event rate may vary significantly depending on the quality of the
% data.
%
% If provided, the optional analysisMode string is used by WTHRESHOLD to
% determine which channels are signal channels, which channels are null
% channels, and which channels to report results for.  For coherent analysis
% modes, a desired white noise veto rate, squared calibration uncertainy
% factor, and required signal correlation factor must also be specified.
%
% The optional maximumSignificants argument provides a safety mechanism to
% limit the total number of events returned by WTHRESHOLD.  If this maximum
% number of significants is exceeded, the overflow flag is set, only the
% maximumSignificants most significant tiles are returned, and a warning is
% issued if debugLevel is set to 1 or higher.  By default, maximumSignificants
% is set to infinity and debugLevel is set to unity.
%
% See also WTILE, WCONDITION, WTRANSFORM, WSELECT, WEXAMPLE, and WSEARCH.
%
% Authors:
% Shourov K. Chatterji <shourov@ligo.caltech.edu>
% Leo C. Stein <lstein@ligo.mit.edu>
%
% $Id: wthreshold.m 3420 2012-07-20 14:09:38Z michal.was@LIGO.ORG $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify correct number of input arguments
error(nargchk(4, 15, nargin, 'struct'));

% apply default arguments
if (nargin < 5) || isempty(referenceTime),
  referenceTime = startTime + tiling.duration / 2;
end
if (nargin < 6) || isempty(timeRange),
  timeRange = 0.5 * (tiling.duration - 2 * tiling.transientDuration) * [-1 +1];
end
if (nargin < 7) || isempty(frequencyRange),
  frequencyRange = [-Inf +Inf];
end
if (nargin < 8) || isempty(qRange),
  qRange = [-Inf +Inf];
end
if (nargin < 9) || isempty(maximumSignificants),
  maximumSignificants = Inf;
end
if (nargin < 10) || isempty(analysisMode),
  analysisMode = 'independent';
end
if (nargin < 11) || isempty(falseVetoRate),
  falseVetoRate = 0;
end
if (nargin < 12) || isempty(uncertaintyFactor),
  uncertaintyFactor = 0;
end
if (nargin < 13) || isempty(correlationFactor),
  correlationFactor = 0;
end
if (nargin < 14) || isempty(debugLevel),
  debugLevel = 1;
end
if (nargin < 15) || isempty(PSD),
  PSD = 0;
end

% force cell arrays
transforms = wmat2cell(transforms);

% force one dimensional cell arrays
transforms = transforms(:);

% determine number of channels
numberOfChannels = length(transforms);

% force ranges to be monotonically increasing column vectors
timeRange = unique(timeRange(:));
frequencyRange = unique(frequencyRange(:));
qRange = unique(qRange(:));

% if only a single Q is requested, find nearest Q plane
if length(qRange) == 1,
  [ignore, qPlane] = min(abs(log(tiling.qs / qRange)));
  qRange = tiling.qs(qPlane) * [1 1];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% validate tiling structure
if ~strcmp(tiling.id, 'Discrete Q-transform tile structure'),
  error('input argument is not a discrete Q transform tiling structure');
end

% validate transform structures
for channelNumber = 1 : numberOfChannels,
  if ~strcmp(transforms{channelNumber}.id, ...
             'Discrete Q-transform transform structure'),
    error('input argument is not a discrete Q transform structure');
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         normalized energy threshold                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% approximate number of statistically independent tiles per second
independentsRate = tiling.numberOfIndependents / tiling.duration;

% apply emperically determined correction factor
independentsRate = independentsRate * 1.5;

% probability associated with desired false event rate
falseEventProbability = falseEventRate / independentsRate;

% probability associated with desired false veto rate
falseVetoProbability = falseVetoRate / independentsRate;

% normalized energy threshold for desired false event rate
eventThreshold = -log(falseEventProbability);

% normalized energy threshold for desired false veto rate
if falseVetoProbability == 0,
  vetoThreshold = Inf;
else
  vetoThreshold = -log(falseVetoProbability);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             apply analysis mode                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% switch on analysis mode
switch lower(analysisMode),

  case {'independent', 'bayesian'},

    % threshold on all signal channels individually
    for channelNumber = 1 : numberOfChannels,
      outputChannels{channelNumber}.channelName = ...
          transforms{channelNumber}.channelName;
      outputChannels{channelNumber}.channelType = 'signal';
      outputChannels{channelNumber}.signalChannel = channelNumber;
      outputChannels{channelNumber}.referenceChannel = [];
    end

  case {'coherent'},

    % threshold on signal channel
    outputChannels{1}.channelName = ...
        regexprep(transforms{1}.channelName, '-.*$', '');
    outputChannels{1}.channelType = 'signal';
    outputChannels{1}.signalChannel = 1;
    outputChannels{1}.referenceChannel = 2;

    % threshold on null channel
    if numberOfChannels > 2,
      outputChannels{2}.channelName = ...
          regexprep(transforms{3}.channelName, '-.*$', '');
      outputChannels{2}.channelType = 'null';
      outputChannels{2}.signalChannel = 3;
      outputChannels{2}.referenceChannel = 4;
    end

  otherwise,
    error(['unknown analysis mode "' analysisMode '"']);

% end switch on analysis mode
end

% number of output channels
numberOfOutputChannels = length(outputChannels);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             initialize statistically significant event structure             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create empty cell array of significant event structures
significants = cell(numberOfOutputChannels, 1);

% begin loop over channels
for outputChannelNumber = 1 : numberOfOutputChannels

  % insert structure identification string
  significants{outputChannelNumber}.id = 'Discrete Q-transform event structure';

  % initialize result vectors
  significants{outputChannelNumber}.time = [];
  significants{outputChannelNumber}.frequency = [];
  significants{outputChannelNumber}.q = [];
  significants{outputChannelNumber}.duration = [];
  significants{outputChannelNumber}.bandwidth = [];
  significants{outputChannelNumber}.normalizedEnergy = [];
  significants{outputChannelNumber}.amplitude = [];

  % initialize overflow flag
  significants{outputChannelNumber}.overflowFlag = 0;

  % include incoherent energy for coherent channels
  if ~isempty(outputChannels{outputChannelNumber}.referenceChannel),
    significants{outputChannelNumber}.incoherentEnergy = [];
  end

  % fill channel names
  significants{outputChannelNumber}.channelName = ...
      outputChannels{outputChannelNumber}.channelName;

% end loop over channels
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           begin loop over Q planes                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% begin loop over Q planes
for plane = 1 : tiling.numberOfPlanes,

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                              threshold on Q                                %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % skip Q planes outside of requested Q range
  if ((tiling.planes{plane}.q < min(qRange)) || ...
      (tiling.planes{plane}.q > max(qRange))),
    continue;
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                      begin loop over frequency rows                        %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % begin loop over frequency rows
  for row = 1 : tiling.planes{plane}.numberOfRows,

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                    threshold on central frequency                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % skip frequency rows outside of requested frequency range
    if ((tiling.planes{plane}.rows{row}.frequency < ...
         min(frequencyRange)) || ...
        (tiling.planes{plane}.rows{row}.frequency > ...
         max(frequencyRange))),
      continue;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                       begin loop over channels                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % begin loop over channels
    for outputChannelNumber = 1 : numberOfOutputChannels,

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %                  extract output channel details                        %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      % extract output channel structure
      outputChannel = outputChannels{outputChannelNumber};

      % extract output channel details
      channelName = outputChannel.channelName;
      channelType = outputChannel.channelType;
      signalChannel = outputChannel.signalChannel;
      referenceChannel = outputChannel.referenceChannel;

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %                    threshold on significance                           %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      % switch on channel type
      switch channelType,
        
        % if signal channel,
        case 'signal',

          % threshold on signal significance
          if isempty(referenceChannel),
            significantTileIndices = find( ...
                transforms{signalChannel}.planes{plane}.rows{row} ...
                  .normalizedEnergies >=  ...
                eventThreshold);
          else
            significantTileIndices = find( ...
                transforms{signalChannel}.planes{plane}.rows{row} ...
                .normalizedEnergies >= ...
                eventThreshold + correlationFactor * ...
                transforms{referenceChannel}.planes{plane}.rows{row} ...
                .normalizedEnergies);
          end

        % if null channel,
        case 'null',

          % threshold on null significance
          significantTileIndices = find( ...
              transforms{signalChannel}.planes{plane}.rows{row} ...
              .normalizedEnergies >= ...
              vetoThreshold + uncertaintyFactor * ...
              transforms{referenceChannel}.planes{plane}.rows{row} ...
              .normalizedEnergies);

      % end test for channel type
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %                    threshold on central time                           %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     times = (0 :  tiling.planes{plane}.rows{row}.numberOfTiles - 1) * ...
         tiling.planes{plane}.rows{row}.timeStep;

      % skip tiles outside requested time range
      keepIndices = ...
          (times(significantTileIndices) >= ...
           (referenceTime - startTime + min(timeRange))) & ...
          (times(significantTileIndices) <= ...
           (referenceTime - startTime + max(timeRange)));
      significantTileIndices = significantTileIndices(keepIndices);

      % number of statistically significant tiles in frequency row
      numberOfSignificants = length(significantTileIndices);

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %      append significant tile properties to event structure             %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      % append center times of significant tiles in row
      significants{outputChannelNumber}.time = ...
          [significants{outputChannelNumber}.time ...
           times(significantTileIndices) + ...
           startTime];

      % append center frequencies of significant tiles in row
      significants{outputChannelNumber}.frequency = ...
          [significants{outputChannelNumber}.frequency ...
           tiling.planes{plane}.rows{row}.frequency * ...
           ones(1, numberOfSignificants)];

      % append qs of significant tiles in row
      significants{outputChannelNumber}.q = ...
          [significants{outputChannelNumber}.q ...
           tiling.planes{plane}.q * ...
           ones(1, numberOfSignificants)];

      % append durations of significant tiles in row
      significants{outputChannelNumber}.duration = ...
          [significants{outputChannelNumber}.duration ...
           tiling.planes{plane}.rows{row}.duration * ...
           ones(1, numberOfSignificants)];

      % append bandwidths of significant tiles in row
      significants{outputChannelNumber}.bandwidth = ...
          [significants{outputChannelNumber}.bandwidth ...
           tiling.planes{plane}.rows{row}.bandwidth * ...
           ones(1, numberOfSignificants)];
        
      % append normalized energies of significant tiles in row
      significants{outputChannelNumber}.normalizedEnergy = ...
          [significants{outputChannelNumber}.normalizedEnergy ...
           (transforms{signalChannel}.planes{plane}.rows{row} ...
            .normalizedEnergies(significantTileIndices))];

      % -- append amplitudes of significant tiles in row
      if PSD
        % Compute the amplitude (in sqrt(Hz) units) based on the PSD floor
        % (harmonic average windowed by the bi-quadratic window of omega)

        % find frequencies in PSD for given tile 
        qPrime = tiling.planes{plane}.q / sqrt(11);
        [tmp nearestIndex] = min(abs(PSD(:,1) - tiling.planes{plane}.rows{row}.frequency));
        inBandMask = abs(PSD(:,1) - tiling.planes{plane}.rows{row}.frequency)*...
            qPrime < tiling.planes{plane}.rows{row}.frequency & ...
            PSD(:,1) >= tiling.highPassCutoff & PSD(:,1) <= tiling.lowPassCutoff;
        psdIndices = union(nearestIndex,find(inBandMask));
        % dimensionless frequency vector for window construction
        windowArgument = abs(PSD(psdIndices,1) - tiling.planes{plane}.rows{row}.frequency) * ...
            qPrime / tiling.planes{plane}.rows{row}.frequency;
        % bi square window function
        window = (1 - windowArgument.^2).^2;
        window = window/norm(window);
        significants{outputChannelNumber}.amplitude = ...
            [significants{outputChannelNumber}.amplitude ...
             sqrt(2*(transforms{signalChannel}.planes{plane}.rows{row} ...
                   .normalizedEnergies(significantTileIndices) - 1) / ...
                  dot(window.^2,1./PSD(psdIndices,2)))];
      else
        significants{outputChannelNumber}.amplitude = ...
            [significants{outputChannelNumber}.amplitude ...
             sqrt((transforms{signalChannel}.planes{plane}.rows{row} ...
                   .normalizedEnergies(significantTileIndices) - 1) * ...
                  transforms{signalChannel}.planes{plane}.rows{row}.meanEnergy)];
      end

      % append incoherent energies of significant tiles in row
      if ~isempty(referenceChannel),
        significants{outputChannelNumber}.incoherentEnergy = ...
            [significants{outputChannelNumber}.incoherentEnergy ...
             (transforms{referenceChannel}.planes{plane}.rows{row} ...
              .normalizedEnergies(significantTileIndices))];
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %             prune excessive significants as we accumulate              %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      % determine number of significant tiles in channel
      numberOfSignificants = length(significants{outputChannelNumber}.time);

      % if maximum allowable number of significant tiles is exceeded
      if numberOfSignificants > maximumSignificants,

        % issue warning
        wlog(debugLevel, 2, '%s: trimming excess significants to maximum (%s).\n', ...
             channelName,maximumSignificants);

        % set overflow flag
        significants{outputChannelNumber}.overflowFlag = 1;

        % sort significant tiles by normalized energy
        [ignore, maximumIndices] = ...
            sort(significants{outputChannelNumber}.normalizedEnergy);

        % find indices of most significant tiles
        maximumIndices = maximumIndices(end - maximumSignificants + 1 : end);

        % extract most significant tile properties
        significants{outputChannelNumber} = ...
            wcopyevents(significants{outputChannelNumber}, maximumIndices);

      % otherwise continue
      end      

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                        end loop over channels                            %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % end loop over channels
    end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                       end loop over frequency rows                         %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % end loop over frequency rows
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            end loop over Q planes                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end loop over Q planes
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    return statistically significant tiles                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return
