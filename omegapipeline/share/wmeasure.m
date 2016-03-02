function measurements = wmeasure(transforms, tiling, startTime, ...
                                 referenceTime, timeRange, frequencyRange, ...
                                 qRange, debugLevel)
% WMEASURE Measure peak and weighted signal properties from Q transforms
%
% WMEASURE reports the peak and significance weighted mean properties of Q
% transformed signals within the specified time-frequency region.
%
% usage:
%
%   measurements = wmeasure(transforms, tiling, startTime, referenceTime, ...
%                           timeRange, frequencyRange, qRange, debugLevel);
%
%   transforms           cell array of input Q transform structures
%   tiling               discrete Q transform tiling structure from WTILE
%   startTime            GPS start time of Q transformed data
%   referenceTime        reference time for time range to search over
%   timeRange            vector range of relative times to search over
%   frequencyRange       vector range of frequencies to search over
%   qRange               scalar Q or vector range of Qs to search over
%   debugLevel           verboseness of debug output
%
%   measurements         cell array of measured signal properties
%
% WMEASURE returns a cell array of measured signal properties, with one cell per
% channel.  The measured signal properties are returned as a structure that
% contains the following fields.
%
%   peakTime                 center time of peak tile [gps seconds]
%   peakFrequency            center frequency of peak tile [Hz]
%   peakQ                    quality factor of peak tile []
%   peakDuration             duration of peak tile [seconds]
%   peakBandwidth            bandwidth of peak tile [Hz]
%   peakNormalizedEnergy     normalized energy of peak tile []
%   peakAmplitude            amplitude of peak tile [Hz^-1/2]
%   signalTime               weighted central time [gps seconds]
%   signalFrequency          weighted central frequency [Hz]
%   signalDuration           weighted duration [seconds]
%   signalBandwidth          weighted bandwidth [Hz]
%   signalNormalizedEnergy   total normalized energy []
%   signalAmplitude          total signal amplitude [Hz^-1/2]
%   signalArea               measurement time frequency area []
%
% The user can focus on a subset of the times and frequencies available in
% the transform data by specifying a desired range of central times,
% central frequencies, and Qs to threshold on.  Ranges should be specified
% as a two component vector, consisting of a minimum and maximum value.
% Alternatively, if only a single Q is specified, WMEASURE is only applied to
% the time-frequency plane which has the nearest value of Q in a
% logarithmic sense to the requested value.
%
% To determine the range of central times to search over, WMEASURE requires
% the start time of the transformed data in addition to a reference time
% and a relative time range.  Both the start time and reference time should
% be specified as absolute quantities, while the range of times to analyze
% should be specified relative to the requested reference time.
%
% By default, WMEASURE is applied to all available frequencies and Qs, and the
% reference time and relative time range arguments are set to exclude data
% potentially corrupted by filter transients as identified by the transient
% duration field of the tiling structure.  The default value can be
% obtained for any argument by passing the empty matrix [].
%
% See also WTILE, WCONDITION, WTRANSFORM, WTHRESHOLD, WSELECT, WEXAMPLE, WSCAN,
% and WSEARCH.

% Notes:
% 1. Compute absolute or normalized energy weighted signal properties?
% 2. Only include tiles with Z>Z0 in integrands?

% Shourov K. Chatterji
% shourov@ligo.caltech.edu

% $Id: wmeasure.m 1716 2009-04-10 17:00:49Z jrollins $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify correct number of input arguments
error(nargchk(3, 8, nargin));

% apply default arguments
if (nargin < 4) || isempty(referenceTime),
  referenceTime = startTime + tiling.duration / 2;
end
if (nargin < 5) || isempty(timeRange),
  timeRange = 0.5 * (tiling.duration - 2 * tiling.transientDuration) * [-1 +1];
end
if (nargin < 6) || isempty(frequencyRange),
  frequencyRange = [-Inf +Inf];
end
if (nargin < 7) || isempty(qRange),
  qRange = [-Inf +Inf];
end
if (nargin < 8) || isempty(debugLevel),
  debugLevel = 1;
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
%                      initialize measurement structures                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create empty cell array of measurement structures
measurements = cell(numberOfChannels, 1);

% begin loop over channels
for channelNumber = 1 : numberOfChannels

  % insert structure identification string
  measurements{channelNumber}.id = 'Discrete Q-transform measurement structure';

  % initialize peak signal properties
  measurements{channelNumber}.peakTime = 0;
  measurements{channelNumber}.peakFrequency = 0;
  measurements{channelNumber}.peakQ = 0;
  measurements{channelNumber}.peakDuration = 0;
  measurements{channelNumber}.peakBandwidth = 0;
  measurements{channelNumber}.peakNormalizedEnergy = 0;
  measurements{channelNumber}.peakAmplitude = 0;

  % initialize integrated signal properties
  measurements{channelNumber}.signalTime = ...
      zeros(1, tiling.numberOfPlanes);
  measurements{channelNumber}.signalFrequency = ...
      zeros(1, tiling.numberOfPlanes);
  measurements{channelNumber}.signalDuration = ...
      zeros(1, tiling.numberOfPlanes);
  measurements{channelNumber}.signalBandwidth = ...
      zeros(1, tiling.numberOfPlanes);
  measurements{channelNumber}.signalNormalizedEnergy = ...
      zeros(1, tiling.numberOfPlanes);
  measurements{channelNumber}.signalAmplitude = ...
      zeros(1, tiling.numberOfPlanes);
  measurements{channelNumber}.signalArea = ...
      zeros(1, tiling.numberOfPlanes);

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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                           calculate times                                %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     times = (0 :  tiling.planes{plane}.rows{row}.numberOfTiles - 1) ...
             * tiling.planes{plane}.rows{row}.timeStep;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                    threshold on central frequency                        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % skip frequency rows outside of requested frequency range
    if ((tiling.planes{plane}.rows{row}.frequency < ...
         min(frequencyRange)) || ...
        (tiling.planes{plane}.rows{row}.frequency > ...
         max(frequencyRange))),
      continue;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                      threshold on central time                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % skip tiles outside requested time range
    tileIndices = ...
        find((times >= ...
              (referenceTime - startTime + min(timeRange))) & ...
             (times <= ...
              (referenceTime - startTime + max(timeRange))));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           differential time-frequency area for integration               %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % differential time-frequency area for integration
    differentialArea = tiling.planes{plane}.rows{row}.timeStep * ...
                       tiling.planes{plane}.rows{row}.frequencyStep;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                       begin loop over channels                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % begin loop over channels
    for channelNumber = 1 : numberOfChannels,

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %                   update peak tile properties                          %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      % vector of row tile normalized energies
      normalizedEnergies = transforms{channelNumber}.planes{plane}.rows{row} ...
                           .normalizedEnergies(tileIndices);

      % find most significant tile in row
      [peakNormalizedEnergy, peakIndex] = max(normalizedEnergies);

      % if peak tile is in this row
      if peakNormalizedEnergy > measurements{channelNumber}.peakNormalizedEnergy,

        % update plane index of peak tile
        peakPlane{channelNumber} = plane;
        
        % extract time index of peak tile
        peakIndex = tileIndices(peakIndex);
      
        % update center time of peak tile
        measurements{channelNumber}.peakTime = ...
            times(peakIndex) + startTime;

        % update center frequency of peak tile
        measurements{channelNumber}.peakFrequency = ...
            tiling.planes{plane}.rows{row}.frequency;

        % update q of peak tile
        measurements{channelNumber}.peakQ = ...
            tiling.planes{plane}.q;

        % update duration of peak tile
        measurements{channelNumber}.peakDuration = tiling.planes{plane}.rows{row}.duration;

        % update bandwidth of peak tile
        measurements{channelNumber}.peakBandwidth = tiling.planes{plane}.rows{row}.bandwidth;
          
        % update normalized energy of peak tile
        measurements{channelNumber}.peakNormalizedEnergy = ...
            (transforms{channelNumber}.planes{plane}.rows{row} ...
             .normalizedEnergies(peakIndex));

        % udpate amplitude of peak tile
        measurements{channelNumber}.peakAmplitude = ...
            sqrt((transforms{channelNumber}.planes{plane}.rows{row} ...
                  .normalizedEnergies(peakIndex) - 1) * ...
                 transforms{channelNumber}.planes{plane}.rows{row}.meanEnergy);

      % end test for peak tile in this row
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %                update weighted signal properties                       %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      % threshold on significance
      normalizedEnergyThreshold = 4.5;
      significantIndices = find(normalizedEnergies > normalizedEnergyThreshold);
      normalizedEnergies = normalizedEnergies(significantIndices);
      significantIndices = tileIndices(significantIndices);

      % vector of row tile calibrated energies
      calibratedEnergies = (normalizedEnergies - 1) * ...
          transforms{channelNumber}.planes{plane}.rows{row}.meanEnergy * ...
          tiling.planes{plane}.normalization;

      % sum of normalized tile energies in row
      sumNormalizedEnergies = sum(normalizedEnergies);

      % sum of calibrated tile enregies in row
      sumCalibratedEnergies = sum(calibratedEnergies);
      
      % update weighted central time integral
      measurements{channelNumber}.signalTime(plane) = ...
          measurements{channelNumber}.signalTime(plane) + ...
          sum(times(significantIndices) .* ...
              calibratedEnergies) * ...
          differentialArea;

      % update weighted central frequency integral
      measurements{channelNumber}.signalFrequency(plane) = ...
          measurements{channelNumber}.signalFrequency(plane) + ...
          tiling.planes{plane}.rows{row}.frequency * ...
          sumCalibratedEnergies * ...
          differentialArea;

      % update weighted duration integral
      measurements{channelNumber}.signalDuration(plane) = ...
          measurements{channelNumber}.signalDuration(plane) + ...
          sum(times(significantIndices).^2 .* ...
              calibratedEnergies) * ...
          differentialArea;

      % update weighted bandwidth integral
      measurements{channelNumber}.signalBandwidth(plane) = ...
          measurements{channelNumber}.signalBandwidth(plane) + ...
          tiling.planes{plane}.rows{row}.frequency^2 * ...
          sumCalibratedEnergies * ...
          differentialArea;

      % update total normalized energy integral
      measurements{channelNumber}.signalNormalizedEnergy(plane) = ...
          measurements{channelNumber}.signalNormalizedEnergy(plane) + ...
          sumNormalizedEnergies * ...
          differentialArea;

      % update total calibrated energy integral
      measurements{channelNumber}.signalAmplitude(plane) = ...
          measurements{channelNumber}.signalAmplitude(plane) + ...
          sumCalibratedEnergies * ...
          differentialArea;

      % update total signal area integral
      measurements{channelNumber}.signalArea(plane) = ...
          measurements{channelNumber}.signalArea(plane) + ...
          length(normalizedEnergies) * ...
          differentialArea;
      
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

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                       normalize signal properties                          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % begin loop over channels
  for channelNumber = 1 : numberOfChannels,

    % normalize weighted signal properties by total normalized energy
    if measurements{channelNumber}.signalAmplitude(plane) ~= 0,
      measurements{channelNumber}.signalTime(plane) = ...
          measurements{channelNumber}.signalTime(plane) ./ ...
          measurements{channelNumber}.signalAmplitude(plane);
      measurements{channelNumber}.signalFrequency(plane) = ...
          measurements{channelNumber}.signalFrequency(plane) ./ ...
          measurements{channelNumber}.signalAmplitude(plane);
      measurements{channelNumber}.signalDuration(plane) = ...
          measurements{channelNumber}.signalDuration(plane) ./ ...
          measurements{channelNumber}.signalAmplitude(plane);
      measurements{channelNumber}.signalBandwidth(plane) = ...
          measurements{channelNumber}.signalBandwidth(plane) ./ ...
          measurements{channelNumber}.signalAmplitude(plane);
    end

    % duration and bandwidth are second central moments in time and frequency
    measurements{channelNumber}.signalDuration(plane) = ...
        sqrt(measurements{channelNumber}.signalDuration(plane) - ...
             measurements{channelNumber}.signalTime(plane).^2);
    measurements{channelNumber}.signalBandwidth(plane) = ...
        sqrt(measurements{channelNumber}.signalBandwidth(plane) - ...
             measurements{channelNumber}.signalTime(plane).^2);

    % convert signal energy to signal amplitude
    measurements{channelNumber}.signalAmplitude(plane) = ...
        sqrt(measurements{channelNumber}.signalAmplitude(plane));

    % add start time to measured central time
    measurements{channelNumber}.signalTime(plane) = ...
        measurements{channelNumber}.signalTime(plane) + startTime;

  % end loop over channels
  end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            end loop over Q planes                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end loop over Q planes
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          report signal properties from plane with peak significance          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% begin loop over channels
for channelNumber = 1 : numberOfChannels,

  % weighted central time estimate from plane with peak tile significance
  measurements{channelNumber}.signalTime = ...
      measurements{channelNumber}.signalTime(peakPlane{channelNumber});

  % weighted central frequency estimate from plane with peak tile significance
  measurements{channelNumber}.signalFrequency = ...
      measurements{channelNumber}.signalFrequency(peakPlane{channelNumber});

  % weighted duration estimate from plane with peak tile significance
  measurements{channelNumber}.signalDuration = ...
      measurements{channelNumber}.signalDuration(peakPlane{channelNumber});

  % weighted bandwidth estimate from plane with peak tile significance
  measurements{channelNumber}.signalBandwidth = ...
      measurements{channelNumber}.signalBandwidth(peakPlane{channelNumber});

  % total signal normalized energy estimate from plane with peak tile significance
  measurements{channelNumber}.signalNormalizedEnergy = ...
      measurements{channelNumber}.signalNormalizedEnergy(peakPlane{channelNumber});

  % total signal amplitude estimate from plane with peak tile significance
  measurements{channelNumber}.signalAmplitude = ...
      measurements{channelNumber}.signalAmplitude(peakPlane{channelNumber});

  % measured time frequency area in plane with peak tile significance
  measurements{channelNumber}.signalArea = ...
      measurements{channelNumber}.signalArea(peakPlane{channelNumber});

  % report peak tile properties for very weak signals
  if measurements{channelNumber}.signalArea < 1,
    measurements{channelNumber}.signalTime = ...
        measurements{channelNumber}.peakTime;
    measurements{channelNumber}.signalFrequency = ...
        measurements{channelNumber}.peakFrequency;
    measurements{channelNumber}.signalDuration = ...
        measurements{channelNumber}.peakDuration;
    measurements{channelNumber}.signalBandwidth = ...
        measurements{channelNumber}.peakBandwidth;
    measurements{channelNumber}.signalNormalizedEnergy = ...
        measurements{channelNumber}.peakNormalizedEnergy;
    measurements{channelNumber}.signalAmplitude = ...
        measurements{channelNumber}.peakAmplitude;
    measurements{channelNumber}.signalArea = 1;
  end

% end loop over channels
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 return most significant tile properties                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return
