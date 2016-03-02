function tiling = wtile(timeRange, qRange, frequencyRange, sampleFrequency, ...
                        maximumMismatch, highPassCutoff, lowPassCutoff, ...
                        whiteningDuration, transientFactor)
% WTILE Tile space of time, frequency, and Q for discrete Q transform analysis
%
% WTILE covers the specified range of time, frequency, and Q with the minimum
% number of measurement tiles such that the fractional energy loss encountered
% by an arbitrary minimum uncertainty signal never exceeds the requested maximum
% mismatch.  WTILE is typically called once for a particular set of search
% parameters.  The resulting tiling structure is then used in subsequent calls
% to WTRANSFORM on different data segments.
%
% usage: tiling = wtile(timeRange, qRange, frequencyRange, sampleFrequency, ...
%                       maximumMismatch, highPassCutoff, lowPassCutoff, ...
%                       whiteningDuration, transientFactor);
%
%  timeRange          duration of analysis
%  qRange             range of Q to search
%  frequencyRange     range of frequency to search
%  sampleFrequency    sample frequency of input data
%  maximumMismatch    fractional loss in squared signal energy due to mismatch
%  highPassCutoff     cutoff frequency for high pass filter
%  lowPassCutoff      cutoff frequency for low pass filter
%  whiteningDuration  duration of whitening filter
%  transientFactor    ratio of transient duration to whitening filter duration
%
%  tiling             output discrete Q transform tiling structure
%
% To define the targeted signal space, WTILE takes as arguments the duration,
% frequency band, and Q range of the search.  The desired duration should be
% specified as a single scalar number, while the desired frequency range and Q
% range should both be two component vectors that specify the minimum and
% maximum frequency and Q for the analysis.
%
% This signal space cannot be specified arbitrarily.  In order to avoid
% frequency domain aliasing, WTILE enforces a minimum permissible Q of sqrt(11)
% and a maximum permissible analysis frequency that depends on Q.  In order to
% ensure a sufficient number of statistically independent tiles in each
% frequency row, WTILE also enforces a minimum permissible analysis frequency
% that depends on Q.  For convenience, the maximum permissible frequency range
% may be obtained for each Q plane by specifying a frequency range of [].
% Alternatively, the minimum permissible frequency may be obtained by specifying
% a lower limit of 0 Hz and the maximum permissible frequency by specifiying an
% upper limit of Inf Hz.
%
% WTILE also reports recommended filter parameters for data conditioning that
% include cutoff frequencies for high pass and low pass filters and whitening
% filter duration.  It also reports a recommended duration to ignore at both
% beginning and end of the transform due to filter transients.  This transient
% duration is simply the product of the whitening filter duration and a user
% specified factor.  If no transient factor is specified, a default value of
% four is used.  The user may also override the suggested data conditioning
% filter parameters by providing alternative parameters as arguments to WTILE.
%
% The output Q transform tiling structure contains the following fields.
%
%   id                    identification string for structure
%   duration              duration of data under analysis
%   minimumQ              minimum Q of search
%   maximumQ              maximum Q of search
%   minimumFrequency      minimum frequency of search
%   maximumFrequency      maximum frequency of search
%   sampleFrequency       sample frequency of the data under analysis
%   maximumMismatch       maximum fractional energy loss due to signal mismatch
%   numberOfPlanes        number of Q planes in analysis
%   qs                    vector of Qs
%   planes                cell array of plane structures
%   numberOfTiles         total number of tiles in analysis
%   numberOfIndependents  total number of statistically independent tiles
%   numberOfFlops         total number of flops in analysis
%   highPassCutoff        cutoff frequency for high pass filter
%   lowPassCutoff         cutoff frequency for low pass filter
%   whiteningDuration     duration of whitening filter
%   transientDuration     duration of filter transients to supress
%
% The planes field is a cell array of plane structures, one for each Q, which
% contain the following fields.
%
%   q                     Q of plane
%   minimumFrequency      Q dependent minimum frequency of search
%   maximumFrequency      Q dependent maximum frequency of search
%   normalization         Q dependent normalization factor
%   frequencies           vector of frequencies
%   numberOfRows          number of frequency rows in plane
%   rows                  cell array of row structures
%   numberOfTiles         number of tiles in plane
%   numberOfIndependents  number of statistically independent tiles in plane
%   numberOfFlops         number of flops to compute plane
%
% The rows field is a cell array of row structures, one for each frequency in
% the plane, which contain the following fields.
%
%   frequency             frequency of row
%   duration              tile duration for coincidence testing
%   bandwidth             tile bandwidth for coincidence testing
%   timeStep              tile time step for integration
%   frequencyStep         tile frequency step for integration
%   times                 vector of times
%       THIS FIELD HAS BEEN REMOVED DUE TO EXCESSVE MEMORY USE
%       WHERE REQUIRED, COMPUTE IT BY:
%          times = (0 :  tiling.planes{plane}.rows{row}.numberOfTiles - 1) ...
%            * tiling.planes{plane}.rows{row}.timeStep;
%   window                window vector
%   dataIndices           index into frequency domain data
%   zeroPadLength         number of zeros to append to windowed data
%   numberOfTiles         number tiles in frequency row
%   numberOfIndependents  number of statistically independent tiles in row
%   numberOfFlops         number of flops to compute frequency row
%
% See also WCONDITION, WTRANSFORM, WTHRESHOLD, WSELECT, WEXAMPLE, and WSEARCH.

% Shourov K. Chatterji <shourov@ligo.mit.edu>

% $Id: wtile.m 3421 2012-07-20 14:53:33Z michal.was@LIGO.ORG $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify correct number of input arguments
error(nargchk(5, 9, nargin, 'struct'));

% apply default arguments
if (nargin < 6) || isempty(highPassCutoff),
  highPassCutoff = [];
end
if (nargin < 7) || isempty(lowPassCutoff),
  lowPassCutoff = [];
end
if (nargin < 8) || isempty(whiteningDuration),
  whiteningDuration = [];
end
if (nargin < 9) || isempty(transientFactor),
  transientFactor = 4;
end

% default frequency range
if isempty(frequencyRange),
  frequencyRange = [0 Inf];
end

% force column vectors
timeRange = timeRange(:);
qRange = qRange(:);
frequencyRange = frequencyRange(:);

% check for scalar time range
if length(timeRange) ~= 1,
  error('invalid time range');
end

% promote scalar Q range to two element vector
if length(qRange) == 1,
  qRange = [qRange qRange];
end

% check for two element Q range
if length(qRange) ~= 2,
  error('invalid Q range');
end

% promote scalar frequency range to two element vector
if length(frequencyRange) == 1,
  frequencyRange = [frequencyRange frequencyRange];
end

% check for two element frequency range
if length(frequencyRange) ~= 2,
  error('invalid frequency range');
end

% extract minimum and maximum Q from Q range
minimumQ = qRange(1);
maximumQ = qRange(2);

% extract minimum and maximum frequency from frequency range
minimumFrequency = frequencyRange(1);
maximumFrequency = frequencyRange(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          compute derived parameters                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% nyquist frequency
nyquistFrequency = sampleFrequency / 2;

% maximum mismatch between neighboring tiles
mismatchStep = 2 * sqrt(maximumMismatch / 3);

% maximum possible time resolution
minimumTimeStep = 1 / sampleFrequency;

% maximum possible frequency resolution
minimumFrequencyStep = 1 / timeRange;

% conversion factor from Q prime to true Q
qPrimeToQ = sqrt(11);

% total number of samples in input data
numberOfSamples = timeRange * sampleFrequency;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       determine parameter constraints                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% minimum allowable Q prime to prevent window aliasing at zero frequency
minimumAllowableQPrime = 1.0;

% minimum allowable Q to avoid window aliasing at zero frequency
minimumAllowableQ = minimumAllowableQPrime * qPrimeToQ;

% reasonable number of statistically independent tiles in a frequency row
minimumAllowableIndependents = 50;

% maximum allowable mismatch parameter for reasonable performance
maximumAllowableMismatch = 0.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             validate parameters                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check for valid time range
if timeRange < 0,
  error('negative time range');
end

% check for valid Q range
if minimumQ > maximumQ,
  error('minimum Q exceeds maximum Q');
end

% check for valid frequency range
if minimumFrequency > maximumFrequency,
  error('minimum frequency exceeds maximum frequency');
end

% check for valid minimum Q
if minimumQ < minimumAllowableQ,
  error('minimum Q less than %3.2f\n', minimumAllowableQ);
end

% check for reasonable maximum mismatch parameter
if  maximumMismatch > maximumAllowableMismatch,
  error('maximum mismatch exceeds %.2f\n', maximumAllowableMismatch);
end

% check for integer power of two data length
if mod(log(timeRange * sampleFrequency) / log(2), 1) ~= 0,
  error('data length is not an integer power of two');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              determine Q planes                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cumulative mismatch across Q range
qCumulativeMismatch = log(maximumQ / minimumQ) / sqrt(2);

% number of Q planes
numberOfPlanes = ceil(qCumulativeMismatch / mismatchStep);

% insure at least one plane
if numberOfPlanes == 0,
  numberOfPlanes = 1;
end

% mismatch between neighboring planes
qMismatchStep = qCumulativeMismatch / numberOfPlanes;

% index of Q planes
qIndices = 0.5 : numberOfPlanes - 0.5;

% vector of Qs
qs = minimumQ * exp(sqrt(2) * qIndices * qMismatchStep);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             validate frequencies                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% minimum allowable frequency to provide sufficient statistics
minimumAllowableFrequency = minimumAllowableIndependents * max(qs) / ...
                            (2 * pi * timeRange);

% maximum allowable frequency to avoid window aliasing
maximumAllowableFrequency = nyquistFrequency / (1 + qPrimeToQ / min(qs));

% check for valid minimum frequency
if (minimumFrequency ~= 0) && ...
   (minimumFrequency < minimumAllowableFrequency),
  error(['requested minimum frequency of %.2f Hz ' ...
        'less than minimum allowable frequency of %.2f Hz\n'], ...
        minimumFrequency, minimumAllowableFrequency);
end

% check for valid maximum frequency
if (maximumFrequency ~= Inf) && ...
   (maximumFrequency > maximumAllowableFrequency),
  error(['requested maximum frequency of %.2f Hz ' ...
         'greater than maximum allowable frequency %.2f Hz\n'], ...
        maximumFrequency, maximumAllowableFrequency);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     create Q transform tiling structure                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% structure type identifier
tiling.id = 'Discrete Q-transform tile structure';

% insert duration into tiling structure
tiling.duration = timeRange;

% insert minimum Q into tiling structure
tiling.minimumQ = minimumQ;

% insert maximum Q into tiling structure
tiling.maximumQ = maximumQ;

% insert minimum frequency into tiling structure
tiling.minimumFrequency = minimumFrequency;

% insert maximum frequency into tiling structure
tiling.maximumFrequency = maximumFrequency;

% insert sample frequency into tiling structure
tiling.sampleFrequency = sampleFrequency;

% insert maximum loss due to mismatch into tiling structure
tiling.maximumMismatch = maximumMismatch;

% insert Q vector into tiling structure
tiling.qs = qs;

% insert number of Q planes into tiling structure
tiling.numberOfPlanes = numberOfPlanes;

% initialize cell array of Q plans in tiling structure
tiling.planes = cell(1, numberOfPlanes);

% initialize total number of tiles counter
tiling.numberOfTiles = 0;

% initialize total number of independent tiles counter
tiling.numberOfIndependents = 0;

% initialize total number of flops counter
tiling.numberOfFlops = numberOfSamples * log(numberOfSamples);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           begin loop over Q planes                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% begin loop over Q planes
for plane = 1 : numberOfPlanes,

  % extract Q of plane from Q vector
  q = qs(plane);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                        determine plane properties                          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % find Q prime for the plane
  qPrime = q / qPrimeToQ;

  % for large qPrime
  if qPrime > 10,

    % use asymptotic value of planeNormalization
    planeNormalization = 1;

  % otherwise
  else

    % polynomial coefficients for plane normalization factor
    coefficients = [+ 1 * log((qPrime + 1) / (qPrime - 1)); - 2; ...
                    - 4 * log((qPrime + 1) / (qPrime - 1)); + 22 / 3; ...
                    + 6 * log((qPrime + 1) / (qPrime - 1)); - 146 / 15; ...
                    - 4 * log((qPrime + 1) / (qPrime - 1)); + 186 / 35; ...
                    + 1 * log((qPrime + 1) / (qPrime - 1));];

    % plane normalization factor
    planeNormalization = sqrt(256 / (315 * qPrime * ...
                                     polyval(coefficients, qPrime)));

  % continue
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                         determine frequency rows                           %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % plane specific minimum allowable frequency to provide sufficient statistics
  minimumAllowableFrequency = minimumAllowableIndependents * q / ...
                              (2 * pi * tiling.duration);

  % plane specific maximum allowable frequency to avoid window aliasing
  maximumAllowableFrequency = nyquistFrequency / (1 + qPrimeToQ / q);

  % use plane specific minimum allowable frequency if requested
  if tiling.minimumFrequency == 0,
    minimumFrequency = minimumAllowableFrequency;
  end

  % use plane specific maximum allowable frequency if requested
  if tiling.maximumFrequency == Inf,
    maximumFrequency = maximumAllowableFrequency;
  end

  % cumulative mismatch across frequency range
  frequencyCumulativeMismatch = log(maximumFrequency / minimumFrequency) * ...
                                sqrt(2 + q^2) / 2;

  % number of frequency rows
  numberOfRows = ceil(frequencyCumulativeMismatch / mismatchStep);

  % insure at least one row
  if numberOfRows == 0,
    numberOfRows = 1;
  end

  % mismatch between neighboring frequency rows
  frequencyMismatchStep = frequencyCumulativeMismatch / numberOfRows;

  % index of frequency rows
  frequencyIndices = 0.5 : numberOfRows - 0.5;

  % vector of frequencies
  frequencies = minimumFrequency * exp((2 / sqrt(2 + q^2)) * ...
                                       frequencyIndices * ...
                                       frequencyMismatchStep);

  % ratio between successive frequencies
  frequencyRatio = exp((2 / sqrt(2 + q^2)) * frequencyMismatchStep);

  % project frequency vector onto realizable frequencies
  frequencies = round(frequencies / minimumFrequencyStep) .* ...
                minimumFrequencyStep;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                    create Q transform plane structure                      %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % insert Q of plane into Q plane structure
  tiling.planes{plane}.q = q;

  % insert minimum search frequency of plane into Q plane structure
  tiling.planes{plane}.minimumFrequency = minimumFrequency;

  % insert maximum search frequency of plane into Q plane structure
  tiling.planes{plane}.maximumFrequency = maximumFrequency;

  % insert plane normalization factor into Q plane structure
  tiling.planes{plane}.normalization = planeNormalization;

  % insert frequency vector into Q plane structure
  tiling.planes{plane}.frequencies = frequencies;

  % insert number of frequency rows into Q plane structure
  tiling.planes{plane}.numberOfRows = numberOfRows;

  % initialize cell array of frequency rows into Q plane structure
  tiling.planes{plane}.rows = cell(1, numberOfRows);

  % initialize number of tiles in plane counter
  tiling.planes{plane}.numberOfTiles = 0;

  % initialize number of independent tiles in plane counter
  tiling.planes{plane}.numberOfIndependents = 0;

  % initialize number of flops in plane counter
  tiling.planes{plane}.numberOfFlops = 0;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                      begin loop over frequency rows                        %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % begin loop over frequency rows
  for row = 1 : numberOfRows,

    % extract frequency of row from frequency vector
    frequency = frequencies(row);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                      determine tile properties                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % bandwidth for coincidence testing
    bandwidth = 2 * sqrt(pi) * frequency / q;

    % duration for coincidence testing
    duration = 1 / bandwidth;

    % frequency step for integration
    frequencyStep = frequency * (frequencyRatio - 1) / sqrt(frequencyRatio);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                         determine tile times                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % cumulative mismatch across time range
    timeCumulativeMismatch = timeRange * 2 * pi * frequency / q;

    % number of time tiles
    numberOfTiles = 2^nextpow2(timeCumulativeMismatch / mismatchStep);

    % mismatch between neighboring time tiles
    timeMismatchStep = timeCumulativeMismatch / numberOfTiles;

    % index of time tiles
    timeIndices = 0 : numberOfTiles - 1;

    % vector of times
    times = q * timeIndices * timeMismatchStep / (2 * pi * frequency);

    % time step for integration
    timeStep = q * timeMismatchStep / (2 * pi * frequency);

    % project time vector onto realizable times
    % times = round(times / minimumTimeStep) .* minimumTimeStep;

    % number of flops to compute row
    numberOfFlops = numberOfTiles * log(numberOfTiles);

    % number of independent tiles in row
    numberOfIndependents = 1 + timeCumulativeMismatch;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                           generate window                                %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % half length of window in samples
    halfWindowLength = floor((frequency / qPrime) / minimumFrequencyStep);

    % full length of window in samples
    windowLength = 2 * halfWindowLength + 1;

    % sample index vector for window construction
    windowIndices = -halfWindowLength : halfWindowLength;

    % frequency vector for window construction
    windowFrequencies = windowIndices * minimumFrequencyStep;

    % dimensionless frequency vector for window construction
    windowArgument = windowFrequencies * qPrime / frequency;

    % bi square window function
    window = (1 - windowArgument.^2).^2;

    % row normalization factor
    rowNormalization = sqrt((315 * qPrime) / (128 * frequency));

    % inverse fft normalization factor
    ifftNormalization = numberOfTiles / numberOfSamples;

    % normalize window
    % window = window * ifftNormalization * rowNormalization * ...
    %          planeNormalization;
    window = window * ifftNormalization * rowNormalization;

    % number of zeros to append to windowed data
    zeroPadLength = numberOfTiles - windowLength;

    % vector of data indices to inverse fourier transform
    dataIndices = round(1 + frequency / minimumFrequencyStep + windowIndices);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                   create Q transform row structure                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % insert frequency of row into frequency row structure
    tiling.planes{plane}.rows{row}.frequency = frequency;

    % insert duration into frequency row structure
    tiling.planes{plane}.rows{row}.duration = duration;

    % insert bandwidth into frequency row structure
    tiling.planes{plane}.rows{row}.bandwidth = bandwidth;

    % insert time step into frequency row structure
    tiling.planes{plane}.rows{row}.timeStep = timeStep;

    % insert frequency step into frequency row structure
    tiling.planes{plane}.rows{row}.frequencyStep = frequencyStep;

    % insert time vector into frequency row structure
    % tiling.planes{plane}.rows{row}.times = times;
    % THIS FIELD HAS BEEN REMOVED DUE TO EXCESSVE MEMORY USE
    % WHERE REQUIRED, COMPUTE IT BY:
    %     times = (0 :  tiling.planes{plane}.rows{row}.numberOfTiles - 1) ...
    %       * tiling.planes{plane}.rows{row}.timeStep;

    % insert window vector into frequency row structure
    tiling.planes{plane}.rows{row}.window = window;

    % insert window vector into frequency row structure
    tiling.planes{plane}.rows{row}.zeroPadLength = zeroPadLength;

    % insert data index vector into frequency row structure
    tiling.planes{plane}.rows{row}.dataIndices = dataIndices;

    % insert number of time tiles into frequency row structure
    tiling.planes{plane}.rows{row}.numberOfTiles = numberOfTiles;

    % insert number of independent tiles in row into frequency row structure
    tiling.planes{plane}.rows{row}.numberOfIndependents = numberOfIndependents;

    % insert number of flops to compute row into frequency row structure
    tiling.planes{plane}.rows{row}.numberOfFlops = numberOfFlops;

    % increment number of tiles in plane counter
    tiling.planes{plane}.numberOfTiles = ...
        tiling.planes{plane}.numberOfTiles + numberOfTiles;

    % increment number of indepedent tiles in plane counter
    tiling.planes{plane}.numberOfIndependents = ...
        tiling.planes{plane}.numberOfIndependents + numberOfIndependents * ...
        (1 + frequencyCumulativeMismatch) / numberOfRows;

    % increment number of flops in plane counter
    tiling.planes{plane}.numberOfFlops = ...
        tiling.planes{plane}.numberOfFlops + numberOfFlops;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                       end loop over frequency rows                         %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % end loop over frequency rows
  end

  % increment total number of tiles counter
  tiling.numberOfTiles = tiling.numberOfTiles + ...
      tiling.planes{plane}.numberOfTiles;

  % increment total number of independent tiles counter
  tiling.numberOfIndependents = tiling.numberOfIndependents + ...
      tiling.planes{plane}.numberOfIndependents * ...
      (1 + qCumulativeMismatch) / numberOfPlanes;

  % increment total number of flops counter
  tiling.numberOfFlops = tiling.numberOfFlops + ...
      tiling.planes{plane}.numberOfFlops;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            end loop over Q planes                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end loop over Q planes
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         determine filter properties                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% default high pass filter cutoff frequency
defaultHighPassCutoff = Inf;
for plane = 1 : tiling.numberOfPlanes,
  defaultHighPassCutoff = min(defaultHighPassCutoff, ...
                              tiling.planes{plane}.minimumFrequency);
end

% default low pass filter cutoff frequency
defaultLowPassCutoff = 0;
for plane = 1 : tiling.numberOfPlanes,
  defaultLowPassCutoff = max(defaultLowPassCutoff, ...
                             tiling.planes{plane}.maximumFrequency);
end

% default whitening filter duration
defaultWhiteningDuration = 0;
for plane = 1 : tiling.numberOfPlanes,
  defaultWhiteningDuration = max(defaultWhiteningDuration, ...
                                 tiling.planes{plane}.q / ...
                                 (2 * tiling.planes{plane}.minimumFrequency));
end
% put duration as an integer power of 2 of seconds
defaultWhiteningDuration = 2^round(log2(defaultWhiteningDuration));


% high pass filter cutoff frequency
if isempty(highPassCutoff),
  tiling.highPassCutoff = defaultHighPassCutoff;
else
  tiling.highPassCutoff = highPassCutoff;
end

% low pass filter cutoff frequency
if isempty(lowPassCutoff),
  tiling.lowPassCutoff = defaultLowPassCutoff;
else
  tiling.lowPassCutoff = lowPassCutoff;
end

% whitening filter duration
if isempty(whiteningDuration),
  tiling.whiteningDuration = defaultWhiteningDuration;
else
  tiling.whiteningDuration = whiteningDuration;
end

% estimated duration of filter transients to supress
tiling.transientDuration = transientFactor * tiling.whiteningDuration;

% test for insufficient data
if (2 * tiling.transientDuration) >= tiling.duration,
  error('duration of filter transients equals or exceeds data duration');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          return Q transform tiling                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
