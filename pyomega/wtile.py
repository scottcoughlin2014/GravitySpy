###############################################################################
##########################                             ########################
##########################      Func: wtile            ########################
##########################                             ########################
###############################################################################

def wtile(blockTime, searchQRange, searchFrequencyRange, sampleFrequency, \
                        searchMaximumEnergyLoss, highPassCutoff, lowPassCutoff, \
                        whiteningDuration, transientFactor);

# extract minimum and maximum Q from Q range
minimumQ = searchQRange[0];
maximumQ = searchQRange[1];

# extract minimum and maximum frequency from frequency range
minimumFrequency = searchFrequencyRange[0];
maximumFrequency = searchFrequencyRange[1];

################################################################################
#                          compute derived parameters                          #
################################################################################

# nyquist frequency
nyquistFrequency = sampleFrequency / 2;

# maximum mismatch between neighboring tiles
mismatchStep = 2 * np.sqrt(searchMaximumEnergyLoss / 3);

# maximum possible time resolution
minimumTimeStep = 1 / sampleFrequency;

# maximum possible frequency resolution
minimumFrequencyStep = 1 / blockTime;

# conversion factor from Q prime to true Q
qPrimeToQ = np.sqrt(11);

# total number of samples in input data
numberOfSamples = blockTime * sampleFrequency;

################################################################################
#                       determine parameter constraints                        #
################################################################################

# minimum allowable Q prime to prevent window aliasing at zero frequency
minimumAllowableQPrime = 1.0;

# minimum allowable Q to avoid window aliasing at zero frequency
minimumAllowableQ = minimumAllowableQPrime * qPrimeToQ;

# reasonable number of statistically independent tiles in a frequency row
minimumAllowableIndependents = 50;

# maximum allowable mismatch parameter for reasonable performance
maximumAllowableMismatch = 0.5;

################################################################################
#                             validate parameters                              #
################################################################################

# check for valid time range
if blockTime < 0:
  print('negative time range')
  sys.exit()

# check for valid Q range
if minimumQ > maximumQ:
  print('minimum Q is larger than maximum Q')
  sys.exit()

# check for valid frequency range
if minimumFrequency > maximumFrequency:
  print('minimum frequency exceeds maximum frequency')
  sys.exit()

# check for valid minimum Q
if minimumQ < minimumAllowableQ:
  print('minimum Q less than {0}'.format(minimumAllowableQ))
  sys.exit()

# check for reasonable maximum mismatch parameter
if searchMaximumEnergyLoss > maximumAllowableMismatch:
  print('maximum mismatch exceeds {0}'.format(maximumAllowableMismatch))
  sys.exit()

# check for integer power of two data length
if not np.mod(np.log(blockTime * sampleFrequency) / np.log(2), 1) == 0:
  print('data length is not an integer power of two')
  sys.exit()

################################################################################
#                              determine Q planes                              #
################################################################################

# cumulative mismatch across Q range
qCumulativeMismatch = np.log(maximumQ / minimumQ) / np.sqrt(2);

# number of Q planes
numberOfPlanes = np.ceil(qCumulativeMismatch / mismatchStep);

# insure at least one plane
if numberOfPlanes == 0:
  numberOfPlanes = 1

# mismatch between neighboring planes
qMismatchStep = qCumulativeMismatch / numberOfPlanes;

# index of Q planes
qIndices = np.linspace(0.5,numberOfPlanes - 0.5,numberOfPlanes)

# vector of Qs
qs = minimumQ * np.exp(np.sqrt(2) * qIndices * qMismatchStep);

################################################################################
#                             validate frequencies                             #
################################################################################

# minimum allowable frequency to provide sufficient statistics
minimumAllowableFrequency = minimumAllowableIndependents * max(qs) / \
                            (2 * np.pi * blockTime);

# maximum allowable frequency to avoid window aliasing
maximumAllowableFrequency = nyquistFrequency / (1 + qPrimeToQ / min(qs));

# check for valid minimum frequency
if (not minimumFrequency == 0) and \
   (minimumFrequency < minimumAllowableFrequency):
   print('requested minimum frequency of {0} Hz  \
        less than minimum allowable frequency of {1} Hz').format(\
        minimumFrequency, minimumAllowableFrequency)
   sys.exit()

# check for valid maximum frequency
if (not np.isinf(maximumFrequency)) and \
   (maximumFrequency > maximumAllowableFrequency):
   print('requested maximum frequency of {0} Hz  \
        less than maximum allowable frequency of {1} Hz').format(\
        maximumFrequency, maximumAllowableFrequency)
   sys.exit()

################################################################################
#                     create Q transform tiling structure                      #
################################################################################

tiling = np.array
# structure type identifier
tiling.id = 'Discrete Q-transform tile structure';

# insert duration into tiling structure
tiling.duration = blockTime;

# insert minimum Q into tiling structure
tiling.minimumQ = minimumQ;

# insert maximum Q into tiling structure
tiling.maximumQ = maximumQ;

# insert minimum frequency into tiling structure
tiling.minimumFrequency = minimumFrequency;

# insert maximum frequency into tiling structure
tiling.maximumFrequency = maximumFrequency;

# insert sample frequency into tiling structure
tiling.sampleFrequency = sampleFrequency;

# insert maximum loss due to mismatch into tiling structure
tiling.searchMaximumEnergyLoss = searchMaximumEnergyLoss;

# insert Q vector into tiling structure
tiling.qs = qs;

# insert number of Q planes into tiling structure
tiling.numberOfPlanes = numberOfPlanes;

# initialize cell array of Q plans in tiling structure
tiling.planes = cell(1, numberOfPlanes);

# initialize total number of tiles counter
tiling.numberOfTiles = 0;

# initialize total number of independent tiles counter
tiling.numberOfIndependents = 0;

# initialize total number of flops counter
tiling.numberOfFlops = numberOfSamples * log(numberOfSamples);

################################################################################
#                           begin loop over Q planes                           #
################################################################################

# begin loop over Q planes
for plane = 1 : numberOfPlanes,

  # extract Q of plane from Q vector
  q = qs(plane);

  ##############################################################################
  #                        determine plane properties                          #
  ##############################################################################

  # find Q prime for the plane
  qPrime = q / qPrimeToQ;

  # for large qPrime
  if qPrime > 10,

    # use asymptotic value of planeNormalization
    planeNormalization = 1;

  # otherwise
  else

    # polynomial coefficients for plane normalization factor
    coefficients = [+ 1 * log((qPrime + 1) / (qPrime - 1)); - 2; ...
                    - 4 * log((qPrime + 1) / (qPrime - 1)); + 22 / 3; ...
                    + 6 * log((qPrime + 1) / (qPrime - 1)); - 146 / 15; ...
                    - 4 * log((qPrime + 1) / (qPrime - 1)); + 186 / 35; ...
                    + 1 * log((qPrime + 1) / (qPrime - 1));];

    # plane normalization factor
    planeNormalization = sqrt(256 / (315 * qPrime * ...
                                     polyval(coefficients, qPrime)));

  # continue
  end

  ##############################################################################
  #                         determine frequency rows                           #
  ##############################################################################

  # plane specific minimum allowable frequency to provide sufficient statistics
  minimumAllowableFrequency = minimumAllowableIndependents * q / ...
                              (2 * pi * tiling.duration);

  # plane specific maximum allowable frequency to avoid window aliasing
  maximumAllowableFrequency = nyquistFrequency / (1 + qPrimeToQ / q);

  # use plane specific minimum allowable frequency if requested
  if tiling.minimumFrequency == 0,
    minimumFrequency = minimumAllowableFrequency;
  end

  # use plane specific maximum allowable frequency if requested
  if tiling.maximumFrequency == Inf,
    maximumFrequency = maximumAllowableFrequency;
  end

  # cumulative mismatch across frequency range
  frequencyCumulativeMismatch = log(maximumFrequency / minimumFrequency) * ...
                                sqrt(2 + q^2) / 2;

  # number of frequency rows
  numberOfRows = ceil(frequencyCumulativeMismatch / mismatchStep);

  # insure at least one row
  if numberOfRows == 0,
    numberOfRows = 1;
  end

  # mismatch between neighboring frequency rows
  frequencyMismatchStep = frequencyCumulativeMismatch / numberOfRows;

  # index of frequency rows
  frequencyIndices = 0.5 : numberOfRows - 0.5;

  # vector of frequencies
  frequencies = minimumFrequency * exp((2 / sqrt(2 + q^2)) * ...
                                       frequencyIndices * ...
                                       frequencyMismatchStep);

  # ratio between successive frequencies
  frequencyRatio = exp((2 / sqrt(2 + q^2)) * frequencyMismatchStep);

  # project frequency vector onto realizable frequencies
  frequencies = round(frequencies / minimumFrequencyStep) .* ...
                minimumFrequencyStep;

  ##############################################################################
  #                    create Q transform plane structure                      #
  ##############################################################################

  # insert Q of plane into Q plane structure
  tiling.planes{plane}.q = q;

  # insert minimum search frequency of plane into Q plane structure
  tiling.planes{plane}.minimumFrequency = minimumFrequency;

  # insert maximum search frequency of plane into Q plane structure
  tiling.planes{plane}.maximumFrequency = maximumFrequency;

  # insert plane normalization factor into Q plane structure
  tiling.planes{plane}.normalization = planeNormalization;

  # insert frequency vector into Q plane structure
  tiling.planes{plane}.frequencies = frequencies;

  # insert number of frequency rows into Q plane structure
  tiling.planes{plane}.numberOfRows = numberOfRows;

  # initialize cell array of frequency rows into Q plane structure
  tiling.planes{plane}.rows = cell(1, numberOfRows);

  # initialize number of tiles in plane counter
  tiling.planes{plane}.numberOfTiles = 0;

  # initialize number of independent tiles in plane counter
  tiling.planes{plane}.numberOfIndependents = 0;

  # initialize number of flops in plane counter
  tiling.planes{plane}.numberOfFlops = 0;

  ##############################################################################
  #                      begin loop over frequency rows                        #
  ##############################################################################

  # begin loop over frequency rows
  for row = 1 : numberOfRows,

    # extract frequency of row from frequency vector
    frequency = frequencies(row);

    ############################################################################
    #                      determine tile properties                           #
    ############################################################################

    # bandwidth for coincidence testing
    bandwidth = 2 * sqrt(pi) * frequency / q;

    # duration for coincidence testing
    duration = 1 / bandwidth;

    # frequency step for integration
    frequencyStep = frequency * (frequencyRatio - 1) / sqrt(frequencyRatio);

    ############################################################################
    #                         determine tile times                             #
    ############################################################################

    # cumulative mismatch across time range
    timeCumulativeMismatch = blockTime * 2 * pi * frequency / q;

    # number of time tiles
    numberOfTiles = 2^nextpow2(timeCumulativeMismatch / mismatchStep);

    # mismatch between neighboring time tiles
    timeMismatchStep = timeCumulativeMismatch / numberOfTiles;

    # index of time tiles
    timeIndices = 0 : numberOfTiles - 1;

    # vector of times
    times = q * timeIndices * timeMismatchStep / (2 * pi * frequency);

    # time step for integration
    timeStep = q * timeMismatchStep / (2 * pi * frequency);

    # project time vector onto realizable times
    # times = round(times / minimumTimeStep) .* minimumTimeStep;

    # number of flops to compute row
    numberOfFlops = numberOfTiles * log(numberOfTiles);

    # number of independent tiles in row
    numberOfIndependents = 1 + timeCumulativeMismatch;

    ############################################################################
    #                           generate window                                #
    ############################################################################

    # half length of window in samples
    halfWindowLength = floor((frequency / qPrime) / minimumFrequencyStep);

    # full length of window in samples
    windowLength = 2 * halfWindowLength + 1;

    # sample index vector for window construction
    windowIndices = -halfWindowLength : halfWindowLength;

    # frequency vector for window construction
    windowFrequencies = windowIndices * minimumFrequencyStep;

    # dimensionless frequency vector for window construction
    windowArgument = windowFrequencies * qPrime / frequency;

    # bi square window function
    window = (1 - windowArgument.^2).^2;

    # row normalization factor
    rowNormalization = sqrt((315 * qPrime) / (128 * frequency));

    # inverse fft normalization factor
    ifftNormalization = numberOfTiles / numberOfSamples;

    # normalize window
    # window = window * ifftNormalization * rowNormalization * ...
    #          planeNormalization;
    window = window * ifftNormalization * rowNormalization;

    # number of zeros to append to windowed data
    zeroPadLength = numberOfTiles - windowLength;

    # vector of data indices to inverse fourier transform
    dataIndices = round(1 + frequency / minimumFrequencyStep + windowIndices);

    ############################################################################
    #                   create Q transform row structure                       #
    ############################################################################

    # insert frequency of row into frequency row structure
    tiling.planes{plane}.rows{row}.frequency = frequency;

    # insert duration into frequency row structure
    tiling.planes{plane}.rows{row}.duration = duration;

    # insert bandwidth into frequency row structure
    tiling.planes{plane}.rows{row}.bandwidth = bandwidth;

    # insert time step into frequency row structure
    tiling.planes{plane}.rows{row}.timeStep = timeStep;

    # insert frequency step into frequency row structure
    tiling.planes{plane}.rows{row}.frequencyStep = frequencyStep;

    # insert time vector into frequency row structure
    # tiling.planes{plane}.rows{row}.times = times;
    # THIS FIELD HAS BEEN REMOVED DUE TO EXCESSVE MEMORY USE
    # WHERE REQUIRED, COMPUTE IT BY:
    #     times = (0 :  tiling.planes{plane}.rows{row}.numberOfTiles - 1) ...
    #       * tiling.planes{plane}.rows{row}.timeStep;

    # insert window vector into frequency row structure
    tiling.planes{plane}.rows{row}.window = window;

    # insert window vector into frequency row structure
    tiling.planes{plane}.rows{row}.zeroPadLength = zeroPadLength;

    # insert data index vector into frequency row structure
    tiling.planes{plane}.rows{row}.dataIndices = dataIndices;

    # insert number of time tiles into frequency row structure
    tiling.planes{plane}.rows{row}.numberOfTiles = numberOfTiles;

    # insert number of independent tiles in row into frequency row structure
    tiling.planes{plane}.rows{row}.numberOfIndependents = numberOfIndependents;

    # insert number of flops to compute row into frequency row structure
    tiling.planes{plane}.rows{row}.numberOfFlops = numberOfFlops;

    # increment number of tiles in plane counter
    tiling.planes{plane}.numberOfTiles = ...
        tiling.planes{plane}.numberOfTiles + numberOfTiles;

    # increment number of indepedent tiles in plane counter
    tiling.planes{plane}.numberOfIndependents = ...
        tiling.planes{plane}.numberOfIndependents + numberOfIndependents * ...
        (1 + frequencyCumulativeMismatch) / numberOfRows;

    # increment number of flops in plane counter
    tiling.planes{plane}.numberOfFlops = ...
        tiling.planes{plane}.numberOfFlops + numberOfFlops;

  ##############################################################################
  #                       end loop over frequency rows                         #
  ##############################################################################

  # end loop over frequency rows
  end

  # increment total number of tiles counter
  tiling.numberOfTiles = tiling.numberOfTiles + ...
      tiling.planes{plane}.numberOfTiles;

  # increment total number of independent tiles counter
  tiling.numberOfIndependents = tiling.numberOfIndependents + ...
      tiling.planes{plane}.numberOfIndependents * ...
      (1 + qCumulativeMismatch) / numberOfPlanes;

  # increment total number of flops counter
  tiling.numberOfFlops = tiling.numberOfFlops + ...
      tiling.planes{plane}.numberOfFlops;

################################################################################
#                            end loop over Q planes                            #
################################################################################

# end loop over Q planes
end

################################################################################
#                         determine filter properties                          #
################################################################################

# default high pass filter cutoff frequency
defaultHighPassCutoff = Inf;
for plane = 1 : tiling.numberOfPlanes,
  defaultHighPassCutoff = min(defaultHighPassCutoff, ...
                              tiling.planes{plane}.minimumFrequency);
end

# default low pass filter cutoff frequency
defaultLowPassCutoff = 0;
for plane = 1 : tiling.numberOfPlanes,
  defaultLowPassCutoff = max(defaultLowPassCutoff, ...
                             tiling.planes{plane}.maximumFrequency);
end

# default whitening filter duration
defaultWhiteningDuration = 0;
for plane = 1 : tiling.numberOfPlanes,
  defaultWhiteningDuration = max(defaultWhiteningDuration, ...
                                 tiling.planes{plane}.q / ...
                                 (2 * tiling.planes{plane}.minimumFrequency));
end
# put duration as an integer power of 2 of seconds
defaultWhiteningDuration = 2^round(log2(defaultWhiteningDuration));


# high pass filter cutoff frequency
if isempty(highPassCutoff),
  tiling.highPassCutoff = defaultHighPassCutoff;
else
  tiling.highPassCutoff = highPassCutoff;
end

# low pass filter cutoff frequency
if isempty(lowPassCutoff),
  tiling.lowPassCutoff = defaultLowPassCutoff;
else
  tiling.lowPassCutoff = lowPassCutoff;
end

# whitening filter duration
if isempty(whiteningDuration),
  tiling.whiteningDuration = defaultWhiteningDuration;
else
  tiling.whiteningDuration = whiteningDuration;
end

# estimated duration of filter transients to supress
tiling.transientDuration = transientFactor * tiling.whiteningDuration;

# test for insufficient data
if (2 * tiling.transientDuration) >= tiling.duration,
  error('duration of filter transients equals or exceeds data duration');
end

################################################################################
#                          return Q transform tiling                           #
################################################################################

# return to calling function
return tiling;

