# ---- Import standard modules to the python path.
from __future__ import division
import sys, os, shutil, math, random, copy, getopt, re, string, popen2, time
import ConfigParser, glob, operator, optparse, json
from glue import segments
from glue import segmentsUtils
from glue import pipeline
from glue.lal import CacheEntry
import numpy as np
from gwpy.timeseries import TimeSeries

###############################################################################
##########################                             ########################
##########################   Func: parse_commandline   ########################
##########################                             ########################
###############################################################################
# Definite Command line arguments here

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--inifile", help="Name of ini file of params")
    parser.add_option("--eventTime", type=float,help="Trigger time of the glitch")
    parser.add_option("--uniqueID", type=float,help="Trigger time of the glitch")
    parser.add_option("--outDir", help="Outdir of omega scan and omega scan webpage (i.e. you html directory)")
    parser.add_option("--NSDF", action="store_true", default=False,help="No framecache file available want to use NSDF server")
    opts, args = parser.parse_args()


    return opts

###############################################################################
##########################                             ########################
##########################   Func: nextpow2            ########################
##########################                             ########################
###############################################################################

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

###############################################################################
##########################                             ########################
##########################      Func: wtile            ########################
##########################                             ########################
###############################################################################

def wtile(blockTime, searchQRange, searchFrequencyRange, sampleFrequency, \
                        searchMaximumEnergyLoss, highPassCutoff, lowPassCutoff, \
                        whiteningDuration, transientFactor):
    # extract minimum and maximum Q from Q range
    minimumQ = searchQRange[0];
    maximumQ = searchQRange[1];

    # extract minimum and maximum frequency from frequency range
    minimumFrequency = searchFrequencyRange[0];
    maximumFrequency = searchFrequencyRange[1];

    ###########################################################################
    #                          compute derived parameters                     #
    ###########################################################################

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

    ############################################################################
    #                       determine parameter constraints                    #
    ############################################################################

    # minimum allowable Q prime to prevent window aliasing at zero frequency
    minimumAllowableQPrime = 1.0;

    # minimum allowable Q to avoid window aliasing at zero frequency
    minimumAllowableQ = minimumAllowableQPrime * qPrimeToQ;

    # reasonable number of statistically independent tiles in a frequency row
    minimumAllowableIndependents = 50;

    # maximum allowable mismatch parameter for reasonable performance
    maximumAllowableMismatch = 0.5;

    ############################################################################
    #                             validate parameters                          #
    ############################################################################

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

    ############################################################################
    #                          determine Q planes                              #
    ############################################################################

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


    ############################################################################
    #                         validate frequencies                             #
    ############################################################################

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

    tiling = {}
    tiling["generalparams"] = {}
    tiling["generalparams"]["duration"] = blockTime;
    tiling["generalparams"]["minimumQ"] = minimumQ;
    tiling["generalparams"]["maximumQ"] = maximumQ;
    tiling["generalparams"]["minimumFrequency"] = minimumFrequency;
    tiling["generalparams"]["maximumFrequency"] = maximumFrequency;
    tiling["generalparams"]["sampleFrequency"] = sampleFrequency;
    tiling["generalparams"]["searchMaximumEnergyLoss"] = searchMaximumEnergyLoss;
    tiling["generalparams"]["qs"] = qs;
    tiling["generalparams"]["numberOfPlanes"] = numberOfPlanes;
    tiling["generalparams"]["numberOfTiles"] = 0;
    tiling["generalparams"]["numberOfIndependents"] = 0;
    tiling["generalparams"]["numberOfFlops"] = numberOfSamples * np.log(numberOfSamples);

    for plane in np.arange(0,numberOfPlanes):

	q = qs[plane]

  	#######################################################################
  	#                 determine plane properties                          #
	#######################################################################

	# find Q prime for the plane
  	qPrime = q / qPrimeToQ;

  	# for large qPrime
  	if qPrime > 10:

    	    # use asymptotic value of planeNormalization
    	    planeNormalization = 1
  	else:

    	    # polynomial coefficients for plane normalization factor
	    coefficients = [\
		          np.log((qPrime + 1) / (qPrime - 1)), -2,\
                    - 4 * np.log((qPrime + 1) / (qPrime - 1)), 22 / 3,\
                      6 * np.log((qPrime + 1) / (qPrime - 1)), - 146 / 15,\
                    - 4 * np.log((qPrime + 1) / (qPrime - 1)), 186 / 35,\
                          np.log((qPrime + 1) / (qPrime - 1))];
	    # Cast as an array
	    coefficients = np.asarray(coefficients)

    	    # plane normalization factor
    	    planeNormalization = np.sqrt(256 / (315 * qPrime * \
                                     np.polyval(coefficients, qPrime)));

  	###################################################################
	#                   determine frequency rows                      #
  	###################################################################

  	# plane specific minimum allowable frequency to provide sufficient statistics
 	minimumAllowableFrequency = minimumAllowableIndependents * q / \
                              (2 * np.pi * tiling['generalparams']['duration']);

	# plane specific maximum allowable frequency to avoid window aliasing
	maximumAllowableFrequency = nyquistFrequency / (1 + qPrimeToQ / q);

	# use plane specific minimum allowable frequency if requested
	if tiling['generalparams']['minimumFrequency'] == 0:
	    minimumFrequency = minimumAllowableFrequency;

	# use plane specific maximum allowable frequency if requested
	if np.isinf(tiling['generalparams']['maximumFrequency']):
	    maximumFrequency = maximumAllowableFrequency;

	# cumulative mismatch across frequency range
	frequencyCumulativeMismatch = np.log(maximumFrequency / \
	    minimumFrequency) * np.sqrt(2 + q**2) / 2;

	# number of frequency rows
	numberOfRows = np.ceil(frequencyCumulativeMismatch / mismatchStep);

	# insure at least one row
	if numberOfRows == 0:
	    numberOfRows = 1

	# mismatch between neighboring frequency rows
	frequencyMismatchStep = frequencyCumulativeMismatch / numberOfRows;

	# index of frequency rows
	frequencyIndices = np.linspace(0.5,numberOfRows - 0.5,numberOfRows);

	# vector of frequencies
	frequencies = minimumFrequency * np.exp((2 / np.sqrt(2 + q**2)) * \
                                       frequencyIndices * \
                                       frequencyMismatchStep);

	# ratio between successive frequencies
	frequencyRatio = np.exp((2 / np.sqrt(2 + q**2)) * frequencyMismatchStep);

	# project frequency vector onto realizable frequencies
	frequencies = np.round(frequencies / minimumFrequencyStep) * \
                minimumFrequencyStep;

	#######################################################################
	#             create Q transform plane structure                      #
	#######################################################################

	planestr = 'planes' +str(plane)
	tiling[planestr] = {}
	tiling["generalparams"]["duration"] = blockTime;
	# insert Q of plane into Q plane structure
	tiling[planestr]['q'] = q;

	# insert minimum search frequency of plane into Q plane structure
	tiling[planestr]['minimumFrequency'] = minimumFrequency;

	# insert maximum search frequency of plane into Q plane structure
	tiling[planestr]['maximumFrequency'] = maximumFrequency;

	# insert plane normalization factor into Q plane structure
	tiling[planestr]['normalization'] = planeNormalization;

	# insert frequency vector into Q plane structure
	tiling[planestr]['frequencies'] = frequencies;

	# insert number of frequency rows into Q plane structure
	tiling[planestr]['numberOfRows'] = numberOfRows;

	# initialize cell array of frequency rows into Q plane structure
	for i in np.arange(0,numberOfRows):
	    rowstr = 'row' + str(i)
	    tiling[planestr][rowstr] = {};

	# initialize number of tiles in plane counter
	tiling[planestr]['numberOfTiles'] = 0;

	# initialize number of independent tiles in plane counter
	tiling[planestr]['numberOfIndependents'] = 0;

	# initialize number of flops in plane counter
	tiling[planestr]['numberOfFlops'] = 0;

	#######################################################################
	#               begin loop over frequency rows                        #
	#######################################################################

	for row in np.arange(0,numberOfRows):

	    rowstr = 'row' + str(row)
    	    # extract frequency of row from frequency vector
    	    frequency = frequencies[row];

    	    ####################################################################
    	    #              determine tile properties                           #
    	    ####################################################################

    	    # bandwidth for coincidence testing
    	    bandwidth = 2 * np.sqrt(np.pi) * frequency / q;

   	    # duration for coincidence testing
    	    duration = 1 / bandwidth;

    	    # frequency step for integration
    	    frequencyStep = frequency * (frequencyRatio - 1) / np.sqrt(frequencyRatio);

    	    ####################################################################
    	    #                 determine tile times                             #
    	    ####################################################################

    	    # cumulative mismatch across time range
    	    timeCumulativeMismatch = blockTime * 2 * np.pi * frequency / q;

    	    # number of time tiles
    	    numberOfTiles = nextpow2(timeCumulativeMismatch / mismatchStep);

    	    # mismatch between neighboring time tiles
    	    timeMismatchStep = timeCumulativeMismatch / numberOfTiles;

    	    # index of time tiles
    	    timeIndices = np.arange(0,numberOfTiles);

    	    # vector of times
    	    times = q * timeIndices * timeMismatchStep / (2 * np.pi * frequency);

    	    # time step for integration
    	    timeStep = q * timeMismatchStep / (2 * np.pi * frequency);

    	    # number of flops to compute row
    	    numberOfFlops = numberOfTiles * np.log(numberOfTiles);

    	    # number of independent tiles in row
    	    numberOfIndependents = 1 + timeCumulativeMismatch;

    	    ####################################################################
    	    #                   generate window                                #
    	    ####################################################################

    	    # half length of window in samples
    	    halfWindowLength = np.floor((frequency / qPrime) / minimumFrequencyStep);

    	    # full length of window in samples
    	    windowLength = 2 * halfWindowLength + 1;

    	    # sample index vector for window construction
    	    windowIndices = np.arange(-halfWindowLength,halfWindowLength+1);

    	    # frequency vector for window construction
    	    windowFrequencies = windowIndices * minimumFrequencyStep;

    	    # dimensionless frequency vector for window construction
    	    windowArgument = windowFrequencies * qPrime / frequency;

    	    # bi square window function
     	    window = (1 - windowArgument**2)**2;

   	    # row normalization factor
    	    rowNormalization = np.sqrt((315 * qPrime) / (128 * frequency));

    	    # inverse fft normalization factor
    	    ifftNormalization = numberOfTiles / numberOfSamples;

    	    # normalize window
    	    window = window * ifftNormalization * rowNormalization;

    	    # number of zeros to append to windowed data
   	    zeroPadLength = numberOfTiles - windowLength;

    	    # vector of data indices to inverse fourier transform
    	    dataIndices = np.round(1 + frequency / minimumFrequencyStep + windowIndices);	
	    ####################################################################
	    #           create Q transform row structure                       #
	    ####################################################################

	    # insert frequency of row into frequency row structure
            tiling[planestr][rowstr]['frequency'] = frequency;

    	    # insert duration into frequency row structure
    	    tiling[planestr][rowstr]['duration'] = duration;

    	    # insert bandwidth into frequency row structure
    	    tiling[planestr][rowstr]['bandwidth'] = bandwidth;

    	    # insert time step into frequency row structure
    	    tiling[planestr][rowstr]['timeStep'] = timeStep;

    	    # insert frequency step into frequency row structure
    	    tiling[planestr][rowstr]['frequencyStep'] = frequencyStep;

    	    # insert window vector into frequency row structure
    	    tiling[planestr][rowstr]['window'] = window;

    	    # insert window vector into frequency row structure
    	    tiling[planestr][rowstr]['zeroPadLength'] = zeroPadLength;

    	    # insert data index vector into frequency row structure
    	    tiling[planestr][rowstr]['dataIndices'] = dataIndices;

    	    # insert number of time tiles into frequency row structure
    	    tiling[planestr][rowstr]['numberOfTiles'] = numberOfTiles;

    	    # insert number of independent tiles in row into frequency row structure
    	    tiling[planestr][rowstr]['numberOfIndependents'] = numberOfIndependents;

    	    # insert number of flops to compute row into frequency row structure
    	    tiling[planestr][rowstr]['numberOfFlops'] = numberOfFlops;

    	    # increment number of tiles in plane counter
    	    tiling[planestr]['numberOfTiles'] = \
        	tiling[planestr]['numberOfTiles'] + numberOfTiles;

    	    # increment number of indepedent tiles in plane counter
    	    tiling[planestr]['numberOfIndependents'] = \
        	tiling[planestr]['numberOfIndependents'] \
			+ numberOfIndependents * \
        		((1 + frequencyCumulativeMismatch) / numberOfRows);

    	    # increment number of flops in plane counter
    	    tiling[planestr]['numberOfFlops'] = \
        	tiling[planestr]['numberOfFlops'] + numberOfFlops;

	    #print tiling[planestr]['numberOfFlops']
  	    ###################################################################
  	    #            end loop over frequency rows                         #
  	    ###################################################################
	
        # increment number of tiles in plane counter
        tiling['generalparams']['numberOfTiles'] = \
             tiling['generalparams']['numberOfTiles']  + tiling[planestr]['numberOfTiles'];

        # increment number of indepedent tiles in plane counter
        tiling['generalparams']['numberOfIndependents'] = \
	    	tiling['generalparams']['numberOfIndependents'] +\
                tiling[planestr]['numberOfIndependents'] *\
                        ((1 + qCumulativeMismatch) / numberOfPlanes);

        # increment number of flops in plane counter
        tiling['generalparams']['numberOfFlops'] = \
                tiling['generalparams']['numberOfFlops'] + tiling[planestr]['numberOfFlops'];

        ########################################################################
        #                        end loop over Q planes                        #
        ########################################################################
    ########################################################################
    #                 determine filter properties                          #
    ########################################################################

    # default high pass filter cutoff frequency
    defaultHighPassCutoff = float('Inf');
    # default low pass filter cutoff frequency
    defaultLowPassCutoff = 0;
    # default whitening filter duration
    defaultWhiteningDuration = 0;
    for plane in np.arange(0,numberOfPlanes):
	planestr = 'planes' + str(plane)
	defaultHighPassCutoff = min(defaultHighPassCutoff, \
                              tiling[planestr]['minimumFrequency']);
	defaultLowPassCutoff = max(defaultLowPassCutoff, \
                             tiling[planestr]['maximumFrequency']);
	defaultWhiteningDuration = max(defaultWhiteningDuration, \
                                 tiling[planestr]['q'] / \
                                 (2 * tiling[planestr]['minimumFrequency']));

    # put duration as an integer power of 2 of seconds
    defaultWhiteningDuration = 2**np.round(np.log2(defaultWhiteningDuration));


    # high pass filter cutoff frequency
    if not highPassCutoff:
	tiling['generalparams']['highPassCutoff'] = defaultHighPassCutoff;
    else:
	tiling['generalparams']['highPassCutoff'] = highPassCutoff;

    # low pass filter cutoff frequency
    if not lowPassCutoff:
	tiling['generalparams']['lowPassCutoff'] = defaultLowPassCutoff;
    else:
	tiling['generalparams']['lowPassCutoff'] = lowPassCutoff;

    # whitening filter duration
    if not whiteningDuration:
	tiling['generalparams']['whiteningDuration'] = defaultWhiteningDuration;
    else:
	tiling['generalparams']['whiteningDuration'] = whiteningDuration;

    # estimated duration of filter transients to supress
    tiling['generalparams']['transientDuration'] = transientFactor * tiling['generalparams']['whiteningDuration'];

    # test for insufficient data
    if (2 * tiling['generalparams']['transientDuration']) >= \
			tiling['generalparams']['duration']:
	error('duration of filter transients equals or exceeds data duration');

    return tiling

###############################################################################
#####################                                  ########################
#####################   medianmeanaveragespectrum      ########################
#####################                                  ########################
###############################################################################
def medianmeanaveragespectrum(data,fs,N,w):

    ###########################################################################
    #   Segment data and FFT.
    ###########################################################################

    # ---- Number of segments (FFTs).
    Ns = 2*length(data)/N-1;  # -- always odd

    # ---- Number of point by which consecutive segments overlap.
    Delta = N/2;

    # ---- Sampling time.
    dt = 1/fs;

    # ---- Enforce unity RMS on the window.
    w = w/mean(w.^2)^0.5;

    # ---- Compute spectrogram of data (array: frequency x time).
    [S, F] = spectrogram(data,w,Delta,N,fs);
    # # ---- S is simply the (complex) FFT.  Square this for the PSD.
    S = real(S).^2 + imag(S).^2;

    # ---- Divide time segments into two sets of non-overlapping segments.
    #      Will compute median PSD on each separately, then average results.
    oddSegs = [1:2:Ns];  
    evenSegs = [2:2:Ns];
    # ---- Note that oddSegs always has 1 more element than evenSegs.  Drop an
    #      element from one so that both contain an odd number of elements. 
    if rem(length(oddSegs),2) == 0,
        oddSegs = oddSegs(2:end);
    else
        evenSegs = evenSegs(2:end);

    Ns_odd = length(oddSegs);
    Ns_even = length(evenSegs);
    # ---- Compute median-based PSD over each set of segments.
    if (Ns_even > 0)
        # ---- Compute median-based PSD over each set of segments.
        S_odd = median(S(:,oddSegs),2) / medianbiasfactor(Ns_odd);
        S_even = median(S(:,evenSegs),2) / medianbiasfactor(Ns_even);
        # ---- Take weighted average of the two median estimates.
        S = (Ns_odd*S_odd + Ns_even*S_even) / (Ns_odd + Ns_even);
    else
    # ---- Have only 1 segment.  No averaging to be done!
    ;
    # ---- Normalize to physical units.
    S = 2/(N*fs)*S;

###############################################################################
#####################                                  ########################
#####################           wtransform             ########################
#####################                                  ########################
###############################################################################

def wtransform(data, tiling, outlierFactor, \
               analysisMode, channelNames, coefficients, coordinate):

    # determine number of channels
    numberOfChannels = length(data);

    # determine required data lengths
    dataLength = tiling['generalparams']['sampleFrequency'] * tiling['generalparams']['blockTime'];
    halfDataLength = dataLength / 2 + 1;

    # validate data length and force row vectors
    if len(data) != halfDataLength:
	sys.exit()

    # determine number of sites
    numberOfSites = 1;

    if length(coordinate) != 2:
	sys.exit()

    #######################################################################
    #                   setup independent analysis                        #
    #######################################################################

    intermediateData = data;
    numberOfIntermediateChannels = numberOfChannels;
    numberOfOutputChannels = numberOfChannels;
    outputChannelNames = channelNames;

    #######################################################################
    #                   Define some variables                             #
    #######################################################################

    numberOfPlanes = tiling['generalparams']['numberOfPlanes']
    #######################################################################
    #             initialize Q transform structures                       #
    #######################################################################

    # create empty cell array of Q transform structures

    transforms = {};

    # begin loop over channels
    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)
        transforms[channelstr] = {}

	# begin loop over Q planes
	for plane in np.arange(0,numberOfPlanes):

	    planestr = plane + str(plane)
	    transforms[channelstr][planestr] = {}

	    for row in tiling[planestr]['numberOfRows']:
		# create empty cell array of frequency row structures
		rowstr = row +str(row)
		transforms[channelstr][planestr][rowstr] = {}


    ############################################################################
    #                       begin loop over Q planes                           #
    ############################################################################

    # begin loop over Q planes
    for plane in np.arange(0,numberOfPlanes):

	########################################################################
	#                begin loop over frequency rows                        #	########################################################################

	# begin loop over frequency rows
	for row in tiling[planestr]['numberOfRows']:
	    ####################################################################
    	    #          extract and window frequency domain data                #
	    ####################################################################

    	    # number of zeros to pad at negative frequencies
    	    leftZeroPadLength = (tiling[channelstr][planestr][rowstr]['zeroPadLength'] - 1) / 2;

    	    # number of zeros to pad at positive frequencies
	    rightZeroPadLength = (tiling[channelstr][planestr][rowstr]['zeroPadLength'] + 1) / 2;

	    # begin loop over intermediate channels
	    for intermediateChannelNumber in np.arange(0,numberOfIntermediateChannels):
		windowedData = {}
		windowedData['intermediateChannelNumber'] ={}
		# extract and window in-band data
		windowedData['intermediateChannelNumber'] = tiling[channelstr][planestr][rowstr]['window'] * \
		intermediateData[tiling[channelstr][planestr][rowstr]['dataIndices']];

		# zero pad windowed data
		windowedData['intermediateChannelNumber'] = [np.zeros(1, leftZeroPadLength) \
                       windowedData['intermediateChannelNumber'] \
                       np.zeros(1, rightZeroPadLength)];

		# reorder indices for fast fourier transform
		windowedData['intermediateChannelNumber'] = \
		windowedData['intermediateChannelNumber']([(end / 2 : end) (1 : end / 2 - 1)]);

    		# end loop over intermediate channels

    	    ################################################################
    	    #        inverse fourier transform windowed data               #
	    ################################################################

	    # begin loop over intermediate channels
	    for intermediateChannelNumber in np.arange(0,numberOfIntermediateChannels):

	    	# complex valued tile coefficients
        	tileCoefficients{intermediateChannelNumber} = ifft(windowedData{intermediateChannelNumber});
		# End loop over intermediate channels

    	    ##################################################################
    	    #              energies directly or indirectly                   #
    	    ##################################################################
      	    # compute energies directly from intermediate data
      	    for channelNumber = in np.arange(0,numberOfIntermediateChannels):
        	energies{channelNumber} = \
          	    real(tileCoefficients{channelNumber}).^2 + \
          	    imag(tileCoefficients{channelNumber}).^2 ;

    	    ####################################################################
   	    #        exclude outliers and filter transients from statistics    #
	    ####################################################################

    	    times = (0 :  tiling.planes{plane}.rows{row}.numberOfTiles - 1) * \
            	tiling.planes{plane}.rows{row}.timeStep;

	    # begin loop over channels
    	    for channelstr in np.arange(0,numberOfChannels):

      	    	# indices of non-transient tiles
      		validIndices{channelstr} = \
          	find((times > \
                	tiling.transientDuration) & \
               	     (times < \
                	tiling.duration - tiling.transientDuration));

      # identify lower and upper quartile energies
      sortedEnergies = ...
          sort(energies{channelstr}(validIndices{channelstr}));
      lowerQuartile{channelstr} = ...
          sortedEnergies(round(0.25 * length(validIndices{channelstr})));
      upperQuartile{channelstr} = ...
          sortedEnergies(round(0.75 * length(validIndices{channelstr})));

      # determine inter quartile range
      interQuartileRange{channelstr} = upperQuartile{channelstr} - ...
                                          lowerQuartile{channelstr};

      # energy threshold of outliers
      outlierThreshold{channelstr} = upperQuartile{channelstr} + ...
          outlierFactor * interQuartileRange{channelstr};

      # indices of non-outlier and non-transient tiles
      validIndices{channelstr} = ...
          find((energies{channelstr} < ...
                outlierThreshold{channelstr}) & ...
               (times > ...
                tiling.transientDuration) & ...
               (times < ...
                tiling.duration - tiling.transientDuration));

    # end loop over channels
    end

    # for reasonable outlier factors,
    if outlierFactor < 100,

      # mean energy correction factor for outlier rejection bias
      meanCorrectionFactor = (4 * 3^outlierFactor - 1) / ...
                             ((4 * 3^outlierFactor - 1) - ...
                             (outlierFactor * log(3) + log(4)));

    # otherwise, for large outlier factors
    else

      # mean energy correction factor for outlier rejection bias
      meanCorrectionFactor = 1;

    # continue
    end

    ############################################################################
    #          determine tile statistics and normalized energies               #
    ############################################################################

    # begin loop over channels
    for channelstr = 1 : numberOfOutputChannels,

      # mean of valid tile energies
      meanEnergy{channelstr} = ...
          mean(energies{channelstr}(validIndices{channelstr}));

      # correct for bias due to outlier rejection
      meanEnergy{channelstr} = meanEnergy{channelstr} * ...
          meanCorrectionFactor;

      # normalized tile energies
      normalizedEnergies{channelstr} = energies{channelstr} / ...
          meanEnergy{channelstr};

    # end loop over channels
    end

    ############################################################################
    #               insert results into transform structure                    #
    ############################################################################

    # begin loop over channels
    for channelstr = 1 : numberOfOutputChannels,

      # insert mean tile energy into frequency row structure
      transforms{channelstr}.planes{plane}.rows{row}.meanEnergy = ...
          meanEnergy{channelstr};

      # insert normalized tile energies into frequency row structure
      transforms{channelstr}.planes{plane}.rows{row}.normalizedEnergies = ...
          normalizedEnergies{channelstr};
      
    # end loop over channels

	########################################################################
  	#                 end loop over frequency rows                         #
  	########################################################################


    ############################################################################
    #                        end loop over Q planes                            #
    ############################################################################

    ############################################################################
    #                return discrete Q transform structure                     #
    ############################################################################

    for channelNumber in np.arange(0,numberOfIntermediateChannels):
    	transforms{channelNumber}.channelName = \
            outputChannelNames{channelNumber};

    return transforms


###############################################################################
##########################                     ################################
##########################      MAIN CODE      ################################
##########################                     ################################
###############################################################################

# Parse commandline arguments

opts = parse_commandline()

################################################################################
#                                   Parse Ini File                             #
################################################################################
# ---- Create configuration-file-parser object and read parameters file.
cp = ConfigParser.ConfigParser()
cp.read(opts.inifile)

# ---- Read needed variables from [parameters] and [channels] sections.
sampleFrequency 	= int(cp.get('parameters','sampleFrequency'));
blockTime 		= int(cp.get('parameters','blockTime'));
searchFrequencyRange 	= json.loads(cp.get('parameters','searchFrequencyRange'));
searchQRange		= json.loads( cp.get('parameters','searchQRange'));
searchMaximumEnergyLoss = float(cp.get('parameters','searchMaximumEnergyLoss'));
searchWindowDuration 	= float(cp.get('parameters','searchWindowDuration'));
plotTimeRanges 		= json.loads(cp.get('parameters','plotTimeRanges'));
plotFrequencyRange 	= json.loads(cp.get('parameters','plotFrequencyRange'));
plotNormalizedERange 	= json.loads(cp.get('parameters','plotNormalizedERange'));
frameCacheFile		= cp.get('channels','frameCacheFile')
frameType		= cp.get('channels','frameType')
channelName		= cp.get('channels','channelName')

################################################################################
#                            hard coded parameters                             #
################################################################################

# search parameters
transientFactor = 2;
outlierFactor = 2.0;

# display parameters
plotHorizontalResolution = 512;


################################################################################
#                           create output directory                            #
################################################################################

# if outputDirectory not specified, make one based on center time
if opts.outDir is None:
	outDir = './scans';
else:
	outDir = opts.outDir

# report status
print('creating event directory');
print('outputDirectory:  {0}'.format(outDir));

# create spectrogram directory
system_call = 'mkdir -p {0}'.format(outDir)
os.system(system_call)

################################################################################
#     Determine if this is a normal omega scan or a Gravityspy omega scan      #
################################################################################
if opts.uniqueID is None:
	IDstring = opts.eventTime
else:
	IDstring = opts.uniqueID

##############################################################################
#               identify statistically significant channels                  #
##############################################################################

# find closest sample time to event time
centerTime = np.floor(opts.eventTime) + \
           np.round((opts.eventTime - np.floor(opts.eventTime)) * \
                 sampleFrequency) / sampleFrequency;
print centerTime
# determine segment start and stop times
startTime = round(centerTime - blockTime / 2);
stopTime = startTime + blockTime;

# Read in the data
if opts.NSDF:
	data = TimeSeries.fetch(channelName,startTime,stopTime)
else:
	data = TimeSeries.read(frameCacheFile,channelName, format='gwf',start=startTime,end=stopTime)

# resample data
data = data.resample(sampleFrequency)
 
# generate search tiling
highPassCutoff = [];
lowPassCutoff = [];
whiteningDuration = [];
tiling = wtile(blockTime, searchQRange, searchFrequencyRange, sampleFrequency, \
             searchMaximumEnergyLoss, highPassCutoff, lowPassCutoff, \
             whiteningDuration, transientFactor);

# high pass filter and whiten data
whitenedData = wcondition(rawData, tiling);

# q transform whitened data
coefficients = [];
coordinate = [pi/2,0]
whitenedTransform = \
  wtransform(whitenedData, tiling, outlierFactor, 'independent', channelName,coefficients, coordinate);

# identify most significant whitened transform tile
wlog(debugLevel, 2, '  measuring peak significance...\n');
thresholdReferenceTime = centerTime;
thresholdTimeRange = 0.5 * searchWindowDuration * [-1 +1];
thresholdFrequencyRange = [];
thresholdQRange = [];
whitenedProperties = \
  wmeasure(whitenedTransform, tiling, startTime, thresholdReferenceTime, \
           thresholdTimeRange, thresholdFrequencyRange, thresholdQRange, \
           debugLevel);
 
# Select most siginficant Q
mostSignificantQ = \
      whitenedProperties.peakQ;

############################################################################
#                      plot whitened spectrogram                           #
############################################################################

# plot whitened spectrogram
wspectrogram(whitenedTransform, tiling, outputDirectory,uniqueID,startTime, centerTime, \
             plotTimeRanges, plotFrequencyRange, \
             mostSignificantQ, plotNormalizedEnergyRange, \
             plotHorizontalResolution);
