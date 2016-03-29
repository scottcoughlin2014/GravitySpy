# ---- Import standard modules to the python path.
from __future__ import division
import sys, os, shutil, math, random, copy, getopt, re, string, time
import ConfigParser, glob, operator, optparse, json
from glue import segments
from glue import segmentsUtils
from glue import pipeline
from glue.lal import CacheEntry
import numpy as np
from gwpy.timeseries import TimeSeries
import scipy 
import rlcompleter
import pdb
pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
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
    parser.add_option("--uniqueID", type=float,help="Unique ID of the glitch (from Gravity Spy project)")
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

    # nyquist frequency (half the sampling rate)
    nyquistFrequency = sampleFrequency / 2;

    # maximum mismatch between neighboring tiles
    '''why is this the formula for max mismatch'''
    mismatchStep = 2 * np.sqrt(searchMaximumEnergyLoss / 3);

    # maximum possible time resolution
    minimumTimeStep = 1 / sampleFrequency;

    # maximum possible frequency resolution
    minimumFrequencyStep = 1 / blockTime;

    # conversion factor from Q' to true Q
    '''why sqrt(11)'''
    qPrimeToQ = np.sqrt(11);

    # total number of samples in input data
    numberOfSamples = blockTime * sampleFrequency;

    ############################################################################
    #                       determine parameter constraints                    #
    ############################################################################

    # minimum allowable Q' to prevent window aliasing at zero frequency
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

    # ensure at least one plane
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

        q = qs[int(plane)]

        #######################################################################
        #                 determine plane properties                          #
        #######################################################################

        # find Q' for the plane
        qPrime = q / qPrimeToQ;

        # for large Q'
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

        # ensure at least one row
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

        planestr = 'plane' +str(plane)
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
    	    frequency = frequencies[int(row)];

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
            '''what?'''
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

    	    # insert zero pad length into frequency row structure
    	    tiling[planestr][rowstr]['zeroPadLength'] = zeroPadLength;

    	    # insert data index vector into frequency row structure
    	    tiling[planestr][rowstr]['dataIndices'] = dataIndices.astype('int');

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

            ##############################################################
            #       end loop over frequency rows                         #
            ##############################################################
	
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
	planestr = 'plane' + str(plane)
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
    Ns = 2*len(data)/N-1;  # -- always odd

    # ---- Number of point by which consecutive segments overlap.
    Delta = N/2;

    # ---- Sampling time.
    dt = 1/fs;

    # ---- Enforce unity RMS on the window.
    w = w/mean(w**2)**0.5;

    # ---- Compute spectrogram of data (array: frequency x time).
    [S, F] = spectrogram(data,w,Delta,N,fs);
    # # ---- S is simply the (complex) FFT.  Square this for the PSD.
    S = (S.real)**2 + (S.imag)**2;

    # ---- Divide time segments into two sets of non-overlapping segments.
    #      Will compute median PSD on each separately, then average results.
    oddSegs = np.arange(0,Ns,2)
    evenSegs = np.arange(1,Ns,2)
    
    # ---- Note that oddSegs always has 1 more element than evenSegs.  Drop an
    #      element from one so that both contain an odd number of elements. 
    if np.mod(len(oddSegs),2) == 0:
        oddSegs = oddSegs(np.arange(1,Ns));
    else:
        evenSegs = evenSegs(np.arange(1,Ns));

    Ns_odd = len(oddSegs);
    Ns_even = len(evenSegs);
    # ---- Compute median-based PSD over each set of segments.
    if (Ns_even > 0):
        # ---- Compute median-based PSD over each set of segments.
        S_odd = median(S[oddSegs],2) / medianbiasfactor(Ns_odd);
        S_even = median(S[evenSegs],2) / medianbiasfactor(Ns_even);
        # ---- Take weighted average of the two median estimates.
        S = (Ns_odd*S_odd + Ns_even*S_even) / (Ns_odd + Ns_even);
    # End If statement

    # ---- Normalize to physical units.
    S = 2/(N*fs)*S;

    return S

###############################################################################
#####################                                  ########################
#####################           highpassfilter         ########################
#####################                                  ########################
###############################################################################

def highpassfilt(data,tiling):

    # determine required data lengths
    dataLength = tiling['generalparams']['sampleFrequency'] * tiling['generalparams']['duration'];
    halfDataLength = dataLength / 2 + 1;

    # nyquist frequency
    nyquistFrequency = tiling["generalparams"]["sampleFrequency"] / 2;

    # linear predictor error filter order
    lpefOrder = int(np.ceil(tiling["generalparams"]["sampleFrequency"] * \
	tiling['generalparams']['whiteningDuration']));

    if tiling['generalparams']['highPassCutoff'] > 0:
	# high pass filter order
	filterOrder = 12;

	# design high pass filter
	[hpfZeros, hpfPoles, hpfGain] = \
	scipy.signal.butter(filterOrder, tiling['generalparams']['highPassCutoff'] \
	/ nyquistFrequency, btype='high',output='zpk');

	data = data.zpk(hpfZeros,hpfPoles,hpfGain)

    # End if statement
    # supress high pass filter transients
    data.value[:lpefOrder] = np.zeros(lpefOrder);
    data.value[(dataLength - lpefOrder) :dataLength] = np.zeros(lpefOrder);

    return data, lpefOrder

###############################################################################
#####################                                  ########################
#####################           makePSD                ########################
#####################                                  ########################
###############################################################################
def makePSD(data,tiling):
    return

###############################################################################
#####################                                  ########################
#####################           wtransform             ########################
#####################                                  ########################
###############################################################################

def wtransform(data, tiling, outlierFactor, \
               analysisMode, channelNames, coefficients, coordinate):

    # determine number of channels
    numberOfChannels = 1;

    # determine required data lengths
    dataLength = tiling['generalparams']['sampleFrequency'] * tiling['generalparams']['duration'];
    halfDataLength = dataLength / 2 + 1;

    # validate data length and force row vectors
    if len(data) != halfDataLength:
	sys.exit()

    # determine number of sites
    numberOfSites = 1;

    if len(coordinate) != 2:
	sys.exit()

    #######################################################################
    #                   Define some variables                             #
    #######################################################################

    outputChannelNames = channelNames;
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
        
    # End loop over channels

	# begin loop over Q planes
	for plane in np.arange(0,numberOfPlanes):

	    planestr = 'plane' + str(plane)
	    transforms[channelstr][planestr] = {}
        
        # Begin loop over frequency rows
	    for row in np.arange(0,tiling[planestr]['numberOfRows']):
		    # create empty cell array of frequency row structures
		    rowstr = 'row' +str(row)
		    transforms[channelstr][planestr][rowstr] = {}
        # End loop over frequency rows

    # End loop over Q planes
    
    # Initialize energies
    energies = {}
    # Initialize windowedData
    windowedData = {}
    # Initialize Tile Coefficients
    tileCoefficients = {}
    # Initialize validIndices
    validIndices = {}
    # Initialize Quatiles strcuture arrays
    lowerQuartile = {}
    upperQuartile = {}
    interQuartileRange = {}
    # Initialize outlier threshold
    outlierThreshold = {}
    # Initialize mean energy
    meanEnergy = {}
    # Initialize Normalized energy
    normalizedEnergies = {}


    ############################################################################
    #                       begin loop over Q planes                           #
    ############################################################################

    # begin loop over Q planes
    for plane in np.arange(0,numberOfPlanes):
	planestr = 'plane' + str(plane)

        ########################################################################
        #                begin loop over frequency rows                        #
        ########################################################################

        # begin loop over frequency rows
        for row in np.arange(0,tiling[planestr]['numberOfRows']):

	    rowstr = 'row' +str(row)

            ####################################################################
    	    #          extract and window frequency domain data                #
            ####################################################################

    	    # number of zeros to pad at negative frequencies
    	    leftZeroPadLength = int((tiling[planestr][rowstr]['zeroPadLength'] - 1) / 2);

    	    # number of zeros to pad at positive frequencies
            rightZeroPadLength = int((tiling[planestr][rowstr]['zeroPadLength'] + 1) / 2);

            # begin loop over intermediate channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)
                windowedData[channelstr] ={}
                # extract and window in-band data
                windowedData[channelstr] = tiling[planestr][rowstr]['window'] * \
                data[tiling[planestr][rowstr]['dataIndices']];
		
                # zero pad windowed data
                windowedData[channelstr] = np.pad(windowedData[channelstr],\
                                    [leftZeroPadLength,rightZeroPadLength],'constant',constant_values=(0,0))
                # reorder indices for fast fourier transform
		lastIndex = len(windowedData[channelstr]) 
		order1 = np.arange(lastIndex / 2,lastIndex)
	        order2 = np.arange(0,lastIndex/2)
	        order1 = order1.astype('int')
		order2 = order2.astype('int')
	        order  = np.concatenate((order1,order2))
		windowedData[channelstr] = windowedData[channelstr][order]

            # end loop over intermediate channels

    	    ################################################################
    	    #        inverse fourier transform windowed data               #
            ################################################################

            # begin loop over intermediate channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)
                
                # Initialize tileCoefficients
                tileCoefficients[channelstr] = {}
                # complex valued tile coefficients
                tileCoefficients[channelstr] = np.fft.ifft(windowedData[channelstr]);
            # End loop over intermediate channels

            ##################################################################
            #              energies directly or indirectly                   #
            ##################################################################

            # compute energies directly from intermediate data
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)
                energies[channelstr] = \
                    tileCoefficients[channelstr].real**2 + \
                    tileCoefficients[channelstr].imag**2 ;

            # End loop over channels

            ####################################################################
            #        exclude outliers and filter transients from statistics    #
            ####################################################################

    	    times = np.arange(0,tiling[planestr][rowstr]['numberOfTiles']) * \
            	tiling[planestr][rowstr]['timeStep'];

            # begin loop over channels
    	    for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)
                
      	    	# indices of non-transient tiles
                validIndices[channelstr] = \
                np.logical_and(times > \
                	tiling['generalparams']['transientDuration'], \
               	     times < \
                	tiling['generalparams']['duration']- tiling['generalparams']['transientDuration']);

                # identify lower and upper quartile energies
                sortedEnergies = \
                    np.sort(energies[channelstr][validIndices[channelstr]]);
                lowerQuartile[channelstr] = \
                sortedEnergies[np.round(0.25 * len(validIndices[channelstr])).astype('int')];
                upperQuartile[channelstr] = \
                sortedEnergies[np.round(0.75 * len(validIndices[channelstr])).astype('int')];

                # determine inter quartile range
                interQuartileRange[channelstr] = upperQuartile[channelstr] - \
                                          lowerQuartile[channelstr];

                # energy threshold of outliers
                outlierThreshold[channelstr] = upperQuartile[channelstr] + \
                    outlierFactor * interQuartileRange[channelstr];

                # indices of non-outlier and non-transient tiles
                validIndices[channelstr] = \
                    np.logical_and(np.logical_and(energies[channelstr] < \
                          outlierThreshold[channelstr],\
                          times > \
                          tiling['generalparams']['transientDuration']), \
                          times < \
                          tiling['generalparams']['duration']- tiling['generalparams']['transientDuration']);

            # end loop over channels

            # for reasonable outlier factors,
    	    if outlierFactor < 100:

                # mean energy correction factor for outlier rejection bias
                meanCorrectionFactor = (4 * 3**outlierFactor - 1) / \
                             ((4 * 3**outlierFactor - 1) - \
                             (outlierFactor * np.log(3) + np.log(4)));

    	    # otherwise, for large outlier factors
    	    else:

                # mean energy correction factor for outlier rejection bias
                meanCorrectionFactor = 1;

            # End if statement

    	    ####################################################################
    	    #       determine tile statistics and normalized energies          #
            ####################################################################

    	    # begin loop over channels
    	    for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)

      	    	# mean of valid tile energies
      	    	meanEnergy[channelstr] = \
          	    np.mean(energies[channelstr][validIndices[channelstr]]);

      	    	# correct for bias due to outlier rejection
      	    	meanEnergy[channelstr] = meanEnergy[channelstr] * \
          	    meanCorrectionFactor;

      	    	# normalized tile energies
                normalizedEnergies[channelstr] = energies[channelstr] / \
                        meanEnergy[channelstr];

            # end loop over channels
            
            ####################################################################
    	    #              insert results into transform structure             #
            ####################################################################


    	    # begin loop over channels
    	    for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)

      	    	# insert mean tile energy into frequency row structure
                transforms[channelstr][planestr][rowstr]['meanEnergy'] = \
                    meanEnergy[channelstr];

                # insert normalized tile energies into frequency row structure
                transforms[channelstr][planestr][rowstr]['normalizedEnergies'] = \
                normalizedEnergies[channelstr];
      
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

    #for channel in np.arange(0,numberOfChannels):
    #    channelstr = 'channel' + str(channel)
    # 	transforms[channelstr]['channelName'] = \
    #         outputChannelNames[channelstr];

    return transforms

###############################################################################
##########################                     ################################
##########################      wmeasure       ################################
##########################                     ################################
###############################################################################

def wmeasure(transforms, tiling, startTime, \
                                 referenceTime, timeRange, frequencyRange, \
                                 qRange):
    # WMEASURE Measure peak and weighted signal properties from Q transforms
    #
    # WMEASURE reports the peak and significance weighted mean properties of Q
    # transformed signals within the specified time-frequency region.
    #
    # usage:
    #
    #   measurements = wmeasure(transforms, tiling, startTime, referenceTime, ...
    #                           timeRange, frequencyRange, qRange, debugLevel);
    #
    #   transforms           cell array of input Q transform structures
    #   tiling               discrete Q transform tiling structure from WTILE
    #   startTime            GPS start time of Q transformed data
    #   referenceTime        reference time for time range to search over
    #   timeRange            vector range of relative times to search over
    #   frequencyRange       vector range of frequencies to search over
    #   qRange               scalar Q or vector range of Qs to search over
    #   debugLevel           verboseness of debug output
    #
    #   measurements         cell array of measured signal properties
    #
    # WMEASURE returns a cell array of measured signal properties, with one cell per
    # channel.  The measured signal properties are returned as a structure that
    # contains the following fields.
    #
    #   peakTime                 center time of peak tile [gps seconds]
    #   peakFrequency            center frequency of peak tile [Hz]
    #   peakQ                    quality factor of peak tile []
    #   peakDuration             duration of peak tile [seconds]
    #   peakBandwidth            bandwidth of peak tile [Hz]
    #   peakNormalizedEnergy     normalized energy of peak tile []
    #   peakAmplitude            amplitude of peak tile [Hz^-1/2]
    #   signalTime               weighted central time [gps seconds]
    #   signalFrequency          weighted central frequency [Hz]
    #   signalDuration           weighted duration [seconds]
    #   signalBandwidth          weighted bandwidth [Hz]
    #   signalNormalizedEnergy   total normalized energy []
    #   signalAmplitude          total signal amplitude [Hz^-1/2]
    #   signalArea               measurement time frequency area []
    #
    # The user can focus on a subset of the times and frequencies available in
    # the transform data by specifying a desired range of central times,
    # central frequencies, and Qs to threshold on.  Ranges should be specified
    # as a two component vector, consisting of a minimum and maximum value.
    # Alternatively, if only a single Q is specified, WMEASURE is only applied to
    # the time-frequency plane which has the nearest value of Q in a
    # logarithmic sense to the requested value.
    #
    # To determine the range of central times to search over, WMEASURE requires
    # the start time of the transformed data in addition to a reference time
    # and a relative time range.  Both the start time and reference time should
    # be specified as absolute quantities, while the range of times to analyze
    # should be specified relative to the requested reference time.
    #
    # By default, WMEASURE is applied to all available frequencies and Qs, and the
    # reference time and relative time range arguments are set to exclude data
    # potentially corrupted by filter transients as identified by the transient
    # duration field of the tiling structure.  The default value can be
    # obtained for any argument by passing the empty matrix [].
    #
    # See also WTILE, WCONDITION, WTRANSFORM, WTHRESHOLD, WSELECT, WEXAMPLE, WSCAN,
    # and WSEARCH.

    # Notes:
    # 1. Compute absolute or normalized energy weighted signal properties?
    # 2. Only include tiles with Z>Z0 in integrands?

    # Shourov K. Chatterji
    # shourov@ligo.caltech.edu

    # $Id: wmeasure.m 1716 2009-04-10 17:00:49Z jrollins $

    ###########################################################################
    #                   process command line arguments                        #
    ###########################################################################

    # apply default arguments
    if not referenceTime:
      referenceTime = startTime + tiling.duration / 2;

#    if not timeRange:
#      timeRange = 0.5 * \
#	(tiling['generalparams']['duration'] - 2 * tiling['generalparams']['transientDuration']) * [-1 +1];

    if not frequencyRange:
      frequencyRange = [float('-Inf'),float('Inf')];

    if not qRange:
      qRange = [float('-Inf'),float('Inf')];

    # determine number of channels
    numberOfChannels = len(transforms);

    # if only a single Q is requested, find nearest Q plane
    if len(qRange) == 1:
      [ignore, qPlane] = min(abs(np.log(tiling['generalparams']['qs']) / \
                                 qRange));
      qRange = tiling['generalparams']['qs'][qPlane] * [1,1];

    ##########################################################################
    #                    validate command line arguments                     #
    ##########################################################################

    # Check for two component range vectors
    if len(timeRange) != 2:
      error('Time range must be two component vector [tmin tmax].');

    if len(frequencyRange) != 2:
      error('Frequency range must be two component vector [fmin fmax].');

    if len(qRange) > 2:
      error('Q range must be scalar or two component vector [Qmin Qmax].');


    ##########################################################################
    #                   initialize measurement structures                    #
    ##########################################################################

    # Initialize measurements cell
    measurements = {}
    peakPlane = {}
    numberOfPlanes = tiling['generalparams']['numberOfPlanes'].astype('int')

    # create empty cell array of measurement structures
    # begin loop over channels
    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)
        measurements[channelstr] = {}
        # End loop over channels

    # begin loop over channels
    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)
        # initialize peak signal properties
        measurements[channelstr]['peakTime'] = 0;
        measurements[channelstr]['peakFrequency'] = 0;
        measurements[channelstr]['peakQ'] = 0;
        measurements[channelstr]['peakDuration'] = 0;
        measurements[channelstr]['peakBandwidth'] = 0;
        measurements[channelstr]['peakNormalizedEnergy'] = 0;
        measurements[channelstr]['peakAmplitude'] = 0;

        # initialize integrated signal properties
        measurements[channelstr]['signalTime'] = \
          np.zeros(numberOfPlanes);
        measurements[channelstr]['signalFrequency'] = \
          np.zeros(numberOfPlanes);
        measurements[channelstr]['signalDuration'] = \
          np.zeros(numberOfPlanes);
        measurements[channelstr]['signalBandwidth'] = \
          np.zeros(numberOfPlanes);
        measurements[channelstr]['signalNormalizedEnergy'] = \
          np.zeros(numberOfPlanes);
        measurements[channelstr]['signalAmplitude'] = \
          np.zeros(numberOfPlanes);
        measurements[channelstr]['signalArea'] = \
          np.zeros(numberOfPlanes);

        # end loop over channels
    numberOfPlanes = numberOfPlanes.astype('float')

    ###########################################################################
    #                      begin loop over Q planes                           #
    ###########################################################################

    # begin loop over Q planes
    for plane in np.arange(0,numberOfPlanes):
        planestr = 'plane' +str(plane)
	plane = plane.astype('int')
        numberOfRows = tiling[planestr]['numberOfRows']
      
        #######################################################################
        #                       threshold on Q                                #
        #######################################################################

        # skip Q planes outside of requested Q range
        if ((tiling[planestr]['q'] < min(qRange)) or \
          (tiling[planestr]['q'] > max(qRange))):
            break;

        #######################################################################
        #               begin loop over frequency rows                        #
        #######################################################################

        # begin loop over frequency rows
        for row in np.arange(0,numberOfRows):
            rowstr = 'row' + str(row)

            ###################################################################
            #                  calculate times                                #
            ###################################################################

            times = np.arange(0,tiling[planestr][rowstr]['numberOfTiles']) * \
                   tiling[planestr][rowstr]['timeStep'];

            ###################################################################
            #           threshold on central frequency                        #
            ###################################################################

            # skip frequency rows outside of requested frequency range
            if ((tiling[planestr][rowstr]['frequency'] < \
                 min(frequencyRange)) or \
                (tiling[planestr][rowstr]['frequency'] > \
                 max(frequencyRange))):
                    break;

            ###################################################################
            #             threshold on central time                           #
            ###################################################################
        
            # skip tiles outside requested time range
            tileIndices = \
                np.logical_and(times >= \
                  (referenceTime - startTime + min(timeRange)), \
                 times <= \
                  (referenceTime - startTime + max(timeRange)));
            ###################################################################
            #  differential time-frequency area for integration               #
            ###################################################################
        
            # differential time-frequency area for integration
            differentialArea = tiling[planestr][rowstr]['timeStep'] * \
                           tiling[planestr][rowstr]['frequencyStep'];

            ###################################################################
            #              begin loop over channels                           #
            ###################################################################
        
            # begin loop over channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)

                ###############################################################
                #        update peak tile properties                          #
                ###############################################################
          
                # vector of row tile normalized energies
                normalizedEnergies = transforms[channelstr][planestr][rowstr]\
                        ['normalizedEnergies'][tileIndices];

                # find most significant tile in row
                peakNormalizedEnergy = np.max(normalizedEnergies);
		peakIndex            = np.argmax(normalizedEnergies).astype('int')

                # if peak tile is in this row
                if peakNormalizedEnergy > \
                        measurements[channelstr]['peakNormalizedEnergy']:

                    # update plane index of peak tile
                    peakPlane[channelstr] = plane;
            
                    # extract time index of peak tile
                    peakIndex = tileIndices[peakIndex];
                  
                    # update center time of peak tile
                    measurements[channelstr]['peakTime'] = \
                        times[peakIndex] + startTime;

                    # update center frequency of peak tile
                    measurements[channelstr]['peakFrequency'] = \
                        tiling[planestr][rowstr]['frequency'];

                    # update q of peak tile
                    measurements[channelstr]['peakQ']= \
                        tiling[planestr]['q'];

                    # update duration of peak tile
                    measurements[channelstr]['peakDuration'] = \
                                tiling[planestr][rowstr]['duration'];

                    # update bandwidth of peak tile
                    measurements[channelstr]['peakBandwidth'] = \
                                tiling[planestr][rowstr]['bandwidth'];
                      
                    # update normalized energy of peak tile
                    measurements[channelstr]['peakNormalizedEnergy'] = \
                        (transforms[channelstr][planestr][rowstr]\
                         ['normalizedEnergies'][peakIndex]);

                    # udpate amplitude of peak tile
                    measurements[channelstr]['peakAmplitude'] = \
                        np.sqrt((transforms[channelstr][planestr][rowstr] \
                              ['normalizedEnergies'][peakIndex] - 1) * \
                             transforms[channelstr][planestr][rowstr]['meanEnergy']);
                            
                # End If Statement
          
                ###############################################################
                #                update weighted signal properties            #
                ###############################################################

                # threshold on significance
                normalizedEnergyThreshold = 4.5;
                significantIndices = np.where(normalizedEnergies > \
                                                normalizedEnergyThreshold);
                normalizedEnergies = normalizedEnergies[significantIndices];

                # vector of row tile calibrated energies
                calibratedEnergies = (normalizedEnergies - 1) * \
                    transforms[channelstr][planestr][rowstr]['meanEnergy'] * \
                        tiling[planestr]['normalization'];

                # sum of normalized tile energies in row
                sumNormalizedEnergies = sum(normalizedEnergies);

                # sum of calibrated tile enregies in row
                sumCalibratedEnergies = sum(calibratedEnergies);
          
                # update weighted central time integral
                measurements[channelstr]['signalTime'][plane] = \
                    measurements[channelstr]['signalTime'][plane] + \
                        sum(times[tileIndices][significantIndices] * \
                            calibratedEnergies) * \
                            differentialArea;

                # update weighted central frequency integral
                measurements[channelstr]['signalFrequency'][plane] = \
                    measurements[channelstr]['signalFrequency'][plane] + \
                        tiling[planestr][rowstr]['frequency'] * \
                            sumCalibratedEnergies * \
                                differentialArea;

                # update weighted duration integral
                measurements[channelstr]['signalDuration'][plane] = \
                    measurements[channelstr]['signalDuration'][plane] + \
                            sum(times[tileIndices][significantIndices]**2 * \
                                calibratedEnergies) * \
                                differentialArea;

                # update weighted bandwidth integral
                measurements[channelstr]['signalBandwidth'][plane] = \
                    measurements[channelstr]['signalBandwidth'][plane] + \
                        tiling[planestr][rowstr]['frequency']**2 * \
                            sumCalibratedEnergies * \
                                differentialArea;

                # update total normalized energy integral
                measurements[channelstr]['signalNormalizedEnergy'][plane] = \
                    measurements[channelstr]['signalNormalizedEnergy'][plane] + \
                        sumNormalizedEnergies * \
                            differentialArea;

                # update total calibrated energy integral
                measurements[channelstr]['signalAmplitude'][plane] = \
                    measurements[channelstr]['signalAmplitude'][plane] + \
                        sumCalibratedEnergies * \
                            differentialArea;

                # update total signal area integral
                measurements[channelstr]['signalArea'][plane] = \
                    measurements[channelstr]['signalArea'][plane] + \
                        len(normalizedEnergies) * \
                            differentialArea;

            ##################################################################
            #              end loop over channels                            #
            ##################################################################


        ######################################################################
        #               end loop over frequency rows                         #
        ######################################################################
        

        ######################################################################
        #               normalize signal properties                          #
        ######################################################################
      
        # begin loop over channels
        for channel in np.arange(0,numberOfChannels):
	    channelstr = 'channel' + str(channel)
            # normalize weighted signal properties by total normalized energy
            if measurements[channelstr]['signalAmplitude'][plane] != 0:
                
              measurements[channelstr]['signalTime'][plane] = \
                  measurements[channelstr]['signalTime'][plane] / \
                  measurements[channelstr]['signalAmplitude'][plane];
              
              measurements[channelstr]['signalFrequency'][plane] = \
                  measurements[channelstr]['signalFrequency'][plane] / \
                  measurements[channelstr]['signalAmplitude'][plane];
              
              measurements[channelstr]['signalDuration'][plane] = \
                  measurements[channelstr]['signalDuration'][plane] / \
                  measurements[channelstr]['signalAmplitude'][plane];
              
              measurements[channelstr]['signalBandwidth'][plane] = \
                  measurements[channelstr]['signalBandwidth'][plane] / \
                  measurements[channelstr]['signalAmplitude'][plane];
            # End If Statement

            # duration and bandwidth are second central
            # moments in time and frequency
            '''Not sure what this means...'''            
            measurements[channelstr]['signalDuration'][plane] = \
                np.sqrt(measurements[channelstr]['signalDuration'][plane] - \
                     measurements[channelstr]['signalTime'][plane]**2);
            measurements[channelstr]['signalBandwidth'][plane] = \
                np.sqrt(measurements[channelstr]['signalBandwidth'][plane] - \
                     measurements[channelstr]['signalTime'][plane]**2);

            # convert signal energy to signal amplitude
            measurements[channelstr]['signalAmplitude'][plane] = \
                np.sqrt(measurements[channelstr]['signalAmplitude'][plane]);

            # add start time to measured central time
            measurements[channelstr]['signalTime'][plane] = \
                measurements[channelstr]['signalTime'][plane] + startTime;

        # end loop over channels

      
    ######################################################################
    #                  end loop over Q planes                            #
    ######################################################################

    ######################################################################
    #   report signal properties from plane with peak significance       #
    ######################################################################

    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)

        # weighted central time estimate from plane with peak tile significance
        measurements[channelstr]['signalTime'] = \
            measurements[channelstr]['signalTime'][peakPlane[channelstr]];

        # weighted central frequency estimate from plane with peak tile significance
        measurements[channelstr]['signalFrequency'] = \
            measurements[channelstr]['signalFrequency'][peakPlane[channelstr]];

        # weighted duration estimate from plane with peak tile significance
        measurements[channelstr]['signalDuration'] = \
            measurements[channelstr]['signalDuration'][peakPlane[channelstr]];

        # weighted bandwidth estimate from plane with peak tile significance
        measurements[channelstr]['signalBandwidth'] = \
            measurements[channelstr]['signalBandwidth'][peakPlane[channelstr]];

        # total signal normalized energy estimate from plane with peak tile significance
        measurements[channelstr]['signalNormalizedEnergy'] = \
            measurements[channelstr]['signalNormalizedEnergy'][peakPlane[channelstr]];

        # total signal amplitude estimate from plane with peak tile significance
        measurements[channelstr]['signalAmplitude'] = \
            measurements[channelstr]['signalAmplitude'][peakPlane[channelstr]];

        # measured time frequency area in plane with peak tile significance
        measurements[channelstr]['signalArea'] = \
            measurements[channelstr]['signalArea'][peakPlane[channelstr]];

        # report peak tile properties for very weak signals
        if measurements[channelstr]['signalArea'] < 1:
            measurements[channelstr]['signalTime'] = \
                measurements[channelstr]['peakTime'];
            measurements[channelstr]['signalFrequency'] = \
                measurements[channelstr]['peakFrequency'];
            measurements[channelstr]['signalDuration'] = \
                measurements[channelstr]['peakDuration'];
            measurements[channelstr]['signalBandwidth'] = \
                measurements[channelstr]['peakBandwidth'];
            measurements[channelstr]['signalNormalizedEnergy'] = \
                measurements[channelstr]['peakNormalizedEnergy'];
            measurements[channelstr]['signalAmplitude'] = \
                measurements[channelstr]['peakAmplitude'];
            measurements[channelstr]['signalArea'] = 1;

        # End If statment

    # end loop over channels

    ###########################################################################
    #           return most significant tile properties                       #
    ###########################################################################

    return measurements

###############################################################################
##########################                     ################################
##########################      wspectrogram   ################################
##########################                     ################################
###############################################################################

def wspectrogram(transforms, tiling, outputDirectory,uniqueID,startTime, \
           referenceTime, timeRange, frequencyRange, qRange,\
	 	normalizedEnergyRange, horizontalResolution):

# WSPECTROGRAM Display time-frequency Q transform spectrograms
#
# WSPECTROGRAM displays multi-resolution time-frequency spectrograms of
# normalized tile energy produced by WTRANSFORM.  A separate figure is produced
# for each channel and for each time-frequency plane within the requested range
# of Q values.  The resulting spectrograms are logarithmic in frequency and
# linear in time, with the tile color denoting normalized energy.
#
# usage:
#
#   handles = wspectrogram(transforms, tiling, startTime, referenceTime, ...
#                          timeRange, frequencyRange, qRange, ...
#                          normalizedEnergyRange, horizontalResolution);
#
#     transforms              cell array of q transform structures
#     tiling                  q transform tiling structure
#     startTime               start time of transformed data
#     referenceTime           reference time of plot
#     timeRange               vector range of relative times to plot
#     frequencyRange          vector range of frequencies to plot
#     qRange                  scalar Q or vector range of Qs to plot
#     normalizedEnergyRange   vector range of normalized energies for colormap
#     horizontalResolution    number of data points across image
#
#     handles                 matrix of axis handles for each spectrogram
#
# The user can focus on a subset of the times and frequencies available in the
# original transform data by specifying a desired time and frequency range.
# Ranges should be specified as a two component vector, consisting of the
# minimum and maximum value.  Additionally, WSPECTROGRAM can be restricted to
# plot only a subset of the available Q planes by specifying a single Q or a
# range of Qs.  If a single Q is specified, WSPECTROGRAM displays the
# time-frequency plane which has the nearest value of Q in a logarithmic sense
# to the requested value.  If a range of Qs is specified, WSPECTROGRAM displays
# all time-frequency planes with Q values within the requested range.  By
# default all available channels, times, frequencies, and Qs are plotted.  The
# default values can be obtained for any argument by passing the empty matrix
# [].
#
# To determine the range of times to plot, WSPECTROGRAM requires the start time
# of the transformed data, a reference time, and relative time range.  Both the
# start time and reference time should be specified as absolute quantities, but
# the range of times to plot should be specified relative to the requested
# reference time.  The specified reference time is used as the time origin in
# the resulting spectrograms and is also reported in the title of each plot.
#
# If only one time-frequency plane is requested, it is plotted in the current
# figure window.  If more than one spectrogram is requested, they are plotted
# in separate figure windows starting with figure 1.
#
# The optional normalizedEnergyRange specifies the range of values to encode
# using the colormap.  By default, the lower bound is zero and the upper bound
# is autoscaled to the maximum normalized energy encountered in the specified
# range of time, frequency, and Q.
#
# The optional cell array of channel names are used to label the resulting
# figures.
#
# The optional horizontal resolution argument specifies the number data points
# in each frequency row of the displayed image.  The vector of normalized
# energies in each frequency row is then interpolated to this resolution in
# order to produce a rectangular array of data for displaying.  The vertical
# resolution is directly determined by the number of frequency rows available in
# the transform data.  By default, a horizontal resolution of 2048 data points
# is assumed, but a higher value may be necessary if the zoom feature will be
# used to magnify the image.  For aesthetic purposes, the resulting image is
# also displayed with interpolated color shading enabled.
#
# WSPECTROGRAM returns a matrix of axis handles for each spectrogram with
# each channel in a separate row and each Q plane in a separate column.
#
# See also WTILE, WCONDITION, WTRANSFORM, WEVENTGRAM, and WEXAMPLE.

# Shourov K. Chatterji <shourov@ligo.mit.edu>

# $Id: wspectrogram.m 2753 2010-02-26 21:33:24Z jrollins $

    ############################################################################
    #                        hard coded parameters                             #
    ############################################################################

    # number of horizontal pixels in image
    defaultHorizontalResolution = 2048;

    # spectrogram boundary
    spectrogramLeft = 0.14;
    spectrogramWidth = 0.80;
    spectrogramBottom = 0.28;
    spectrogramHeight = 0.62;
    spectrogramPosition = [spectrogramLeft spectrogramBottom ...
                       spectrogramWidth spectrogramHeight];

    # colorbar position
    colorbarLeft = spectrogramLeft;
    colorbarWidth = spectrogramWidth;
    colorbarBottom = 0.12;
    colorbarHeight = 0.02;
    colorbarPosition = [colorbarLeft colorbarBottom ...
                    colorbarWidth colorbarHeight];

    # time scales for labelling
    millisecondThreshold = 0.5;
    secondThreshold = 3 * 60;
    minuteThreshold = 3 * 60 * 60;
    hourThreshold = 3 * 24 * 60 * 60;
    dayThreshold = 365.25 * 24 * 60 * 60;

    ############################################################################
    #                      identify q plane to display                         #
    ############################################################################
    # find plane with Q nearest the requested value
    planeIndices = np.argmin(abs(np.log(tiling['generalparams']['qs'] / qRange)));

    # number of planes to display
    numberOfPlanes = len(planeIndices);

    # index of plane in tiling structure
    planeIndex = planeIndices[0];
    planeIndexstr = 'plane' + str(planeIndex)
    ########################################################################
    #               identify frequency rows to display                     #
    ########################################################################

    pdb.set_trace()
    # vector of frequencies in plane
    frequencies = tiling['planeIndexstr']['frequencies'];

    # find rows within requested frequency range
    rowIndices = np.logical_and(frequencies >= np.min(frequencyRange),
                  frequencies <= np.max(frequencyRange));

    # pad by one row if possible
    #if rowIndices(1) > 1,
    #  rowIndices = [rowIndices(1) - 1 rowIndices];
    #if rowIndices(end) < length(frequencies),
    #  rowIndices = [rowIndices rowIndices(end) + 1];

    # vector of frequencies to display
    frequencies = frequencies[rowIndices];

    # number of rows to display
    numberOfRows = len(rowIndices);

    ############################################################################
    #                       initialize display matrix                          #
    ############################################################################

    # initialize matrix of normalized energies for display
    for iN in np.arange(0,len(timeRange)):
        normalizedEnergies{iN} = np.zeros(numberOfRows, horizontalResolution);

    ############################################################################
    #                     begin loop over frequency rows                       #
    ############################################################################

    # loop over rows
    for row = 1 : numberOfRows,

        # index of row in tiling structure
        rowIndex = rowIndices(row);

        # vector of times in plane
        rowTimes = ...
           (0 :  tiling.planes{planeIndexstr}.rows{rowIndex}.numberOfTiles - 1) ... 
          * tiling.planes{planeIndexstr}.rows{rowIndex}.timeStep + ...
             (startTime - referenceTime);

        # find tiles within requested time range
        for iN in np.arange(0,len(timeRange)):
            # vector of times to display
            timeRange1{iN} = timeRange(iN) * [-1 +1]*0.5;
            times{iN} = linspace(min(timeRange1{iN}), max(timeRange1{iN}), horizontalResolution);
            padTime = 1.5 * tiling.planes{planeIndexstr}.rows{rowIndex}.timeStep;
            tileIndices = find((rowTimes >= min(timeRange1{iN}) - padTime) & ...
                     (rowTimes <= max(timeRange1{iN}) + padTime));

            # vector of times to display
            rowTimestemp = rowTimes(tileIndices);

            # corresponding tile normalized energies
            rowNormalizedEnergies = transforms{channelNumber}.planes{planeIndexstr} ...
                            .rows{rowIndex}.normalizedEnergies(tileIndices);

            # interpolate to desired horizontal resolution
            rowNormalizedEnergies = interp1(rowTimestemp, rowNormalizedEnergies, times{iN}, ...
                                  'pchip');

            # insert into display matrix
            normalizedEnergies{iN}(row, :) = rowNormalizedEnergies;

############################################################################
#                      end loop over frequency rows                        #
############################################################################

   
############################################################################
#                      Plot spectrograms                                   #
############################################################################ 
for iN in np.arange(0,len(timeRange)):

    return

###########
#def whiten(data, asd):
#    freq_data = data.fft()
#    assert asd.df == freq_data.df
#    freq_data /= asd
    # FIXME: No ifft method on the frequency data
#    return freq_data.ifft()
#white_data = whiten(data, asd)
#############
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

'''describe these...'''
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

########################################################################
#     Determine if this is a normal omega scan or a Gravityspy         #
#	omega scan with unique ID                                      #
########################################################################

if opts.uniqueID is None:
	IDstring = opts.eventTime
else:
	IDstring = opts.uniqueID

##############################################################################
#               Process Channel Data                                         #
##############################################################################

# find closest sample time to event time
centerTime = np.floor(opts.eventTime) + \
           np.round((opts.eventTime - np.floor(opts.eventTime)) * \
                 sampleFrequency) / sampleFrequency;

# determine segment start and stop times
startTime = round(centerTime - blockTime / 2);
stopTime = startTime + blockTime;

# Read in the data
if opts.NSDF:
	data = TimeSeries.fetch(channelName,startTime,stopTime)
else:
	data = TimeSeries.read(frameCacheFile,channelName, format='gwf',start=startTime,end=stopTime)

# Plot Time Series at given plot durations
plot = data.plot()
plot.set_title('TimeSeries')
plot.set_ylabel('Gravitational-wave strain amplitude')
for iTime in np.arange(0,len(plotTimeRanges)):
    halfTimeRange = plotTimeRanges[iTime]*0.5
    plot.set_xlim(opts.eventTime - halfTimeRange,opts.eventTime + halfTimeRange)
    plot.save('/home/scoughlin/public_html/test3/timeseries' + str(plotTimeRanges[iTime]) + '.png')

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
data,lpefOrder = highpassfilt(data,tiling)

# Plot HighPass Filtered Time  Series at given plot durations
plot = data.plot()
plot.set_title('HighPassFilter')
plot.set_ylabel('Gravitational-wave strain amplitude')
for iTime in np.arange(0,len(plotTimeRanges)):
    halfTimeRange = plotTimeRanges[iTime]*0.5
    plot.set_xlim(opts.eventTime - halfTimeRange,opts.eventTime + halfTimeRange)
    plot.save('/home/scoughlin/public_html/test3/highpass' + str(plotTimeRanges[iTime]) + '.png')

# Time to whiten the times series data
# Our FFTlength is determined by the predetermined whitening duration found in wtile
FFTlength = tiling['generalparams']['whiteningDuration']

# We will condition our data using the median-mean approach. This means we will take two sets of non overlaping data from our time series and find the median S for both and then average them in order to be both safe against outliers and less bias. By default this ASD is one-sided.
'''what is S? ASD?'''

asd = data.asd(FFTlength, FFTlength/2., method='median-mean')

# Apply ASD to the data to whiten it
white_data = data.whiten(FFTlength, FFTlength/2., asd=asd)

plot = white_data.plot()
plot.set_title('Whitened')
plot.set_ylabel('Gravitational-wave strain amplitude')
plot.save('/home/scoughlin/public_html/test3/whitenedwhole.png')
for iTime in np.arange(0,len(plotTimeRanges)):
    halfTimeRange = plotTimeRanges[iTime]*0.5
    plot.set_xlim(opts.eventTime - halfTimeRange,opts.eventTime + halfTimeRange)
    plot.save('/home/scoughlin/public_html/test3/whitened' + str(plotTimeRanges[iTime]) + '.png')

# Extract one-sided frequency-domain conditioned data.
dataLength = tiling['generalparams']['sampleFrequency'] * tiling['generalparams']['duration'];
halfDataLength = int(dataLength / 2 + 1);
white_data_fft = white_data.fft()

# q transform whitened data
coefficients = [];
coordinate = [np.pi/2,0]
whitenedTransform = \
  wtransform(white_data_fft, tiling, outlierFactor, 'independent', channelName,coefficients, coordinate);

# identify most significant whitened transform tile
thresholdReferenceTime = centerTime;
thresholdTimeRange = 0.5 * searchWindowDuration * np.array([-1,1]);
thresholdFrequencyRange = [];
thresholdQRange = [];
whitenedProperties = \
  wmeasure(whitenedTransform, tiling, startTime, thresholdReferenceTime, \
           thresholdTimeRange, thresholdFrequencyRange, thresholdQRange);

pdb.set_trace() 
# Select most siginficant Q
mostSignificantQ = \
      whitenedProperties['channel0']['peakQ'];

############################################################################
#                      plot whitened spectrogram                           #
############################################################################

# plot whitened spectrogram
wspectrogram(whitenedTransform, tiling, outputDirectory,uniqueID,startTime, centerTime, \
             plotTimeRanges, plotFrequencyRange, \
             mostSignificantQ, plotNormalizedEnergyRange, \
             plotHorizontalResolution);
