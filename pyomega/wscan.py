# ---- Import standard modules to the python path.
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
    tiling["generalparams"]["maximumMismatch"] = maximumMismatch;
    tiling["generalparams"]["sampleFrequency"] = sampleFrequency;
    tiling["generalparams"]["qs"] = qs;
    tiling["generalparams"]["numberOfPlanes"] = numberOfPlanes;
    tiling["generalparams"]["sampleFrequency"] = sampleFrequency;
    tiling["generalparams"]["qs"] = qs;

    return qs

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

data = data.resample(sampleFrequency)
 
# generate search tiling
highPassCutoff = [];
lowPassCutoff = [];
whiteningDuration = [];
tiling = wtile(blockTime, searchQRange, searchFrequencyRange, sampleFrequency, \
             searchMaximumEnergyLoss, highPassCutoff, lowPassCutoff, \
             whiteningDuration, transientFactor);
print tiling
# read data from frame file
wlog(debugLevel, 2, '  reading data...\n');
timeShifts = [];
[rawData, rawSampleFrequency] = \
	wreaddata(frameCache, channelName, frameType, \
            startTime, stopTime, timeShifts, debugLevel);

# resample data
wlog(debugLevel, 2, '  resampling data...\n');
rawData = wresample(rawData, rawSampleFrequency, sampleFrequency);

# high pass filter and whiten data
wlog(debugLevel, 2, '  conditioning data...\n');
whitenedData = wcondition(rawData, tiling);

# q transform whitened data
wlog(debugLevel, 2, '  transforming whitened data...\n');
whitenedTransform = \
  wtransform(whitenedData, tiling, outlierFactor, [], channelName);

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
