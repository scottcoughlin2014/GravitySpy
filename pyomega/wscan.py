# ---- Import standard modules to the python path.
import sys, os, shutil, math, random, copy, getopt, re, string, popen2, time
import ConfigParser, glob, operator, optparse
from glue import segments
from glue import segmentsUtils
from glue import pipeline
from glue.lal import CacheEntry
from numpy import loadtxt

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
    opts, args = parser.parse_args()


    return opts

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
searchFrequencyRange 	= int(cp.get('parameters','searchFrequencyRange'));
searchQRange		= int(cp.get('parameters','searchQRange'));
searchMaximumEnergyLoss = int(cp.get('parameters','searchMaximumEnergyLoss'));
searchWindowDuration 	= int(cp.get('parameters','searchWindowDuration'));
plotTimeRanges 		= int(cp.get('parameters','plotTimeRanges'));
plotFrequencyRange 	= int(cp.get('parameters','plotFrequencyRange'));
plotNormalizedERange 	= int(cp.get('parameters','plotNormalizedERange'));
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
#                            load frame cache file                             #
################################################################################

# load frame file cache
frameCache = loadframecache(frameCacheFile);

################################################################################
#                       initialize random number generators                    #
################################################################################

# set random number generator seeds based on event time
rand('state', eventTime);
randn('state', eventTime);

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
centerTime = floor(eventTime) + ...
           round((eventTime - floor(eventTime)) * ...
                 sampleFrequency) / sampleFrequency;

# determine segment start and stop times
startTime = round(centerTime - blockTime / 2);
stopTime = startTime + blockTime;

# generate search tiling
wlog(debugLevel, 2, '  tiling for search...\n');
highPassCutoff = [];
lowPassCutoff = [];
whiteningDuration = [];
tiling = wtile(blockTime, qRange, frequencyRange, sampleFrequency, ...
             maximumEnergyLoss, highPassCutoff, lowPassCutoff, ...
             whiteningDuration, transientFactor);

# read data from frame file
wlog(debugLevel, 2, '  reading data...\n');
timeShifts = [];
[rawData, rawSampleFrequency] = ...
  wreaddata(frameCache, channelName, frameType, ...
            startTime, stopTime, timeShifts, debugLevel);

# resample data
wlog(debugLevel, 2, '  resampling data...\n');
rawData = wresample(rawData, rawSampleFrequency, sampleFrequency);

# high pass filter and whiten data
wlog(debugLevel, 2, '  conditioning data...\n');
[~, ~, whitenedData] = ...
  wcondition(rawData, tiling);

# q transform whitened data
wlog(debugLevel, 2, '  transforming whitened data...\n');
whitenedTransform = ...
  wtransform(whitenedData, tiling, outlierFactor, [], channelName);

# identify most significant whitened transform tile
wlog(debugLevel, 2, '  measuring peak significance...\n');
thresholdReferenceTime = centerTime;
thresholdTimeRange = 0.5 * searchWindowDuration * [-1 +1];
thresholdFrequencyRange = [];
thresholdQRange = [];
whitenedProperties = ...
  wmeasure(whitenedTransform, tiling, startTime, thresholdReferenceTime, ...
           thresholdTimeRange, thresholdFrequencyRange, thresholdQRange, ...
           debugLevel);
 
# Select most siginficant Q
mostSignificantQ = ...
      whitenedProperties{1}.peakQ;

clear whitenedProperties whitenedData channelName eventTime frameCache frameCacheFile
clear frameType highPassCutoff lowPassCutoff outlierFactor qRange rawData rawSampleFrequency
clear searchWindow stopTime blockTime timeShifts transientFactor

toc;

############################################################################
#                      plot whitened spectrogram                           #
############################################################################

# plot whitened spectrogram
wlog(debugLevel, 2, '    plotting whitened spectrogram...\n');
clf;
wspectrogram(whitenedTransform, tiling, outputDirectory,uniqueID,startTime, centerTime, ...
             plotTimeRanges, plotFrequencyRange, ...
             mostSignificantQ, plotNormalizedEnergyRange, ...
             plotHorizontalResolution);
toc;

################################################################################
#                                     exit                                     #
################################################################################
# close all figures
close all;

# return to calling function
return;
