function wscan(eventTime, uniqueID, channelName,frameType,frameCacheFile, ...
               outputDirectory, debugLevel)
% WSCAN Create spectrograms of significant channels for candidate events
%
% WSCAN Examines gravitational-wave channels, auxiliary interferometer channels,
% and environmental channels around times of interest.  Spectrograms are
% generated for those channels that exhibit statistically significant behavior.
%
% usage: wscan(eventTime, configurationFile, frameCacheFile, ...
%              outputDirectory, , uniqueid, debugLevel);
%
%   eventTime             GPS time of candidate event
%   frameCacheFile        path name of frame file cache file
%   outputDirectory       directory to write results
%   uniqueID		      glitch ID tag
%   debugLevel            verboseness of debug level output
%
% To allow use as a stand-alone executable, the specified eventTime should be
% a string, not a number.
%
% The configuration file is an ASCII text file describing the parameters for
% each channel to be analyzed.  The entries should have the following syntax.
%
% {
%   channelName:                 'H1:LSC-AS_Q'
%   frameType:                   'RDS_R_L1'
%   sampleFrequency:             4096
%   searchTimeRange:             16
%   searchFrequencyRange:        [64 1024]
%   searchQRange:                [4 64]
%   searchMaximumEnergyLoss:     0.2
%   whiteNoiseFalseRate:         1e-2
%   searchWindowDuration:        0.1
%   plotTimeRanges:              [0.1 1.0 10.0]
%   plotFrequencyRange:          [64 1024]
%   plotNormalizedEnergyRange:   [0 25.5]
%   alwaysPlotFlag:              0
% }
%
% Groups of related channels may optionally be separated by a section title
% specifier with the following form.
%
% [index_entry:section_title]
%
% This will be used to generate a index entry and section title in the resulting
% web page.
%
% The WCONFIGURE.SH script can be used to automatically generate a reasonable
% configuration file for sample frame files.
%
% If no configuration file is specified, WSCAN looks for the file
% configuration.txt in the current working directory.  Similarly, if not frame
% cache file is specified, WSCAN looks for the file framecache.txt in the
% current working directory.
%
% For information on the frameCacheFile, see READFRAMEDATA.
%
% The resulting spectrograms for statistically significant channels are written
% to an event sub-directory that is created in the specified output directory
% and is labelled by the GPS time of the event under analysis.  If no output
% directory is specified, a default output directory called wscans is created in
% the current working directory.  A web page named index.html is also created in
% each event sub-directory and presents the resulting spectrograms in table
% form.
%
% The specified debugLevel controls the amount of detail in the output log.
% A debugLevel of unity is assumed by default.
%

% Shourov K. Chatterji <shourov@ligo.mit.edu>

% $Id: wscan.m 3431 2013-11-02 22:35:51Z brennan.hughey@LIGO.ORG $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify correct number of input arguments
narginchk(1, 7);

% apply default arguments
if (nargin < 5) || isempty(frameCacheFile),
  frameCacheFile = 'framecache.txt';
end
if (nargin < 6)
  outputDirectory = [];
end
if (nargin < 7) || isempty(debugLevel),
  debugLevel = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   defaults                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
% default configuration parameters
sampleFrequency = 16384;
timeRange = 64;
frequencyRange = [10 2048];
qRange = [4 64];
maximumEnergyLoss = 0.2;
searchWindowDuration = 0.5;
plotTimeRanges = [0.5 1 2 4];
plotFrequencyRange = [10 2048];
plotNormalizedEnergyRange = [0 25.5];
% convert string event time and debug level to numbers
debugLevel = str2num(debugLevel);
eventTime = str2num(eventTime);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            hard coded parameters                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% search parameters
transientFactor = 2;
outlierFactor = 2.0;

% display parameters
plotHorizontalResolution = 512;

% name of text summary file
textSummaryFile = 'summary.txt';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            load frame cache file                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report status
wlog(debugLevel, 1, 'reading framecache file %s...\n', frameCacheFile);

% load frame file cache
frameCache = loadframecache(frameCacheFile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       initialize random number generators                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set random number generator seeds based on event time
rand('state', eventTime);
randn('state', eventTime);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           create output directory                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if outputDirectory not specified, make one based on center time
if isempty(outputDirectory),
  outputDirectory = sprintf('scans/%s', uniqueID);
end

% report status
wlog(debugLevel, 1, 'creating event directory\n');
wlog(debugLevel, 1, '  outputDirectory:         %s\n', outputDirectory);

% create spectrogram directory
unix(['mkdir -p ' outputDirectory]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        initialize text summary report                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report status
wlog(debugLevel, 1, 'opening text summary...\n');

% open text summary file
textSummaryFID = fopen([outputDirectory '/' textSummaryFile], 'w');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               identify statistically significant channels                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% find closest sample time to event time
centerTime = floor(eventTime) + ...
           round((eventTime - floor(eventTime)) * ...
                 sampleFrequency) / sampleFrequency;

% determine segment start and stop times
startTime = round(centerTime - timeRange / 2);
stopTime = startTime + timeRange;

% generate search tiling
wlog(debugLevel, 2, '  tiling for search...\n');
highPassCutoff = [];
lowPassCutoff = [];
whiteningDuration = [];
tiling = wtile(timeRange, qRange, frequencyRange, sampleFrequency, ...
             maximumEnergyLoss, highPassCutoff, lowPassCutoff, ...
             whiteningDuration, transientFactor);

% read data from frame file
wlog(debugLevel, 2, '  reading data...\n');
timeShifts = [];
[rawData, rawSampleFrequency] = ...
  wreaddata(frameCache, channelName, frameType, ...
            startTime, stopTime, timeShifts, debugLevel);

% resample data
wlog(debugLevel, 2, '  resampling data...\n');
rawData = wresample(rawData, rawSampleFrequency, sampleFrequency);

% high pass filter and whiten data
wlog(debugLevel, 2, '  conditioning data...\n');
[~, ~, whitenedData] = ...
  wcondition(rawData, tiling);

% q transform whitened data
wlog(debugLevel, 2, '  transforming whitened data...\n');
whitenedTransform = ...
  wtransform(whitenedData, tiling, outlierFactor, [], channelName);

% identify most significant whitened transform tile
wlog(debugLevel, 2, '  measuring peak significance...\n');
thresholdReferenceTime = centerTime;
thresholdTimeRange = 0.5 * searchWindowDuration * [-1 +1];
thresholdFrequencyRange = [];
thresholdQRange = [];
whitenedProperties = ...
  wmeasure(whitenedTransform, tiling, startTime, thresholdReferenceTime, ...
           thresholdTimeRange, thresholdFrequencyRange, thresholdQRange, ...
           debugLevel);
 
% Select most siginficant Q
mostSignificantQ = ...
      whitenedProperties{1}.peakQ;
clear whitenedProperties whitenedData channelName eventTime frameCache frameCacheFile
clear frameType highPassCutoff lowPassCutoff outlierFactor qRange rawData rawSampleFrequency
clear searchWindow stopTime timeRange timeShifts transientFactor

toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      plot whitened spectrogram                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot whitened spectrogram
wlog(debugLevel, 2, '    plotting whitened spectrogram...\n');
clf;
wspectrogram(whitenedTransform, tiling, outputDirectory,uniqueID,startTime, centerTime, ...
             plotTimeRanges, plotFrequencyRange, ...
             mostSignificantQ, plotNormalizedEnergyRange, ...
             plotHorizontalResolution);
toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          close text summary report                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report status
wlog(debugLevel, 1, 'closing text summary...\n');

% close text summary file
fclose(textSummaryFID);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     exit                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report completion
wlog(debugLevel, 1, 'finished on %s at %s\n', ...
     datestr(clock, 29), datestr(clock, 13));

% close all figures
close all;

% return to calling function
return;
