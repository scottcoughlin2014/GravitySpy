function [data, sampleFrequencies] = ...
    wreaddata(frameCache, channelNames, frameTypes, startTime, stopTime, ...
              timeShifts, debugLevel)
% WREADDATA Read multiple channels of data from frame files
%
% WREADDATA finds and retrieves multiple channels of time series data from a set
% of frame files.  The data are specified by the frame file type, channel name,
% start time, and duration or stop time.  The necessary frame files are located
% using a file name caching scheme.
%
% WREADDATA is build on top of the READFRAMEDATA set of high level functions
% for reading data from frame files.
%
% usage:
%
%   [data, sampleFrequencies] = ...
%     wreaddata(frameCache, channelNames, frameTypes, startTime, stopTime, ...
%               timeShifts, debugLevel);
%
%   frameCache           frame file cache structure from LOADFRAMECACHE
%   channelNames         cell array of channel names
%   frameTypes           cell array of frame types
%   startTime            gps start time of data to extract
%   stopTime             gps stop time (or duration) of data to extract
%   timeShifts           vector of time shifts [seconds]
%   debugLevel           verboseness of debug output
%
%   data                 cell array of extracted data
%   sampleFrequencies    vector of sample frequencies
%
% The data is returned as a cell array of row vectors in the same order as in
% the cell array of channel names.
%
% WREADDATA retrieves data from the requested start time up to, but not
% including, the requested stop time, such that stop minus start seconds are
% retrieved.  Alternatively, the desired duration in seconds may be specified
% instead of the GPS stop time parameter.
%
% The optional time shift argument should be a vector with one element per
% channel that specifies an additional time shift to apply to the start time for
% each detector.  By convention, a positive time shift corresponds to a delay of
% the corresponding time series.  If no time offsets are specified, a default
% time shift of zero is assumed for all detectors.
%
% To avoid the effects of round-off error, the requested start time, stop
% time, and time shifts should all be aligned with samples of the input data
% stream.
%
% If it is unable to load the requested data, WREADDATA returns empty result
% vectors and zero sample frequency for the failed channel as well as issuing a
% warning if debugLevel is set to 1 or higher.  By default, a debugLevel of
% unity is assumed.
%
% See also READFRAMEDATA, LOADFRAMECACHE, CREATEFRAMECACHE, FRGETVECT,
% WSEARCH, and WEXAMPLE.

% Shourov K. Chatterji <shourov@ligo.caltech.edu>

% $Id: wreaddata.m 3416 2012-06-02 17:50:48Z michal.was@LIGO.ORG $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            hard coded parameters                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% permit redundant frame data
allowRedundantFlag = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check for sufficient command line arguments
error(nargchk(5, 7, nargin, 'struct'));

% apply default arguments
if (nargin < 6) || isempty(timeShifts),
  timeShifts = [];
end
if (nargin < 7) || isempty(debugLevel),
  debugLevel = 1;
end

% force cell arrays
channelNames = wmat2cell(channelNames);
frameTypes = wmat2cell(frameTypes);

% provide default time shifts
if isempty(timeShifts),
  timeShifts = zeros(size(channelNames));
end

% force one dimensional cell arrays and vectors
channelNames = channelNames(:);
frameTypes = frameTypes(:);
timeShifts = timeShifts(:);

% determine number of channels
numberOfChannels = length(channelNames);

% convert duration to absolute stop time
if stopTime < startTime,
  stopTime = startTime + stopTime;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% validate frame types
if length(frameTypes) ~= numberOfChannels,
  error('number of frame types is inconsistent with number of channels');
end

% validate time shifts
if length(timeShifts) ~= numberOfChannels,
  error('number of time shifts is inconsistent with number of channels');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         initialize result structures                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize result structures
data = cell(numberOfChannels, 1);
sampleFrequencies = zeros(numberOfChannels, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  read data                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% begin loop over channels
for channelNumber = 1 : numberOfChannels,

  % switch on frametype
  switch upper(frameTypes{channelNumber}),

    % if no data is requested,
    case 'NONE',

     % return empty result
     data{channelNumber} = [];
     sampleFrequencies(channelNumber) = 0;

    % if whitenoise is requested,
    case {'WHITE', 'WHITENOISE'},

      % generate white noise data
      sampleFrequencies(channelNumber) = 16384;
      data{channelNumber} = sqrt(sampleFrequencies(channelNumber) / 2) * ...
                            normrnd(0, 1, 1, ...
                                    sampleFrequencies(channelNumber) * ...
                                    (stopTime - startTime));

    % otherwise,
    otherwise,

      % read requested data
      % KLUDGE for geoclient, site specification should be a single character
      % for a normal framecache, requires LTPDA and geoda plugin
      if length(frameCache.sites{1}) > 1
        pl_signal = plist('built-in','geoclient', 'hostname', frameCache.frameTypes{1}, ...
                           'type', frameCache.sites{1}, 'port', frameCache.startTimes,  'channels', frameCache.directories{1}, ...
                           'starttimes', startTime - timeShifts(channelNumber), ...
                           'nsecs', stopTime-startTime);
        signal = ao(pl_signal);
        data{channelNumber} = signal.y;
        sampleFrequencies(channelNumber) = signal.fs;
      else
        % read data from frame
        [data{channelNumber}, sampleFrequencies(channelNumber)] = ...
            readframedata(frameCache, channelNames{channelNumber}, ...
                          frameTypes{channelNumber}, ...
                          startTime - timeShifts(channelNumber), ...
                          stopTime - timeShifts(channelNumber), ...
                          allowRedundantFlag, debugLevel);
      end
  % end switch on frametype
  end

  % force row vector data
  data{channelNumber} = data{channelNumber}(:)';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          return to calling function                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
