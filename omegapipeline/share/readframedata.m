function [data, sampleFrequency, time] = ...
           readframedata(frameCache, channelName, frameType, ...
                         startTime, stopTime, allowRedundantFlag, ...
                         debugLevel);
% READFRAMEDATA Read a single channel of data from frame files
%
% READFRAMEDATA finds and retrieves the requested time series data from a
% set of frame files.  The data is specified by the frame file type,
% channel name, start time, and duration or stop time.  The necessary frame
% files are located using a file name caching scheme.
%
% usage: [data, sampleFrequency, time] = ...
%          readframedata(frameCache, channelName, frameType, ...
%                        startTime, stopTime, allowRedundantFlag, ...
%                        debugLevel);
%
%   frameCache           file name cache
%   channelName          channel name
%   frameType            frame file type
%   startTime            GPS start time
%   stopTime             GPS stop time (or duration)
%   allowRedundantFlag   permit redundant frame data
%   debugLevel           verboseness of debug output
%
%   data                 data vector
%   sampleFrequency      sample frequency [Hz]
%   time                 time vector
%
% READFRAMEDATA expects frame cache information in the format produced by
% LOADFRAMECACHE which contains the site, frame type, duration, time, and
% location of available frame data files.
%
% The requested site designator is determined from the first character of
% the requested channel name.  The frame file type may contain wildcards.
% Unless redundant frame data is permitted, it is an error if the same
% frame file appears more than once in the frame cache file.  If redundant
% frame data is permitted, the first matching frame file from the cache is
% used.  It is always an error if two frame files overlap but do not cover
% the same time range or differ in type.  By default, redundant frame data
% is permitted.
%
% READFRAMEDATA retrieves data from the requested start time up to, but not
% including, the requested stop time, such that stop minus start seconds
% are retrieved.  Alternatively, the desired duration in seconds may be
% specified instead of the GPS stop time parameter.
%
% The resulting time series is returned as two row vectors containing the
% data sequence and the corresponding GPS timestamps, as well as the scalar
% sample frequency.  To protect against roundoff error, an integer sample
% frequency is assumed.
%
% If it is unable to load the requested data, READFRAMEDATA returns empty
% result vectors and zero sample frequency as well as a warning if
% debugLevel is set to 1 or higher.  By default, a debugLevel of unity is
% assumed.
%
% READFRAMEDATA is built on top of the FRGETVECT function from the FrameL
% library, which is available from the following URL.
%
% http://lappweb.in2p3.fr/virgo/FrameL/
%
% See also LOADFRAMECACHE, CREATEFRAMECACHE.pl, and CONVERTLALCACHE.pl.

% Shourov K. Chatterji <shourov@ligo.mit.edu>
% Jameson Rollins <jrollins@phys.columbia.edu>

% $Id: readframedata.m 3445 2016-01-23 02:02:11Z brennan.hughey@LIGO.ORG $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify number of input arguments
error(nargchk(5, 7, nargin, 'struct'));

% apply default arguments
if (nargin < 6) || isempty(allowRedundantFlag),
  allowRedundantFlag = true;
end
if (nargin < 7) || isempty(debugLevel),
  debugLevel = 1;
end

% if specified stop time precedes start time,
if stopTime < startTime,

  % treat the specified stop time as a duration
  stopTime = startTime + stopTime;

% otherwise, continue
end

% determine site designator from channel name
site = channelName(1);

% translate wildcard in requested frame type to regexp syntax
frameType = ['^' regexprep(frameType, '\*', '.*') '$'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if specified frame cache is invalid,
if ~isstruct(frameCache),

  % issue warning
  if debugLevel >= 1,
    warning('Invalid frame file cache.');
  end

  % return empty results
  data = [];
  time = [];
  sampleFrequency = 0;
  return;

% otherwise, continue
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 identify matching segments from frame cache                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% find overlap of cache segments with requested data
segmentStartTimes = max(startTime, frameCache.startTimes);
segmentStopTimes = min(stopTime, frameCache.stopTimes);

% identify cache segments which overlap with requested times
segments = find(segmentStopTimes > segmentStartTimes);

% if no segments overlap with requested times,
if isempty(segments),

  % issue warning
  if debugLevel >= 1,
    warning(['No data available for [' ...
             num2str(startTime) ', ' num2str(stopTime) ').']);
  end

  % return empty results
  data = [];
  time = [];
  sampleFrequency = 0;
  return;

% otherwise, if only one overlapping segment
elseif length(segments) == 1,

  % test for requested site and frame type
  siteMatches = ...
      ~isempty(regexp(frameCache.sites(segments), site));
  frameTypeMatches = ...
      ~isempty(regexp(frameCache.frameTypes(segments), frameType));
  segments = segments(siteMatches & frameTypeMatches);

% otherwise, if more than one matching segment
else

  % identify cache segments with requested site and frame type
  siteMatches = ...
      ~cellfun('isempty', regexp(frameCache.sites(segments), site));
  frameTypeMatches = ...
      ~cellfun('isempty', regexp(frameCache.frameTypes(segments), frameType));
  segments = segments(siteMatches & frameTypeMatches);

% otherwise, continue
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        identify available frame files                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% intialize list of available frame files
frameFilePaths = [];
frameFileTypes = [];
frameFileStartTimes = [];
frameFileStopTimes = [];

% loop over matching segments
for segment = segments.',

  % frame type of frame files in segment
  frameFileType = frameCache.frameTypes{segment};

  % find time stamp of first frame file in segment
  firstFrameFileStartTime = frameCache.startTimes(segment) + ...
                            frameCache.durations(segment) * ...
                            floor((segmentStartTimes(segment) - ...
                                   frameCache.startTimes(segment)) / ...
                                  frameCache.durations(segment));

  % find time stamp of last frame file in segment
  lastFrameFileStartTime = frameCache.startTimes(segment) + ...
                           frameCache.durations(segment) * ...
                           ceil((segmentStopTimes(segment) - ...
                                 frameCache.startTimes(segment)) / ...
                                frameCache.durations(segment) - 1);

  % loop over frame file start times in segment
  for frameFileStartTime = firstFrameFileStartTime : ...
                           frameCache.durations(segment) : ...
                           lastFrameFileStartTime;

    % stop time of frame file
    frameFileStopTime = frameFileStartTime + frameCache.durations(segment);

    % full path name of frame file for 9 digit gps time naming
    frameFilePath = [frameCache.directories{segment} '/' ...
                     frameCache.sites{segment} '-' ...
                     frameCache.frameTypes{segment} '-' ...
                     sprintf('%09d',frameFileStartTime) '-' ...
                     int2str(frameCache.durations(segment)) '.gwf'];

    if debugLevel >= 2,
      fprintf(1, 'searching for %s...', frameFilePath);
    end

    % if frame file does not exist
    if (system(['test -r ' frameFilePath]) ~= 0),

      % try full path name of frame file with 10 digit gps time naming
      frameFilePath = [frameCache.directories{segment} '/' ...
                       frameCache.sites{segment} '-' ...
                       frameCache.frameTypes{segment} '-' ...
                       sprintf('%010d',frameFileStartTime) '-' ...
                       int2str(frameCache.durations(segment)) '.gwf'];

      if debugLevel >= 2,
        fprintf(1, ' not found.\n');
      end
      if debugLevel >= 3,
        command = ['date; ls -l ' frameCache.directories{segment} '/ ; df'];
        [status, result] = system(command);
        fprintf(1, [command '\n' result]);
      end
      if debugLevel >= 2,
        fprintf(1, 'searching for %s...', frameFilePath);
      end
    end

    % if frame file exists
    if (system(['test -r ' frameFilePath]) == 0),

      % record frame file information
      frameFilePaths{length(frameFilePaths) + 1} = frameFilePath;
      frameFileTypes{length(frameFileTypes) + 1} = frameFileType;
      frameFileStartTimes = [frameFileStartTimes frameFileStartTime];
      frameFileStopTimes = [frameFileStopTimes frameFileStopTime];

      if debugLevel >= 2,
        fprintf(1, ' found.\n');
      end
    else
      if debugLevel >= 2,
        fprintf(1, ' not found.\n');
      end
      if debugLevel >= 3,
        command = ['date; ls -l ' frameCache.directories{segment} '/'];
        [status, result] = system(command);
        fprintf(1, [command '\n' result]);
      end
    end

  % end loop over frame files
  end

% end loop over matching segments
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           test for redundant data                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number of available frame files
numberOfFrameFiles = length(frameFilePaths);

% initialize list of frame files to read
keepFrameFileNumbers = [];

% begin loop over available frame files
for frameFileNumber = 1 : numberOfFrameFiles,

  % intialize frame file status
  keepFrameFileFlag = true;

  % begin loop over previously tested frame files
  for previousFrameFileNumber = 1 : frameFileNumber - 1,

    % find overlap between current frame file and previously tested one
    overlapStartTime = max(frameFileStartTimes(frameFileNumber), ...
                           frameFileStartTimes(previousFrameFileNumber));
    overlapStopTime = min(frameFileStopTimes(frameFileNumber), ...
                          frameFileStopTimes(previousFrameFileNumber));

    % if current frame overlaps previously tested one,
    if overlapStartTime < overlapStopTime,

      % if redundant frame data is permitted,
      if allowRedundantFlag,

        % if current frame file same as previously tested frame file
        if ((frameFileStartTimes(frameFileNumber) == ...
             frameFileStartTimes(previousFrameFileNumber)) & ...
            (frameFileStopTimes(frameFileNumber) == ...
             frameFileStopTimes(previousFrameFileNumber)) & ...
            strcmp(frameFileTypes{frameFileNumber}, ...
                   frameFileTypes{previousFrameFileNumber})),

          % ignore this frame file
          keepFrameFileFlag = false;
          continue;

        % otherwise, if frame files are different
        else

          % issue warning
          if debugLevel >= 1,
            warning(['Overlapping but dissimilar frame files ' ...
                     frameFilePaths{frameFileNumber} ' and ' ...
                     frameFilePaths{previousFrameFileNumber} '.']);
          end

          % return empty results
          data = [];
          time = [];
          sampleFrequency = 0;
          return;

        % end test for same frame file
        end

      % otherwise, if redundant data not permitted
      else

        % issue warning
        if debugLevel >= 1,
          warning(['Redundant frame files ' ...
                   frameFilePaths{frameFileNumber} ' and ' ...
                   frameFilePaths{previousFrameFileNumber} '.']);
        end

        % return empty results
        data = [];
        time = [];
        sampleFrequency = 0;
        return;

      % end test if redundant data permitted
      end

    % otherwise
    end

  % end loop over previously tested frame files
  end

  % if not excluded,
  if keepFrameFileFlag,

    % keep this frame for reading
    keepFrameFileNumbers = [keepFrameFileNumbers frameFileNumber];

  % otherwise,
  end

% end loop over available frame files
end

% eliminate redundant frame files
frameFilePaths = frameFilePaths(keepFrameFileNumbers);
frameFileTypes = frameFileTypes(keepFrameFileNumbers);
frameFileStartTimes = frameFileStartTimes(keepFrameFileNumbers);
frameFileStopTimes = frameFileStopTimes(keepFrameFileNumbers);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            test for missing data                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sort frame files in order of increasing start time
[ignore, sortedIndices] = sort(frameFileStartTimes);
frameFilePaths = frameFilePaths(sortedIndices);
frameFileTypes = frameFileTypes(sortedIndices);
frameFileStartTimes = frameFileStartTimes(sortedIndices);
frameFileStopTimes = frameFileStopTimes(sortedIndices);

% test for non-contiguous data
continuityStartTimes = [max(startTime, frameFileStartTimes) stopTime];
continuityStopTimes = [startTime min(stopTime, frameFileStopTimes)];
discontinuities = find(continuityStartTimes ~= continuityStopTimes);

% if data is missing
if any(discontinuities),

  % issue warning
  if debugLevel >= 1,
    warning(['Missing ' channelName ' ' frameType(2 : end - 1) ' data at ' ...
             int2str(round(continuityStopTimes(discontinuities(1)))) '.']);
  end
  if debugLevel >= 4,
    continuityStartTimes
    continuityStopTimes
    continuityStopTimes - continuityStartTimes
  end
  
  % return empty results
  data = [];
  time = [];
  sampleFrequency = 0;
  return;

% otherwise, continue
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               read frame data                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize result vectors
data = [];
time = [];
sampleFrequency = [];
firstCorrectionFlag = 1;

% number of frame files to read
numberOfFrameFiles = length(frameFilePaths);

% loop over matching frame files
for frameFileNumber = 1 : numberOfFrameFiles,

  % full path name to frame file
  frameFilePath = frameFilePaths{frameFileNumber};

  % display path name of frame file being read
  if debugLevel >= 2,
    fprintf(1, 'reading %s...\n', frameFilePath);
  end

  % determine real channel name from virtual channel name
  realChannelName = regexprep(channelName, '^.*;', '');
  readGPSframeStart = 1;
  % read start time from frame, assume that the duration in frame file
  % name is correct
  warning off frgetvect:info
  try
    [ignore1 ignore2 ignore3 readGPSframeStart] = ...
        frgetvect(frameFilePath, realChannelName);
  catch
      exception = lasterror;
      if debugLevel >= 2,
        warning(exception.message);
      end
  end
  frameFileStartTime = readGPSframeStart;
  frameFileStopTime = frameFileStartTime + ...
      frameFileStopTimes(frameFileNumber) - ...
      frameFileStartTimes(frameFileNumber);
  % align requested start/stop times to follow discrepency between frame
  % file name and stored time stamps, do it only once
  if firstCorrectionFlag
    startTimeCorrection = ...
        frameFileStartTime - frameFileStartTimes(frameFileNumber);
    startTime = startTime + startTimeCorrection;
    stopTime = stopTime + startTimeCorrection;
    firstCorrectionFlag = 0;
  end

  % initialize error status
  lasterr('');

  % initialize data array
  readData = [];

  % start time of data to read from frame file
  readStartTime = max(startTime, frameFileStartTime);

  % duration of data to read from frame file
  readDuration = min(stopTime, frameFileStopTime) - readStartTime;

  % turn off warnings
  if debugLevel < 2,
    warning off frgetvect:info
  end

  % read data from frame file, catching any errors
  try
      [readData, readTime, readF, readGPS] = ...
          frgetvect(frameFilePath, realChannelName, ...
                    readStartTime, ...
                    readDuration, ...
                    0);
  catch
      exception = lasterror;
      if debugLevel >= 2,
        warning(exception.message);
      end
  end

  % if error on reading frame file,
  if (length(lasterr) ~= 0) | isempty(readData) | any(isnan(readData)),
      
    % issue warning
    if debugLevel >= 1,
      warning(['Error reading ' channelName ' from ' ...
               frameFilePath '.']);
    end
    
    % return empty results
    data = [];
    time = [];
    sampleFrequency = 0;
    return;

  % otherwise, continue
  end

  % determine time step of read data
  if length(readTime) == 1,
    readTimeStep = [];
  else
    readTimeStep = readTime(2) - readTime(1);
  end

  % determine sample frequency of read data
  if isempty(readTimeStep),
    readSampleFrequency = [];
  else
    readSampleFrequency = round(1 / readTimeStep);
  end

  % if initial frame file,
  if isempty(sampleFrequency),

    % initialize resulting sample frequency
    sampleFrequency = readSampleFrequency;

  % otherwise, if sample frequency has changed,
  elseif sampleFrequency ~= readSampleFrequency,

    % issue warning
    if debugLevel >= 1,
      warning(['Inconsistent sample frequency for ' channelName ...
               ' in ' frameFilePath '.']);
    end

    % return empty results
    data = [];
    time = [];
    sampleFrequency = 0;
    return;

  % otherwise, continue
  end

  % if duration of read data is not compatible with requested data
  if ~isempty(readTimeStep) && ...
          ((length(readTime) < floor(readDuration / readTimeStep)) || ...
           (length(readTime) > ceil (readDuration / readTimeStep)));
      
    % issue warning
    if debugLevel >= 1,
      warning(['Unexpected data length for ' channelName ...
               ' in ' frameFilePath '.']);
    end
    
    % return empty results
    data = [];
    time = [];
    sampleFrequency = 0;
    return;

  % otherwise, continue
  end

  % force row vectors
  readData = readData(:).';
  readTime = readTime(:).';

  % absolute time vector
  readTime = readTime + readGPS;

  % append to result vectors
  data = [data readData];
  if nargout > 2,
    time = [time readTime];
  end

% end loop over matching frame files
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      return results to calling function                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
