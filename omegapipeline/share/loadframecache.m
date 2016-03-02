function cache = loadframecache(cachePath,cacheType)
% LOADFRAMECACHE Load lal/frame file cache information
%
% LOADFRAMECACHE reads the lal/frame file cache information stored in the
% specified file.  The resulting cache structure is used to locate frame data
% during subsequent calls to READFRAMEDATA.
%
% usage: cache = loadframecache(cachePath,cacheType)
%
%   cachePath     path to lal/frame cache file
%   cacheType     'lal' or 'frame' ('frame')
%   cache         lal/frame cache structure
%
% formats:
%
% FRAMECACHE: The frame cache file should consist of whitespace delimited ASCII
% text and contains one line for each contiguous data segment with a common
% site, frame type, duration, and directory.  Each line should consist of the
% following six columns.
%
%   * site designator (e.g. 'H' or 'L')
%   * frame file type (e.g. 'RDS_R_L3')
%   * GPS start time of segment
%   * GPS stop time of segment
%   * frame file duration in seconds
%   * full path name of directory
%
% The resulting frame cache structure consists of the following six
% fields.
%
%   .sites        cell array of segment site designators
%   .frameTypes   cell array of segment frame file types
%   .startTimes   vector of segment GPS start times
%   .stopTimes    vector of segment GPS stop times
%   .durations    vector of segment frame file durations
%   .directories  cell array of segment directory names
%
% The data segments are inclusive of the specified start time, but
% exclusive of the specified stop time, such that the segment duration
% is simply the difference between the stop and start times.
%
% LALCACHE: The lal cache file should consist of whitespace delimited ASCII text
% and contains one line for each data frame.  Each line should consist of the
% following five columns.
%
%   * site designator (e.g. 'H' or 'L')
%   * frame type (e.g. 'RDS_R_L3')
%   * GPS start time of frame
%   * frame duration in seconds
%   * full path name of frame
%
% The resulting lal cache structure consists of the following six
% fields.
%
%   .sites        cell array of segment site designators
%   .frameTypes   cell array of segment frame file types
%   .startTimes   vector of frame GPS start times
%   .durations    vector of frame duration
%   .frames       cell array of frame file paths
%
% See also READFRAMEDATA, CREATEFRAMECACHE.pl, and CONVERTLALACHE.pl.

% Shourov K. Chatterji <shourov@ligo.mit.edu>
% Jameson Rollins <jrollins@phys.columbia.edu>

% $Id: loadframecache.m 2753 2010-02-26 21:33:24Z jrollins $

% verify correct number of input arguments
error(nargchk(0, 2, nargin, 'struct'));

if (nargin < 1),
  cachePath = './framecache.txt';
end
if (nargin < 2),
  cacheType = 'frame';
end

switch cacheType
  case 'frame'
    % read requested lal cache file
    [cache.sites, cache.frameTypes, cache.startTimes, ...
     cache.stopTimes, cache.durations, cache.directories] = ...
        textread(cachePath, '%s %s %u %u %u %s');
  case 'lal'
    % read requested frame cache file
    [cache.sites, cache.frameTypes, cache.startTimes, ...
     cache.durations, cache.frames] = ...
        textread(cachePath, '%s %s %u %u %s');
  otherwise
    error(['Unknown cache type "' cacheType '".'])
end
