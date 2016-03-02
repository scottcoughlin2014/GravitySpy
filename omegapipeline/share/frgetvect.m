function [data,varargout] = frgetvect(fname,cname,varargin)
% FRGETVECT - extract from a frame file the data for a channel
%
% [data,tsamp,fsamp,gps0,strStart,unitX,unitY] = ...
%    frgetvect(fname,cname,gpsStart,duration,dbgLvl)
%
% fname      - file name 
% cname      - channel name (ADC or PROC or SIM)
% gpsStart   - (optional) starting GPS time (default: first frame in file)
% duration   - (optional) duration in seconds  (default: 1 second)
% dbgLvl     - (optional) debug level (default: 0 for no output)
% 
% data       - channel data (time or frequency series stored as double)
% tsamp      - (optional) time values relative to the first data point
% fsamp      - (optional) frequency values (for FFT's)
% gps0       - (optional) GPS starting time (in second, stored in double)
% strStart   - (optional) starting time as a string
% unitX      - (optional) vector unitX as a string
% unitY      - (optional) vector unitY as a string
%
% Warning types:
%   frgetvect:info    -- informational messages
%   frgetvect:dt==0   -- channel metadata dt == 0
%
%  Error types:
%   frgetvect:inputArgCount    -- too many or too few input arguments
%   frgetvect:outputArgCount   -- too many or too few output arguments
%   frgetvect:inputArg         -- first two arguments are not strings
%   frgetvect:nameLength       -- file or channel name too long
%   frgetvect:fileOpen         -- failure to open file
%   frgetvect:channelNotFound  -- channel not in frame
%   frgetvect:noMatlabType     -- no matlab type for channel data

%#mex
%#external
