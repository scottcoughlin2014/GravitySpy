function wlog(debugLevel,setLevel,varargin)
% WLOG write message to stdout
%
% usage: wlog(debugLevel,setLevel,format,A)
% if format, A are empty, a newline is output

% Authors:
% Jameson Rollins <jrollins@phys.columbia.edu>

% apply default arguments
if (nargin < 2) || isempty(setLevel),
  setLevel = 1;
end
if isempty(varargin),
  varargin{1} = '\n';
end

% write log if debug level is greater than the set level
if debugLevel >= setLevel,

  % log to stderr
  fprintf(1, varargin{:});

end
