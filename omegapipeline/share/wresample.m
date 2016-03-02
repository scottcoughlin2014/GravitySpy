function data = wresample(data, sampleFrequencies, sampleFrequency)
% WRESAMPLE Resample time series data at a common sample frequency
%
% WRESAMPLE resamples the given time series data at the specified
% common sample frequency.  The input time series data must have the
% same length in time and the ratio of input sample frequencies must
% be expressable as a ratio of integers.
%
% usage:
%
%   resampledData = wresample(data, sampleFrequencies, sampleFrequency);
%
%   data                 cell array of input time series data
%   sampleFrequencies    vector of sample frequencies of input data [Hz]
%   sampleFrequency      desired sample frequency of resampled data [Hz]
%
%   resampledData        cell array of resampled time series data
%
% Note that any decrease in the sample frequency of a channel requires
% the application of a low pass filter to prevent aliasing at the new
% Nyquist frequency.  This filtering is performed using a zero phase
% filter to avoid introducing unwanted time delays into the resampled
% data stream.  However, filter transients are still expected at the
% beginning and end of the resample time series.  In addition, the
% resampled data stream has limited high frequency content since the
% anti-alias filter has a cutoff frequency of approximately 85 percent
% of the new Nyquist frequency.  It is up to the user to account for
% these effects.
%
% WRESAMPLE uses the Matlab RESAMPLE function in the Signal Processing
% Toolbox.
%
% See also RESAMPLE, WCONDITION, and WSEARCH.

% Shourov K. Chatterji <shourov@ligo.mit.edu>

% $Id: wresample.m 2753 2010-02-26 21:33:24Z jrollins $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check for sufficient command line arguments
error(nargchk(3, 3, nargin, 'struct'));

% force cell array
data = wmat2cell(data);

% force one dimensional cell arrays and vectors
data = data(:);
sampleFrequencies = sampleFrequencies(:);

% determine number of channels
numberOfChannels = length(data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% validate number of sample frequencies
if length(sampleFrequencies) ~= numberOfChannels,
  error('number of sample frequencies and channels are inconsistent');
end

% force row vectors and determine data lengths
dataLengths = zeros(numberOfChannels, 1);
for channelNumber = 1 : numberOfChannels,
  data{channelNumber} = data{channelNumber}(:).';
  dataLengths(channelNumber) = length(data{channelNumber});
end

% determine data durations
dataDurations = dataLengths ./ sampleFrequencies;

% validate equal data durations
if ~all(dataDurations == dataDurations(1)),
  error('data durations are not equal');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                resample data                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% begin loop over channels
for channelNumber = 1 : numberOfChannels,

  % factor ratio of sample frequencies
  [upSampleFactor, downSampleFactor] = rat(sampleFrequency / ...
                                           sampleFrequencies(channelNumber));

  % design anti-alias filter
  filterOrder = 2 * 256 * max(upSampleFactor, downSampleFactor);
  filterCutoff = 0.99 / max(upSampleFactor, downSampleFactor);
  filterFrequencies = [0 filterCutoff filterCutoff 1];
  filterMagnitudes = [1 1 0 0];
  filterCoefficients0 = firls(filterOrder, filterFrequencies, filterMagnitudes);
  filterCoefficients1 = hanning(filterOrder + 1);
  filterCoefficients = ...
      upSampleFactor * ...
      filterCoefficients0(:) .* filterCoefficients1(:);

  % resample data
  data{channelNumber} = resample(data{channelNumber}, ...
                                 upSampleFactor, downSampleFactor, ...
                                 filterCoefficients);

% end loop over channels
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    return                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
