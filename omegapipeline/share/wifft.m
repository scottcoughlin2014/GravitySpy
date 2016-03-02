function data = wifft(data)
% WIFFT Inverse Fourier transform one-sided frequency domain data
%
% WIFFT computes the inverse Fourier transform of one-sided frequency domain
% data produced by WFFT, WCONDITION, and WSCANCONDITION.  The frequency domain
% data are assumed to extend from zero frequency to the Nyquist frequency.  As a
% result the frequency domain data are of length N / 2 + 1 while the time domain
% data are of length N.
%
% WIFFT can be used to convert to time domain the output of WCONDITION and WFFT.
%
% usage:
%
% timeDomainData = wifft(frequencyDomainData);
%
%   frequencyDomainData   cell array of one-sided frequency domain data
%
%   timeDomainData        cell array of time domain data
%
% See also WFFT, WCONDITION, and WSCANCONDITION.

% Shourov K. Chatterji
% shourov@ligo.mit.edu

% $Id: wifft.m 3070 2010-05-14 21:38:01Z jrollins $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check for sufficient command line arguments
error(nargchk(1, 1, nargin, 'struct'));

% force cell array of data
data = wmat2cell(data);

% force one dimensional cell array
data = data(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% determine number of channels
numberOfChannels = length(data);

% force row vectors
for channelNumber = 1 : numberOfChannels,
  data{channelNumber} = data{channelNumber}(:).';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        inverse fast fourier transform                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% inverse Fourier transform
for channelNumber = 1 : numberOfChannels,
  data{channelNumber} = real(ifft([data{channelNumber} ...
                        conj(fliplr(data{channelNumber}(2 : end - 1)))]));
  data{channelNumber} = data{channelNumber};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    return                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
