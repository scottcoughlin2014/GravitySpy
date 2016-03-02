function varargout = wcondition(rawData, tiling, doubleWhiten, savePSD, filename_PSD)
% WCONDITION High pass filter and whiten time series data
%
% WCONDITION high pass filters and whitens time series data prior to analysis
% by the Q transform.  The data is first zero-phase high pass filtered at the
% minimum analysis frequency specified in the tiling structure.  The resulting
% high pass filtered data is then whitened at
% a frequency resolution equal to the minimum analysis bandwidth requested in
% the tiling structure.  Note that the resulting whitened data is returned as
% a frequency series, not a time series.  In addition, the returned frequency
% series extends from zero frequency to the Nyquist frequency.  As a result,
% they are of length N / 2 + 1, where N is the length of the individual input
% time series.
%
% To enable recovery of calibrated amplitude and phase information, WCONDITION
% also returns the effective frequency domain coefficients of the combined high
% pass and whitening filters for each channel.
%
% WCONDITION also permits double whitening of the data to support true matched
% filtering.  Regardless of whether double whitening is requested or not, the
% returned frequency-domain filter coefficients always correspond to the single
% whitened case.
%
% usage:
%
%   [conditionedData, coefficients] = wcondition(rawData, tiling, doubleWhiten);
%
%   rawData               cell array of input time series
%   tiling                discrete Q transform tiling structure from WTILE
%   doubleWhiten          bool for double whitening
%
%   conditionedData       cell array of conditioned output frequency series
%   coefficients          cell array of frequency domain filter coefficients
%   doubleWhiten          boolean flag to enable double whitening
%
% There is also an alternative output syntax, which provides access to the
% intermediate raw and high pass filtered data for use by WSCAN.
%
%   [rawData, highPassedData, whitenedData, coefficients] = ...
%       wcondition(rawData, tiling, doubleWhiten);
%
%   rawData               cell array of unconditioned frequency series
%   highPassedData        cell array of high pass filtered frequency series
%   whitenedData          cell array of whitened frequency series data
%   coefficients          cell array of frequency domain filter coefficients
%
% See also WTILE, WTRANSFORM, WTHRESHOLD, WSELECT, WSEARCH, WEVENT, WSCAN,
% and WEXAMPLE.

% Shourov K. Chatterji <shourov@ligo.mit.edu>
% Antony Searle <antony.searle@anu.edu.au>
% Jameson Rollins <jrollins@phys.columbia.edu>

% $Id: wcondition.m 3442 2016-01-14 22:59:57Z brennan.hughey@LIGO.ORG $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify correct number of input arguments
error(nargchk(2, 5, nargin, 'struct'));

% apply default arguments
if (nargin < 3) || isempty(doubleWhiten),
  doubleWhiten = 0;
end

if (nargin < 4)
  savePSD = 0; 
end

if (nargin < 5) 
  filename_PSD= [];
end
 

% determine necessary output arguments
switch nargout,
  case 1,
    returnIntermediateData = 0;
    returnCoefficients = 0;
  case 2,
    returnIntermediateData = 0;
    returnCoefficients = 1;
  case 3,
   if savePSD
     returnIntermediateData = 0;
     returnCoefficients = 1;
   else
    returnIntermediateData = 1;
    returnCoefficients = 0;
   end
  case 4,
    returnIntermediateData = 1;
    returnCoefficients = 1;
end

% force cell arrays
if ~iscell(rawData),
  rawData = mat2cell(rawData, size(rawData, 1), size(rawData, 2));
end

% force one dimensional cell arrays
rawData = rawData(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% determine number of channels
numberOfChannels = length(rawData);

% validate tiling structure
if ~strcmp(tiling.id, 'Discrete Q-transform tile structure'),
  error('input argument is not a discrete Q transform tiling structure');
end

% determine required data lengths
dataLength = tiling.sampleFrequency * tiling.duration;
halfDataLength = dataLength / 2 + 1;

% validate data length and force row vectors
for channelNumber = 1 : numberOfChannels,
  rawData{channelNumber} = rawData{channelNumber}(:).';
  if length(rawData{channelNumber}) ~= dataLength,
    error('data length not consistent with tiling');
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                design filters                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% nyquist frequency
nyquistFrequency = tiling.sampleFrequency / 2;

% linear predictor error filter order
lpefOrder = ceil(tiling.sampleFrequency * tiling.whiteningDuration);

% if high pass filtering is requested,
if tiling.highPassCutoff > 0,
    
  % high pass filter order
  hpfOrder = 12;

  % design high pass filter
  [hpfZeros, hpfPoles, hpfGain] = ...
      butter(hpfOrder, tiling.highPassCutoff / nyquistFrequency, 'high');
  hpfSOS = zp2sos(hpfZeros, hpfPoles, hpfGain);

  % magnitude response of high pass filter
  minimumFrequencyStep = 1 / tiling.duration;
  frequencies = 0 : minimumFrequencyStep : nyquistFrequency;
  hpfArgument = (frequencies / tiling.highPassCutoff).^(2 * hpfOrder);
  hpfResponse = hpfArgument ./ (1 + hpfArgument);

  highPassCutoffIndex = ceil(tiling.highPassCutoff / minimumFrequencyStep);
    
% end test for high pass filtering
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            initialize cell arrays                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize cell array of high pass filtered data vectors
highPassedData = cell(numberOfChannels, 1);

% initialize cell array of whitened data vectors
whitenedData = cell(numberOfChannels, 1);

% initialize cell array of conditioning filter coefficients
if returnCoefficients,
  coefficients = cell(numberOfChannels, 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           begin loop over channels                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% begin loop over channels
for channelNumber = 1 : numberOfChannels,

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                   initialize conditioning coefficients                     %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % initialize conditioning filter coefficients
  if returnCoefficients,
    coefficients{channelNumber} = ones(1, halfDataLength);
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                             high pass filter                               %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % if high pass filtering is requested,
  if tiling.highPassCutoff > 0,

    % apply high pass filter
    highPassedData{channelNumber} = sosfiltfilt(hpfSOS, rawData{channelNumber});

    % include high pass filter in conditioning coefficients
    if returnCoefficients,
      coefficients{channelNumber} = coefficients{channelNumber} .* hpfResponse;
    end
    
  % if high pass filtering is not requested,
  else,

    % do nothing
    highPassedData{channelNumber} = rawData{channelNumber};

  % end test for high pass filtering
  end

  % supress high pass filter transients
  highPassedData{channelNumber}(1 : lpefOrder) = ...
      zeros(1, lpefOrder);
  highPassedData{channelNumber}(dataLength - lpefOrder + 1 : dataLength) = ...
      zeros(1, lpefOrder);
      
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                          fast fourier transform                            %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % fourier transform high pass filtered data
  highPassedData{channelNumber} = fft(highPassedData{channelNumber});

  % fourier transform raw data
  if returnIntermediateData,
    rawData{channelNumber} = fft(rawData{channelNumber});
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                 compute accurate power spectrum                            %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Use the median mean average algorithm (as detailed in the FINDCHIRP paper) 
  % to calculate an initial PSD. This reduces the effect of large glitches and/or
  % injections.
  
  % Generate PSD with whitening duration resolution.
  [PSDintermediate Fraw PSDvar] = ...
      medianmeanaveragespectrum(...
          real(ifft(highPassedData{channelNumber})),...
          tiling.sampleFrequency,...
          tiling.sampleFrequency*tiling.whiteningDuration);
  
  %we will save the output of the PSD looking to parameters and number of ouput( not to do it for scans)
  if savePSD 
    PSD=[Fraw PSDintermediate PSDvar];
    PSD = PSD(2:end,:);
    if not(isempty(filename_PSD)),
      save(filename_PSD,'PSD','-ascii');
   end
  end
  % -- try to correct normalization:
  PSDintermediate = PSDintermediate*tiling.sampleFrequency*length(highPassedData{channelNumber});
  % -- interpolate to finer resolution

  PSDintermediate = interp1(Fraw,PSDintermediate,[0:1/tiling.duration:tiling.sampleFrequency/2]','pchip','extrap');
  

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                      whitening filter                                      %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % if whitening is requested,
  if tiling.whiteningDuration > 0,

    % make theCoefficients directly from the power spectrum
    theCoefficients = sqrt(2)./sqrt(PSDintermediate.');
    
    % extract one-sided frequency-domain high pass filtered data
    highPassedData{channelNumber} = ...
        highPassedData{channelNumber}(1 : halfDataLength);

    % extract one-sided frequency-domain raw data
    if returnIntermediateData,
      rawData{channelNumber} = rawData{channelNumber}(1 : halfDataLength);
    end

    if tiling.highPassCutoff > 0
        theCoefficients(1:highPassCutoffIndex) = 0;
    end
        
    % include whitening filter in conditioning coefficients
    if returnCoefficients,
      coefficients{channelNumber} = ...
          coefficients{channelNumber} .* theCoefficients;
    end
  
    % apply whitening filter
    whitenedData{channelNumber} = ...
        theCoefficients .* highPassedData{channelNumber};
  
    % reapply whitening filter if double whitening requested
    if doubleWhiten,
      whitenedData{channelNumber} = ...
        theCoefficients .* whitenedData{channelNumber};
    end

  % if whitening is not requested,
  else

    % extract one-sided frequency-domain high pass filtered data
    highPassedData{channelNumber} = ...
        highPassedData{channelNumber}(1 : halfDataLength);

    % extract one-sided frequency-domain raw data
    if returnIntermediateData,
      rawData{channelNumber} = ...
          rawData{channelNumber}(1 : halfDataLength);
    end

    % do nothing
    whitenedData{channelNumber} = highPassedData{channelNumber};
  
  % end test for whitening
  end
  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            end loop over channels                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
% end loop over channels
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          construct output arguments                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% construct output arguments
switch nargout,
  case 1,
    varargout{1} = whitenedData;
  case 2,
    varargout{1} = whitenedData;
    varargout{2} = coefficients;
  case 3,
   if savePSD
    varargout{1} = whitenedData;
    varargout{2} = coefficients;
    varargout{3} = PSD;
   else
    varargout{1} = rawData;
    varargout{2} = highPassedData;
    varargout{3} = whitenedData;
   end
  case 4,
    varargout{1} = rawData;
    varargout{2} = highPassedData;
    varargout{3} = whitenedData;
    varargout{4} = coefficients;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    return                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
