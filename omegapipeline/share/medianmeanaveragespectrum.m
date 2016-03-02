function [S, F, PSDvar] = medianmeanaveragespectrum(data,fs,N,w)
% MEDIANMEANAVERAGESPECTRUM - power spectrum estimation a la FINDCHIRP.
%
% usage
%
%   S = medianmeanaveragespectrum(data,fs,N,w)
%
% data      Vector of noise data.
% fs        Scalar.  Sample rate [Hz].  Must be a positive integer.
% N         Scalar.  Desired FFT length [samples].  Must be a power of 2.
% w         Optional vector.  Window to be used in FFT [default hann].
%           If specified, it must be of length N.
%
% S         Vector of estimated one-sided power-spectrum density values.
% F         Vector of frequencies at which S is estimated.
%
% The power spectrum S is estimated using the median-mean algorithm of the
% FINDCHIRP pipeline (gr-qc/0509116).  The data is segmented in segments of
% length N samples with 50% overlap between consecutive segments and FFTed.  
% The segments are grouped into two sets on non-overlapping segments, the
% median power in each set is computed, and the two median estimates are
% then averaged to give S.  The use of the median gives some robustness
% against glitches in the input data stream.
%
% Note that length(data) must be an integer multiple of N.  Also, the
% window, if specified, is normalized to unity RMS for computing S.
%
% The output power spectrum is one-sided and has units Hz^(-1/2).  For
% example, white noise of variance sigma^2 will have a power spectrum of 
%
%   S = 2 sigma^2 / fs;
% 
% The first and last frequency bins (DC and Nyquist) typically have about
% 2/3 the value of S as the other bins.
%
% EXAMPLE:
% 
% % ---- Construct 256 sec of simulated detector noise from 40Hz to 2048Hz.
% T = 256;
% sampleFrequency = 4096;
% data = SimulatedDetectorNoise('LIGO',T,sampleFrequency,40,sampleFrequency/2);
% % ---- Compute PSD with 1 Hz resolution (good enough to avoid leakage).
% Smeas = medianmeanaveragespectrum(data,sampleFrequency,sampleFrequency);
% % ---- Design curve.
% f = [32:2048]';
% S = SRD('LIGO',f);
% % ---- Compare:
% figure;
% loglog(f,[Smeas(33:end) S],'linewidth',2); legend('measured','design');
% grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Checks on input.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---- Check for sufficient command line arguments.
error(nargchk(3, 4, nargin, 'struct'));

% ---- Verify that data is a vector.
if ~isvector(data)
    error('data must be a vector.');
end

% ---- Verify that sample rate is a positive integer.
if (fs ~= round(fs) | fs <= 0)
    error('sample rate fs must be a positive integer.');
end

% ---- Verify that data length and FFT length are commensurate.
if (gcd(length(data),N) ~= N)
    error('data length must be an integer multiple of N.');
end

% ---- Assign window if necessary.
if (nargin<4)
    w = hann(N);
else
    if (length(w) ~= N)
        error('if specified, w must have length N.');
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Segment data and FFT.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---- Number of segments (FFTs).
Ns = 2*length(data)/N-1;  % -- always odd

% ---- Number of point by which consecutive segments overlap.
Delta = N/2;

% ---- Sampling time.
dt = 1/fs;

% ---- Enforce unity RMS on the window.
w = w/mean(w.^2)^0.5;

% ---- Compute spectrogram of data (array: frequency x time).
[S, F] = spectrogram(data,w,Delta,N,fs);
% % ---- S is simply the (complex) FFT.  Square this for the PSD.
S = real(S).^2 + imag(S).^2;

% ---- If needed by user, compute gaussianity (PSD standard deviation)
if nargin >= 3
    PSDvar = (2/(N*fs))^2*var(S,0,2);
end

% ---- Divide time segments into two sets of non-overlapping segments.
%      Will compute median PSD on each separately, then average results.
oddSegs = [1:2:Ns];  
evenSegs = [2:2:Ns];
% ---- Note that oddSegs always has 1 more element than evenSegs.  Drop an
%      element from one so that both contain an odd number of elements. 
if rem(length(oddSegs),2) == 0,
    oddSegs = oddSegs(2:end);
else
    evenSegs = evenSegs(2:end);
end
Ns_odd = length(oddSegs);
Ns_even = length(evenSegs);
% ---- Compute median-based PSD over each set of segments.
if (Ns_even > 0)
    % ---- Compute median-based PSD over each set of segments.
    S_odd = median(S(:,oddSegs),2) / medianbiasfactor(Ns_odd);
    S_even = median(S(:,evenSegs),2) / medianbiasfactor(Ns_even);
    % ---- Take weighted average of the two median estimates.
    S = (Ns_odd*S_odd + Ns_even*S_even) / (Ns_odd + Ns_even);
else
    % ---- Have only 1 segment.  No averaging to be done!
    ;
end
% ---- Normalize to physical units.
S = 2/(N*fs)*S;

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Helper functions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function alpha = medianbiasfactor(n)
% MEDIANBIASFACTOR - Compute bias factor for median estimate of mean.
%
% usage:
%
%   alpha = medianbiasfactor(n)
%
%  n        Scalar.  Number of samples used to estimate median.  Must be a
%           positive, odd integer.  
%  alpha    Scalar.  Factor by which meidan must be divided to give an
%           unbiased estimate of the mean, assuming an exponential
%           distribution.    

% ---- Verify that n is a positive, odd, integer scalar.
if ~isscalar(n) || (rem(n,2) ~= 1),
    error('n must be a positive, odd, integer scalar.');
end

% ---- Compute bias factor alpha.
ii = 1:n;
alpha = sum((-1).^(ii+1) ./ ii);

return
