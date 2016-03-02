function [locations, responses] = wdetectors(detectors)
% WDETECTORS Location and orientation data for gravitational wave detectors
%
% WDETECTORS returns the location and response matrix for the specified
% gravitational wave detectors.
%
%  usage:
%
%    [locations, responses] = wdetectors(detectors);
%
%    detectors      cell array of site, detector, or channel names
%
%    locations      cell array of detector locations [s]
%    responses      cell array of detector response matrices []
%
% The list of detectors should be specified as a cell array of site identifiers
% or channel names.  Site identifiers are determined by taking the first
% character of each channel name.  The following sites are recognized.
%
%   G  GEO
%   H  LIGO Hanford
%   L  LIGO Livingston
%   V  Virgo
%   0  Fiducial reference detecor
%
% The returned detector location and response data are cell arrays, with one
% cell per detector.
%
% The returned detector location data correspond to the speed of light travel
% time from the center of the Earth, as used by the WRESPONSE and WBASELINE
% functions.
%
% The returned detector response data are dimensionless response matrices as
% used by the WRESPONSE function.
%
% The fiducial reference detector is provided for validation purposes.  It is
% positioned in the direction of the North pole, at a distance equal to one
% speed of light travel time from the Earth center, and has arms along the
% positive X and Y axis in the Earth fixed geocentric coordinate system.
%
% See also WRSESPONE, WBASELINE, WTILESKY, and WSKYMAP.

%   Shourov Chatterji   shourov@ligo.caltech.edu
%   Albert Lazzarini    lazz@ligo.caltech.edu
%   Antony Searle       antony.searle@anu.edu.au
%   Leo Stein           lstein@ligo.caltech.edu
%   Patrick Sutton      psutton@ligo.caltech.edu
%   Massimo Tinto       massimo.tinto@jpl.nasa.gov

% $Id$

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify correct number of input arguments
error(nargchk(1, 1, nargin, 'struct'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% force cell array of detector names
detectors = wmat2cell(detectors);

% force one dimensional cell array
detectors = detectors(:);

% number of detectors
numberOfDetectors = length(detectors);

% force uppercase detector names
detectors = upper(detectors);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          begin loop over detectors                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize results cell arrays
locations = cell(1, numberOfDetectors);
responses = cell(1, numberOfDetectors);

% begin loop over detectors
for detectorNumber = 1 : numberOfDetectors,

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                          detector response data                            %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  switch detectors{detectorNumber}(1),

    % handle case of GEO600 detector
    case 'G',

      % cartesian coordinates of detector [light seconds]
      locations{detectorNumber} = ...
          [+0.012863270; +0.002223531; +0.016743719];

      % detector response matrix
      responses{detectorNumber} = ...
          [-0.096843926; -0.365712231; +0.122125440; ...
           -0.365712231; +0.222996416; +0.249808025; ...
           +0.122125440; +0.249808025; -0.126152489;];

    % handle cases of LIGO Hanford 2k and 4km detectors
    case 'H',

      % cartesian coordinates of detector [light seconds]
      locations{detectorNumber} = ...
          [-0.007209704; -0.012791166; +0.015345117;];

      % detector response matrix
      responses{detectorNumber} = ...
          [-0.392614702; -0.077612253; -0.247388405; ...
           -0.077612253; +0.319524089; +0.227998294; ...
           -0.247388405; +0.227998294; +0.073090613;];

    % handle case of LIGO Livingston 2k detector
    case 'L',

      % cartesian coordinates of detector [light seconds]
      locations{detectorNumber} = ...
          [-0.000247758; -0.018333629; +0.010754964;];

      % detector response matrix
      responses{detectorNumber} = ...
          [+0.411281744; +0.140209630; +0.247293475; ...
           +0.140209630; -0.109005943; -0.181616031; ...
           +0.247293475; -0.181616031; -0.302275801;];

    % handle case of Virgo detector
    case 'V',

      % cartesian coordinates of detector [light seconds]
      locations{detectorNumber} = ...
          [+0.015165071; +0.002811912; +0.014605361;];

      % detector response matrix
      responses{detectorNumber} = ...
          [+0.243874678; -0.099086615; -0.232575796; ...
           -0.099086615; -0.447827872; +0.187828535; ...
           -0.232575796; +0.187828535; +0.203953193;];

    % handle case of test detector
    case '0',

      % cartesian coordinates of detector [light seconds]
      locations{detectorNumber} = ...
          [0; 0; 1;];

      % detector response matrix
      responses{detectorNumber} = ...
          [+0.5; +0.0; +0.0; ...
           +0.0; -0.5; +0.0; ...
           +0.0; +0.0; +0.0;];

  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                         end loop over detectors                            %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end loop over detectors
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          return to calling function                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
