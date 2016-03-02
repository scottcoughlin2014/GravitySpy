function [fplus, fcross, deltat] = wresponse(coordinates, detectors)
% WRESPONSE Antenna response for a given detector and source coordinates
%
% WRESPONSE computes the response of the specified gravitational wave detector
% to both polarizations of gravitational wave signal coming from the specified
% directions on the sky with
%
%  usage:
%
%    [fplus, fcross, deltat] = wresponse(coordinates, detectors);
%
%    coordinates    matrix of sky positions and polarization angles [radians]
%    detectors      cell array of site, detector, or channel names
%
%    fplus          matrix of plus polarization antenna response factors
%    fcross         matrix of cross polarization antenna response factors
%    deltat         matrix of time delays relative to earth center
%
% The sky positions and polarizations should be provided as a three column
% matrix of spherical coordinates in the form [theta phi psi].  The coordinate
% theta is a geocentric colatitude running from 0 at the North pole to pi at the
% South pole, and the coordinate phi is the geocentric longitude in Earth fixed
% coordinates with 0 on the prime meridian.  The polarization psi is defined as
% the right handed angle about the direction of propagation from the negative
% phi direction to the plus polarization direction of the source.  The range of
% theta is [0, pi] and the range of phi is [0, 2 pi).  The range of psi is [0,
% pi).  If the coordinate matrix has only two columns, they are assumed to be
% theta and phi values and psi is assumed to be zero.  If the coordinate matrix
% has more than three columns, the additional columns are simply ignored.
%
% The list of detectors should be specified as a cell array of site identifiers
% or channel names, which are passed to the WDETECTORS function to retrieve the
% corresponding detector location and response data.  Site identifiers are
% determined by taking the first character of each channel name.  The following
% sites are recognized by the WDETECTORS function.
%
%   G  GEO
%   H  LIGO Hanford
%   L  LIGO Livingston
%   V  Virgo
%   0  Fiducial reference detecor
%
% The resulting fplus, fcross, deltat matrices contain a column for each detector
% and a row for each coordinate.
%
% The fiducial reference detector is provided for validation purposes.  It is
% positioned in the direction of the North pole, at a distance equal to one
% speed of light travel time from the Earth center, and has arms along the
% positive X and Y axis in the Earth fixed geocentric coordinate system.
%
% See also WDETECTORS, WBASELINE, WTILESKY, and WSKYMAP.

%   Shourov Chatterji   shourov@ligo.caltech.edu
%   Albert Lazzarini    lazz@ligo.caltech.edu
%   Antony Searle       antony.searle@anu.edu.au
%   Leo Stein           lstein@ligo.caltech.edu
%   Patrick Sutton      psutton@ligo.caltech.edu
%   Massimo Tinto       massimo.tinto@jpl.nasa.gov

% $Id: wresponse.m 2753 2010-02-26 21:33:24Z jrollins $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify correct number of input arguments
error(nargchk(2, 2, nargin, 'struct'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       validate command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% validate coordinate matrix
if size(coordinates, 2) < 2,
  error('coordinates must be at least a two column matrix [theta phi]');
elseif size(coordinates, 2) == 2,
  coordinates = [coordinates zeros(size(coordinates, 1), 1)];
end

% extract spherical coordinates and polarizations
theta = coordinates(:, 1);
phi = coordinates(:, 2);
psi = coordinates(:, 3);

% validate coordinates
if any((theta < 0) | (theta > pi)),
  error('theta outside of range [0, pi]');
end
if any((phi < 0) | (phi >= 2 * pi)),
  error('phi outside of range [0, 2 pi)');
end
if any((psi < 0) | (psi >= pi)),
  error('psi outside of range [0, pi)');
end

% force cell array of detector names
detectors = wmat2cell(detectors);

% force one dimensional cell array
detectors = detectors(:);

% number of detectors
numberOfDetectors = length(detectors);

% force uppercase detector names
detectors = upper(detectors);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             direction to source                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cartesian source direction
sourceDirection = [sin(theta) .* cos(phi) ...
                   sin(theta) .* sin(phi) ...
                   cos(theta)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             polarization tensors                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute polarization tensors
m1 = -sin(phi) .* cos(psi) + cos(phi) .* cos(theta) .* sin(psi);
m2 = +cos(phi) .* cos(psi) + sin(phi) .* cos(theta) .* sin(psi);
m3 = -sin(theta) .* sin(psi);
n1 = +sin(phi) .* sin(psi) + cos(phi) .* cos(theta) .* cos(psi);
n2 = -cos(phi) .* sin(psi) + sin(phi) .* cos(theta) .* cos(psi);
n3 = -sin(theta) .* cos(psi);
mm = [m1.*m1, m1.*m2, m1.*m3, m2.*m1, m2.*m2, m2.*m3, m3.*m1, m3.*m2, m3.*m3];
mn = [m1.*n1, m1.*n2, m1.*n3, m2.*n1, m2.*n2, m2.*n3, m3.*n1, m3.*n2, m3.*n3];
nm = [n1.*m1, n1.*m2, n1.*m3, n2.*m1, n2.*m2, n2.*m3, n3.*m1, n3.*m2, n3.*m3];
nn = [n1.*n1, n1.*n2, n1.*n3, n2.*n1, n2.*n2, n2.*n3, n3.*n1, n3.*n2, n3.*n3];
eplus = mm - nn;
ecross = mn + nm;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              detector geometry                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% retrieve detector location and response data
[locations, responses] = wdetectors(detectors);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          compute detector response                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize results
fplus = [];
fcross = [];
deltat = [];

% begin loop over detectors
for detectorNumber = 1 : numberOfDetectors,

  % determine detector response
  fplus = [fplus eplus * responses{detectorNumber}];
  fcross = [fcross ecross * responses{detectorNumber}];
  deltat = [deltat -sourceDirection * locations{detectorNumber}];

% end loop over detectors
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          return to calling function                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to calling function
return;
