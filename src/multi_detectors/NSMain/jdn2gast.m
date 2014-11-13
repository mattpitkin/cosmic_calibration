function y = jdn2gast(x)
% JDN2GAST converts Julian day number to Greenwich apparent sidereal time
%       y = jdn2gast(x)
%
%    x:  Julian day number (JDN) in days
%    y: Greenwich apparent sidereal time (GAST) in hours 
%
%    for details see http://aa.usno.navy.mil/faq/docs/GAST.html
%
%  - supports input vectors
%
%  $Id: jdn2gast.m,v 1.1 2007/02/16 23:00:54 shantanu Exp $

% first calculate Greenwich mean sidereal time (GMST) [in hours]
   y = jdn2gmst(x);

% the number of days from Jan.1 2000, 12h UT when 
% Julian date is 2451545.0
   D = x - 2451545.0;

% calculate the Longitude of the ascending node of the Moon [in deg]
   Omega = 125.04 - 0.052954 * D;

% calculate the Mean Longitude of the Sun [in deg]
   L = 280.47 + 0.98565 * D;

% calculate the nutation in longitude approximately [in hours]
   deltaPsi = -0.000319 * sind(Omega) - 0.000024 * sind(2*L);

% calculate the obliquity [in deg]
   Epsilon = 23.4393 - 0.0000004 * D;

% The Greenwich apparent sidereal time is obtained by adding a correction 
% to the Greenwich mean sidereal time. The correction term is called the 
% nutation in right ascension or the equation of the equinoxes (EqEq):
   EqEq = deltaPsi .* cosd(Epsilon);
   y = y + EqEq;

return
