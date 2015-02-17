function y = jdn2gmst(x)
% JDN2GMST converts Julian day number to Greenwich mean sidereal time
%       y = jdn2gmst(x)
%
%    x:  Julian day number (JDN) in days
%    y: Greenwich mean sidereal time (GMST) in hours 
%
%    for details see http://aa.usno.navy.mil/faq/docs/GAST.html
%
%  - supports input vectors
%
% $Id: jdn2gmst.m,v 1.1 2007/02/16 23:00:54 shantanu Exp $

% calculate Julian day of the previous midnight (0h UT)
   x0 = floor(x - 0.5) + 0.5;

% calculate the number of hours passed since that time
   H = (x - x0) * 24;

% For both 'x' and 'x0' compute the number of days 
% from 2000 January 1, 12h UT, Julian day 2451545.0
  D  = x  - 2451545.0;
  D0 = x0 - 2451545.0;

% The number of centuries since the year 2000
  T = D/36525;

% The Greenwich mean sidereal time in hours
   F = 6.697374558 + 0.06570982441908 * D0 + ...
       1.00273790935 * H + 0.000026 * (T.^2);

% reduce the GMST to the 24-hours range
   y = mod(F, 24);

return
