function y = gps2sidereal(x)
% GPS2SIDEREAL converts GPS time to sidereal time
%   y = gps2sidereal(x)
%       x:  GPS time in seconds (x)
%       y: Greenwich apparent sidereal time in hours (y)
%
%  - supports input vectors of GPS times
%
%  $Id: gps2sidereal.m,v 1.1 2007/02/16 23:00:54 shantanu Exp $

% convert to integer in case of running in compiled matlab
% x = strassign(x);

% calculate UTC
  a = gps2utcCorr(x);

% calculate Julian day
  JD = calen2jdn(a);

% calculate Greenwich apparent sidereal time (GAST) in hours
  y = jdn2gast(JD);

return
