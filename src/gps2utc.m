function y = gps2utc(x)
% GPS2UTC converts GPS time to UTC time
%      y = gps2utc(x)
%   x:  GPS time in seconds
%   y: UTC time as year, month, day, hour, min, sec
%
%  - vectorized
%
% $Id: gps2utc.m,v 1.1 2007/02/16 23:00:54 shantanu Exp $

% the number of days within this GPS time
daySec = 24*3600;
Nday = x / daySec;

% The GPS clock started on Jan.6, 1980. This corresponds to
N0 = datenum('6-Jan-1980');

% shift the starting date from Jan.6, 1980 to Jan.1, year 0
Nday = Nday + N0;

%  Output as date vector
y = datevec(Nday);
return
