function y = gps2utcCorr(x)


% GPS2UTCCORR converts GPS time to UTC time with leap second correction
%
%   input:  GPS time in seconds
%   output: UTC time as year, month, day, hour, min, sec



% first remove the correction due leap seconds

x = lsCorrRem(x);


% then calculate UTC from GPS

y = gps2utc(x);
