function JD = calen2jdn(year, month, day, hour, min, sec)
% CALEN2JDN converts the calendar date to Julian day number (JDN)
%   JD = calen2jdn(year, month, day, hour, min, sec)
%       input:  calendar date as year, month, day, hour, min, sec
%       output: Julian Day (JD) in days
%
%   JD = calen2jdn(dateVec)
%       input: date vector with columns (year,month,day,hour,min,sec)
%       output: vector of Julian Day (JD) in days
%
%   for details see http://en.wikipedia.org/wiki/Julian_day
%
%   $Id: calen2jdn.m,v 1.1 2007/02/16 23:00:54 shantanu Exp $

if(nargin == 1)
    [mRow,nCol] = size(year);
    if(nCol ~= 6)
        msgId = 'calen2jdn:tooFewInputs';
        error(msgId,'%s: date vector needs 6 columns!',msgId);
    end
    inData = year;
    year = inData(:,1);
    month = inData(:,2);
    day = inData(:,3);
    hour = inData(:,4);
    min = inData(:,5);
    sec = inData(:,6);
else
    if (nargin < 6)
        msgId = 'calen2jdn:tooFewInputs';
        error(msgId,'%s: # of inputs %d less than 6',msgId,nargin);
    end
end

a = floor((14 - month)/12);
Y = year + 4800 - a;
M = month + 12 * a - 3;

% For a date in the Gregorian calendar (at noon):
JDN = day + floor((153 * M + 2)/5) + 365 * Y + floor(Y/4) - ...
      floor(Y/100) + floor(Y/400) - 32045;

% Full Julian Date, not counting leap seconds
JD = JDN + (hour - 12)/24 + min/1440 + sec/86400;

return
