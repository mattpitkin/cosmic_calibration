function y = lsCorrRem(x)
% LSCORRREM removes the correction due leap seconds
%   y = lsCorrRem(x)
%       x:  GPS time in seconds (corrected)
%       y: GPS time in seconds (uncorrected)
%
%   - vectorized for 2-D arrays
% $Id: lsCorrRem.m,v 1.1 2007/02/16 23:00:54 shantanu Exp $

% load the table of GPS times when leap second corrections were applied
% Note: the times are CORRECTED for leap seconds
% -- the following may not work in compiled matlab, 
%       store values in z directly
z = load('leapSecGPStimesCorr.dat');
z = z' ;    
%z = [46828801 78364802 109900803 173059204 252028805 315187206 346723207 ...
%     393984008  425520009 457056010 504489611 551750412 599184013 820108814] ;

% FIND largest leap second times less that GPS time
% IF none are found (i.e. GPS time is earlier than first leap second)
%   SET output to input GPS
% ELSE
%   CALCULATE output = input GPS - index of largest lesser leap-second
% ENDIF
y = x;
[mRow,nCol] = size(x);
for iRow = 1:mRow
    for jCol = 1:nCol
        xTest = x(iRow,jCol);
        maxLeapIdx = max(find(z <= xTest));
        if(~isempty(maxLeapIdx) && (maxLeapIdx > 0))
            y(iRow,jCol) = xTest - maxLeapIdx;
        end
    end
end
return
