function [val, idx] = closest(a, b)

% [val, idx] = closest(a, b)
%
% Find the value in vector b that is closest to the value given in a.

[y, idx] = min(abs(b-a));

val = b(idx);