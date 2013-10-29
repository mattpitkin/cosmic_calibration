function [ z ] = logplus( x, y )

%   z = logplus( x, y )
%   
%   This function performs the addition of two logarithmic values of the
%   form log(exp(x) + exp(y)). It avoids problems of dynamic range when the
%   exponentiated values of x or y are very large or small. If x and y are
%   vectors then adjacent values in each will be added.

% make sure x and y are vectors
if ~isvector(x) || ~isvector(y)
    error('Either x or y is not a vector!');
end

% make sure both are the same length
lx = length(x);
ly = length(y);

if lx ~= ly
    error('Vectors x and y are not the same length!');
end

% allocate z vector (default to inf)
z = ones(lx,1)*inf;

% deal with both values being -infinity
j = isinf(x) & isinf(y) & (x < 0) & (y < 0);
z(j) = -inf;

i = (x > y) & ~(isinf(x) & isinf(y));
z(i) = x(i) + log(1+exp(y(i) - x(i)));
i = (x <= y) & ~(isinf(x) & isinf(y));
z(i) = y(i) + log(1+exp(x(i) - y(i)));

end

