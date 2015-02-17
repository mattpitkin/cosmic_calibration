function logL = logL(data, model, parnames, parvals)

% logL = logL(data, model, parnames, parvals)
%
% This function will compute the log likelihood of a multivariate
% gaussian:
%
%     L = 1/sqrt((2 pi)^N det C)
%         exp[-0.5*(y - model(x,params))^T * inv(C) * (y - model(x,params))]
%
% The input parameters are:
%     data - a cell array with three columns
%            { x values, y values, C: covariance matrix }
%     NOTE: if C is a single number, convert to a diag covariance matrix
%     model - the function handle for the signal model.
%     parnames - a cell array listing the names of the model parameters
%     parvals - a cell array containing the values of the parameters given
%         in parnames. These must be in the same order as in parnames. 
%         If parvals is an empty vector the noise-only likelihood will be 
%         calculated.
%
% -------------------------------------------------------------------------
%           This is the format required by nested_sampler.m.
% -------------------------------------------------------------------------

% check whether model is a string or function handle
if ischar(model)
    fmodel = str2func(model);
elseif isa(model, 'function_handle')
    fmodel = model;
else
    error('Error... Expecting a model function!');
end

% get data values from cell array
fs = data{1}; % frequency series
y = data{2}; % frequency data
C = data{3}; % uncalibrated noise psd 

% evaluate the model
if isempty(parvals)
    % if parvals is not defined get the null likelihood (noise model
    % likelihood)
    md = 0;
else
    md = feval(fmodel, fs, parnames, parvals);
    
    % if the model returns a NaN then set the likelihood to be zero (e.g. 
    % loglikelihood to be -inf
    %if isnan(md)
    %    logL = -inf;
    %    return;
    %end
end

v = ~isnan(md);
ys = y(v);
Cs = C(v);
mds = md(v);

df = fs(2)-fs(1);

% calculate the log likelihood
dh = real((conj(ys).' ./ Cs.')*mds); % model crossed with data
hh = real((conj(mds).' ./ Cs.')*mds); % model crossed with model
dd = real((conj(ys).' ./ Cs.')*ys); % data crossed with data

logL = 4*df*(dh - 0.5*(hh + dd));

fprintf(1, '%le\n', logL);

if isnan(logL)
    error('Error: log likelihood is NaN!');
end

return
