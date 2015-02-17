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
y = data{2}; % frequency data for detector 1
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
df = fs(2)-fs(1);


%ys = zeros(length(y{1}(:,1)), length(y));
%Cs = zeros(length(C{1}(:,1)), length(C));
%mds = zeros(size(md));
logL= zeros(length(data),1);
dh = zeros(length(data),1);
hh = zeros(length(data),1);
dd = zeros(length(data),1);

for i = 1:length(C)
    % ys, Cs, and mds ingore all NaN values in data set
    ys = y{i}(v(:,i));
    Cs = C{i}(v(:,i));
    mds = md(v(:,i),i);
    
    
    dh(i) = real((conj(ys.') ./ Cs.')*mds); % model crossed with data
    hh(i) = real((conj(mds.') ./ Cs.')*mds); % model crossed with model
    dd(i) = real((conj(ys.') ./ Cs.')*ys); % data crossed with data
    
    logL(i) = 4*df*(dh(i) - 0.5*(hh(i)+dd(i))); % finds the log likelihood
    
end

logL = sum(logL);
fprintf(1, '%le\n', logL);

if isnan(logL)
    error('Error: log likelihood is NaN!');
end

return