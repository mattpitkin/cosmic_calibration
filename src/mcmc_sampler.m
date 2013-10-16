function [post_samples, logPsamples] = mcmc_sampler(data, likelihood, ...
          model, prior, extraparams, varargin)

% function [post_samples, logPsamples] = mcmc_sampler(data, likelihood, ...
%           model, prior, extraparams, optargs)
%
% This function performs an MCMC exploration of a posterior defined by the
% input likelihood and prior (given a set of data, a model, and a set of
% extra model parameters).
%
% Required arguments:
%   data - a cell array containing information required for the model and
%          likelihood calculation, such as a set of data.
%   likelihood - a function handle, or string, giving the log likelihood
%                function calculted using the data and model.
%   model - a function handle, or string, giving the model based on a set
%           of input parameters. This will be passed to the likelihood
%           function.
%   prior - the prior should be a cell array with each cell containing five
%           values:
%               parameter name (string)
%               prior type (string) e.g. 'uniform', 'gaussian' or
%                 'jeffreys'
%               minimum value (for uniform prior), or mean value (for
%                 Gaussian prior)
%               maximum value (for uniform prior), or width (for Gaussian
%                 prior)
%               parameter behaviour (string):
%                   'reflect' - if the parameters reflect off the
%                     boundaries
%                   'cyclic'  - if the parameter space is cyclic
%                   'fixed'   - if the parameters have fixe boundaries
%                   ''        - for gaussian priors
%           e.g., prior = {'h0', 'uniform', 0, 1, 'reflect'; 
%                          'r', 'gaussian', 0, 5, '';
%                          'phi', 'uniform', 0, 2*pi, 'cyclic'};
%
%   extraparams - a cell array of fixed extra parameters (in addition
%                 to those specified by prior) used by the model 
%                 e.g.  extraparams = {'phi', 2;
%                                      'x', 4};
%                 This can be empty of no additional parameters are
%                 required
%
%
% Optional arguments:
%  Set these via e.g. 'Nmcmc', 100000
%   Nmcmc - the number of MCMC iterations that will be used to form the
%           output chain of posterior samples. If not supplied this will
%           default to 100000.
%   Nburnin - the number of MCMC iterations that will be used as a burn-in
%             period. These will by default be discarded from the posterior
%             samples. If not supplied this will default to 100000. 
%   temperature - the temperature used for a simulated annealing stage
%                 during the burn-in. This should be between 0 and 1. If
%                 not suppled this will default to 1 e.g. no annealing.
%   dimupdate - the number of parameters/dimensions to update via the
%               proposal on each MCMC iteration. If not set this will
%               default to updating all parameters.
%   outputbi - if set to 1 this will output the burn-in samples as well as
%              the posterior samples. If not set this will default to not
%              output the burn-in samples.
%   propscale - a positive value with which the scale the proposal
%               distribution. If not set the default is 0.1
%   recalcprop - the number of samples after which to recalculate the
%                proposal distribution during burn-in. Default is 1000.

global verbose;

% get the number of parameters from the prior array
D = size(prior,1);

% get optional input arguments
optargin = size(varargin,2);

% set all optional values to their default values
Nmcmc = 100000; % default number of MCMC posterior samples to 100000
Nburnin = 100000; % default number of burn in samples to 100000
temperature = 1; % set temperature to 1 i.e. no annealing during burn in
dimupdate = 0; % update all dimensions duing each MCMC sample
outputBi = 0; % do not output the burn in samples
propscale = 0.1; % scale the proposal distribution by this factor
recalciter = 1000; % number of samples from which to recalculate the proposal

if optargin > 1
    for i = 1:2:optargin
        if strcmpi(varargin{i}, 'Nmcmc') % number of MCMC samples
            if ~isempty(varargin{i+1})
                if varargin{i+1} < 1
                    fprintf(1, 'Number of MCMC samples is silly. Reverting to default of %d\n', Nmcmc);
                else
                    Nmcmc = varargin{i+1};
                end
            end
        elseif strcmpi(varargin{i}, 'Nburnin') % number of burn in samples
            if ~isempty(varargin{i+1})
                if varargin{i+1} < 1
                    fprintf(1, 'Number of burn in samples is silly. Reverting to default of %d\n', Nburnin);
                else
                    Nburnin = varargin{i+1};
                end
            end
        elseif strcmpi(varargin{i}, 'temperature') || strcmpi(varargin{i}, 'temp') % burn-in chain initial temperature
            if ~isempty(varargin{i+1})
                if varargin{i+1} < 0 || varargin{i+1} > 1
                    fprintf(1, 'Temperature is out of range 0->1. Reverting to default of %d\n', temperature);
                else
                    temperature = varargin{i+1};
                end
            end
        elseif strcmpi(varargin{i}, 'dimupdate') % number of dimensions/parameters to update on each sample
            if ~isempty(varargin{i+1})
                if varargin{i+1} < 0 || varargin{i+1} > D
                    fprintf(1, 'Number of parameters to update on each sample is out of range 0->%d. Reverting to default of all parameters\n', D);
                else
                    dimupdate = varargin{i+1};
                end
            end
        elseif strcmpi(varargin{i}, 'outputBi') % output the burn in chain along with the posterior
            if ~isempty(varargin{i+1})
                if varargin{i+1} ~= 1 || varargin{i+1} ~= 0
                    fprintf(1, 'Argument for outputing burn in samples should be 0 (False) or 1 (True). Defaulting to 0\n');
                else
                    outputBi = varargin{i+1};
                end
            end
        elseif strcmpi(varargin{i}, 'propscale') % value by which to scale the proposal distribution
            if ~isempty(varargin{i+1})
                if varargin{i+1} < 0
                    fprintf(1, 'Proposal scale factor should be greater than 0. Defaulting to 0.1\n');
                else
                    propscale = varargin{i+1};
                end
            end
        elseif strcmpi(varargin{i}, 'recalcprop') % number of samples after which to recalculate the proposal
            if ~isempty(varargin{i+1})
                if varargin{i+1} < 0 || mod(varargin{i+1}, 1) ~= 0
                    fprintf(1, 'Number of samples must be an integer greater than 0. Defaulting to 1000\n');
                else
                    recalciter = varargin{i+1};
                end
            end
        end
    end
end
    
% get all parameter names
parnames = prior(:,1);

if ~isempty(extraparams)
    extraparnames = extraparams(:,1);
    extraparvals = extraparams(:,2);
    parnames = cat(1, parnames, extraparnames);
else
    extraparvals = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% some initial values for the MCMC

% check whether likelihood is a function handle, or a string that is a 
% function name
if ischar(likelihood)
    flike = str2func(likelihood);
elseif isa(likelihood, 'function_handle')
    flike = likelihood;
else
    error('Error... Expecting a model function!');
end

% allocate memory for the samples
Ntot = Nmcmc + Nburnin;
post_samples = zeros(Ntot, D); % posterior samples
logPsamples = zeros(Ntot, 1); % log posterior values

% acceptance ratio
acc = 0;
accBi = 0;

if verbose
    acctmp = 0;
    niter = 100;
end

% draw initial sample from the prior, get posterior and then rescale
inisample = zeros(1, D);
newPrior = -inf;
for i=1:D
    priortype = char(prior(i,2));
    p4 = cell2mat(prior(i,4));
    p3 = cell2mat(prior(i,3));
    
    % currently only handles uniform or Gaussian priors
    if strcmp(priortype, 'uniform')
        inisample(i) = p3 + (p4-p3)*rand;
        
        pv = -log(p4-p3);
        newPrior = logplus(newPrior, pv);
    elseif strcmp(priortype, 'gaussian')
        p3 = cell2mat(prior(i,3));
        inisample(i) = p3 + p4*randn;
        
        pv = -l2p - inisample(i)^2/2;
        newPrior = logplus(newPrior, pv);
    elseif strcmp(priortype, 'jeffreys')
        % uniform in log space
        inisample(i) = 10.^(log10(p3) + (log10(p4)-log10(p3))*rand);
        
        pv = -log(10^(inisample(i)*(log10(p4) - log10(p3)) + log10(p3)));
        newPrior = logplus(newPrior, pv);
    end
end

parvals = cat(1, num2cell(inisample'), extraparvals);
logL = feval(flike, data, model, parnames, parvals);

% now scale the parameters, so that uniform parameters range from 0->1, 
% and Gaussian parameters have a mean of zero and unit standard deviation
inisample = scale_parameters(prior, inisample);

post_samples(1,:) = inisample;
logPsamples(1) = logL + newPrior;

% set up initial covariance matrix to be unity
cholmat = propscale*eye(D); % an identity matrix

loginvtemp = log(1/temperature); 

i = 2;

% run the MCMC
while i < Ntot+1
    if i < (1 + Nburnin/2)
        % use annealing, so scale the temperature - have it fall and reach
        % 1 half way through the burn in - also only update every
        % recalciter samples
        if i == 2 || mod(i, recalciter) == 0
            temptemp = temperature*exp(loginvtemp * (i-1)/(Nburnin/2));
        end
    else
        temptemp = 1;
    end
    
    if i < Nburnin && mod(i, recalciter) == 0
        % if no new points have been accepted rescale the covariance matrix
        if accBi == 0
            covmat = cholmat*cholmat';
            % randomly make proposal bigger or smaller
            [l, d] = mchol(2*rand*covmat);
            cholmat = l.'*sqrt(d);
            i = i-recalciter;
        else
            [l, d] = mchol(propscale*cov(post_samples(i-(recalciter-1):i,:)));
            cholmat = l.'*sqrt(d);
        end
        
        accBi = 0;
    end
    
    [post_samples(i,:), logPsamples(i), ar] = ...
        draw_mcmc_sample(post_samples(i-1,:), cholmat, ...
        logPsamples(i-1), prior, data, likelihood, model, parnames, ...
        extraparvals, temptemp, dimupdate);
    
    % check acceptance rate
    if i < Nburnin
        if ar
            accBi = accBi + 1;
        end
    else
        if ar
            acc = acc + 1;
        end
    end
    
    % print out acceptance rate
    if verbose
        if ar
            acctmp = acctmp + 1;
        end
        
        if mod(i, niter) == 0
            fprintf(1, '%i: Acceptance rate: %.1f%%\n', i, ...
                100*acctmp/niter);
            acctmp = 0;
        end
    end
    
    i = i + 1;
end

% rescale the samples back to their true ranges
for i=1:Ntot
    post_samples(i,:) = ...
        rescale_parameters(prior, post_samples(i,:));
end

% only output points after burn-in unless specified to output all points
if ~outputBi
    post_samples = post_samples(end-(Nmcmc-1):end,:);
    logPsamples = logPsamples(end-(Nmcmc-1):end);
end
    
return
