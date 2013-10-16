function [sampleOut, logPOut, ar] = draw_mcmc_sample(sampleIn, cholmat, ...
    logPIn, prior, data, likelihood, model, parnames, extraparvals, ...
    temperature, dimupdate)

% function [sample, logL, ar] = draw_mcmc_sample(livepoints, cholmat, ...
%    logLmin, prior, data, likelihood, model, Nmcmc, parnames, ...
%    extraparvals, temperature, dimupdate)
%
% This function will draw a new multi-dimensional sample using the proposal
% distribution defined by cholmat. The MCMC will use a Students-t (with N=2
% degrees of freedon) proposal distribution based on the Cholesky
% decomposed covariance matrix, cholmat. The posterior probability will be
% calculated using the prior and the likelihood. extraparvals is a vector
% of additional parameters needed by the model. The sample to be returned
% will be based on the Metropolis-Hastings rejection criterion. The
% proposal used will always be symmetric, so the proposal ratio is unity.
% 
% The dimupdate specifies the number of dimensions/parameters that you wish
% to update with each newly drawn sample. If this is 0 or greater than the
% actual number of dimensions then all parameters will be updated,
% otherwise the number of specified parameters will be updated, with the
% updated parameters chosen at random.
%
% The returned value ar is 1 is the sample is accepted, or 0 if rejected.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ar = 0;

l2p = 0.5*log(2*pi); % useful constant
Ndegs = 2; % degrees of freedom of Students't distribution

Npars = length(sampleIn);

% draw new points from mulitvariate Gaussian distribution 
gasdevs = randn(Npars, 1);
sampletmp = (cholmat*gasdevs)';

% check how many of the dimensions you want to update
if dimupdate > 0 && dimupdate < Npars
    r = randperm(Npars);
    sampletmpnew = zeros(1, Npars);
    sampletmpnew(r(1:dimupdate)) = sampletmp(r(1:dimupdate));
    sampletmp = sampletmpnew;
end

% calculate chi-square distributed value
chi = sum(randn(Ndegs,1).^2);
            
% add value onto old sample
sampletmp = sampleIn + sampletmp*sqrt(Ndegs/chi);

% check sample is within the (scaled) prior
newPrior = -inf;
outofbounds = 0;

for j=1:Npars
    priortype = char(prior(j,2));
    p3 = cell2mat(prior(j,3));
    p4 = cell2mat(prior(j,4));
            
    if strcmp(priortype, 'uniform')
        behaviour = char(prior(j,5));
                
        dp = 1;
                
        if sampletmp(j) < 0 || sampletmp(j) > 1
            if strcmp(behaviour, 'reflect')
                % reflect the value from the boundary
                sampletmp(j) = 1 - mod(sampletmp(j), dp);
            elseif strcmp(behaviour, 'cyclic')
                % wrap parameter from one side to the other
                while sampletmp(j) > 1
                    sampletmp(j) = sampletmp(j) - 1;
                end
                        
                while sampletmp(j) < 0
                    sampletmp(j) = sampletmp(j) + 1;
                end
            else
                outofbounds = 1;
                break;
            end
        end
                
        pv = -log(p4-p3);
        newPrior = logplus(newPrior, pv);
                
    elseif strcmp(priortype, 'gaussian')
        pv = -l2p - sampletmp(j)^2/2;
        newPrior = logplus(newPrior, pv);
    elseif strcmp(priortype, 'jeffreys')
        behaviour = char(prior(j,5));
                
        dp = 1;
               
        if sampletmp(j) < 0 || sampletmp(j) > 1
            if strcmp(behaviour, 'reflect')
                % reflect the value from the boundary
                sampletmp(j) = 1 - mod(sampletmp(j), dp);
            else
                outofbounds = 1;
                break;
            end
        end
                
        pv = -log(10^(sampletmp(j)*(log10(p4) - log10(p3)) + log10(p3)));
        newPrior = logplus(newPrior, pv);
    end
end

if outofbounds % reject point
    sampleOut = sampleIn;
    logPOut = logPIn;
    return;
end

% rescale sample back to its proper range for likelihood
sc = rescale_parameters(prior, sampletmp);
    
% get likelihood of new point
logLnew = feval(likelihood, data, model, parnames, cat(1, num2cell(sc), ...
    extraparvals));
    
% get posterior probability
logPnew = logLnew + newPrior;
    
% accpet/reject point using Metropolis-Hastings criterion
if log(rand) > (logPnew - logPIn) * temperature
    % reject the new sample
    sampleOut = sampleIn;
    logPOut = logPIn;
else
    % accept the new sample
    sampleOut = sampletmp;
    logPOut = logPnew;
    ar = 1;
end

return
