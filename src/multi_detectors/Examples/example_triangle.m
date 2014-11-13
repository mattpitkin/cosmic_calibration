% example: estimate amplitude and initial phase of a 
% sinusoidal signal in additive white gaussian noise

global verbose;
verbose = 1;
global DEBUG;
DEBUG = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define nested sampling parameters
Nlive = 500;

Nmcmc = input('Input number of iterations for MCMC sampling: (enter 0 for multinest sampling)\n');
tolerance = 0.1;
likelihood = @logL_model_likelihood;
model = @triangle_model;
prior = {'x', 'uniform', 0, 1, 'fixed'};
extraparams = {'gradient', 1}; % fixed signal frequency

data = 1; % dummy data

% called nested sampling routine
[logZ, nest_samples, post_samples] = nested_sampler(data, Nlive, Nmcmc, ...
  tolerance, likelihood, model, prior, extraparams);

