% perform the tests in calibrationtest.m, but this time with an aLIGO noise
% curve and an true inspiral signal.

global verbose;
verbose = 1;

global DEBUG;
DEBUG = 0;

% create an inspiral signal of two neutron stars
m1 = 1.4; % mass in Msun
m2 = 1.4; % mass in Msun
iota = 0; % orientation angle (rads) ( 0=face on system)
psi = 0.3; % polarisation angle (rads) 
det = 'H1'; % use H1 detector
z = 0; % approximately redsift of zero (as close)
D = 15; % luminosity distance of 50 Mpc
tc = 900000000; % coalesence time
phic = 4.5; % phase at coalescence (rads)
ra = 0.3; % right ascension (rads)
dec = -0.76; % declination (rads)
fmin = 20; % initial frequency of signal
fmax = 250; % final frequency of signal

% create antenna pattern look-up table
psis = linspace(-pi/4, pi/4, 100);
fps = zeros(100,1);
fcs = zeros(100,1);
for i=1:length(psis)
    [fp, fc] = antenna_pattern(det, ra, dec, psis(i), tc);
    fps(i) = fp;
    fcs(i) = fc;
end
resplut = [psis', fps, fcs]';
size(resplut)

parnames = {'fmin', 'fmax', 'D', 'm1', 'm2', 'z', 'tc', 'iota', 'psi', 'ra', 'dec', 'phic', 'det', 'resplut'};
parvals = {fmin, fmax, D, m1, m2, z, tc, iota, psi, ra, dec, phic, det, resplut};

Nd = 2000;
dt = 1/Nd;

% set frequencies
%fbins = linspace(0, (Nd/2), (Nd/2)+1);
fbins = linspace(0, 1000, 200);

% create a signal
hf = freqdomaininspiral(fbins, parnames, parvals);

% create a coloured spectrum using aLIGO full design sensitivity noise curve
fp = fopen('../data/aligo_sensitivity.txt', 'r'); 
N = textscan(fp, '%f%f%f%f%f%f', 'CommentStyle', '#');
fclose(fp);

alfreqs = N{1}; % advanced ligo frequencies - column 1
alamp = N{5}; % use fifth column for full design sensitivity (other columns have early run estimated sensitivities)

% prepend values at 0 frequency
alfreqs = [0; alfreqs]; % extend values to start at 0
alamp = [alamp(1); alamp];

cn = interp1(alfreqs, alamp, fbins); % Interpolates to return alamp values for fbins points.
cn = cn';

% add in the signal to create the real dataset
truedata = hf; % coloured noise plus created signal

idx = fbins >= fmin & fbins <= fmax;
% fbins between fmin and fmax

% run the model at new frequencies to update hp and hc global variables
tic
hf = freqdomaininspiral(fbins(idx), parnames, parvals);
toc

scalefactor = 1.0; % a "calibration difference" scale factor

data{1} = fbins(idx);
data{2} = truedata(idx)*scalefactor;
data{3} = (scalefactor*cn(idx)).^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define MCMC parameters
likelihood = @logL;
model = @freqdomaininspiral;

%prior =  {'sf1', 'uniform', 0, 2, 'reflect';
%          'D', 'gaussian', 15, 0.2, '';
%          'iota', 'gaussian', 0, 0.1, ''};
prior =  {'sf1', 'jeffreys', 1e-2, 1e3, '';
          'D', 'gaussian', 15, 0.2, '';
          'iota', 'gaussian', 0, 0.1, '';
          'm1', 'uniform', 1.39, 1.41, '';
          'm2', 'uniform', 1.39, 1.41, ''};

      
extraparams = {'fmin', fmin; ...
               'fmax', fmax; ...
               %'D', D; ...
               %'m1', m1; ...
               %'m2', m2; ...
               'z', z; ...
               'tc', tc; ...
               %'iota', iota; ...
               'psi', psi; ...
               'ra', ra; ...
               'dec', dec; ...
               'phic', phic; ...
               'det', det; ...
               'resplut', resplut'; ...
               'update', 1}; % no need to update the inspiral signal
               
% call MCMC routine
%Nmcmc = 100000;
%Nburnin = 100000;

%[post_samples, logP, logPrior] = mcmc_sampler(data, likelihood, model, prior, ...
%    extraparams, 'Nmcmc', Nmcmc, 'Nburnin', Nburnin, 'NensembleStretch', 5, ...
%    'temperature', 1e-30, 'outputbi', 1);

%[post_samples, logP, logPrior] = mcmc_sampler(data, likelihood, model, prior, ...
%    extraparams, 'Nmcmc', Nmcmc, 'Nburnin', Nburnin, ...
%    'temperature', 1e-30, 'outputbi', 1);

tolerance = 0.1;
Nlive = 1000;
Nmcmc = 200;

[logZ, nest_samples, post_samples] = nested_sampler(data, Nlive, Nmcmc, ...
  tolerance, likelihood, model, prior, extraparams);

