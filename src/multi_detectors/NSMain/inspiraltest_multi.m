    % perform the tests in calibrationtest.m, but this time with an aLIGO noise
    % curve and an true inspiral signal.


clear all
close all

global verbose;
verbose = 1;

global DEBUG;
DEBUG = 0;

% create an inspiral signal of two neutron stars
m1 = 1.4; % mass in Msun
m2 = 1.4; % mass in Msun
iota = 0; % orientation angle (rads) ( 0=face on system)
psi = 0.3; % polarisation angle (rads) 

% Multiple Detectors 
det = {'H1','L1','V1'}; 


z = 0; % approximately redsift of zero (as close)
D = 15; % luminosity distance of 50 Mpc
tc = 900000000; % coalesence time
phic = 4.5; % phase at coalescence (rads)
ra = 0.3; % right ascension (rads)
dec = -0.76; % declination (rads)
fmin = 50; % initial frequency of signal
fmax = 1600; % final frequency of signal

% create antenna pattern look-up table
psis = linspace(-pi/4, pi/4, 100);
fps = zeros(100,length(det));
fcs = zeros(100,length(det));
fp = zeros(1,length(det));
fc = zeros(1,length(det));

for i=1:length(psis)
    for j = 1:length(det)
        [fp(j), fc(j)] = antenna_pattern(det{j}, ra, dec, psis(i), tc);
        fps(i,j) = fp(j);
        fcs(i,j) = fc(j);
    end
end
resplut = [psis', fps, fcs]';
% size(resplut)

parnames = {'fmin', 'fmax', 'D', 'm1', 'm2', 'z', 'tc', 'iota', 'psi', 'ra', 'dec', 'phic', 'det', 'resplut'};
parvals = {fmin, fmax, D, m1, m2, z, tc, iota, psi, ra, dec, phic, det, resplut};

Nd = 2000;
dt = 1/Nd;

% set frequencies
%fbins = linspace(0, (Nd/2), (Nd/2)+1);
fbins = linspace(0, 1000, 200);

% create a signal
hf = freqdomaininspiral(fbins, parnames, parvals);



% create a coloured spectrum using aLIGO H1 full design sensitivity noise curve
fp = fopen('data/aligo_sensitivity.txt', 'r'); 
N = textscan(fp, '%f%f%f%f%f%f', 'CommentStyle', '#');
fclose(fp);
alfreqs = N{1}; % advanced ligo frequencies - column 1
alamp = N{5}; % use fifth column for full design sensitivity (other columns have early run estimated sensitivities)
% prepend values at 0 frequency
alfreqs = [0; alfreqs]; % extend values to start at 0
alamp = [alamp(1); alamp];
cnH1 = interp1(alfreqs, alamp, fbins); % Interpolates to return alamp values for fbins points.
cnH1 = cnH1';
% add in the signal to create the real dataset
truedataH1 = hf(:,1); % coloured noise plus created signal
truedataL1 = hf(:,2);
% create a coloured spectrum using Virgo V1 design sensitivity noise curve



fpv1 = fopen('data/VIRGO_DesignSensitivityH_nolines.txt','r'); 
Nv1 = textscan(fpv1, '%f%f%f%f%f%f');
fclose(fpv1);
v1freqs = N{1};
v1amp = N{2};
v1freqs = [0;v1freqs];
v1amp = [v1amp(1); v1amp];
cnV1 = interp1(v1freqs, v1amp, fbins);
cnV1 = cnV1';



truedataV1 = hf(:,3);

idx = fbins >= fmin & fbins <= fmax;
% fbins between fmin and fmax

% run the model at new frequencies to update hp and hc global variables
% tic
hf = freqdomaininspiral(fbins(idx), parnames, parvals);
% toc

scfacH1 = 15; % a "calibration difference" scale factor
scfacL1 = 0.5; % calibration difference for LIGO Livingston Observatory
scfacV1 = 6; % calibration difference for Vigro  

data{1} = fbins(idx);
data{2} = {truedataH1(idx)*scfacH1, truedataL1(idx)*scfacL1, truedataV1(idx)*scfacV1};
data{3} = {(scfacH1*cnH1(idx)).^2, (scfacL1*cnH1(idx)).^2, (scfacV1*cnV1(idx)).^2};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define MCMC parameters
likelihood = @logL;
model = @freqdomaininspiral;


%prior =  {'sf1', 'uniform', 0, 2, 'reflect';
%          'D', 'gaussian', 15, 0.2, '';
%          'iota', 'gaussian', 0, 0.1, ''};
prior =  {'sf1', 'jeffreys', 1e-2, 1e2, '';...
          'sf2', 'jeffreys', 1e-2, 1e2, '';...
          'sf3', 'jeffreys', 1e-2, 1e2, '';...
%           'D', 'gaussian', 15, 0.2, '';...
%           'iota', 'gaussian', 0, 0.1, '';...
%           'iota', 'uniform', 0, (pi), '';...
%           'm1', 'uniform', 1.39, 1.41, '';...
%           'm2', 'uniform', 1.39, 1.41, '';...

};


extraparams = {'fmin', fmin; ...
                'fmax', fmax; ...
                'D', D; ...
                'm1', m1; ...
                'm2', m2; ...
                'z', z; ...
                'tc', tc; ...
                'iota', iota; ...
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
Nmcmc = 1000;

[logZ, nest_samples, post_samples] = nested_sampler(data, Nlive, Nmcmc, ...
tolerance, likelihood, model, prior, extraparams);

