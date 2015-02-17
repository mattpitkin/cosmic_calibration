function [snr, logZ, nest_samples, post_samples] = inspiral_sd_rand_sp(D, ra, dec, iota, psi, counter)

% perform the tests in calibrationtest.m, but this time with an aLIGO noise
% curve and an true inspiral signal.

global verbose;
verbose = 1;

global DEBUG;
DEBUG = 0;

% create an inspiral signal of two neutron stars
m1 = 1.4; % mass in Msun
m2 = 1.4; % mass in Msun
%iota = 0; % orientation angle (rads) ( 0=face on system)
%psi = 0.3; % polarisation angle (rads) 
det = 'H1'; % use H1 detector
z = 0; % approximately redsift of zero (as close)
% D = 25; % luminosity distance
tc = 900000000; % coalesence time
phic = 4.5; % phase at coalescence (rads)
%ra = 0.3; % right ascension (rads)
%dec = -0.76; % declination (rads)
fmin = 40; % initial frequency of signal
fmax = 1600; % final frequency of signal

[a, b] = antenna_pattern(det, ra, dec, 0, tc);
resplut = [a, b]';
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

% create a coloured spectrum using aLIGO full design sensitivity noise curve
fp = fopen('data/aligo_sensitivity.txt', 'r'); 
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
fbinsidx = fbins(idx);
df = fbinsidx(2) - fbinsidx(1);

%Find the SNR
snr = sqrt(4*sum(hf(idx).*conj(hf(idx))./(cn(idx)).^2).*df);


% run the model at new frequencies to update hp and hc global variables
tic
hf = freqdomaininspiral(fbins(idx), parnames, parvals);
toc

scalefactor = 0.8; % a "calibration difference" scale factor

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
prior =  {'sf1', 'uniform', 0, 2, 'fixed';
          'phic', 'uniform', 0, pi, '';
          'psi', 'uniform', 0, (pi./2), '';
          'iota', 'gaussian', 0, 0.1, '';
          'tc', 'uniform', (900000000 - 0.01), (900000000 + 0.01), 'fixed';
};


      
extraparams = {'fmin', fmin; ...
               'fmax', fmax; ...
               'D', D; ...
               'm1', m1; ...
               'm2', m2; ...
               'z', z; ...
               'tc', tc; ...
%                'iota', iota; ...
%                'psi', psi; ...
               'ra', ra; ...
               'dec', dec; ...
%                'phic', phic; ...
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

[logZ, nest_samples, post_samples] = nested_sampler(data, Nlive, ...
  tolerance, likelihood, model, prior, extraparams, 'Nmcmc', Nmcmc);


post_sf = post_samples(:,1);
std_post = std(post_sf);
mean_post = mean(post_sf);
err = (post_sf)./mean(post_sf);



filename_logZ = ['logZ' num2str(counter) '.txt'];
filename_post = ['post_samples' num2str(counter)  '.txt'];
filename_std_post = ['std' num2str(counter) '.txt'];
filename_snr = ['snr' num2str(counter) '.txt'];

filename = ['variable' num2str(counter) '.mat'];

save(fullfile('variables/mat_files', filename));


save(fullfile('variables/post_samples', filename_post), 'post_sf', '-ascii');
save(fullfile('variables/logZ', filename_logZ), 'logZ', '-ascii');
save(fullfile('variables/std', filename_std_post), 'std_post', '-ascii');
save(fullfile('variables/snr', filename_snr), 'snr', '-ascii');  
save(fullfile('variables/mat_files', filename));


