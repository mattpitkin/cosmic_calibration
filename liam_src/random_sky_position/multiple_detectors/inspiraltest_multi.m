    % perform the tests in calibrationtest.m, but this time with an aLIGO noise
    % curve and an true inspiral signal.
function [snr, err, logZ, nest_samples, post_samples] = inspiraltest_multi(D, ra, dec, iota, psi, counter)



global verbose;
verbose = 1;

global DEBUG;
DEBUG = 0;

% create an inspiral signal of two neutron stars
m1 = 1.4; % mass in Msun
m2 = 1.4; % mass in Msun
% iota = 0; % orientation angle (rads) ( 0=face on system)
% psi = 0.3; % polarisation angle (rads) 

% Multiple Detectors 
det = {'H1','L1','V1'}; 


z = 0; % approximately redsift of zero (as close)
% D = 124; % luminosity distance
tc = 900000000; % coalesence time
phic = 4.5; % phase at coalescence (rads)
% ra = 0.3; % right ascension (rads)
% dec = -0.76; % declination (rads)
fmin = 50; % initial frequency of signal
fmax = 1600; % final frequency of signal


resplut = cell(1, length(det));
for i = 1:length(det)
    [a, b] = antenna_pattern(det{i}, ra, dec, 0, tc);
    resplut{i}  = [a, b];
end

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
fpl1 = fopen('data/aligo_sensitivity.txt', 'r'); 
N = textscan(fpl1, '%f%f%f%f%f%f', 'CommentStyle', '#');
fclose(fpl1);
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


%%% Compute the SNR for all detectors 
df = fbins(2) - fbins(1);

snrH1 = sqrt(4*sum(hf(idx,1).*conj(hf(idx,1))./(cnH1(idx)).^2).*df);
snrL1 = sqrt(4*sum(hf(idx,2).*conj(hf(idx,2))./(cnH1(idx)).^2).*df);
snrV1 = sqrt(4*sum(hf(idx,3).*conj(hf(idx,3))./(cnV1(idx)).^2).*df);

snr = [snrH1 snrL1 snrV1];


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
prior =  {'sf1', 'uniform', 0, 2, '';...
          'sf2', 'uniform', 0, 2, '';...
          'sf3', 'uniform', 0, 2, '';...
          'phic', 'uniform', 0, pi, '';
          'psi', 'uniform', 0, (pi./2), '';...
          'iota', 'gaussian', 0, 0.1,'';...
          'tc', 'uniform', (900000000 - 0.01), (900000000 + 0.01), 'fixed';
%           'm1', 'uniform', 1, 2.4,'';...
%           'm2', 'uniform', 1, 2.4,'';...

};


extraparams = {'fmin', fmin; ...
                'fmax', fmax; ...
                'D', D; ...
                'm1', m1; ...
                'm2', m2; ...
                'z', z; ...
%                 'tc', tc; ...
%                 'iota', iota; ...
%                 'psi', psi; ...
                'ra', ra; ...
                'dec', dec; ...
%                 'phic', phic; ...
                'det', det; ...
                'resplut', resplut;...
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


post_sf = [post_samples(:,1) post_samples(:,2) post_samples(:,3)];
std_post = std(post_sf);
% mean_post = mean(post_sf);
err = (post_sf)./mean(post_sf);


filename_post = ['post_samples' num2str(counter)  '.txt'];
filename_logZ = ['logZ' num2str(counter) '.txt'];
filename_std_post = ['std' num2str(counter) '.txt'];
filename_snr = ['snr' num2str(counter) '.txt'];

filename = ['variable' num2str(counter) '.mat'];

save(fullfile('variables/mat_files', filename));
save(fullfile('variables/logZ', filename_logZ), 'logZ', '-ascii');
save(fullfile('variables/post_samples', filename_post), 'post_samples', '-ascii');
save(fullfile('variables/std', filename_std_post), 'std_post', '-ascii');
save(fullfile('variables/snr', filename_snr), 'snr', '-ascii');    


