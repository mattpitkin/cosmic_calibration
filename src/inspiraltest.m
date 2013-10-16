% perform the tests in calibrationtest.m, but this time with an aLIGO noise
% curve and an true inspiral signal.

global verbose;
verbose = 1;

% create an inspiral signal of two neutron stars
m1 = 1.4; % mass in Msun
m2 = 1.4; % mass in Msun
iota = 0; % orientation angle (rads) ( 0=face on system)
psi = 0.3; % polarisation angle (rads_
det = 'H1'; % use H1 detector
z = 0; % approximately redsift of zero (as close)
D = 15; % luminosity distance of 50 Mpc
tc = 900000000; % coalesence time
phic = 4.5; % pase at coalescence (rads)
ra = 0.3; % right ascension (rads)
dec = -0.76; % declination (rads)
fmin = 20; % initial frequency of signal
fmax = 250; % final frequency of signal

parnames = {'fmin', 'fmax', 'D', 'm1', 'm2', 'z', 'tc', 'iota', 'psi', 'ra', 'dec', 'phic', 'det'};
parvals = {fmin, fmax, D, m1, m2, z, tc, iota, psi, ra, dec, phic, det};

Nd = 2000;
dt = 1/Nd;

% set frequencies
fbins = linspace(0, (Nd/2), (Nd/2)+1);

% create a signal
hf = freqdomaininspiral(fbins, parnames, parvals);

% create some white noise with a 1000 Hz Nyquist frequency and 1 Hz freq
% bins
x = randn(Nd,1);

% fft the data
y = fft(x);

% get the average rms amplitude of an fft of length length(x)
a = sqrt(length(x));

% make noise have rms amplitude of 1 for easier scaling
y = y/a;

% create a coloured spectrum using aLIGO full design sensitivity noise curve
fp = fopen('../data/aligo_sensitivity.txt', 'r');
N = textscan(fp, '%f%f%f%f%f%f', 'CommentStyle', '#');
fclose(fp);

alfreqs = N{1};
alamp = N{5}; % use fifth column for full design sensitivity (other columns have early run estimated sensitivities)

% prepend values at 0 frequency
alfreqs = [0; alfreqs];
alamp = [alamp(1); alamp];

cn = interp1(alfreqs, alamp, fbins);

% colour the noise to the true shape
yn = y(1:length(fbins)).*cn';

% add in the signal to create the real dataset
truedata = yn;
truedata = truedata + hf;

% plot signal and noise
%figure;
loglog(fbins, abs(yn), 'k', fbins, abs(truedata), 'b', fbins, abs(hf), 'r');
xlabel('Frequency (Hz)');
ylabel('Amplitude');

% re-colour the data to give the un-calibrated detector data
dasd = ones(length(fbins),1); % for simplicity have the raw uncalibrated data have a flat amplitude spectrum

obsdata = truedata .* (dasd ./ cn');

% now set up MCMC to estimate to scaling factors between the
% observed asd and the true asd

% as we can't model the noise curve well between frequecies (in the future
% we could use some analystic model of the expected noise spectrum and fit
% various amplitude and spectral index values in that, but for now we won't
% assume an knowledge of the true noise curve) just use data that is +/-2
% frequency bins around the fitting frequencies. 

% set frequencies of scale factors
scalefreqs = [fmin, 75, 100, 150, 200, fmax];

%fsdat = [];
%obs = [];
%ds = [];
%nbins = 5;
%for i=1:length(scalefreqs);
%    [val, idx] = closest(scalefreqs(i), fs);
%    fsdat = [fsdat, fs(idx-nbins:idx+nbins)];
%    obs = [obs; obsdata(21+(idx-nbins):21+(idx+nbins))];
%    ds = [ds; dasd(21+(idx-nbins):21+(idx+nbins))];
%end

idx = fbins >= fmin & fbins <= fmax;

% run the model at new frequencies to update hp and hc global variables
hf = freqdomaininspiral(fbins(idx), parnames, parvals);

%data{1} = fsdat;
%data{2} = obs;
%data{3} = ds.^2;

data{1} = fbins(idx);
data{2} = obsdata(idx);
data{3} = dasd(idx).^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define nested sampling parameters
likelihood = @logL;
model = @freqdomaininspiral;
prior = {'sf1', 'jeffreys', 1e-26, 1e-20, ''; ...
         'sf2', 'jeffreys', 1e-26, 1e-20, ''; ...
         'sf3', 'jeffreys', 1e-26, 1e-20, ''; ...
         'sf4', 'jeffreys', 1e-26, 1e-20, ''; ...
         'sf5', 'jeffreys', 1e-26, 1e-20, ''; ...
         'sf6', 'jeffreys', 1e-26, 1e-20, ''};
     
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
               'scalefreqs', scalefreqs; ...
               %'nbins', nbins; ... % number of frequency bins either side of scalefreqs to use
               'update', 0}; % no need to update the inspiral signal

% call MCMC routine
Nmcmc = 100000;
Nburnin = 100000;
temperature = 1e-5;
recalcprop = 10000; % number of samples after which to recalculate the proposal distribution
[post_samples, logP] = mcmc_sampler(data, likelihood, model, prior, ...
    extraparams, 'Nmcmc', Nmcmc, 'Nburnin', Nburnin, ...
    'temperature', temperature, 'recalcprop', recalcprop);

% scale the output posteriors by the uncalibrated data psd at the given
% frequencies
figure;
hold on;
for i=1:length(scalefreqs)
    post_samples(:,i) = post_samples(:,i)*dasd(fbins == scalefreqs(i));
    
    % plot the posteriors estimates of the posterior samples
    subplot(1,length(scalefreqs),i), hist(post_samples(:,i), 20);
    set(gca, 'fontsize', 16, 'fontname', 'helvetica');
    xlabel('Strain', 'fontsize', 14, 'fontname', 'avantgarde');
    legend(sprintf('%d Hz', scalefreqs(i)));
    title(sprintf('Fractional Error %.1f%%', 100*std(post_samples(:,i))/mean(post_samples(:,i))));
end
hold off;

% plot the estimates of strain on the true strain
figure;
semilogy(fbins, cn, 'k', fbins, abs(yn), 'b')
hold on;
for i=1:length(scalefreqs)
    % add one sigma error bars
    errorbar(scalefreqs(i), mean(post_samples(:,i)), std(post_samples(:,i)), 'r')
end
hold off;
