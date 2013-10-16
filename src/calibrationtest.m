global verbose;
verbose = 1;

% create a chirp-like signal
a0 = 1e-20; % initial amplitude
t0 = 0; % initial time
fmin = 50; % initial frequency of signal
fmax = 500; % final frequency of signal

parnames = {'a0', 't0', 'fmin', 'fmax'};
parvals = {a0, t0, fmin, fmax};

% set frequencies
fs = 50:500;

% create a signal
hf = chirplikesignal(fs, parnames, parvals);

% create some white noise with a 500 Hz Nyquist frequency and 1 Hz freq
% bins
x = randn(1000,1);
fbins = 0:1:500;

% fft the data
y = fft(x);

% get the average rms amplitude of an fft of length length(x)
a = sqrt(length(x));

% make noise have rms amplitude of 1 for easier scaling
y = y/a;

% create a coloured spectrum with a quadratic shape in logspace
cn = 5e-21*exp(((fbins-250)/100).^2);

% colour the noise to the true shape
yn = y(1:501).*cn';

% add in the signal to create the real dataset
truedata = yn;
truedata(51:end) = truedata(51:end) + hf;

% plot signal and noise
%figure;
semilogy(fbins, abs(yn), 'k', fbins, abs(truedata), 'b', fs, hf, 'r');
xlabel('Frequency (Hz)');
ylabel('Amplitude');

% re-colour the data to give the un-calibrated detector data
dasd = ones(501,1); % for simplicity have the raw uncalibrated data have a flat amplitude spectrum

obsdata = truedata .* (dasd ./ cn');

% now set up nested sampling to estimate to scaling factors between the
% observed asd and the true asd
data{1} = fs;
data{2} = obsdata(51:end);
data{3} = dasd(51:end).^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define nested sampling parameters
%Nlive = 500;

%Nmcmc = 100;
%tolerance = 0.1;
likelihood = @logL;
model = @chirplikesignal;
prior = {'sf1', 'jeffreys', 1e-23, 1e-17, ''; ...
         'sf2', 'jeffreys', 1e-23, 1e-17, ''; ...
         'sf3', 'jeffreys', 1e-23, 1e-17, ''; ...
         'sf4', 'jeffreys', 1e-23, 1e-17, ''};
% set frequencies of scale factors
scalefreqs = [100, 200, 300, 400];
     
extraparams = {'a0', a0; 't0', t0; 'fmin', fmin; 'fmax', fmax; 'scalefreqs', scalefreqs};

% called nested sampling routine
%[logZ, nest_samples, post_samples] = nested_sampler(data, Nlive, Nmcmc, ...
%  tolerance, likelihood, model, prior, extraparams);

% call MCMC routine
Nmcmc = 100000;
Nburnin = 100000;
[post_samples, logP] = mcmc_sampler(data, likelihood, model, prior, ...
    extraparams, 'Nmcmc', Nmcmc, 'Nburnin', Nburnin);

% scale the output posteriors by the uncalibrated data psd at the given
% frequencies
figure;
hold on;
for i=1:4
    post_samples(:,i) = post_samples(:,i)*dasd(fbins == scalefreqs(i));
    
    % plot the posteriors estimates of the posterior samples
    subplot(1,4,i), hist(post_samples(:,i), 20);
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
for i=1:4
    % add one sigma error bars
    errorbar(scalefreqs(i), mean(post_samples(:,i)), std(post_samples(:,i)), 'r')
end
hold off;
