function hf = freqdomaininspiral(fs, parnames, parvals)

% see LALSimInspiralTaylorF2.c
% creates an TaylorF2 frequency domain signal at 3.5pN order in phase and
% 0pN order in amplitude, no spin or tidal effects are included.

global hp hc

lpn = length(parnames);
lpv = length(parvals);
if lpn ~= lpv
    error('Error: parnames and parvals are not the same length!'); 
end

nparams = lpn;
% length(parnames)-see inspiraltest.m
scalefreqs = []; %empty matrix
nbins = 0;

sf1 = 1;

update = 1; % by default update the inspiral signal

resplut = []; % 

% extract parameter values
% returns parvals for each parnames (case is true when case and switch are
% equal) - see inspiraltest.m (prior)
for ii=1:nparams % between 1 and length of parnames
    switch parnames{ii}
        case 'sf1'
            sf1 = parvals{ii};
        %case 'sf2'
        %    sf2 = parvals{ii};
        %case 'sf3'
        %    sf3 = parvals{ii};
        %case 'sf4'
        %    sf4 = parvals{ii};
        %case 'sf5'
        %    sf5 = parvals{ii};
        %case 'sf6'
        %    sf6 = parvals{ii};    
        case 'fmin' % minimum start frequency of inspiral
            fmin = parvals{ii};
        case 'fmax' % end frequency of inspiral
            fmax = parvals{ii}; 
        %case 'scalefreqs'
        %    scalefreqs = parvals{ii};
        %case 'nbins'
        %    nbins = parvals{ii};
        case 'D' % luminosity distance (Mpc)
            Dl = parvals{ii};
        case 'm1' % mass of component 1 (solar masses)
            m1 = parvals{ii};
        case 'm2' % mass of component 2 (solar masses)
            m2 = parvals{ii};
        case 'z' % redshift
            z = parvals{ii};
        case 'tc' % coalescence time
            tc = parvals{ii};
        case 'iota' % source inclination angle (rads)
            iota = parvals{ii};
        case 'psi' % source polarisation angle (rads)
            psi = parvals{ii};
        case 'ra' % source right ascension (rads)
            ra = parvals{ii};
        case 'dec' % source declination (rads)
            dec = parvals{ii};
        case 'phic' % coalescence phase (rads)
            phic = parvals{ii};
        case 'det' % detector
            det = char(parvals{ii});
        case 'resplut' % antenna pattern look-up table
            resplut = parvals{ii};
        case 'update' % say whether or not to update the inspiral signal
            update = parvals{ii};
    end
end

if ~isempty(scalefreqs)
    % ie if scalefreqs is not an empty matrix
  % NOTE: Using this sort or spline interpolation can mean that introduce
  % features from the spline fitting that are not really indicative of the
  % noise if the frequency does not have a well constrained scale factor.
    
  % interpolate scale factor values to the full range of frequencies
  scaleint = interp1(scalefreqs, log10([sf1, sf2, sf3, sf4, sf5, sf6]), fs, 'spline');
  % spline interpolation of scalefreqs to range of frequencies fs (from
  % calibrationtest?)
  %if isnan(scaleint)
  %    fprintf(1, 'WARNING! WARNING!\n');
  %end
  
  % for now just use +/- nbins around each frequency 
  %if nbins == 0
  %    error('nbins should be greater than 0!')
  %end
  
  %sfs = [sf1, sf2, sf3, sf4, sf5, sf6];
  
  %allbins = 2*nbins+1;
  
  %lf = length(scalefreqs);
  %scaleint = zeros(lf*allbins,1);
  %for i=1:lf
  %  scaleint((i-1)*allbins+1:i*allbins) = sfs(i);
  %end
end

hf = zeros(length(fs), 1);

% some constants
GAMMA = 0.5772156649015328606065120900824024; % Euler's constant
MTSUN_SI = 4.9254923218988636432342917247829673e-6; % geometrised solar mass in seconds 
MRSUN_SI = 1.4766254500421874513093320107664308e3; % geometrised solar mass in metres
PC_SI = 3.0856775807e16; % parsec in metres 

if update == 1
    m = (1+z)*(m1 + m2);
    m_sec = m * MTSUN_SI;  % total mass in seconds
    eta = m1 * m2 / (m * m);
    piM = pi * m_sec;
    vISCO = 1. / sqrt(6.);
    fISCO = vISCO * vISCO * vISCO / piM;
    v0 = (piM*fmin)^(1/3);

    lambda = -1987./3080.;
    theta = -11831./9240.;

    pfaN = 3./(128.*eta);
    pfa2 = 5.*((743./84.) + 11.*eta)/9.;
    pfa3 = -16.*pi;
    pfa4 = 5.*((3058.673/7.056) + (5429./7.)*eta + 617.*eta^2)/72.;
    pfa5 = (5./9.)*((7729./84.) - 13.*eta)*pi;
    pfl5 = (5./3.)*((7729./84.) - 13.*eta)*pi;
    pfa6 = ((11583.231236531/4.694215680) - (640./3.)*pi^2 - ...
        (6848./21.)*GAMMA) + ...
        eta*((-15335.597827/3.048192) + (2255./12.)*pi^2 - ...
        (1760./3.)*theta + (12320./9)*lambda) + ...
        eta^2*(76055./1728.) - eta^3*(127825./1296.);
    pfl6 = -6848./21.;
    pfa7 = pi*(5./756.)*((15419335./336.) + (75703./2.)*eta - 14809.*eta^2);

    amp0 = -(4.*m1*m2)*MRSUN_SI*MTSUN_SI*sqrt(pi/12);
    shft = 2*pi*tc;

    % get indexes of frequencies to calculate
    if fISCO < fmax
        % is fmax is greater than frequency of the inner most stable circular
        % orbit then switch to use that
        fmax = fISCO;
    end
    idx = fs >= fmin & fs <= fmax;

    log4 = log(4.0);
    logv0 = log(v0);

    v = (piM * fs(idx)).^(1/3);
    logv = log(v);
    v2 = v.*v;
    v3 = v2.*v;
    v4 = v3.*v;
    v5 = v4.*v;
    v6 = v5.*v;
    v7 = v6.*v;
    v10 = v7.*v3;

    % flux coefficient
    FTaN = 32.*eta*eta/5.0;

    % dEnergy coefficient
    dETaN = -2.*(eta/2.);

    phasing = (pfaN./v5).*(pfa7*v7 + (pfa6+pfl6*(log4+logv)).*v6 + ...
        (pfa5+pfl5*(logv-logv0)).*v5 + pfa4*v4 + pfa3*v3 + pfa2*v2 + 1) + ...
        shft*fs(idx) - 2*phic - pi/4;
    amp = amp0*sqrt(-(dETaN*v)./(FTaN*v10)).*v;

    hf(idx) = amp.*cos(phasing) - 1i*amp.*sin(phasing);

    % get antenna pattern at coalesence time
    if isempty(resplut)
        [fp, fc] = antenna_pattern(det, ra, dec, psi, tc);
    else
        psis = resplut(:,1);
        dpsi = psis(2)-psis(1);
        psibin = round((psi - psis(1))/dpsi);
        fp = resplut(psibin,2);
        fc = resplut(psibin,3);
    end

    hc = 1i*hf*fc;
    
    hp = hf*fp;
end

ci = cos(iota);

% cross polarisation
hcn = hc*ci;

% cross polarisation
hpn = 0.5*(1+ci^2)*hp;

% add plus and cross components
hf = hcn + hpn;

% scale by distance
hf = hf / (PC_SI*1e6*Dl);

%if ~isempty(scalefreqs)    
    % divide data by scale factor
%    hf = hf ./ 10.^(scaleint');
    %hf = hf ./ scaleint;
%end

hf = hf*sf1;
