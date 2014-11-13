function [ Fp, Fc ] = antenna_pattern( site, ra, dec, psi, t )
%[Fp, Fc] = antenna_pattern(site, ra, dec, psi, t)
%   This function computes the antenna pattern for a given detector (site),
%   given a sky position in right ascension and declination (in radians),
%   and a polarisation angle psi. It outputs the plus (Fp) and cross (Fc)
%   polarisation responses. This function uses the conventions (and
%   detector positions) taken from eqns 10-13 of Jaranowski, Krolak and
%   Schutz, PRD, 58, 063001 (1998). The function will be calculated at an
%   arbitrary time t (in GPS secs), where t can be a vector of times.
%
%   The site must be either 'LLO', 'LHO', 'GEO', 'Virgo', or 'TAMA'
%
%   Matt Pitkin 03/11/09

% set the site positions and orientations as given in table 1 of JKS
switch site
    case {'GEO','G1'}
        lambda = pi * 52.25 / 180; % latitude
        gamma = pi * 68.775 / 180; % orientation of arms
        lde = pi * -9.81 /180;     % longitude
        xi = pi * 94.44 / 180; % angle between arms
    case {'LHO','H1','H2'}
        lambda = pi * 46.45 / 180; % latitude
        gamma = pi * 171.8 / 180; % orientation of arms
        lde = pi * 119.41 / 180;  % longitude 
        xi = pi / 2; % angle between arms
    case {'LLO','L1'}
        lambda = pi * 30.56 / 180; % latitude
        gamma = pi * 243.0 / 180; % orientation of arms
        lde = pi * 90.77 / 180; % longitude
        xi = pi / 2; % angle between arms
    case {'Virgo','VIRGO','V1'}
        lambda = pi * 43.63 / 180; % latitude
        gamma = pi * 116.5 / 180; % orientation of arms
        lde = pi * -10.5 /180; % longitude
        xi = pi / 2; % angle between arms
    case {'TAMA','T1'}
        lambda = pi * 35.68 / 180; % latitude
        gamma = pi * 225.0 / 180; % orientation of arms
        lde = pi * -135.54 / 180; % longitude
        xi = pi / 2; % angle between arms
    case 'ET'
        % for ET set the site as the same as Virgo (perpendicular arms)
        lambda = pi * 43.63 / 180; % latitude
        gamma = pi * 116.5 / 180; % orientation of arms
        lde = pi * -10.5 /180; % longitude
        xi = pi / 2; % angle between arms
    case 'ETX'
        % for ET set the site as the same as Virgo in triagular
        % confiuration (three detectors each rotated by 120 degrees)
        lambda = pi * 43.63 / 180; % latitude
        gamma = pi * 116.5 / 180; % orientation of arms
        lde = pi * -10.5 /180; % longitude
        xi = pi / 3; % angle between arms
    case 'ETY'
        % for ET set the site as the same as Virgo in triagular
        % confiuration (three detectors each rotated by 120 degrees)
        lambda = pi * 43.63 / 180; % latitude
        gamma = (pi * 116.5 / 180) + (2 * pi / 3); % orientation of arms
        lde = pi * -10.5 /180; % longitude
        xi = pi / 3; % angle between arms 
     case 'ETZ'
        % for ET set the site as the same as Virgo in triagular
        % confiuration (three detectors each rotated by 120 degrees)
        lambda = pi * 43.63 / 180; % latitude
        gamma = (pi * 116.5 / 180) + (4 * pi / 3); % orientation of arms
        lde = pi * -10.5 /180; % longitude
        xi = pi / 3; % angle between arms    
    case 'EQ'
        % detector on the equator
        lambda = 0; % latitude
        gamma = 0;
        lde = 0;
        xi = pi/2;
    case 'POLE'
        % detector at the north pole
        lambda = pi * 90 / 180; % latitude
        gamma = 0;
        lde = 0;
        xi = pi/2;
end

% get JKS a and b values
alpha = ra;
delta = dec;

% get local sidereal time
% convert gps time to GMST
gmst = gps2sidereal(t);

% convert gmst from degs into rads
gmst = (gmst/24)*2*pi;

% covert to LST
lst = mod(gmst - lde, 2*pi);

a = (1/16) .* sin(2*gamma) .* (3 - cos(2*lambda)) .* (3 - cos(2*delta)) .* ...
    cos(2*(alpha - lst)) - ...
    (1/4) .* cos(2*gamma) .* sin(lambda) .* (3 - cos(2*delta)) .* ...
    sin(2*(alpha - lst)) + ...
    (1/4) .* sin(2*gamma) .* sin(2*lambda) .* sin(2*delta) .* ...
    cos(alpha - lst) - ...
    (1/2) .* cos(2*gamma) .* cos(lambda) .* sin(2*delta) .* ...
    sin(alpha - lst) + ...
    (3/4) .* sin(2*gamma) .* cos(lambda)^2 .* cos(delta)^2;

b = cos(2*gamma) .* sin(lambda) .* sin(delta) .* ...
    cos(2*(alpha - lst)) + ...
    (1/4) .* sin(2*gamma) .* (3 - cos(2*lambda)) .* sin(delta) .* ...
    sin(2*(alpha - lst)) + ...
    cos(2*gamma) .* cos(lambda) .* cos(delta) .* ...
    cos(alpha - lst) + ...
    (1/2) .* sin(2*gamma) .* sin(2*lambda) .* cos(delta) .* ...
    sin(alpha - lst);

% get Fp and Fc
Fp = sin(xi) * (a .* cos(2*psi) + b .* sin(2*psi));
Fc = sin(xi) * (b .* cos(2*psi) - a .* sin(2*psi));