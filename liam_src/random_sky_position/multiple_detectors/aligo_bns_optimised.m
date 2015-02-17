function [asd] = aligo_bns_optimised(fs, A, f0, a1, a2, b1, b2, b3, b4, b5)

% An analytic version of the aLIGO amplitude spectral density taken from
% Table 1 of Sathyaprakash & Schutz, Living Reviews in Relativity, 2009.
% The ASD is of the form:
%   asd = A*sqrt(x.^a1 + b1*x.^a2 + b2.*(1 + b3*x.^2 +
%   b4*x.^4)./(1+b5*x.^2))
% where x = fs/f0. The values given in the reference for the spectral
% indices and coefficients are:
% A = sqrt(1e-49), f0 = 215Hz, a1 = -4.14, a2 = -2, b1 = -5, b2 = 111, b3 =
% -1, b4 = 0.5 and b5 = 0.5. However, it should be noted that these values
% don't appear to be that representative of the current aLIGO noise
% estimates e.g. given here https://dcc.ligo.org/LIGO-P1200087-v18/public.
% The curve also seems to be optimised for BNS systems.

x = fs./f0;

asd = A*sqrt(x.^a1 + b1*x.^a2 + b2.*(1 + b3*x.^2 + b4*x.^4)./(1+b5*x.^2));