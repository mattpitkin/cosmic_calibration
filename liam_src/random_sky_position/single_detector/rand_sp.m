%%% Random sky positions
% This script allows one to iterate through the inspiral code a number of Nr times with random sky positions ra and dec, between [0, 2pi] and [-pi/2, pi/2] respectively. Iota and psi are also randomly chosen. D is the luminosity distance. 



D= 20;
Nr = 10;
ra = 2*pi*rand(Nr,1);
dec = -(pi/2) + acos(2*rand(Nr,1) - 1);
iota = (-pi/2) + acos(2* rand(Nr,1)-1);
psi = pi*(rand(Nr,1));
for j = 1:Nr
    [snr, logZ, nest_samples, post_samples] = inspiral_sd_rand_sp(D, ra(j), dec(j), iota(j), psi(j), j);
end



