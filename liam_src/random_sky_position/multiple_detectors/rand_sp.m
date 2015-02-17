% Random sky positions

Nr = 100;
ra = 2*pi*rand(Nr,1);
dec = -(pi/2) + acos(2*rand(Nr,1) - 1);
iota = (-pi/2) + acos(2* rand(Nr,1)-1);
psi = pi*(rand(Nr,1));
D = [20, 120];


for i = 1:length(D);
    for j = 1:length(Nr)
      [snr, logZ, nest_samples, post_samples] = inspiraltest_multi(D(i), ra(j), dec(j), iota(j), psi(j), j);
    end
end



