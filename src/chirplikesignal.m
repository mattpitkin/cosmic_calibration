function hf = chirplikesignal(fs, parnames, parvals)

lpn = length(parnames);
lpv = length(parvals);
if lpn ~= lpv
    error('Error: parnames and parvals are not the same length!');
end

nparams = lpn;

scalefreqs = [];

% extract parameter values
for ii=1:nparams
    switch parnames{ii}
        case 'sf1'
            sf1 = parvals{ii};
        case 'sf2'
            sf2 = parvals{ii};
        case 'sf3'
            sf3 = parvals{ii};
        case 'sf4'
            sf4 = parvals{ii};
        case 'a0'
            a0 = parvals{ii};
        case 't0'
            t0 = parvals{ii};
        case 'fmin'
            fmin = parvals{ii};
        case 'fmax'
            fmax = parvals{ii}; 
        case 'scalefreqs'
            scalefreqs = parvals{ii};
    end
end

if ~isempty(scalefreqs)
  % interpolate scale factor values to the full range of frequencies
  scaleint = interp1(scalefreqs, log10([sf1, sf2, sf3, sf4]), fs, 'spline');
  %if isnan(scaleint)
  %    fprintf(1, 'WARNING! WARNING!\n');
  %end
end
  
hf = zeros(length(fs), 1);

v = find(fs >= fmin & fs <= fmax);

hf(v) = a0*(fs(v)/fmin).^(7/6) .* exp(1i*(2*pi*fs(v)*t0));

if ~isempty(scalefreqs)
    % divide data by scale factor
    hf = hf ./ 10.^(scaleint');
end