#!/usr/bin/env python

"""
A script to run the calibration scale factor check using the emcee MCMC sampler
"""
import numpy as np
import copy
import sys
import os
from optparse import OptionParser
import json

from matplotlib import pyplot as pl

# swig lal modules for signal generation
import lal
import lalsimulation

# MCMC code
import emcee

# a function to compute the antenna response
def antenna_response( gpsTime, ra, dec, psi, det ):
  gps = lal.LIGOTimeGPS( gpsTime )
  gmst_rad = lal.GreenwichMeanSiderealTime(gps)

  # create detector-name map
  detMap = {'H1': lal.LALDetectorIndexLHODIFF, \
            'H2': lal.LALDetectorIndexLHODIFF, \
            'L1': lal.LALDetectorIndexLLODIFF, \
            'G1': lal.LALDetectorIndexGEO600DIFF, \
            'V1': lal.LALDetectorIndexVIRGODIFF, \
            'T1': lal.LALDetectorIndexTAMA300DIFF}

  try:
    detector=detMap[det]
  except KeyError:
    raise ValueError, "ERROR. Key %s is not a valid detector name." % (det)

  # get detector
  detval = lal.CachedDetectors[detector]

  response = detval.response

  # actual computation of antenna factors
  fp, fc = lal.ComputeDetAMResponse(response, ra, dec, psi, gmst_rad)

  return fp, fc

"""
a function to generate a frequency domain inspiral waveform (see e.g. 
https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsSanityChecks?action=AttachFile&do=view&target=
lalsimutils.py)
This function sets spin parameters to zero, so all output are non-spinning waveforms. The reference frequency
is set to fmin.

Inputs are:
 - phiref - the reference phase (rads)
 - deltaF - the frequency bin size (Hz)
 - m1 - the mass of the first component (solar masses)
 - m2 - the mass of the second component (solar masses)
 - fmin - the lower bound on frequency (Hz)
 - fmax - the upper bound on frequency (Hz)
 - dist - the source distance (in Mpc)
 - incl - the source inclination angle (rads)

This function does not apply the antenna pattern to the output.
"""
def fdwaveform(phiref, deltaF, m1, m2, fmin, fmax, dist, incl):
  ampO = 0   # 0 pN order in amplitude
  phaseO = 7 # 3.5 pN order in phase
  approx = lalsimulation.TaylorF2 # Taylor F2 approximant
  fref = fmin
 
  hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(phiref, deltaF,
    m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., fmin,
    fmax, fref, dist*1e6*lal.PC_SI, incl, 0., 0., None, None, ampO, phaseO, approx)
  
  # return the frequency domain plus and cross waveforms
  return hptilde.data.data, hctilde.data.data


# A function to generate frequency domain coloured noise
# This function is adapted from pyCBC in the noise module:
# https://github.com/ligo-cbc/pycbc/blob/master/pycbc/noise/gaussian.py
# copyright of Alex Nitz (2012)
def frequency_noise_from_psd(psd, deltaF, seed = None):
    """ Create noise with a given psd.
    
    Return noise coloured with the given psd. The returned noise 
    has the same length and frequency step as the given psd. 
    Note that if unique noise is desired a unique seed should be provided.
    Parameters
    ----------
    psd : np.array
        The noise weighting to color the noise.
    deltaF: float
        The frequency step size
    seed : {0, int} or None
        The seed to generate the noise. If None specified,
        the seed will not be reset.
        
    Returns
    --------
    noise : numpy array
        A numpy array containing gaussian noise colored by the given psd. 
    """
    sigma = 0.5 * np.sqrt(psd / deltaF)
    if seed is not None:
        np.random.seed(seed)
    
    not_zero = (sigma != 0) & np.isfinite(sigma)
    
    sigma_red = sigma[not_zero]
    noise_re = np.random.normal(0, sigma_red)
    noise_co = np.random.normal(0, sigma_red)
    noise_red = noise_re + 1j * noise_co
    
    noise = np.zeros(len(sigma), dtype='complex')
    noise[not_zero] = noise_red
    
    return noise


# define the log posterior function:
#  - theta - a vector of the varying parameter values
#  - y - the frequency domain data divided by the ASD
#  - dd - the data cross product conj(d)*d for each detector
#  - iotawidth - the width (standard deviation) of a Gaussian prior to use for iota
#  - tccentre - the centre of the prior range on coalescence time
#  - m1 - mass 1
#  - m2 - mass 2
#  - dist - distance
#  - fmin - the lower frequency range
#  - fmax - the upper frequency range
#  - deltaF - the frequency bin size
#  - resps - the antenna respsonse functions
#  - asds - the detector noise ASDs
def lnprob(theta, y, dd, iotaprior, tccentre, m1, m2, dist, fmin, fmax, deltaF, resps, asds): 
  lp = lnprior(theta, iotaprior, tccentre, len(resps))
  
  if not np.isfinite(lp):
    return -np.inf
  
  return lp + lnlike(theta, y, dd, m1, m2, dist, fmin, fmax, deltaF, resps, asds)


# define the log prior function
def lnprior(theta, iotawidth, tccentre, ndets):
  # unpack theta
  psi, phi0, iota, tc = theta[0:4]
  ss = theta[-ndets:]

  lp = 0. # logprior

  # outside prior ranges
  if 0. < psi < np.pi/2. and 0. < phi0 < 2.*np.pi and tccentre-0.01 < tc < tccentre+0.01:
    lp = 0.
  else:
    return -np.inf
  
  # Jeffrey's prior of scale factor within range 0.01 to 10
  for s in ss:
    if 0.01 < s < 10.:
      lp = lp - np.log(s)
    else:
      return -np.inf
  
  # add iota prior
  lp = lp - 0.5*(iota/iotawidth)**2

  return lp


# define the log likelihood function
def lnlike(theta, y, dd, m1, m2, dist, fmin, fmax, deltaF, resps, asds):
  # unpack theta
  psi, phi0, iota, tc = theta[0:4]
  ss = theta[-len(resps):]
  
  spsi = np.sin(2.*psi)
  cpsi = np.cos(2.*psi)
  
  # generate waveform
  hp, hc = fdwaveform(phi0, deltaF, m1, m2, fmin, fmax, dist, iota)

  L = 0. # log likelihood

  # add response function and scale waveform and calculate likelihood
  for i in range(len(resps)):
    Ap, Ac = resps[i]

    H = (hp*(Ap*cpsi + Ac*spsi) + hc*(Ac*cpsi - Ap*spsi))*ss[i]

    Hs = H/asds[i]
    
    # Hs[~np.isfinite(Hs)] = 0.
    
    dh = np.vdot(y[i], Hs)
    hh = np.vdot(Hs, Hs)
    
    L = L + 4.*deltaF*(dh.real - 0.5*(hh.real + dd[i]))
  
  return L


# a function to get the credible intervals using a greedy binning method
def credible_interval(dsamples, ci):
    n, binedges = np.histogram(dsamples, bins=100)
    dbins = binedges[1]-binedges[0] # width of a histogram bin
    bins = binedges[0:-1]+dbins/2. # centres of bins

    histIndices=np.argsort(n)[::-1]  # indices of the points of the histogram in decreasing order

    frac = 0.0
    j = 0
    for i in histIndices:
        frac += float(n[i])/float(len(dsamples))
        j = j+1
        if frac >= ci:
            break

    return (np.min(bins[histIndices[:j]]), np.max(bins[histIndices[:j]]))


if __name__=='__main__':
  
  usage = "Usage: %prog [options]"

  parser = OptionParser( usage = usage )

  parser.add_option("-o", "--outpath", dest="outpath", help="The path for "
                    "the analysis output (a sub-directory based on the pulsar "
                    "name will be created here that contains the pulsar "
                    "information)", metavar="DIR")

  parser.add_option("-g", "--det", dest="dets",
                    help="An interferometer to be used (multiple interferometers can be set \
with multiple uses of this flag, e.g. \"-g H1 -g L1 -g V1\") [default: H1]", 
                    action="append")

  parser.add_option("-S", "--scalefac", dest="scales", action="append", type="float",
                    help="The simulation calibration scale factors for each detector \
- unless the value is 1 for all detectors there must be the same number of these values \
set as detectors given [default: 1.]")

  parser.add_option("-N", "--Niter", dest="Niter",
                    help="Number of MCMC iterations [default: %default]", type="int",
                    default=10000)
  
  parser.add_option("-B", "--Nburnin", dest="Nburnin",
                    help="Number of MCMC burn-in iterations [default: %default]", type="int",
                    default=5000)

  parser.add_option("-E", "--Nensemble", dest="Nensemble",
                    help="Number of MCMC ensemble points [default: %default]", type="int",
                    default=20)

  parser.add_option("-s", "--intseed", dest="intseed",
                    help="A unique integer for use in output generation [default: %default].",
                    default="0")

  parser.add_option("-D", "--dist", dest="dist", type="float",
                    help="Inspiral distance in Mpc [default: %default]", default=10.)
  
  parser.add_option("-r", "--ra", dest="ra", type="float",
                    help="Right ascension (rads) - if not specified the RA will be drawn randomly \
from a uniform distribution on the sky")
  
  parser.add_option("-d", "--dec", dest="dec", type="float",
                    help="Declination (rads) - if not specified the dec will be drawn randomly \
from a uniform distribution on the sky")

  parser.add_option("-i", "--iota", dest="iota", type="float",
                    help="Inclination of insprial (rads) - if not specified iota will be drawn from \
a Gaussian with zero mean and standard devaition specified by \"iotawidth\"")

  parser.add_option("-w", "--iotawidth", dest="iotawidth", type="float",
                    help="Width of iota simulation distribution (rads) [default: %default]",
                    default=0.1)

  parser.add_option("-t", "--t0", dest="t0", type="float",
                    help="Time of coalescence (GPS) [default: %default]", default=900000000.)

  parser.add_option("-p", "--phi0", dest="phi0", type="float",
                    help="The phase at coalescence (rads) - if not specified phi0 will be drawn from \
a uniform distribution between 0 and 2pi.")

  parser.add_option("-a", "--psi", dest="psi", type="float",
                    help="The polarisation angle (rads) - if not specified psi will be drawn from \
a uniform distribution between 0 and pi/2.")

  parser.add_option("-f", "--fmin", dest="fmin", type="float",
                    help="Lower frequency bound (Hz) [default: %default]", default=40.)
  
  parser.add_option("-m", "--fmax", dest="fmax", type="float",
                    help="Upper frequency bound (Hz) [default: %default]", default=1600.)

  parser.add_option("-x", "--deltaF", dest="deltaF", type="float",
                    help="Frequency bins size (Hz) [default: %default]", default=2.)

  parser.add_option("-z", "--noise", dest="noise", default=False, action="store_true",
                    help="If this flag is set a noise spectrum is added to the data")

  parser.add_option("-k", "--psd-noise", dest="psdnoise", default=0, type="int",
                     help="If this flag is non-zero the PSD is estimated as the average of \
the given number of noisy PSD estimates.")
  
  parser.add_option("-P", "--plot", dest="plot", default=False, action="store_true",
                    help="If this flag is set the posteriors will be plotted (requires triangle.py).")

  parser.add_option("-T", "--seed", dest="seed", type="int",
                     help="A numpy random number generator seed value.")

  # parse input options
  (opts, args) = parser.parse_args()

  # check that output path has been given
  if not opts.__dict__['outpath']:
    print >> sys.stderr, "Must specify an output path"
    parser.print_help()
    sys.exit(0)
  else:
    outpath = opts.outpath

  intseed = opts.intseed

  # MCMC options
  Niter = opts.Niter
  Nburnin = opts.Nburnin
  Nensemble = opts.Nensemble

  # detectors
  if not opts.__dict__['dets']:
    dets = ['H1']
  else:
    dets = opts.dets

  # integration options
  fmin = opts.fmin
  fmax = opts.fmax
  deltaF = opts.deltaF

  # injected source options
  dist = opts.dist
  t0 = opts.t0
  
  if not opts.__dict__['ra']:
    ra = 2.*np.pi*np.random.rand() # generate RA uniformly between 0 and 2pi
  else:
    ra = opts.ra

  if not opts.__dict__['dec']:
    dec = -(np.pi/2.) + np.arccos(2.*np.random.rand()-1.)
  else:
    dec = opts.dec
    
  iotawidth = opts.iotawidth
  if not opts.__dict__['iota']:
    iota = np.random.randn()*iotawidth
  else:
    iota = opts.iota

  if not opts.__dict__['psi']:
    psi = 0.5*np.pi*np.random.rand() # draw from between 0 and pi/2
  else:
    psi = opts.psi
    
  if not opts.__dict__['phi0']:
    phi0 = 2.*np.pi*np.random.rand() # draw from between 0 and 2pi
  else:
    phi0 = opts.phi0

  if not opts.__dict__['scales']:
    scales = [1.]
  else:
    scales = opts.scales

  if opts.__dict__['seed']:
    np.random.seed(opts.seed) # set the random seed

  # check whether to add noise
  addnoise = opts.noise
  
  # check whether to calculate the PSD from noisy data
  psdnoise = opts.psdnoise

  if len(scales) != len(dets):
    if len(scales) == 1 and scales[0] == 1.: # all scale factors will be one
      scales = [1. for i in range(len(dets))]
    else:
      print >> sys.stderr, "Must specify the same number of calibration scale factors as detectors"
      sys.exit(0)

  # create a simulated waveform in each detector
  # the masses will both be fixed at 1.4 solar masses
  m1 = 1.4
  m2 = 1.4

  # waveform
  hp, hc = fdwaveform(phi0, deltaF, m1, m2, fmin, fmax, dist, iota)

  resps = [] # list to hold detector plus polarisation responses
  H = [] # the strain

  # waveforms for each detector (accounting for antenna pattern and calibration scale)
  for i in range(len(dets)):
    apt, act = antenna_response( t0, ra, dec, psi, dets[i] )
    H.append((hp*apt + hc*act)*scales[i])
    
    # save response function for psi=0 for further computations
    apt, act = antenna_response( t0, ra, dec, 0.0, dets[i] )
    resps.append([apt, act])

  # create frequency series for PSDs
  psd = lal.CreateREAL8FrequencySeries('name', t0, 0., deltaF, lal.Unit(), len(hp))

  # generate the ASD esimates (use values from P1200087)
  asds = []
  SNRs = []
  dd = [] # the data cross product
  
  for i in range(len(dets)):
    if dets[i] in ['H1', 'L1']:
      ret = lalsimulation.SimNoisePSDaLIGODesignSensitivityP1200087(psd, fmin)
      psd.data.data[psd.data.data == 0.] = np.inf
    elif dets[i] in 'V1':
      ret = lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd, fmin)
    
    psd.data.data[psd.data.data == 0.] = np.inf
    
    if psdnoise:
      # get average of 32 noise realisations for PSD
      psdav = np.zeros(len(hp))
      for j in range(psdnoise):
        noisevals = frequency_noise_from_psd(psd.data.data, deltaF)
        psdav = psdav + 2.*deltaF*np.abs(noisevals)**2
      psdav = psdav/float(psdnoise)
      
      psdav[psdav == 0.] = np.inf
      
      asds.append(np.sqrt(psdav)*scales[i])
    else:
      asds.append(np.sqrt(psd.data.data)*scales[i])
   
    freqs = np.linspace(0., deltaF*(len(hp)-1.), len(hp))
    #pl.plot(freqs, np.sqrt(psd.data.data), colours[i])
    #pl.plot(freqs, np.abs(H[i]), colours[i])
    
    # get htmp for snr calculation
    htmp = H[i]/asds[i]
    #htmp[~np.isfinite(htmp)] = 0.
    snr = np.sqrt(4.*deltaF*np.vdot(htmp, htmp).real)
    SNRs.append(snr)
    print >> sys.stderr, "%s: SNR = %.2f" % (dets[i], snr) 
    
    # create additive noise
    if addnoise:
      noisevals = frequency_noise_from_psd(psd.data.data, deltaF)
      #pl.plot(freqs, np.abs(noisevals), colours[i])

      H[i] = H[i] + noisevals*scales[i]
      
    H[i] = H[i]/asds[i]
    #H[i][~np.isfinite(H[i])] = 0.
    
    dd.append(np.vdot(H[i], H[i]).real)
 
  #pl.show()
  #sys.exit(0)
 
  # set up MCMC
  # get initial seed points
  pos = []
  for i in range(Nensemble):
    psiini = np.random.rand()*np.pi*0.5 # psi
    phi0ini = np.random.rand()*2.*np.pi # phi0
    iotaini = iotawidth*np.random.randn() # iota
    tcini = -0.01 + 2.*0.01*np.random.rand() + t0 # time of coalescence
    sfs = 0.01 + (10.-0.01)*np.random.rand(len(scales)) # scale factors
    
    thispos = [psiini, phi0ini, iotaini, tcini]
    for s in sfs:
      thispos.append(s)
    
    pos.append(np.array(thispos))
  
  ndim = len(pos[0])
  # Multiprocessing version
  #sampler = emcee.EnsembleSampler(Nensemble, ndim, lnprob, args=(H, dd, iotawidth, t0, m1, m2, 
  #                                dist, fmin, fmax, deltaF, resps, asds), threads=3)
  sampler = emcee.EnsembleSampler(Nensemble, ndim, lnprob, args=(H, dd, iotawidth, t0, m1, m2,
                                  dist, fmin, fmax, deltaF, resps, asds))
  
  sampler.run_mcmc(pos, Niter+Nburnin)
  
  #  remove burn-in and flatten
  samples = sampler.chain[:, Nburnin:, :].reshape((-1, ndim))
 
  # output samples to gzipped file
  samplefile = os.path.join(outpath, 'samples_'+intseed+'.txt.gz')
  np.savetxt(samplefile, samples, fmt='%.5f')
 
  # output injection information and scale factor recovery information (JSON format)
  infofile = os.path.join(outpath, 'info_'+intseed+'.txt')
  outdict = {} # output dictionary
  
  outdict['InjectionParameters'] = {}
  outdict['InjectionParameters']['psi'] = psi
  outdict['InjectionParameters']['phi0'] = phi0
  outdict['InjectionParameters']['iota'] = iota
  outdict['InjectionParameters']['tc'] = t0
  outdict['InjectionParameters']['m1'] = m1
  outdict['InjectionParameters']['m2'] = m2
  outdict['InjectionParameters']['dist'] = dist
  outdict['InjectionParameters']['fmin'] = fmin
  outdict['InjectionParameters']['fmax'] = fmax
  outdict['InjectionParameters']['deltaF'] = deltaF
  outdict['InjectionParameters']['scales'] = scales
  outdict['InjectionParameters']['SNRs'] = SNRs
 
  outdict['Detectors'] = dets
 
  outdict['MCMC'] = {}
  outdict['MCMC']['Niterations'] = Niter
  outdict['MCMC']['Nburnin'] = Nburnin
  outdict['MCMC']['Nensemble'] = Nensemble

  commandline = ''
  for arg in sys.argv:
    commandline += arg + ' '
  outdict['CommandLine'] = commandline

  # get scale factor standard devaitions and 95% credible intervals
  scstds = []
  scstdsfrac = []
  cis = []
  cisfrac = []
  for i in range(len(dets)):
    mean_s = np.mean(samples[:,4+i])
    std_s = np.std(samples[:,4+i])
    scstds.append(std_s)
    scstdsfrac.append(std_s/mean_s)
    ci = credible_interval(samples[:,4+i], 0.95)
    cis.append(ci)
    cisfrac.append(np.diff(ci)[0]/mean_s)
  
  outdict['Results'] = {}
  outdict['Results']['ScaleSigma'] = scstds
  outdict['Results']['Scale95%CredibleInterval'] = cis
  outdict['Results']['ScaleSigmaFrac'] = scstdsfrac
  outdict['Results']['Scale95%CredibleIntervalFrac'] = cisfrac
 
  f = open(infofile, 'w')
  json.dump(outdict, f, indent=2)
  f.close()
 
  if opts.plot:
    try:
      import triangle
    except:
      print >> sys.stderr, "Can't load triangle.py, so no plot will be produced"
      sys.exit(0)
      
    labels = ["$\psi$", "$\phi_0$", "$\iota$", "$t_c$"]
    truths = [psi, phi0, iota, t0]
    for i in range(len(dets)):
      labels.append("$s_{\mathrm{%s}}}$" % dets[i])
      truths.append(scales[i])
      print np.std(samples[:,4+i])
    
    fig = triangle.corner(samples, labels=labels, truths=truths)
    
    plotfile = os.path.join(outpath, 'posterior_plot_'+intseed+'.png')
    fig.savefig(plotfile)