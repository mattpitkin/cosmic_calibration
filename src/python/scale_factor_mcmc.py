#!/usr/bin/env python

"""
A script to run the calibration scale factor check using the emcee MCMC sampler
"""
import numpy as np
import copy
import sys
from optparse import OptionParser

# swig lal modules for signal generation
import lal
import lalsimulation

# MCMC code
import emcee

# pyCBC code for generating coloured frequency noise
from pycbc.types import frequencyseries
from pycbc import noise

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
            'T1': lal.LALDetectorIndexTAMA300DIFF, \
            'AL1': lal.LALDetectorIndexLLODIFF, \
            'AH1': lal.LALDetectorIndexLHODIFF, \
            'AV1': lal.LALDetectorIndexVIRGODIFF}

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
  lp = lnprior(theta, iotaprior, tccentre)
  
  if not np.isfinite(lp):
    return -np.inf
  
  return lp + lnlike(theta, y, dd, m1, m2, dist, fmin, fmax, deltaF, resps, asds)


# define the log prior function
def lnprior(theta, iotawidth, tccentre):
  # unpack theta
  psi, phi0, iota, tc, sf1, sf2, sf3 = theta
  lp = 0. # logprior

  ss = [sf1, sf2, sf3]

  # outside prior ranges
  if 0. < psi < np.pi and 0. < phi0 < 2.*np.pi and tccentre-0.01 < tc < tccentre+0.01:
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
  psi, phi0, iota, tc, sf1, sf2, sf3 = theta
  
  spsi = np.sin(2.*psi)
  cpsi = np.cos(2.*psi)
  ss = [sf1, sf2, sf3]
  
  # generate waveform
  hp, hc = fdwaveform(phi0, deltaF, m1, m2, fmin, fmax, dist, iota)

  L = 0. # log likelihood

  # add response function and scale waveform and calculate likelihood
  for i in range(len(resps)):
    Ap, Ac = resps[i]

    H = (hp*(Ap*cpsi + Ac*spsi) + hc*(Ac*cpsi - Ap*spsi))*ss[i]

    Hs = H/asds[i]
    
    Hs[~np.isfinite(Hs)] = 0.
    
    dh = np.vdot(y[i], Hs)
    hh = np.vdot(Hs, Hs)
    
    L = L + 4.*deltaF*(dh.real - 0.5*(hh.real + dd[i]))
  
  return L



if __name__=='__main__':
  
  usage = "Usage: %prog [options]"

  parser = OptionParser( usage = usage )

  parser.add_option("-o", "--outpath", dest="outpath", help="The path for "
                    "the analysis output (a sub-directory based on the pulsar "
                    "name will be created here that contains the pulsar "
                    "information)", metavar="DIR")

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
                    help="The phase at coalescence (rads) [default: %default]", default=0.)

  parser.add_option("-a", "--psi", dest="psi", type="float",
                    help="The polarisation angle (rads) [default: %default]", default=0.)

  parser.add_option("-f", "--fmin", dest="fmin", type="float",
                    help="Lower frequency bound (Hz) [default: %default]", default=40.)
  
  parser.add_option("-m", "--fmax", dest="fmax", type="float",
                    help="Upper frequency bound (Hz) [default: %default]", default=1600.)

  parser.add_option("-x", "--deltaF", dest="deltaF", type="float",
                    help="Frequency bins size (Hz) [default: %default]", default=2.)

  parser.add_option("-z", "--noise", dets="noise", type="boolean", default=False,
                    help="If this flag is set a noise spectrum is added to the data")


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

  # integration options
  fmin = opts.fmin
  fmax = opts.fmax
  deltaF = opts.deltaF

  # injected source options
  dist = opts.dist
  t0 = opts.t0
  phi0 = opts.phi0
  psi = opts.psi
  
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

  # check whether to add noise
  addnoise = opts.noise

  # we will use three detectors H1, L1 and V1
  dets = ['H1', 'L1', 'V1']

  # we will hardcode the calibration scale factors for each detector
  scales = [0.8, 1.2, 1.4]

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

  if addnoise:
    freqvals = np.linspace(0., deltaF*(len(hp)-1), len(hp))
    freqarray = frequencyseries.FrequencySeries(freqvals, delta_f=deltaF)

  # generate the ASD esimates (use values from P1200087)
  asds = []
  SNRs = []
  dd = [] # the data cross product
  for i in range(len(dets)):
    if dets[i] in ['H1', 'L1']:
      ret = lalsimulation.SimNoisePSDaLIGODesignSensitivityP1200087(psd, fmin)
      asds.append(np.sqrt(psd.data.data)*scales[i])
    elif dets[i] in 'V1':
      ret = lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd, fmin)
      asds.append(np.sqrt(psd.data.data)*scales[i])
    
    # create additvie noise
    if addnoise:
        psdarray = frequencyseries.FrequencySeries(psd.data.data, delta_f=deltaF)
        noisevals = noise.gaussian.frequency_noise_from_psd(psdarray)
        H[i] = H[i] + noisevals*scales[i]
    
    H[i] = H[i]/(np.sqrt(psd.data.data)*scales[i])
    H[i][~np.isfinite(H[i])] = 0.
    
    dd.append(np.vdot(H[i], H[i]).real)
    
    snr = np.sqrt(4.*deltaF*dd[i])
    SNRs.append(snr)
    print >> sys.stderr, "%s: SNR = %.2f" % (dets[i], snr) 
 
  # set up MCMC
  # get initial seed points
  pos = []
  for i in range(Nensemble):
    # psi
    psiini = np.random.rand()*np.pi
    
    # phi0
    phi0ini = np.random.rand()*2.*np.pi/2.
    
    # iota
    iotaini = iotawidth*np.random.randn()
  
    # time of coalescence
    tcini = -0.01 + 2.*0.01*np.random.rand() + t0
  
    # scale factors
    sfs = 0.01 + (10.-0.01)*np.random.rand(3)
    
    pos.append(np.array([psiini, phi0ini, iotaini, tcini, sfs[0], sfs[1], sfs[2]]))
  
  ndim = len(pos[0])
  # Multiprocessing version
  #sampler = emcee.EnsembleSampler(Nensemble, ndim, lnprob, args=(H, dd, iotawidth, t0, m1, m2, 
  #                                dist, fmin, fmax, deltaF, resps, asds), threads=3)
  sampler = emcee.EnsembleSampler(Nensemble, ndim, lnprob, args=(H, dd, iotawidth, t0, m1, m2,
                                  dist, fmin, fmax, deltaF, resps, asds))
  
  sampler.run_mcmc(pos, Niter)
  
  #  remove burn-in and flatten
  samples = sampler.chain[:, Nburnin:, :].reshape((-1, ndim))
  
  import triangle
  fig = triangle.corner(samples, labels=["$\psi$", "$\phi_0$", "$\iota$", "$t_c$", "$s_1$", \
                        "$s_2$", "$s_3$"],
                        truths=[psi, phi0, iota, t0, scales[0], scales[1], scales[2]])
  fig.savefig("test"+intseed+".png")

  print np.std(samples[:,4]), np.std(samples[:,5]), np.std(samples[:,6])