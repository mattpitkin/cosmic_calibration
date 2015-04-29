#!/usr/bin/env python

"""
A script to check that the injections are as expected
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


if __name__=='__main__':
  
  usage = "Usage: %prog [options]"

  parser = OptionParser( usage = usage )

  parser.add_option("-g", "--det", dest="dets",
                    help="An interferometer to be used (multiple interferometers can be set \
with multiple uses of this flag, e.g. \"-g H1 -g L1 -g V1\") [default: H1]", 
                    action="append")

  parser.add_option("-N", "--Nsim", dest="Nsim",
                    help="Number of simulations [default: %default]", type="int",
                    default=1000)
 
  parser.add_option("-D", "--dist", dest="dist", type="float",
                    help="Inspiral distance in Mpc [default: %default]", default=10.)
  
  parser.add_option("-r", "--ra", dest="ra", type="float",
                    help="Right ascension (rads) - if not specified the RA will be drawn randomly \
from a uniform distribution on the sky")
  
  parser.add_option("-d", "--dec", dest="dec", type="float",
                    help="Declination (rads) - if not specified the dec will be drawn randomly \
from a uniform distribution on the sky")

  parser.add_option("-i", "--iota", dest="iota", type="float",
                    help="Inclination of insprial (rads) - if not specified cos(iota) will be drawn from \
a uniform distribution between -1 and 1")

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

  parser.add_option("-k", "--psd-noise", dest="psdnoise", default=0, type="int",
                     help="If this flag is non-zero the PSD is estimated as the average of \
the given number of noisy PSD estimates.")
 

  # parse input options
  (opts, args) = parser.parse_args()

  Nsim = opts.Nsim

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

  # generate simulations
  if not opts.__dict__['ra']:
    ras = 2.*np.pi*np.random.rand(Nsim) # generate RA uniformly between 0 and 2pi
  else:
    ras = opts.ra*np.ones(Nsim)

  if not opts.__dict__['dec']:
    decs = -(np.pi/2.) + np.arccos(2.*np.random.rand(Nsim)-1.)
  else:
    decs = opts.dec*np.ones(Nsim)

  if not opts.__dict__['iota']:
    cosiotas = -1. + 2.*np.random.rand(Nsim)
  else:
    cosiotas = np.cos(opts.iota)*np.ones(Nsim)

  if not opts.__dict__['psi']:
    psis = 0.5*np.pi*np.random.rand(Nsim) # draw from between 0 and pi/2
  else:
    psis = opts.psi*np.ones(Nsim)
    
  if not opts.__dict__['phi0']:
    phi0s = 2.*np.pi*np.random.rand(Nsim) # draw from between 0 and 2pi
  else:
    phi0s = opts.phi0*np.ones(Nsim)

  # check whether to calculate the PSD from noisy data
  psdnoise = opts.psdnoise

  # create a simulated waveform in each detector
  # the masses will both be fixed at 1.4 solar masses
  m1 = 1.4
  m2 = 1.4

  indSNRs = []
  netSNRs = []

  asds = []
  # create frequency series for PSDs
  psd = lal.CreateREAL8FrequencySeries('name', t0, 0., deltaF, lal.Unit(), 1+int(fmax/deltaF))

  for i in range(len(dets)):
    if dets[i] in ['H1', 'L1']:
      ret = lalsimulation.SimNoisePSDaLIGODesignSensitivityP1200087(psd, fmin)
      psd.data.data[psd.data.data == 0.] = np.inf
    elif dets[i] in 'V1':
      ret = lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd, fmin)
    
    psd.data.data[psd.data.data == 0.] = np.inf
    
    if psdnoise:
      # get average of 32 noise realisations for PSD
      psdav = np.zeros(psd.data.length)
      for j in range(psdnoise):
        noisevals = frequency_noise_from_psd(psd.data.data, deltaF)
        psdav = psdav + 2.*deltaF*np.abs(noisevals)**2
      psdav = psdav/float(psdnoise)
      
      psdav[psdav == 0.] = np.inf
      
      asds.append(np.sqrt(psdav))
    else:
      asds.append(np.sqrt(psd.data.data))

  for i in range(Nsim):
    # waveform
    hp, hc = fdwaveform(phi0s[i], deltaF, m1, m2, fmin, fmax, dist, np.arccos(cosiotas[i]))

    snrs = []
    netsnr = 0.

    # waveforms for each detector (accounting for antenna pattern and calibration scale)
    for j in range(len(dets)):
      apt, act = antenna_response( t0, ras[i], decs[i], psis[i], dets[j] )
      H = (hp*apt + hc*act)

      H = H/asds[j]
      snr = np.sqrt(4.*deltaF*np.vdot(H, H).real)
      snrs.append(snr)
      netsnr = netsnr + snr**2
 
    indSNRs.append(snrs)
    netSNRs.append(np.sqrt(netsnr))
 
  # plot histogram of SNRs
  fig = pl.figure(figsize=(6,5), dpi=200)
 
  print np.mean(netSNRs), np.std(netSNRs)
  pl.hist(netSNRs, bins=20, histtype='step', normed=True)
  ax = pl.gca()
  pl.plot([np.mean(netSNRs), np.mean(netSNRs)], [0., ax.get_ylim()[1]], 'k--')
  pl.show()
  fig.savefig('SNRhist_%.2fMpc.png' % dist)
