#!/usr/bin/env python

"""
Script to go through a set of directories and gather up information from all the
info_* files to produce a histrogram of how well the calibration scale factor
can be recovered for different distance.
"""

import json
import sys
import os

import numpy as np
from matplotlib import pyplot as pl


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


pl.rc('text', usetex=True)
pl.rc('font', family='serif')
pl.rc('font', size=14)

dists = [50, 100, 150, 200, 250, 300, 450]
colours = ['r', 'm', 'c', 'g', 'b', 'p', 'k']

prefix = '/home/sismp2/projects/cosmic_calibration/no_noise/distances'

# directories
dirs = [os.path.join(prefix, '%dMpc' % dist) for dist in dists]
print dirs
# file name prefix
fnpre = 'info_'

#fig, ax = pl.subplots(3, figsize=(7,14), dpi=200)
fig = pl.figure(figsize=(7,5), dpi=200)

for i, d in enumerate(dirs):
  print d

  relsf = []
  
  files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and fnpre in f]
  
  # go through files and extract the relative standard devaition on the scale factors for each detector
  for f in files:
    fo = open(f, 'r')
    info = json.load(fo)
    fo.close()

    relsf.append(info['Results']['ScaleSigmaFrac'])
    
  nprelsf = np.array(relsf)
  print nprelsf.shape
   
  # plot output
  #ax[0].hist(nprelsf[:,0], bins=20, histtype='step', normed=True, label='%d Mpc'%dists[i])
  #ax[1].hist(nprelsf[:,1], bins=20, histtype='step', normed=True, label='%d Mpc'%dists[i])
  #ax[2].hist(nprelsf[:,2], bins=20, histtype='step', normed=True, label='%d Mpc'%dists[i])
  ci1 = credible_interval(nprelsf[:,0], 0.90)
  ci2 = credible_interval(nprelsf[:,1], 0.90)
  ci3 = credible_interval(nprelsf[:,2], 0.90)
  
  pl.plot(dists[i]-7.5, np.mean(nprelsf[:,0]), 'bo', lw=2, ms=6)
  pl.plot(dists[i], np.mean(nprelsf[:,1]), 'ro', lw=2, ms=6)
  pl.plot(dists[i]+7.5, np.mean(nprelsf[:,2]), 'go', lw=2, ms=6)
  
  pl.plot(dists[i]-7.5, np.median(nprelsf[:,0]), 'bx', lw=2, ms=6)
  pl.plot(dists[i], np.median(nprelsf[:,1]), 'rx', lw=2, ms=6)
  pl.plot(dists[i]+7.5, np.median(nprelsf[:,2]), 'gx', lw=2, ms=6)
  
  if i == 0:
    pl.plot([dists[i]-7.5, dists[i]-7.5], ci1, 'b', label='H1', lw=2)
    pl.plot([dists[i], dists[i]], ci2, 'r', label='L1', lw=2)
    pl.plot([dists[i]+7.5, dists[i]+7.5], ci3, 'g', label='V1', lw=2)
  else:
    pl.plot([dists[i]-7.5, dists[i]-7.5], ci1, 'b', lw=2)
    pl.plot([dists[i], dists[i]], ci2, 'r', lw=2)
    pl.plot([dists[i]+7.5, dists[i]+7.5], ci3, 'g', lw=2)
  
pl.legend(loc='best')
pl.xlabel('distance (Mpc)')
pl.ylabel('90\% $\sigma_{\\textrm{frac}}$')
  
fig.savefig('relative_error.png')
