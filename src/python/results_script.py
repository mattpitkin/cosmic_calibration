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

fig, ax = pl.subplots(3, figsize=(7,14), dpi=200)

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
  ax[0].hist(nprelsf[:,0], bins=20, histtype='step', normed=True, label='%d Mpc'%dists[i])
  ax[1].hist(nprelsf[:,1], bins=20, histtype='step', normed=True, label='%d Mpc'%dists[i])
  ax[2].hist(nprelsf[:,2], bins=20, histtype='step', normed=True, label='%d Mpc'%dists[i])
  
fig.savefig('relative_error.png')
