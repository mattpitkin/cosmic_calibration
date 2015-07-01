#!/usr/bin/env python

"""
Script to go through a set of directories and gather up information from all the
info_* files to produce a histrogram of how well the calibration scale factor
can be recovered for different distance.
"""

import json
import sys
import os
import sys

from optparse import OptionParser

import numpy as np
from matplotlib import pyplot as pl

usage = "Usage: %prog [options]"

parser = OptionParser( usage = usage )

parser.add_option("-o", "--outfile", dest="outfile", help="The output plot file")

parser.add_option("-i", "--indir", dest="indir", help="The input directory", metavar="DIR")

# parse input options
(opts, args) = parser.parse_args()

# check that output path has been given
if not opts.__dict__['outfile']:
  print >> sys.stderr, "Must specify an output file"
  parser.print_help()
  sys.exit(0)
else:
  outfile = opts.outfile

# check that output path has been given
  if not opts.__dict__['indir']:
    print >> sys.stderr, "Must specify an input path"
    parser.print_help()
    sys.exit(0)
  else:
    indir = opts.indir
    
    if not os.path.isdir(indir):
      print >> sys.stderr, "Must specify an input path"
      parser.print_help()
      sys.exit(0)

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

dists = [50, 100, 150, 200, 250, 300, 350, 400, 450]
colours = ['r', 'm', 'c', 'g', 'b', 'p', 'k']

prefix = indir

# directories
dirs = [os.path.join(prefix, '%dMpc' % dist) for dist in dists]
print dirs
# file name prefix
fnpre = 'info_'

fig, ax = pl.subplots(figsize=(8,5))

data = []

for i, d in enumerate(dirs):
  print d

  relsf = []
  
  files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and fnpre in f]
  
  # go through files and extract the relative standard devaition on the scale factors for each detector
  for f in files:
    fo = open(f, 'r')
    info = json.load(fo)
    fo.close()

    vals = []
    for k in range(len(info['InjectionParameters']['scales'])):
      vals.append(info['Results']['Scale68%CredibleInterval'][k][1]-info['Results']['Scale68%CredibleInterval'][k][0])

    relsf.append(vals)
    
  nprelsf = np.array(relsf)
  print nprelsf.shape
  
  # divide by two to get the half widths
  nprelsf[:,0] = nprelsf[:,0]/(2.*info['InjectionParameters']['scales'][0])
  nprelsf[:,1] = nprelsf[:,1]/(2.*info['InjectionParameters']['scales'][1]) 
  nprelsf[:,2] = nprelsf[:,2]/(2.*info['InjectionParameters']['scales'][2])

  data.append(nprelsf[:,0])
  data.append(nprelsf[:,1])
  data.append(nprelsf[:,2])

  # plot output
  #ci1 = credible_interval(nprelsf[:,0], 0.90)
  #ci2 = credible_interval(nprelsf[:,1], 0.90)
  #ci3 = credible_interval(nprelsf[:,2], 0.90)
  
  #pl.plot(dists[i]-7.5, np.mean(nprelsf[:,0]), 'bo', lw=2, ms=6)
  #pl.plot(dists[i], np.mean(nprelsf[:,1]), 'ro', lw=2, ms=6)
  #pl.plot(dists[i]+7.5, np.mean(nprelsf[:,2]), 'go', lw=2, ms=6)
  
  #pl.plot(dists[i]-7.5, np.median(nprelsf[:,0]), 'bx', lw=2, ms=6)
  #pl.plot(dists[i], np.median(nprelsf[:,1]), 'rx', lw=2, ms=6)
  #pl.plot(dists[i]+7.5, np.median(nprelsf[:,2]), 'gx', lw=2, ms=6)
  
  #if i == 0:
  #  pl.plot([dists[i]-7.5, dists[i]-7.5], ci1, 'b', label='H1', lw=2)
  #  pl.plot([dists[i], dists[i]], ci2, 'r', label='L1', lw=2)
  #  pl.plot([dists[i]+7.5, dists[i]+7.5], ci3, 'g', label='V1', lw=2)
  #else:
  #  pl.plot([dists[i]-7.5, dists[i]-7.5], ci1, 'b', lw=2)
  #  pl.plot([dists[i], dists[i]], ci2, 'r', lw=2)
  #  pl.plot([dists[i]+7.5, dists[i]+7.5], ci3, 'g', lw=2)


# check for directory at distance 10000 (i.e. noise only) and if so plot the
# *minimum* value from that, to show what to expect from noise

minH1 = None
minL1 = None
minV1 = None

ldir = os.path.join(prefix, '10000Mpc')
if os.path.isdir(ldir):
  relsf = []

  files = [os.path.join(ldir, f) for f in os.listdir(ldir) if os.path.isfile(os.path.join(ldir, f)) and fnpre in f]

  # go through files and extract the relative standard devaition on the scale factors for each detector
  for f in files:
    fo = open(f, 'r')
    info = json.load(fo)
    fo.close()

    vals = []
    for k in range(len(info['InjectionParameters']['scales'])):
      vals.append(info['Results']['Scale68%CredibleInterval'][k][1]-info['Results']['Scale68%CredibleInterval'][k][0])

    relsf.append(vals)

  nprelsf = np.array(relsf)

  # divide by two to get the half widths
  nprelsf[:,0] = nprelsf[:,0]/(2.*info['InjectionParameters']['scales'][0])
  nprelsf[:,1] = nprelsf[:,1]/(2.*info['InjectionParameters']['scales'][1])
  nprelsf[:,2] = nprelsf[:,2]/(2.*info['InjectionParameters']['scales'][2])

  minH1 = np.min(nprelsf[:,0])
  minL1 = np.min(nprelsf[:,1])
  minV1 = np.min(nprelsf[:,2])


positions = []
for dist in dists:
  positions.append(dist-10)
  positions.append(dist)
  positions.append(dist+10)
 
#bp = pl.boxplot(data, notch=0, sym='+', positions=positions, widths=8)
bp = pl.boxplot(data, notch=0, sym='', positions=positions, widths=8) 

pl.setp(bp['boxes'], color='black')
pl.setp(bp['whiskers'], color='black')
pl.setp(bp['fliers'], color='black')

# Now fill the boxes with desired colors
from matplotlib.patches import Polygon
boxColors = ['b', 'r', 'g']
numBoxes = len(data)
medians = range(numBoxes)
for i in range(numBoxes):
  box = bp['boxes'][i]
  boxX = []
  boxY = []
  for j in range(5):
    boxX.append(box.get_xdata()[j])
    boxY.append(box.get_ydata()[j])
  boxCoords = zip(boxX,boxY)
  
  k = i % 3
  boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
  ax.add_patch(boxPolygon)
  
  # pl.setp(bp['fliers'][i], markerfacecolor=boxColors[k]) # this line doesn't work!

  # Now draw the median lines back over what we just filled in
  med = bp['medians'][i]
  medianX = []
  medianY = []
  for j in range(2):
      medianX.append(med.get_xdata()[j])
      medianY.append(med.get_ydata()[j])
      pl.plot(medianX, medianY, 'k')
      medians[i] = medianY[0]
  # Finally, overplot the sample averages, with horizontal alignment
  # in the center of each box
  pl.plot([np.average(med.get_xdata())], [np.average(data[i])],
           color='w', marker='*', markeredgecolor='k')

if minH1 is not None and minL1 is not None and minV1 is not None:
  pl.plot([0., 500], [minH1, minH1], 'b--')
  pl.plot([0., 500], [minL1, minL1], 'r--')
  pl.plot([0., 500], [minV1, minV1], 'g--')


ax.set_xticklabels(dists)
ax.set_xticks(dists)

ax.set_xlim((0, 500))

#pl.legend(loc='best')
pl.xlabel('distance (Mpc)')
#pl.ylabel('90\% $\sigma_{\\textrm{frac}}$')

try:
  fig.savefig(outfile)
except:
  print >> sys.stderr, "Cannot create plot!"
  sys.exit(0)
