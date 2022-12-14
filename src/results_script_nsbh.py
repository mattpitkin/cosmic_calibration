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
pl.rc('grid', linestyle=':')
pl.rc('grid', alpha=0.5)
pl.rc('grid', linewidth=0.5)

dists = [100, 200, 300, 400, 500, 600, 700, 800, 900] #, 1000]

prefix = indir

# directories
dirs = [os.path.join(prefix, '%dMpc' % dist) for dist in dists]
print dirs
# file name prefix
fnpre = 'info_'

fig, ax = pl.subplots(figsize=(8,5))

data = []
rates = []

for i, d in enumerate(dirs):
  print d

  relsf = []
  totaltime = 0.
  
  files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and fnpre in f]
  
  # go through files and extract the relative standard devaition on the scale factors for each detector
  acc = 0
  for f in files:
    fo = open(f, 'r')
    info = json.load(fo)
    fo.close()

    vals = []
    for k in range(len(info['InjectionParameters']['scales'])):
      # check whether chain looks like it's converged with a simple check of the histogram
      histd = info['Results']['ScaleHist'][k]
      histd = np.array(histd)
      # get number of nonzero values
      nonzero = np.zeros(len(histd[0]))[np.array(histd[0]) > 0]

      # say converged chain must have more than 90% of posterior points being non-zero
      # and the standard deviation of the points must be within a sixth of the width of the histogram
      if float(len(nonzero))/float(len(histd[0])) > 0.9 and info['Results']['ScaleSigma'][k] < (histd[1][-1]-histd[1][0])/6.:
        # divide by two to get the half widths and convert to percentage
        vals.append(100.*(info['Results']['Scale68%CredibleInterval'][k][1]-info['Results']['Scale68%CredibleInterval'][k][0])/(2.*info['InjectionParameters']['scales'][k]))

    if len(vals) == 3:
      relsf.append(vals)
      acc += 1

    totaltime += info['Attempts']   

  rates.append(100.*len(files)/totaltime)
  nprelsf = np.array(relsf)

  data.append(nprelsf[:,0])
  data.append(nprelsf[:,1])
  data.append(nprelsf[:,2])

  print acc
  print np.mean(nprelsf[:,0]), np.mean(nprelsf[:,1]), np.mean(nprelsf[:,2])
  print np.median(nprelsf[:,0]), np.median(nprelsf[:,1]), np.median(nprelsf[:,2])

positions = []
for dist in dists:
  positions.append(dist-20)
  positions.append(dist)
  positions.append(dist+20)

bp = pl.boxplot(data, whis=[5, 95], notch=0, sym='', positions=positions, widths=16) 
#bp = pl.boxplot(data, notch=0, sym='x', positions=positions, widths=16)

pl.setp(bp['boxes'], color='black')
pl.setp(bp['whiskers'], color='black')
pl.setp(bp['fliers'], color='black')

hs = []

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
  hs.append(ax.add_patch(boxPolygon))

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

ax.set_xticklabels(dists)
ax.set_xticks(dists)

ax.set_xlim((0, 1100))
ax.set_ylim((0, 100))
ax.grid(True)

pl.legend(hs, ['H1', 'L1', 'V1'], loc='best')
pl.xlabel('distance (Mpc)')
pl.ylabel('\% calibration scaling error (1$\sigma$ equivalent)')

ax2 = ax.twinx()
ax2.plot(dists, rates, 'mo--', markerfacecolor='None', markeredgecolor='m')
ax2.set_ylabel('\% of source distribution detectable')
ax2.yaxis.label.set_color('magenta')
#ax2.set_xlim((0, 1100))
ax2.set_xlim((0, 1000))

try:
  fig.savefig(outfile)
except:
  print >> sys.stderr, "Cannot create plot!"
  sys.exit(0)
