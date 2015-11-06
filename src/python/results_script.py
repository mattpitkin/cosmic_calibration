#!/usr/bin/env python

"""
Script to go through a set of directories and gather up information from all the
info_* files to produce a histrogram of how well the calibration scale factor
can be recovered for different distance and the P-P plots of injected versus
recovered confidence intervals.
"""

import json
import sys
import os
import sys
from scipy.interpolate import interp1d
from scipy.stats import beta
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


# get the credible region in which the injected value is found (bounded from zero for convenience)
def credible_inj(histbins, histvals, injval):
  # normalise histogram and get cumulative sum
  cumvals = np.cumsum(histvals.astype(float)/np.sum(histvals))

  if injval <= histbins[0] or injval >= histbins[-1]:
    return 1. # injection outside posterior
  else:
    return cumvals[(np.abs(histbins-injval)).argmin()]


pl.rc('text', usetex=True)
pl.rc('font', family='serif')
pl.rc('font', size=14)
pl.rc('grid', linestyle=':')
pl.rc('grid', alpha=0.5)
pl.rc('grid', linewidth=0.5)

dists = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

prefix = indir

# directories
dirs = [os.path.join(prefix, '%dMpc' % dist) for dist in dists]
print dirs
# file name prefix
fnpre = 'info_'

fig, ax = pl.subplots(figsize=(8,5))

data = []
rates = []

pp50 = []
pp500 = []

for i, d in enumerate(dirs):
  print d

  relsf = []
  
  files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and fnpre in f]
  
  totaltime = 0.
  accepted = 0

  # get info for p-p plots for first and last distance
  pps = []

  # go through files and extract the relative standard devaition on the scale factors for each detector
  for f in files:
    fo = open(f, 'r')
    info = json.load(fo)
    fo.close()

    
    ppss = []

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
        if i == 0 or i == len(dirs)-1:
          ppss.append(credible_inj(np.array(histd[1]), np.array(histd[0]), info['InjectionParameters']['scales'][k]))

    if len(vals) == 3:
      relsf.append(vals)
      if i == 0 or i == len(dirs)-1:
        pps.append(ppss)
      accepted += 1

    totaltime += info['Attempts']
  
  rates.append(100.*len(files)/totaltime)
  nprelsf = np.array(relsf)

  data.append(nprelsf[:,0])
  data.append(nprelsf[:,1])
  data.append(nprelsf[:,2])

  if i == 0:
    pp50 = np.copy(np.array(pps))
  if i == len(dirs)-1:
    pp500 = np.copy(np.array(pps))

  print accepted
  print np.mean(nprelsf[:,0]), np.mean(nprelsf[:,1]), np.mean(nprelsf[:,2])
  print np.median(nprelsf[:,0]), np.median(nprelsf[:,1]), np.median(nprelsf[:,2])

positions = []
for dist in dists:
  positions.append(dist-10)
  positions.append(dist)
  positions.append(dist+10)
 
#bp = pl.boxplot(data, notch=0, sym='+', positions=positions, widths=8)
bp = pl.boxplot(data, whis=[5, 95], notch=0, sym='', positions=positions, widths=8) 

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

ax.set_xlim((0, 550))
ax.set_ylim((0, 100))

pl.legend(hs, ['H1', 'L1', 'V1'], loc='best')
pl.xlabel('distance (Mpc)')
pl.ylabel('\% calibration scaling error (1$\sigma$ equivalent)')
ax.grid(True)

ax2 = ax.twinx()
ax2.plot(dists, rates, 'mo--', markerfacecolor='None', markeredgecolor='m')
ax2.set_ylabel('\% of source distribution detectable')
ax2.yaxis.label.set_color('magenta')
ax2.set_xlim((0, 550))

try:
  fig.savefig(outfile)
except:
  print >> sys.stderr, "Cannot create plot!"
  sys.exit(0)

fig.clf()
pl.close(fig)

# create P-P plots for 50 and 500 Mpc for all detectors
fig, ax = pl.subplots(figsize=(6,5))

# info for error bars
ps = np.linspace(0., 1., 1000)

# ('even tailed') confidence interval for 95%
alpha = 1.-0.95

for i in range(3):
  # get cumulative histogram of found regions
  if i == 0: print pp50[:,i]
  ax.hist(pp50[:,i], bins=len(pp50[:,i]), cumulative=True, normed=True, histtype='step', color=boxColors[i])

ax.set_xlabel('Credible interval (CI)')
ax.set_ylabel('Cumulative fraction of true values within CI')
ax.set_xlim((0., 1.))
ax.set_ylim((0., 1.))
pl.legend(['H1', 'L1', 'V1'], loc='best')
ax.grid(True)
ax.plot([0., 1.], [0., 1.], 'k--') # plot diagonal

# plot target confidence band
bins = np.linspace(0.01, 0.99, 50)
ntot = len(pp50[:,0])
cs = np.round(bins*ntot)
errortop = np.zeros(len(bins))
errorbottom = np.zeros(len(bins))

for i, v in enumerate(cs):
  a = v + 1
  b = ntot - v + 1

  Bcs = beta.cdf(ps, a, b)
  csu, ui = np.unique(Bcs, return_index=True)
  intf = interp1d(csu, ps[ui], kind='linear') # interpolation function

  errortop[i] = intf(1.-alpha/2.)
  errorbottom[i] = intf(alpha/2.)

pl.fill_between(bins, errorbottom, errortop, alpha=0.25, facecolor='grey', edgecolor='grey')

fig.savefig(os.path.join(os.path.dirname(outfile), 'pp50Mpc.pdf'))

fig.clf()
pl.close(fig)

fig, ax = pl.subplots(figsize=(6,5))

for i in range(3):
  # get cumulative histogram of found regions
  ax.hist(pp500[:,i], bins=100, cumulative=True, normed=True, histtype='step', color=boxColors[i])

ax.set_xlabel('Credible interval (CI)')
ax.set_ylabel('Cumulative fraction of true values within CI')
ax.set_xlim((0., 1.))
ax.set_ylim((0., 1.))
pl.legend(['H1', 'L1', 'V1'], loc='best')
ax.grid(True)
ax.plot([0., 1.], [0., 1.], 'k--') # plot diagonal

# plot target confidence band
bins = np.linspace(0.01, 0.99, 50)
ntot = len(pp50[:,0])
cs = np.round(bins*ntot)
for i, v in enumerate(cs):
  a = v + 1
  b = ntot - v + 1

  Bcs = beta.cdf(ps, a, b)
  csu, ui = np.unique(Bcs, return_index=True)
  intf = interp1d(csu, ps[ui], kind='linear') # interpolation function

  errortop[i] = intf(1.-alpha/2.)
  errorbottom[i] = intf(alpha/2.)

pl.fill_between(bins, errorbottom, errortop, alpha=0.25, facecolor='grey', edgecolor='grey')

fig.savefig(os.path.join(os.path.dirname(outfile), 'pp500Mpc.pdf'))
