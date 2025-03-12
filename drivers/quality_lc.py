import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.io import fits
import os
from os.path import join
from aesthetic.plot import savefig, set_style

fitsdir = '/Users/luke/Dropbox/proj/cpv/data/photometry/tess'
fitspath = join(fitsdir, 'tess2024030031500-s0075-0000000141146667-0270-s_lc.fits')

hl = fits.open(fitspath)
d = hl[1].data


plt.close("all")

set_style("science")
fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(d['TIME'], np.log2(d['QUALITY']), c='k', s=0.5, label='log2(quality)')

ax.scatter(d['TIME'], np.zeros(len(d['TIME']))-4, s=0.5, c='b',
            label='window fn')

ax.scatter(d['TIME'], d['SAP_FLUX']/np.nanmedian(d['SAP_FLUX']), c='g',
            s=0.1, label='sap_flux');

ax.scatter(d['TIME'], d['SAP_BKG']/np.nanmedian(d['SAP_BKG'])/40-2, c='k',
            s=0.2, label='SAP_BKG')

ax.legend(fontsize='x-small')

plt.show();

savefig(fig, 'results/quality_lc/quality_lc.png')
