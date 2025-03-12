import os
from numpy import array as nparr
from scipy.interpolate import interp1d
from astropy.io import fits
from glob import glob
from os.path import join
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from complexrotators.paths import DATADIR, RESULTSDIR
from astropy import units as u
from aesthetic.plot import set_style, savefig

from cdips_followup.spectools import (
    _get_full_hires_spectrum, bin_spectrum
)
from complexrotators.getters import get_bochanski2007_m_standard

def make_sptype_comparison(chip, order):

    fitsdir = join(DATADIR, 'spectra/HIRES/TIC141146667_DEBLAZED')
    fitspaths = np.sort(glob(join(fitsdir, f"{chip}j*fits")))[1:-1]
    N = len(fitspaths)

    set_style('science')
    fig, ax = plt.subplots(figsize=(4,7))
    colors = [plt.cm.viridis_r(i) for i in np.linspace(0, 0.8, N)]

    bflxs = []

    for spectrum_path, c in zip(fitspaths, colors):

        verbose = 0
        flx_2d, norm_flx_2d, wav_2d, mjd, ra, dec = _get_full_hires_spectrum(
            spectrum_path, verbose=verbose
        )

        flx, wav = norm_flx_2d[order, :], wav_2d[order, :]

        bin_width = 0.2 # angstrom
        bwav, bflx, bflxerr = bin_spectrum(wav, flx, bin_width)
        bflxs.append(bflx)

        #plt.plot(bwav, bflx, c=c, lw=0.5, alpha=0.2)

    bflxs = np.array(bflxs)
    bflxs_tavg = np.nanmedian(bflxs, axis=0)
    bflxs_tstd = np.nanstd(bflxs, axis=0) / np.sqrt(N)
    bflxs_tstd[bflxs_tstd > 0.05]  = np.nanmean(bflxs_tstd)

    dlam = 1.5
    ax.errorbar(bwav+dlam, bflxs_tavg, yerr=bflxs_tstd, c='k', lw=1, label='data')

    sptypes = [f"M{ix}" for ix in range(4,8)]
    colors = [plt.cm.magma_r(i) for i in np.linspace(0.3, 0.8, len(sptypes))]
    for ix, (sptype, c) in enumerate(zip(sptypes, colors)):
        swav, sflx = get_bochanski2007_m_standard(sptype=sptype, activestr='all')
        sel = (swav > bwav.min()+5) & (swav < bwav.max()-5)
        swav = swav[sel]
        sflx = sflx[sel] / np.nanmedian(sflx[sel])
        ax.plot(swav, sflx - 0.15*(ix-4), c=c, lw=1, label=sptype)

    ax.legend(fontsize='xx-small')

    ax.set_xlim(bwav.min()+5, bwav.max()-5)

    savpath = join('results', 'sptypecomp', f'sptypecomp_{chip}_{str(order).zfill(2)}.png')
    savefig(fig, savpath, writepdf=0)

if __name__ == "__main__":

    for order in range(0,16):
        make_sptype_comparison('r', order)
    for order in range(0,9):
        make_sptype_comparison('i', order)

