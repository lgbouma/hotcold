"""
Make a time-averaged spectrum around Li 6708A
and compare it against a spectral template from Bochanski+2007 (take M6)
"""
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

def make_li_plot(chip, order):

    fitsdir = join(DATADIR, 'spectra/HIRES/TIC141146667_DEBLAZED')
    fitspaths = np.sort(glob(join(fitsdir, f"{chip}j*fits")))[1:-1]
    N = len(fitspaths)

    set_style('science')
    fig, ax = plt.subplots(figsize=(2,2))
    colors = [plt.cm.viridis_r(i) for i in np.linspace(0, 0.8, N)]

    bflxs = []

    for spectrum_path, c in zip(fitspaths, colors):

        verbose = 0
        flx_2d, norm_flx_2d, wav_2d, mjd, ra, dec = _get_full_hires_spectrum(
            spectrum_path, verbose=verbose
        )

        flx, wav = norm_flx_2d[order, :], wav_2d[order, :]

        bin_width = 0.5 # angstrom
        bwav, bflx, bflxerr = bin_spectrum(wav, flx, bin_width)
        bflxs.append(bflx)

        # NOTE: uncomment to show individual spectra
        #plt.plot(bwav, bflx, c=c, lw=0.5, alpha=0.2)

    bflxs = np.array(bflxs)
    bflxs_tavg = np.nanmedian(bflxs, axis=0)
    bflxs_tstd = np.nanstd(bflxs, axis=0) / np.sqrt(N)
    bflxs_tstd[bflxs_tstd > 0.05]  = np.nanmean(bflxs_tstd)

    dlam = 1.0
    bwav += dlam

    ax.errorbar(bwav, bflxs_tavg -0.1, yerr=bflxs_tstd, c='k', lw=1,
                label='TIC 141146667')

    sptypes = [f"M{ix}" for ix in range(6,7)]
    colors = [plt.cm.magma_r(i) for i in np.linspace(0.3, 0.8, len(sptypes))]
    for ix, (sptype, c) in enumerate(zip(sptypes, colors)):
        swav, sflx = get_bochanski2007_m_standard(sptype=sptype, activestr='all')
        sel = (swav > bwav.min()+5.1) & (swav < bwav.max()-5.1)
        swav = swav[sel]
        sflx = sflx[sel] / np.nanmedian(sflx[sel])

        # NOTE: this measurement is highly contingent on "aligning" the
        # continuum, which to me seems like a suspicious procedure
        scale, offset = 1.31, -0.33

        # then interpolate down to observed wavelength grid
        interpolator = interp1d(swav, sflx, kind="quadratic",
                                fill_value="extrapolate")
        bsflx = interpolator(bwav)

        from scipy.ndimage import gaussian_filter1d
        fn = lambda x: gaussian_filter1d(x, sigma=2)

        ax.plot(bwav, fn(scale * bsflx + offset)  -0.1, c=c, lw=1,
                label=f"Template")

    ax.legend(fontsize='x-small')

    ax.set_xlim(bwav.min()+5, bwav.max()-5)
    ax.set_xlim(6708-6, 6708+6)
    ax.set_ylim((0.91, 1.09))

    ylim = ax.get_ylim()
    ax.vlines([6707.76, 6707.91], ylim[0], ylim[1], colors='lightgray', alpha=0.5,
              lw=1, ls=':', zorder=-1)

    ax.update({
        'xlabel': 'Wavelength [$\AA$]',
        'ylabel': 'Relative flux'
    })

    savpath = join('results', 'li_ew', f'li_ew_{chip}_{str(order).zfill(2)}.png')
    savefig(fig, savpath, writepdf=1)

    ##########################################

    resids = 1 - ( (scale * bsflx + offset) - bflxs_tavg )

    from cdips_followup.spectools import get_Li_6708_EW

    outpath = join('results', 'li_ew', f'measure_li_ew_{chip}_{str(order).zfill(2)}.png')
    ew_df = get_Li_6708_EW('manual', delta_wav=5, outpath=outpath,
                           writecsvresults=True, verbose=True,
                           montecarlo_errors=True, wavflx=(bwav, resids))

if __name__ == "__main__":

    order = 1
    make_li_plot('i', order)
