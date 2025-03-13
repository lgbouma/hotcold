"""
UCL/LCC : Kerr
IC2602+/IC2391 : Hunt&Reffert24 2024A&A...686A..42H
Pleiades : ditto
TIC1411
"""
import os
from numpy import array as nparr
from scipy.interpolate import interp1d
from astropy.io import fits
from glob import glob
from os.path import join
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.table import Table
from complexrotators.paths import DATADIR, RESULTSDIR
from astropy import units as u
from aesthetic.plot import set_style, savefig
from matplotlib import rcParams

from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data

from rudolf.helpers import get_gaia_catalog_of_nearby_stars

def plot_hrd(smalllims=0):

    ##############################
    # collect cluster data
    fitspath = '../data/literature/Hunt_2024_table2.fits'
    hl = fits.open(fitspath)
    df_h24 = Table(hl[1].data).to_pandas()

    # Pleiades
    df_ple = df_h24[df_h24['Name'] == 'Melotte_22']
    # IC2602
    df_ic2602 = df_h24[df_h24['Name'] == 'IC_2602']

    # Sco-Cen / USco
    fitspath = '../data/literature/Kerr_2021_table1.fits'
    hl = fits.open(fitspath)
    df_k21t1 = Table(hl[1].data).to_pandas()

    df_usco = df_k21t1[(df_k21t1['TLC'] == 22) & (df_k21t1['EOM'] == 17) &
                       (df_k21t1['Prob'] > 0.95) & (df_k21t1['Leaf'] == ' ')]
    df_usco = df_usco.rename({'bp-rp':'BP-RP', 'plx':'Plx'}, axis='columns')

    dr2_source_ids = np.array(df_usco['Gaia'])
    groupname = 'Kerr21t1_USco'
    gdf = given_source_ids_get_gaia_data(dr2_source_ids, groupname,
                                         n_max=10000, overwrite=False,
                                         enforce_all_sourceids_viable=True,
                                         which_columns='*',
                                         table_name='gaia_source',
                                         gaia_datarelease='gaiadr2')
    gdf_usco = gdf.rename(
        {'bp_rp': 'BP-RP', 'parallax': 'Plx', 'phot_g_mean_mag': 'Gmag'},
        axis='columns'
    )

    df_bkgd = get_gaia_catalog_of_nearby_stars()
    df_bkgd = df_bkgd.rename(
        {'phot_g_mean_mag': 'Gmag', 'parallax':'Plx'}, axis='columns'
    )
    #df_bkgd = df_bkgd[df.Plx > 0] # < 50pc
    df_bkgd['BP-RP'] = df_bkgd['phot_bp_mean_mag'] - df_bkgd['phot_rp_mean_mag']

    # TIC 1411
    df_1411 = pd.DataFrame({
        'BP-RP': 3.276,
        'Gmag': 14.701,
        'Plx': 17.324
    }, index=[0])

    #TODO FIXME REDDENING CORRECTION
    #TODO FIXME REDDENING CORRECTION
    #TODO FIXME REDDENING CORRECTION


    ##############################
    # plot
    dfs = [gdf_usco, df_ic2602, df_ple, df_bkgd, df_1411]
    colors = ['limegreen', 'C1', 'cyan', 'gray', 'yellow']
    names = ['USco (11 Myr)', 'IC2602 (40 Myr)', 'Pleiades (112 Myr)',
             'Nearby Stars', 'TIC 141146667']
    zorders = [1,2,3,-1,5]

    set_style("clean")
    rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(2.5,2.5))

    for df, c, l, i in zip(dfs, colors, names, zorders):

        abs_g = df["Gmag"] - 5 * np.log10(1000 / df["Plx"]) + 5
        bprp = df["BP-RP"]

        print(len(abs_g), len(bprp))

        s = 3
        m = 'o'
        lw = 0.2
        r = 0
        _l = l
        if '1411' in l:
            s = 60
            m = '*'
        if 'Nearby' in l:
            s = 0.01
            m = '.'
            lw = 0.03
            r = 1
            _l = None

        ax.scatter(
            bprp, abs_g, color=c,
            edgecolor="k", zorder=i, label=_l,
            linewidth=lw, s=s, marker=m, rasterized=r
        )

        if 'Nearby' in l:
            # trick to get legend to work
            ax.scatter(
                -99, 10, color=c, edgecolor="k", zorder=i,
                label='Nearby Stars', linewidth=0.05, s=2, marker=m,
                rasterized=r
            )

    ax.legend(fontsize='x-small', framealpha=1, handletextpad=-0.5,
              borderpad=0.1, borderaxespad=0., edgecolor='white')

    ax.set_ylim(ax.get_ylim()[::-1])
    ax.update({
        'xlabel': '$G_{\mathrm{BP}}-G_{\mathrm{RP}}$ [mag]',
        'ylabel': 'Absolute $\mathrm{M}_{G}$ [mag]'
    })

    ax.set_xlim([-2,6])
    if smalllims:
        ax.set_xlim([0.85,4.15])
        ax.set_ylim([12.5,4.7])

    plot_dir = "results/hrd"
    os.makedirs(plot_dir, exist_ok=True)

    lims = 'fulllim' if not smalllims else 'smalllim'

    s = f'_{lims}'

    savefig(fig, os.path.join(plot_dir, f"hrd{s}.png"), writepdf=1)


if __name__ == "__main__":
    plot_hrd(smalllims=1)
    plot_hrd(smalllims=0)
