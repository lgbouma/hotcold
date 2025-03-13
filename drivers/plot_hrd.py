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
from rudolf.extinction import append_corrected_gaia_phot_Gaia2018

def get_kerr21_usco():

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
    return gdf_usco

def get_ratzenbock23_lcc_ucl():

    fitspath = '../data/literature/Ratzenboeck_2023_table1.fits'
    hl = fits.open(fitspath)
    df = Table(hl[1].data).to_pandas()

    # select nu Cen (UCL greatest density  and sigma Cen (LCC greatest density)
    sel = (
        ( (df['SigMA'] == 20) | (df['SigMA'] == 22) )
        &
        (df['stability'] > 95)
    )
    df = df[sel]

    dr3_source_ids = np.array(df['GaiaDR3'])
    groupname = 'Ratzenboeck_2023_table1_lcc_ucl'
    gdf = given_source_ids_get_gaia_data(dr3_source_ids, groupname,
                                         n_max=20000, overwrite=False,
                                         enforce_all_sourceids_viable=True,
                                         which_columns='*',
                                         table_name='gaia_source_lite',
                                         gaia_datarelease='gaiadr3')

    return gdf

def get_tic1411():

    dr3_source_ids = np.array([np.int64(860453786736413568)])
    groupname = 'tic1411'
    gdf = given_source_ids_get_gaia_data(dr3_source_ids, groupname,
                                         n_max=2, overwrite=False,
                                         enforce_all_sourceids_viable=True,
                                         which_columns='*',
                                         table_name='gaia_source_lite',
                                         gaia_datarelease='gaiadr3')

    return gdf

def AV_to_EBmV(A_V):
    R_V = 3.1
    EBmV = A_V / R_V
    return EBmV


def get_pleiades():

    fitspath = '../data/literature/Hunt_2024_table2.fits'
    hl = fits.open(fitspath)
    df_h24 = Table(hl[1].data).to_pandas()

    # Pleiades
    df_ple = df_h24[(df_h24['Name'] == 'Melotte_22') & (df_h24.Prob > 0.5)]

    dr3_source_ids = np.array(df_ple['GaiaDR3'])
    groupname = 'Hunt2024_t2_pleiades'
    gdf = given_source_ids_get_gaia_data(dr3_source_ids, groupname,
                                         n_max=20000, overwrite=False,
                                         enforce_all_sourceids_viable=True,
                                         which_columns='*',
                                         table_name='gaia_source_lite',
                                         gaia_datarelease='gaiadr3')

    return gdf


def get_ic2602():

    fitspath = '../data/literature/Hunt_2024_table2.fits'
    hl = fits.open(fitspath)
    df_h24 = Table(hl[1].data).to_pandas()

    # Pleiades
    df_ple = df_h24[(df_h24['Name'] == 'IC_2602') & (df_h24.Prob > 0.5)]

    dr3_source_ids = np.array(df_ple['GaiaDR3'])
    groupname = 'Hunt2024_t2_ic2602'
    gdf = given_source_ids_get_gaia_data(dr3_source_ids, groupname,
                                         n_max=20000, overwrite=False,
                                         enforce_all_sourceids_viable=True,
                                         which_columns='*',
                                         table_name='gaia_source_lite',
                                         gaia_datarelease='gaiadr3')

    return gdf


def plot_hrd(deredden=0, smalllims=0):

    EXTINCTIONDICT = {
        'Pleiades': 0.10209450, # from Hunt+24, table1, A_V
        'IC_2602': 0.11434174, # from Hunt+24, table1, A_V
        'UCL/LCC': 0.12 # Pecaut&Mamajek 2016 table7 median
    }

    ##############################
    # collect cluster data

    df_ple = get_pleiades()
    df_ple['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['Pleiades'])
    df_ple = append_corrected_gaia_phot_Gaia2018(df_ple)

    # IC2602
    df_ic2602 = get_ic2602()
    df_ic2602['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['IC_2602'])
    df_ic2602 = append_corrected_gaia_phot_Gaia2018(df_ic2602)

    # Sco-Cen / USco (deprecated)
    df_usco = get_kerr21_usco()

    # UCL/LCC
    df_ucllcc = get_ratzenbock23_lcc_ucl()
    df_ucllcc['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['UCL/LCC'])
    df_ucllcc = append_corrected_gaia_phot_Gaia2018(df_ucllcc)

    # GCNS
    df_bkgd = get_gaia_catalog_of_nearby_stars()
    df_bkgd['E(B-V)'] = 0
    df_bkgd = append_corrected_gaia_phot_Gaia2018(df_bkgd)

    # TIC1411
    df_1411 = get_tic1411()
    df_1411['E(B-V)'] = 0
    df_1411 = append_corrected_gaia_phot_Gaia2018(df_1411)

    ##############################
    # plot
    dfs = [df_ucllcc, df_ic2602, df_ple, df_bkgd, df_1411]
    colors = ['limegreen', 'C1', 'cyan', 'gray', 'yellow']
    names = ['UCL/LCC (15 Myr)', 'IC2602 (40 Myr)', 'Pleiades (112 Myr)',
             'Nearby Stars', 'TIC 141146667']
    zorders = [1,2,3,-1,5]

    set_style("clean")
    rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(2.5,2.5))

    for df, c, l, i in zip(dfs, colors, names, zorders):


        if not deredden:
            bprp = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']
            abs_g = df["phot_g_mean_mag"] - 5 * np.log10(1000 / df["parallax"]) + 5
        else:
            bprp = df['phot_bp_mean_mag_corr'] - df['phot_rp_mean_mag_corr']
            abs_g = df["phot_g_mean_mag_corr"] - 5 * np.log10(1000 / df["parallax"]) + 5

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
    if not deredden:
        ax.update({
            'xlabel': '$G_{\mathrm{BP}}-G_{\mathrm{RP}}$ [mag]',
            'ylabel': 'Absolute $\mathrm{M}_{G}$ [mag]'
        })
    else:
        ax.update({
            'xlabel': '$(G_{\mathrm{BP}}-G_{\mathrm{RP}})_0$ [mag]',
            'ylabel': 'Absolute $\mathrm{M}_{G,0}$ [mag]'
        })

    ax.set_xlim([-2,6])
    if smalllims:
        ax.set_xlim([0.85,4.15])
        ax.set_ylim([12.5,4.7])

    plot_dir = "results/hrd"
    os.makedirs(plot_dir, exist_ok=True)

    lims = 'fulllim' if not smalllims else 'smalllim'
    drs = 'rawphot' if not deredden else 'dereddened'

    s = f'_{lims}_{drs}'

    savefig(fig, os.path.join(plot_dir, f"hrd{s}.png"), writepdf=1)


if __name__ == "__main__":
    for smalllim in [1,0]:
        for dr in [1,0]:
            plot_hrd(smalllims=smalllim, deredden=dr)
