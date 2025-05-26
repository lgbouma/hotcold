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
from typing import Dict, Tuple

from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data

from rudolf.helpers import get_gaia_catalog_of_nearby_stars
from rudolf.extinction import append_corrected_gaia_phot_Gaia2018

EXTINCTIONDICT = {
    'Pleiades': 0.10209450, # from Hunt+24, table1, A_V
    'IC_2602': 0.11434174, # from Hunt+24, table1, A_V
    'UCL/LCC': 0.12, # Pecaut&Mamajek 2016 table7 median
    'NGC_2516': 0.2046, # from Hunt+24, table1, A_V
    'NGC_6475': 0.20822802, # from Hunt+24, table1, A_V
    'NGC_2632': 0.15869616, # Hunt+24 table1 Praesepe
}

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


def get_ratzenbock23_usco():

    fitspath = '../data/literature/Ratzenboeck_2023_table1.fits'
    hl = fits.open(fitspath)
    df = Table(hl[1].data).to_pandas()

    # select delta sco and sigma sco
    sel = (
        ( (df['SigMA'] == 3) | (df['SigMA'] == 5) )
        &
        (df['stability'] > 95)
    )
    df = df[sel]

    dr3_source_ids = np.array(df['GaiaDR3'])
    groupname = 'Ratzenboeck_2023_table1_usco'
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


def get_ngc2516():

    fitspath = '../data/literature/Hunt_2024_table2.fits'
    hl = fits.open(fitspath)
    df_h24 = Table(hl[1].data).to_pandas()

    # Pleiades
    df_ple = df_h24[(df_h24['Name'] == 'NGC_2516') & (df_h24.Prob > 0.5)]

    dr3_source_ids = np.array(df_ple['GaiaDR3'])
    groupname = 'Hunt2024_t2_ngc2516'
    gdf = given_source_ids_get_gaia_data(dr3_source_ids, groupname,
                                         n_max=20000, overwrite=False,
                                         enforce_all_sourceids_viable=True,
                                         which_columns='*',
                                         table_name='gaia_source_lite',
                                         gaia_datarelease='gaiadr3')

    return gdf


def get_ngc6475():

    fitspath = '../data/literature/Hunt_2024_table2.fits'
    hl = fits.open(fitspath)
    df_h24 = Table(hl[1].data).to_pandas()

    # Pleiades
    df_ple = df_h24[(df_h24['Name'] == 'NGC_6475') & (df_h24.Prob > 0.5)]

    dr3_source_ids = np.array(df_ple['GaiaDR3'])
    groupname = 'Hunt2024_t2_ngc6475'
    gdf = given_source_ids_get_gaia_data(dr3_source_ids, groupname,
                                         n_max=20000, overwrite=False,
                                         enforce_all_sourceids_viable=True,
                                         which_columns='*',
                                         table_name='gaia_source_lite',
                                         gaia_datarelease='gaiadr3')

    return gdf

def get_ngc2632():

    fitspath = '../data/literature/Hunt_2024_table2.fits'
    hl = fits.open(fitspath)
    df_h24 = Table(hl[1].data).to_pandas()

    # Pleiades
    df_ple = df_h24[(df_h24['Name'] == 'NGC_2632') & (df_h24.Prob > 0.5)]

    dr3_source_ids = np.array(df_ple['GaiaDR3'])
    groupname = 'Hunt2024_t2_ngc2632'
    gdf = given_source_ids_get_gaia_data(dr3_source_ids, groupname,
                                         n_max=20000, overwrite=False,
                                         enforce_all_sourceids_viable=True,
                                         which_columns='*',
                                         table_name='gaia_source_lite',
                                         gaia_datarelease='gaiadr3')

    return gdf



def get_cpv_hrd_data():

    csvpath = ('../data/literature/20240304_CPV_lit_compilation_'
               'R16_S17_S18_B20_S21_Z19_G22_P23_B24_TIC8_obs_truncated.csv')
    df = pd.read_csv(csvpath, dtype={'tic8_GAIA':str})

    sel = (
        (
        (df.cluster == 'USco')
        #|
        #(df.cluster == 'USco/rhoOph')
        |
        (df.cluster == 'TucHor')
        |
        (df.cluster == 'IC2602')
        |
        (df.cluster == 'PLE')
        |
        (df.cluster == 'ABDMG')
        |
        (df.cluster == 'ABDoradus')
        )
        &
        (~pd.isnull(df['tic8_GAIA']))
    )
    df = df[sel]
    df = df.reset_index(drop=True)

    dr2_source_ids = np.array(df['tic8_GAIA']).astype(np.int64)
    groupname = '20250313_cpv_compilation'
    gdf = given_source_ids_get_gaia_data(dr2_source_ids, groupname,
                                         n_max=100, overwrite=0,
                                         enforce_all_sourceids_viable=True,
                                         which_columns='*',
                                         table_name='gaia_source',
                                         gaia_datarelease='gaiadr2')


    sel = df.cluster.str.contains("USco")
    gdf['E(B-V)'] = np.zeros(len(gdf))
    gdf['Cluster'] = ''
    gdf.loc[sel, 'E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['UCL/LCC'])
    gdf.loc[sel, 'Cluster'] = 'USco'

    sel = df.cluster.str.contains("IC2602")
    gdf.loc[sel, 'E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['IC_2602'])
    sel = (
        df.cluster.str.contains("IC2602")
        |
        df.cluster.str.contains("TucHor")
    )
    gdf.loc[sel, 'Cluster'] = 'IC2602'

    sel = (
        df.cluster.str.contains("PLE")
        |
        df.cluster.str.contains("ABD")
    )
    gdf.loc[sel, 'E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['Pleiades'])
    gdf.loc[sel, 'Cluster'] = 'Pleiades'

    gdf = append_corrected_gaia_phot_Gaia2018(gdf)

    return gdf



def label_panels_figure(
    fig: plt.Figure,
    label_dict: Dict[str, Tuple[float, float]],
    fontsize: str = 'medium'
) -> None:
    """Place boldface panel labels in figure (not axes) coordinates.

    Generated by ChatGPT o3-mini-high on March 20 2025.

    Args:
        fig (matplotlib.figure.Figure):
            The figure object to place text in.
        label_dict (Dict[str, Tuple[float, float]]):
            A dictionary whose keys are single-letter panel labels (e.g. 'a'),
            and whose values are (x, y) in figure coordinates (0–1 range).

    Returns:
        None
    """
    for letter, (xx, yy) in label_dict.items():
        fig.text(
            xx,
            yy,
            f"{letter}",
            transform=fig.transFigure,
            fontsize=fontsize,
            fontweight='bold',
            ha="left",
            va="top"
        )


def plot_hrd(deredden=0, smalllims=0, cpvcomparison=0, addngc2516=0, showgrp=0,
             addngc6475=0, addngc2632=0):

    ##############################
    # collect data for clusters and CPVs

    df_cpv = get_cpv_hrd_data()
    df_cpv_usco = df_cpv[df_cpv['Cluster'] == 'USco']
    df_cpv_ple = df_cpv[df_cpv['Cluster'] == 'Pleiades']
    df_cpv_ic2602 = df_cpv[df_cpv['Cluster'] == 'IC2602']

    # Pleiades
    df_ple = get_pleiades()
    df_ple['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['Pleiades'])
    df_ple = append_corrected_gaia_phot_Gaia2018(df_ple)

    # IC2602
    df_ic2602 = get_ic2602()
    df_ic2602['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['IC_2602'])
    df_ic2602 = append_corrected_gaia_phot_Gaia2018(df_ic2602)

    # Sco-Cen / USco
    df_usco = get_ratzenbock23_usco()
    df_usco['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['UCL/LCC'])
    df_usco = append_corrected_gaia_phot_Gaia2018(df_usco)

    # UCL/LCC
    df_ucllcc = get_ratzenbock23_lcc_ucl()
    df_ucllcc['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['UCL/LCC'])
    df_ucllcc = append_corrected_gaia_phot_Gaia2018(df_ucllcc)

    # NGC2516
    df_ngc2516 = get_ngc2516()
    df_ngc2516['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['NGC_2516'])
    df_ngc2516 = append_corrected_gaia_phot_Gaia2018(df_ngc2516)

    # NGC6475
    df_ngc6475 = get_ngc6475()
    df_ngc6475['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['NGC_6475'])
    df_ngc6475 = append_corrected_gaia_phot_Gaia2018(df_ngc6475)

    # NGC2632 = Praesepe
    df_ngc2632 = get_ngc2632()
    df_ngc2632['E(B-V)'] = AV_to_EBmV(EXTINCTIONDICT['NGC_2632'])
    df_ngc2632 = append_corrected_gaia_phot_Gaia2018(df_ngc2632)

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
    dfs = [df_usco, df_ic2602, df_ple, df_bkgd, df_1411]
    colors = ['limegreen', 'C1', 'cyan', 'gray', 'yellow']
    names = ['8 Myr', '40 Myr', '112 Myr', 'Nearby Stars',
             'TIC 141146667']
    #names = ['USco (8 Myr)', 'IC2602 (40 Myr)', 'Pleiades (112 Myr)',
    #         'Nearby Stars', 'TIC 141146667']
    zorders = [1,2,3,-1,99]

    if int(addngc6475) + int(addngc2516) + int(addngc2632) > 1:
        raise NotImplementedError('max one of ngc6475 or ngc2516 or NGC2632 allowed')

    if addngc2516:
        dfs.append(df_ngc2516)
        colors.append('blueviolet')
        names.append("150 Myr")
        zorders.append(4)

    if addngc6475:
        dfs.append(df_ngc6475)
        colors.append('rebeccapurple')
        names.append("220 Myr")
        zorders.append(4)

    if addngc2632:
        dfs.append(df_ngc2632)
        colors.append('purple')
        names.append("700 Myr")
        zorders.append(4)

    if cpvcomparison:
        dfs = [df_usco, df_ic2602, df_ple, df_bkgd, df_1411,
               df_cpv_usco, df_cpv_ic2602, df_cpv_ple]
        colors = ['limegreen', 'C1', 'cyan', 'gray', 'yellow', 'limegreen', 'C1', 'cyan']
        names = ['USco (8 Myr)', 'IC2602 (40 Myr)', 'Pleiades (112 Myr)',
                 'Nearby Stars', 'TIC 141146667', '', '', '']
        zorders = [1,2,3,-1,99,6,7,8]

    #'UCL/LCC (15 Myr)',

    set_style("science")
    rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(3,3))

    for df, c, l, i in zip(dfs, colors, names, zorders):

        if not deredden:
            bprp = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']
            abs_g = df["phot_g_mean_mag"] - 5 * np.log10(1000 / df["parallax"]) + 5
        elif deredden:
            bprp = df['phot_bp_mean_mag_corr'] - df['phot_rp_mean_mag_corr']
            abs_g = df["phot_g_mean_mag_corr"] - 5 * np.log10(1000 / df["parallax"]) + 5
        if showgrp and not deredden:
            grp = df['phot_g_mean_mag'] - df['phot_rp_mean_mag']
        elif showgrp and deredden:
            grp = df['phot_g_mean_mag_corr'] - df['phot_rp_mean_mag_corr']

        color = bprp if not showgrp else grp

        print(len(abs_g), len(color))

        s = 2.5
        m = 'o'
        lw = 0.15
        r = 0
        _l = l
        if '1411' in l:
            s = 110
            m = '*'
            lw = 0.3
        if 'Nearby' in l:
            s = 0.01
            m = '.'
            lw = 0.03
            r = 1
            _l = None
        if l == '':
            s = 50
            m = '*'
            _l = None

        ax.scatter(
            color, abs_g, color=c,
            edgecolor="k", zorder=i, label=_l,
            linewidth=lw, s=s, marker=m, rasterized=r
        )

        if 'Nearby' in l:
            # trick to get legend to work
            ax.scatter(
                -99, 10, color=c, edgecolor="k", zorder=i,
                label='Stars near Sun', linewidth=0.05, s=2, marker=m,
                rasterized=r
            )

    ax.legend(fontsize='x-small', handletextpad=0.5,
              handlelength=0.5,
              #framealpha=1, borderpad=0.1, borderaxespad=0.,
              #edgecolor='white',
              loc='upper right')

    ax.set_ylim(ax.get_ylim()[::-1])

    if not deredden and not showgrp:
        xlabel = '$G_{\mathrm{BP}}-G_{\mathrm{RP}}$ [mag]'
    elif deredden and not showgrp:
        xlabel = '$(G_{\mathrm{BP}}-G_{\mathrm{RP}})_0$ [mag]'
    if not deredden and showgrp:
        xlabel = '$G-G_{\mathrm{RP}}$ [mag]'
    elif deredden and showgrp:
        xlabel = '$(G-G_{\mathrm{RP}})_0$ [mag]'

    if not deredden:
        ax.update({
            'xlabel': xlabel,
            'ylabel': 'Absolute $\mathrm{M}_{G}$ [mag]'
        })
    else:
        ax.update({
            'xlabel': xlabel,
            'ylabel': 'Absolute $\mathrm{M}_{G,0}$ [mag]'
        })

    ax.set_xlim([-2,6])
    ax.set_ylim([18,-3])
    if smalllims:
        ax.set_xlim([0.85,4.15])
        ax.set_ylim([12.5,4.7])
        if showgrp:
            ax.set_xlim([0.85,1.55])

    labels = {
        "a": (-0.07, 0.88),
    }
    label_panels_figure(fig, labels, fontsize='large')



    plot_dir = "results/hrd"
    os.makedirs(plot_dir, exist_ok=True)

    sgrp = 'bprp' if not showgrp else 'grp'
    lims = 'fulllim' if not smalllims else 'smalllim'
    drs = 'rawphot' if not deredden else 'dereddened'
    cpvc = '' if not cpvcomparison else '_cpvcomp'
    ngc2516 = '' if not addngc2516 else '_ngc2516'
    ngc6475= '' if not addngc6475 else '_ngc6475'
    ngc2632= '' if not addngc2632 else '_ngc2632'

    s = f'_{sgrp}_{lims}_{drs}{cpvc}{ngc2516}{ngc6475}{ngc2632}'

    savefig(fig, os.path.join(plot_dir, f"hrd{s}.png"), writepdf=1)


if __name__ == "__main__":

    for showgrp in [0,1]:
        plot_hrd(cpvcomparison=0, smalllims=1, deredden=1, addngc2516=0,
                 showgrp=showgrp, addngc6475=0, addngc2632=1)
    assert 0

    # cluster comparison
    for smalllim in [1,0]:
        for dr in [1,0]:
            plot_hrd(smalllims=smalllim, deredden=dr)
