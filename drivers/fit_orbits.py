import pandas as pd
import numpy as np
import os
from os.path import join
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def load_rvs(component_ix=None):

    # innermost (~2veq) clump 
    mask0 = [1,1,1,1,1,
             1,1,1,0,0, #9 = phi 0.54
             0,1,1,1,1,
             0,0,1,1,1,   #16(start) = phi 0.93       
             0]
    # middle (~2.8veq) clump 
    mask1 = [1,1,1,1,1,
             1,1,1,1,0,
             1,1,1,1,1,
             0,0,1,1,1,
             0]
    # outer clump
    # 8&9 : absorption cancelled out the emission??
    mask2 = [0,0,0,0,1,
             1,1,1,0,0, #10 = phi 0.54.  
             1,1,1,1,1,
             0,0,0,0,0, #16(start) = phi 0.93
             0]

    assert component_ix in [1,2,3]

    if component_ix == 1:
        mask = np.array(mask0).astype(bool)
    elif component_ix == 2:
        mask = np.array(mask1).astype(bool)
    elif component_ix == 3:
        mask = np.array(mask2).astype(bool)

    csvpath = 'results/halpha_to_rv_timerseries/multigauss_parametervaltable.csv'
    tdf = pd.read_csv(csvpath)
    csvpath = 'results/halpha_to_rv_timerseries/multigauss_parameterunctable.csv'
    tdf_unc = pd.read_csv(csvpath)

    time = np.array(tdf.BTJD)
    rv = np.array(tdf[f'Mean{component_ix}'])
    rv_err = np.array(tdf[f'Sigma{component_ix}'])

    time = time[mask]
    rv = rv[mask]
    rv_err = rv_err[mask]

    return time, rv, rv_err

def rv_circular(t, K, P, t0):
    return K * np.sin(2*np.pi/P * (t - t0))

def solve_kepler(M, e, tol=1e-6, max_iter=100):
    E = M.copy() if isinstance(M, np.ndarray) else M
    for _ in range(max_iter):
        delta = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        E -= delta
        if np.all(np.abs(delta) < tol):
            break
    return E

def rv_eccentric(t, K, P, t_peri, e):
    M = 2*np.pi/P * (t - t_peri)
    E = solve_kepler(M, e)
    f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
    return K * (np.cos(f) + e)  # assuming ω=0

def compute_stats(model, t, rv, rv_err, popt):
    fit = model(t, *popt)
    residuals = rv - fit
    chi2 = np.sum((residuals/rv_err)**2)
    dof = len(rv) - len(popt)
    red_chi2 = chi2 / dof if dof>0 else np.nan
    N = len(rv)
    k = len(popt)
    bic = chi2 + k * np.log(N)
    return red_chi2, bic

def save_fit_table(filename, params, perr, red_chi2, bic, param_names):
    df = pd.DataFrame({
        'Parameter': param_names,
        'Value': params,
        'Uncertainty': perr
    })
    df['Reduced Chi2'] = red_chi2
    df['BIC'] = bic
    df.to_csv(filename, index=False)

def main():
    circ_summary = []
    ecc_summary = []
    for ix in [1, 2, 3]:
        time, rv, rv_err = load_rvs(component_ix=ix)
        
        # Estimate initial parameters
        K0 = (np.max(rv) - np.min(rv)) / 2
        P0 = 4/24
        t0_0 = time.min()
        # Circular orbit fit
        try:
            popt_circ, pcov_circ = curve_fit(rv_circular, time, rv, sigma=rv_err, p0=[K0, P0, t0_0])
        except Exception as e:
            print(f"Component {ix} circular fit failed: {e}")
            continue
        perr_circ = np.sqrt(np.diag(pcov_circ))
        red_chi2_circ, bic_circ = compute_stats(rv_circular, time, rv, rv_err, popt_circ)
        
        # Save circular fit table
        circ_csv = join('results/halpha_to_rv_timerseries', f'circular_fit_component_{ix}.csv')
        param_names_circ = ['K', 'P', 't0']
        save_fit_table(circ_csv, popt_circ, perr_circ, red_chi2_circ, bic_circ, param_names_circ)
        
        # Plot circular fit
        t_fit = np.linspace(time.min(), time.max(), 1000)
        plt.errorbar(time, rv, yerr=rv_err, fmt='o', label='Data')
        plt.plot(t_fit, rv_circular(t_fit, *popt_circ), 'r-', label='Circular Fit')
        plt.xlabel('Time')
        plt.ylabel('RV')
        plt.title(f'Circular Orbit Fit (Component {ix})')
        plt.legend()
        plot_path = join('results/halpha_to_rv_timerseries', f'circular_fit_component_{ix}.png')
        plt.savefig(plot_path)
        plt.clf()
        
        circ_summary.append({
            'component_ix': ix,
            'K': popt_circ[0],
            'P': popt_circ[1],
            't0': popt_circ[2],
            'Reduced_Chi2': red_chi2_circ,
            'BIC': bic_circ
        })
        
        # Eccentric orbit fit: add initial guess for eccentricity
        e0 = 0.1
        try:
            popt_ecc, pcov_ecc = curve_fit(rv_eccentric, time, rv, sigma=rv_err, p0=[K0, P0, t0_0, e0])
        except Exception as e:
            print(f"Component {ix} eccentric fit failed: {e}")
            continue
        perr_ecc = np.sqrt(np.diag(pcov_ecc))
        red_chi2_ecc, bic_ecc = compute_stats(rv_eccentric, time, rv, rv_err, popt_ecc)
        
        # Save eccentric fit table
        ecc_csv = join('results/halpha_to_rv_timerseries', f'eccentric_fit_component_{ix}.csv')
        param_names_ecc = ['K', 'P', 't_peri', 'e']
        save_fit_table(ecc_csv, popt_ecc, perr_ecc, red_chi2_ecc, bic_ecc, param_names_ecc)
        
        # Plot eccentric fit
        plt.errorbar(time, rv, yerr=rv_err, fmt='o', label='Data')
        plt.plot(t_fit, rv_eccentric(t_fit, *popt_ecc), 'r-', label='Eccentric Fit')
        plt.xlabel('Time')
        plt.ylabel('RV')
        plt.title(f'Eccentric Orbit Fit (Component {ix})')
        plt.legend()
        plot_path = join('results/halpha_to_rv_timerseries', f'eccentric_fit_component_{ix}.png')
        plt.savefig(plot_path)
        plt.clf()
        
        ecc_summary.append({
            'component_ix': ix,
            'K': popt_ecc[0],
            'P': popt_ecc[1],
            't_peri': popt_ecc[2],
            'e': popt_ecc[3],
            'Reduced_Chi2': red_chi2_ecc,
            'BIC': bic_ecc
        })
    
    # Save summary tables
    df_circ = pd.DataFrame(circ_summary)
    df_circ.to_csv(join('results/halpha_to_rv_timerseries', 'circular_summary.csv'), index=False)
    df_ecc = pd.DataFrame(ecc_summary)
    df_ecc.to_csv(join('results/halpha_to_rv_timerseries', 'eccentric_summary.csv'), index=False)


if __name__ == '__main__':
    main()