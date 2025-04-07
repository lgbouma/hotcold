import pandas as pd
import numpy as np
import os, pickle
from os.path import join
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams
from aesthetic.plot import set_style, savefig
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
numpyro.set_host_device_count(2)

def load_rvs(component_ix=None, applymask=True):

    # innermost (~2veq) clump 
    mask0 = [1,1,1,1,1,
             1,1,0,0,0, #9 = phi 0.54
             0,1,1,1,1,
             0,0,1,1,0,   #16(start) = phi 0.93       
             0]
    # middle (~2.8veq) clump 
    mask1 = [1,1,1,1,1,
             1,1,1,1,0,
             1,1,1,1,1,
             0,0,1,1,0,
             0]
    # outer clump
    # 8&9 : absorption cancelled out the emission??
    
    # based on "is there a good signal"?
    mask2 = [0,0,0,0,1,
             1,1,1,0,1, #9 = phi 0.54.  
             1,1,1,1,1,
             0,0,0,0,0, #16(start) = phi 0.93
             0]
    # based on sinusoid
    mask2 = [0,0,0,0,0,
             0,0,1,1,1, #9 = phi 0.54.  
             1,1,1,1,1,
             1,0,0,0,0, #16(start) = phi 0.93
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
    #rv_err = np.array(tdf[f'Sigma{component_ix}'])
    rv_err = np.array(tdf_unc[f'Mean{component_ix}'])

    if applymask:
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

# JAX-compatible functions for use in numpyro models
def rv_circular_jax(t, K, P, t0):
    return K * jnp.sin(2 * jnp.pi / P * (t - t0))

def solve_kepler_jax(M, e, tol=1e-6, max_iter=100):
    E = M
    for _ in range(max_iter):
        delta = (E - e * jnp.sin(E) - M) / (1 - e * jnp.cos(E))
        E = E - delta
        if jnp.all(jnp.abs(delta) < tol):
            break
    return E

def rv_eccentric_jax(t, K, P, t_peri, e):
    M = 2 * jnp.pi / P * (t - t_peri)
    E = solve_kepler_jax(M, e)
    f = 2 * jnp.arctan(jnp.sqrt((1+e)/(1-e)) * jnp.tan(E/2))
    return K * (jnp.cos(f) + e)

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

def model_circular(time, rv_err, rv_obs):
    # Priors: K Uniform(0,6), P Uniform(2/24,6/24), t0 Uniform(min(time),max(time))
    K = numpyro.sample('K', dist.Uniform(0.0, 6.0))
    P = numpyro.sample('P', dist.Uniform(2/24, 6/24))
    t0 = numpyro.sample('t0', dist.Uniform(time.min(), time.max()))
    jitter = numpyro.sample('jitter', dist.Exponential(1.0))
    mu = rv_circular_jax(time, K, P, t0)
    effective_err = jnp.sqrt(rv_err**2 + jitter**2)  # Add jitter in quadrature
    with numpyro.plate('data', len(time)):
        numpyro.sample('obs', dist.Normal(mu, effective_err), obs=rv_obs)

def mcmc_fit_circular(time, rv, rv_err, rng_key=None, num_warmup=1000, num_samples=2000, num_chains=2):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    kernel = NUTS(model_circular)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key, time=time, rv_err=rv_err, rv_obs=rv)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    # compute mean jitter parameter from the samples
    jitter_circ = np.asarray(samples['jitter']).mean()
    popt = np.asarray([
        samples['K'].mean(),
        samples['P'].mean(),
        samples['t0'].mean()
    ], dtype=float)
    params = np.vstack([samples['K'], samples['P'], samples['t0']])
    pcov = np.cov(params)
    return popt, pcov, jitter_circ, samples

def mcmc_fit_eccentric(time, rv, rv_err, rng_key=None, num_warmup=1000, num_samples=2000, num_chains=2):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(1)
    kernel = NUTS(model_eccentric)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key, time=time, rv_err=rv_err, rv_obs=rv)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    # compute mean jitter parameter from the samples
    jitter_ecc = np.asarray(samples['jitter']).mean()
    popt = np.asarray([
        samples['K'].mean(),
        samples['P'].mean(),
        samples['t_peri'].mean(),
        samples['e'].mean()
    ], dtype=float)
    params = np.vstack([samples['K'], samples['P'], samples['t_peri'], samples['e']])
    pcov = np.cov(params)
    return popt, pcov, jitter_ecc

def model_eccentric(time, rv_err, rv_obs):
    # Priors: K Uniform(0,6), P Uniform(2/24,6/24), t_peri Uniform(min(time),max(time)), e Uniform(0,1)
    K = numpyro.sample('K', dist.Uniform(0.0, 6.0))
    P = numpyro.sample('P', dist.Uniform(2/24, 6/24))
    t_peri = numpyro.sample('t_peri', dist.Uniform(time.min(), time.max()))
    e = numpyro.sample('e', dist.Uniform(0.0, 1.0))
    jitter = numpyro.sample('jitter', dist.Exponential(1.0))  # New jitter parameter
    mu = rv_eccentric_jax(time, K, P, t_peri, e)
    effective_err = jnp.sqrt(rv_err**2 + jitter**2)  # Add jitter in quadrature
    with numpyro.plate('data', len(time)):
        numpyro.sample('obs', dist.Normal(mu, effective_err), obs=rv_obs)

def main(fittingstyle='leastsquares'):
    circ_summary = []
    ecc_summary = []
    all_fits = []
    for ix in [1, 2, 3]:
        time, rv, rv_err = load_rvs(component_ix=ix)
        _time, _rv, _rv_err = load_rvs(component_ix=ix, applymask=False)

        # store original error arrays for scaling separately in each fit
        original_rv_err = rv_err.copy()
        original__rv_err = _rv_err.copy()
        
        # Estimate initial parameters
        K0 = (np.max(rv) - np.min(rv)) / 2
        P0 = 4/24
        t0_0 = time.min()
        
        if fittingstyle == 'leastsquares':
            # Circular orbit fit using curve_fit
            try:
                popt_circ, pcov_circ = curve_fit(rv_circular, time, rv, sigma=rv_err, p0=[K0, P0, t0_0])
            except Exception as e:
                print(f"Component {ix} circular fit failed: {e}")
                continue
        elif fittingstyle == 'mcmc':
            try:
                popt_circ, pcov_circ, jitter_circ, samples = mcmc_fit_circular(time, rv, rv_err)
            except Exception as e:
                print(f"Component {ix} MCMC circular fit failed: {e}")
                continue
        else:
            raise ValueError("Invalid fittingstyle. Choose 'leastsquares' or 'mcmc'.")
        
        # Compute stats and scale errors for circular fit (leastsquares only)
        perr_circ = np.sqrt(np.diag(pcov_circ))
        red_chi2_circ, bic_circ = compute_stats(rv_circular, time, rv, original_rv_err, popt_circ)
        if fittingstyle == 'leastsquares':
            scale_circ = np.sqrt(red_chi2_circ) if red_chi2_circ > 0 else 1.0
            rv_err = original_rv_err * scale_circ
            perr_circ = perr_circ * scale_circ
            red_chi2_circ = 1.0
        elif fittingstyle == 'mcmc':
            scale_circ = jitter_circ if jitter_circ > 0 else 1.0
            rv_err = original_rv_err * scale_circ
        # Save circular fit table
        circ_csv = join('results/halpha_to_rv_timerseries', f'circular_fit_component_{ix}.csv')
        param_names_circ = ['K', 'P', 't0']
        save_fit_table(circ_csv, popt_circ, perr_circ, red_chi2_circ, bic_circ, param_names_circ)
        
        # Append data for combined plot (use scaled _rv_err for consistency)
        if fittingstyle == 'leastsquares':
            rv_err = rv_err * scale_circ
            _rv_err = original__rv_err * scale_circ
        elif fittingstyle == 'mcmc':
            rv_err = np.sqrt(rv_err**2  + jitter_circ**2)
            _rv_err = np.sqrt(_rv_err**2  + jitter_circ**2)

        all_fits.append({
            'ix': ix,
            'time': time,
            'rv': rv,
            'rv_err': rv_err,
            '_time': _time,
            '_rv': _rv,
            '_rv_err': _rv_err,
            'popt_circ': popt_circ,
            # For MCMC, store the posterior samples for plotting draws.
            **({'samples': samples} if fittingstyle=='mcmc' else {})
        })
        
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
            'K_err': perr_circ[0],
            'P': popt_circ[1]*24,
            'P_err': perr_circ[1]*24,
            't0': popt_circ[2],
            'Reduced_Chi2': red_chi2_circ,
            'BIC': bic_circ
        })
        
        # Eccentric orbit fit: add initial guess for eccentricity
        e0 = 0.1
        if fittingstyle == 'leastsquares':
            try:
                popt_ecc, pcov_ecc = curve_fit(rv_eccentric, time, rv, sigma=original_rv_err, p0=[K0, P0, t0_0, e0])
            except Exception as e:
                print(f"Component {ix} eccentric fit failed: {e}")
                continue
        elif fittingstyle == 'mcmc':
            try:
                popt_ecc, pcov_ecc, jitter_ecc = mcmc_fit_eccentric(time, rv, rv_err)
            except Exception as e:
                print(f"Component {ix} MCMC eccentric fit failed: {e}")
                continue
        perr_ecc = np.sqrt(np.diag(pcov_ecc))
        red_chi2_ecc, bic_ecc = compute_stats(rv_eccentric, time, rv, original_rv_err, popt_ecc)
        if fittingstyle == 'leastsquares':
            scale_ecc = np.sqrt(red_chi2_ecc) if red_chi2_ecc > 0 else 1.0
            # Use original_rv_err so that both fits use independent scaling factors
            rv_err = original_rv_err * scale_ecc
            perr_ecc = perr_ecc * scale_ecc
            red_chi2_ecc = 1.0
        elif fittingstyle == 'mcmc':
            scale_ecc = jitter_ecc if jitter_ecc > 0 else 1.0
            rv_err = original_rv_err * scale_ecc
        ecc_csv = join('results/halpha_to_rv_timerseries', f'eccentric_fit_component_{ix}.csv')
        param_names_ecc = ['K', 'P', 't_peri', 'e']
        save_fit_table(ecc_csv, popt_ecc, perr_ecc, red_chi2_ecc, bic_ecc, param_names_ecc)
        
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
            'K_err': perr_ecc[0],
            'P': popt_ecc[1]*24,
            'P_err': perr_ecc[1]*24,
            't_peri': popt_ecc[2],
            'e': popt_ecc[3],
            'Reduced_Chi2': red_chi2_ecc,
            'BIC': bic_ecc
        })
    
    df_circ = pd.DataFrame(circ_summary)
    df_circ.to_csv(join('results/halpha_to_rv_timerseries', 'circular_summary.csv'), index=False)
    df_ecc = pd.DataFrame(ecc_summary)
    df_ecc.to_csv(join('results/halpha_to_rv_timerseries', 'eccentric_summary.csv'), index=False)

    # load in t0, period
    pkldir = '/Users/luke/Dropbox/proj/cpv/results/movie_sixpanel_specriver/TIC_141146667_science_wob'
    pklpath = join(pkldir, '141146667_specriver_cache.pkl')
    with open(pklpath, 'rb') as f:
        d = pickle.load(f)
    t0 = d['t0'] # time of transit center
    period = d['period'] # period of the transit

    # FINAL VISUALIZATION PLOT
    for phaseunits in [0, 1]:

        set_style('science')
        rcParams['font.family'] = 'Arial'
        f = 0.85
        fig, ax = plt.subplots(figsize=(f*3.5, f*3))
        colors = ['#000000', '#E69F00', '#009E73']
        
        if phaseunits:
            # Initialize the global grid and accumulator for combining gaussian images.
            fn_global = lambda x: ((x - t0) / period - 110)
            all_times = np.concatenate([fit['_time'] for fit in all_fits])
            global_min = all_times.min() - 1/24
            global_max = all_times.max() + 1/24
            global_t_fit = np.linspace(global_min, global_max, 5000)
            global_x_vals = fn_global(global_t_fit)
            y_min = -5
            y_max = 5
            y_grid = np.linspace(y_min, y_max, 300)
            X_global, Y_global = np.meshgrid(global_x_vals, y_grid)
            gauss_total = np.zeros_like(X_global)

        for ix, fit in enumerate(all_fits):
            t = fit['time']
            rv = fit['rv']
            rv_err = fit['rv_err']
            mask = ~np.in1d(fit['_time'], t)
            _t = fit['_time'][mask]
            _rv = fit['_rv'][mask]
            _rv_err = fit['_rv_err'][mask]
            c = colors[ix]
            t_fit = np.linspace(fit['_time'].min()-1/24,
                                fit['_time'].max()+1/24, 5000)
            if not phaseunits:
                fn = lambda x: 24 * (x - fit['_time'].min())
            else:
                fn = lambda x: ((x - t0) / period - 110)
            ax.errorbar(fn(t), rv, yerr=rv_err, fmt='o', c=c, ms=2)
            ax.errorbar(fn(_t), _rv, yerr=_rv_err, fmt='x', alpha=0.3, zorder=-1, c=c, ms=4)
            if fittingstyle == 'leastsquares':
                popt = fit['popt_circ']
                ax.plot(fn(t_fit), rv_circular(t_fit, *popt), '-', c=c, zorder=-2, alpha=0.7)
            elif fittingstyle == 'mcmc':
                samp = fit['samples']
                inds = np.linspace(0, samp['K'].shape[0]-1, 6, dtype=int)[1:]
                for idx in inds:
                    draw = [samp['K'][idx], samp['P'][idx], samp['t0'][idx]]
                    if not phaseunits:
                        ax.plot(fn(t_fit), rv_circular(t_fit, *draw), '-', c=c, zorder=-2, alpha=0.3, lw=0.4)
                if phaseunits:
                    popt = fit['popt_circ']
                    ax.plot(fn(t_fit), rv_circular(t_fit, *popt), '-', c=c, zorder=-2, alpha=0.9, lw=0.8)
                    sigma = 0.25
                    # Compute the mean fit over the global t_fit grid.
                    y_mean = rv_circular(global_t_fit, *popt)
                    mean_matrix = np.tile(y_mean, (y_grid.size, 1))
                    local_gauss = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((Y_global - mean_matrix)/sigma)**2)
                    gauss_total += local_gauss

        if phaseunits:
            cmap_custom = plt.get_cmap("Blues").copy()
            cmap_custom.set_bad(color='none')
            extent = [global_x_vals[0], global_x_vals[-1], y_min, y_max]
            norm = plt.Normalize(0, (1/3) * np.max(gauss_total))  # Normalize for better visualization
            ax.imshow(gauss_total, extent=extent, origin='lower', aspect='auto', cmap=cmap_custom, alpha=1, zorder=-3, norm=norm)

        ax.set_ylim([-4.9, 4.9])
        if not phaseunits:
            ax.set_xlabel('Time from start [hours]')
            ax.set_xlim([-0.22, 5.42])
        else:
            ax.set_xlabel('Phase φ')
            ax.set_xlim([-0.1, 1.3])
            ax.minorticks_on()
            ax.grid(True, which='both', alpha=0.5)

        ax.set_ylabel('Clump RV [v$_{\\mathrm{eq}}$]')
        s = '' if not phaseunits else '_phaseunits'
        s1 = '_leastsq' if fittingstyle == 'leastsquares' else ''
        combined_plot_path = join(
            'results/halpha_to_rv_timerseries',
            f'combined_circular_fit{s}{s1}.png'
        )
        savefig(fig, combined_plot_path)
        plt.clf()

    # PRINT FINAL CIRCULAR ORBIT FIT RESULTS
    print(df_circ)


if __name__ == '__main__':
    main(fittingstyle='mcmc')
    assert 0
    main(fittingstyle='leastsquares')
