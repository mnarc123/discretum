#!/usr/bin/env python3
"""
DISCRETUM — Extrapolation of observables to the thermodynamic limit N → ∞.

Three fit models:
  1. O(N) = O_∞ + A · N^{-ω}                   (power-law correction)
  2. O(N) = O_∞ + A · N^{-ω} + B · N^{-2ω}     (correction-to-scaling)
  3. O(N) = O_∞ + A / ln(N)                      (logarithmic)

Selects best model by χ²/ndof closest to 1.
Bootstrap (2000 resamples) for confidence intervals.

Reads: data/results/scaling_4d_v3/scaling_summary.json
Writes: data/results/scaling_4d_v3/extrapolation.json
        paper/figures/fig_scaling_*.pdf
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
})

FIGDIR = 'paper/figures'
os.makedirs(FIGDIR, exist_ok=True)

COL_WIDTH = 3.375


# ─── Fit models ───

def power_law(N, O_inf, A, omega):
    return O_inf + A * np.power(N, -omega)

def correction_to_scaling(N, O_inf, A, omega, B):
    return O_inf + A * np.power(N, -omega) + B * np.power(N, -2 * omega)

def logarithmic(N, O_inf, A):
    return O_inf + A / np.log(N)


def try_fit(func, N, y, yerr, p0, bounds, name):
    """Attempt a single fit, return dict or None."""
    try:
        popt, pcov = curve_fit(func, N, y, sigma=yerr, absolute_sigma=True,
                               p0=p0, maxfev=20000, bounds=bounds)
        residuals = (y - func(N, *popt)) / yerr
        chi2 = float(np.sum(residuals**2))
        ndof = max(len(N) - len(popt), 1)
        return {
            'name': name, 'func': func,
            'popt': popt, 'pcov': pcov,
            'chi2': chi2, 'ndof': ndof,
            'chi2_ndof': chi2 / ndof,
            'O_inf': float(popt[0]),
            'O_inf_err': float(np.sqrt(pcov[0, 0])),
        }
    except Exception as e:
        return None


def fit_observable(N, y, yerr, obs_name, target=None):
    """Fit all models, select best, return result dict."""
    results = []

    # Power-law
    g = target if target is not None else y[-1]
    r = try_fit(power_law, N, y, yerr,
                p0=[g, y[0] - y[-1], 0.5],
                bounds=([-50, -500, 0.01], [50, 500, 5.0]),
                name='power_law')
    if r: results.append(r)

    # Correction-to-scaling (need ≥5 points)
    if len(N) >= 5:
        r = try_fit(correction_to_scaling, N, y, yerr,
                    p0=[g, y[0] - y[-1], 0.5, 0.0],
                    bounds=([-50, -500, 0.01, -500], [50, 500, 5.0, 500]),
                    name='correction_to_scaling')
        if r: results.append(r)

    # Logarithmic
    r = try_fit(logarithmic, N, y, yerr,
                p0=[g, (y[0] - y[-1]) * np.log(N[-1])],
                bounds=([-50, -500], [50, 500]),
                name='logarithmic')
    if r: results.append(r)

    if not results:
        print(f"  WARNING: No fit converged for {obs_name}")
        return None

    # Select best by chi2/ndof closest to 1
    best = min(results, key=lambda r: abs(r['chi2_ndof'] - 1.0))

    print(f"\n  {obs_name}:")
    for r in results:
        marker = " ← BEST" if r is best else ""
        print(f"    {r['name']:25s}: O_∞ = {r['O_inf']:+.4f} ± {r['O_inf_err']:.4f}, "
              f"χ²/ndof = {r['chi2']:.2f}/{r['ndof']}{marker}")

    # Bootstrap CI
    n_boot = 2000
    boot_inf = []
    func = best['func']
    for _ in range(n_boot):
        y_boot = y + np.random.randn(len(N)) * yerr
        try:
            npar = len(best['popt'])
            lb = [-50] + [-500] * (npar - 1)
            ub = [50] + [500] * (npar - 1)
            pb, _ = curve_fit(func, N, y_boot, p0=best['popt'],
                              sigma=yerr, absolute_sigma=True, maxfev=5000,
                              bounds=(lb, ub))
            boot_inf.append(pb[0])
        except:
            pass

    boot_inf = np.array(boot_inf)
    ci_lo = float(np.percentile(boot_inf, 2.5)) if len(boot_inf) > 10 else np.nan
    ci_hi = float(np.percentile(boot_inf, 97.5)) if len(boot_inf) > 10 else np.nan
    boot_median = float(np.median(boot_inf)) if len(boot_inf) > 10 else np.nan

    print(f"    Bootstrap (n={len(boot_inf)}): median={boot_median:.4f}, "
          f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        'obs_name': obs_name,
        'model': best['name'],
        'O_inf': best['O_inf'],
        'O_inf_err': best['O_inf_err'],
        'chi2': best['chi2'],
        'ndof': best['ndof'],
        'chi2_ndof': best['chi2_ndof'],
        'boot_median': boot_median,
        'boot_ci_lo': ci_lo,
        'boot_ci_hi': ci_hi,
        'popt': best['popt'].tolist(),
        'func': best['func'],  # not serialized
        'boot_values': boot_inf,  # not serialized
    }


def plot_scaling(N, y, yerr, fit_result, obs_name, ylabel, target=None, filename=None):
    """Create scaling plot with fit, CI band, and extrapolation."""
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

    # Data
    ax.errorbar(N, y, yerr=yerr, fmt='ko', capsize=2, markersize=4, label='Data', zorder=5)

    func = fit_result['func']
    popt = np.array(fit_result['popt'])

    # Fit curve + extrapolation
    N_ext = np.logspace(np.log10(min(N) * 0.8), np.log10(max(N) * 10), 300)
    ax.plot(N_ext, func(N_ext, *popt), 'b-', linewidth=1.0,
            label=f'Fit: $O_\\infty = {fit_result["O_inf"]:.2f} \\pm {fit_result["O_inf_err"]:.2f}$')

    # Bootstrap CI band
    boot_vals = fit_result.get('boot_values', np.array([]))
    if len(boot_vals) > 50:
        boot_curves = np.zeros((len(boot_vals), len(N_ext)))
        # We need to re-fit to get curves; use saved popt as starting point
        for i in range(min(500, len(boot_vals))):
            y_boot = y + np.random.randn(len(N)) * yerr
            try:
                pb, _ = curve_fit(func, N, y_boot, p0=popt, sigma=yerr,
                                  absolute_sigma=True, maxfev=3000)
                boot_curves[i] = func(N_ext, *pb)
            except:
                boot_curves[i] = np.nan
        lo = np.nanpercentile(boot_curves[:min(500, len(boot_vals))], 16, axis=0)
        hi = np.nanpercentile(boot_curves[:min(500, len(boot_vals))], 84, axis=0)
        ax.fill_between(N_ext, lo, hi, alpha=0.15, color='blue', label='$1\\sigma$ CI')

    if target is not None:
        ax.axhline(target, color='gray', ls=':', lw=0.5, alpha=0.7,
                   label=f'Target = {target}')

    ax.set_xlabel('$N$')
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    if filename:
        fig.savefig(os.path.join(FIGDIR, filename + '.pdf'))
        fig.savefig(os.path.join(FIGDIR, filename + '.png'))
    plt.close()


def main():
    # Load scaling data
    with open('data/results/scaling_4d_v3/scaling_summary.json') as f:
        summary = json.load(f)

    scales = summary['scales']

    # Filter: only use points with ≥ 3 connected runs for reliable stats
    MIN_CONNECTED = 2
    good = [s for s in scales if s['n_connected'] >= MIN_CONNECTED]

    if len(good) < 3:
        print(f"ERROR: Only {len(good)} scales with ≥{MIN_CONNECTED} connected runs. "
              f"Need at least 3 for fitting.")
        # Fall back to all scales
        good = scales

    N = np.array([s['N'] for s in good], dtype=float)
    print("=" * 60)
    print("EXTRAPOLATION TO THERMODYNAMIC LIMIT")
    print("=" * 60)
    print(f"Using {len(good)} scales: L = {[s['L'] for s in good]}")
    print(f"Min connected runs per scale: {MIN_CONNECTED}")

    # ─── d_H ───
    dH = np.array([s['d_H'] for s in good])
    dH_err = np.array([max(s['d_H_err'], 0.01) for s in good])  # floor error
    dH_result = fit_observable(N, dH, dH_err, 'd_H', target=4.0)

    if dH_result:
        plot_scaling(N, dH, dH_err, dH_result, 'd_H', '$d_H$',
                     target=4.0, filename='fig6_scaling_dH')

    # ─── d_s ───
    ds = np.array([s['d_s'] for s in good])
    ds_err = np.array([max(s['d_s_err'], 0.01) for s in good])
    ds_result = fit_observable(N, ds, ds_err, 'd_s', target=4.0)

    if ds_result:
        plot_scaling(N, ds, ds_err, ds_result, 'd_s', '$d_s$',
                     target=4.0, filename='fig7_scaling_ds')

    # ─── κ ───
    kappa = np.array([s['kappa'] for s in good])
    kappa_err = np.array([max(s['kappa_err'], 0.001) for s in good])
    kappa_result = fit_observable(N, kappa, kappa_err, 'kappa', target=0.0)

    if kappa_result:
        plot_scaling(N, kappa, kappa_err, kappa_result, '\\kappa',
                     '$\\langle \\kappa \\rangle$',
                     target=0.0, filename='fig8_scaling_kappa')

    # ─── Connectivity trend ───
    N_all = np.array([s['N'] for s in scales], dtype=float)
    conn_frac = np.array([s['n_connected'] / s['n_total'] for s in scales])
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.0))
    ax.plot(N_all, conn_frac * 100, 'ro-', ms=5)
    ax.set_xlabel('$N$')
    ax.set_ylabel('Connected (%)')
    ax.set_xscale('log')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.2)
    ax.set_title('Connectivity vs graph size')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_connectivity.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig_connectivity.png'))
    plt.close()

    # ─── Save results ───
    def serialize(r):
        if r is None:
            return None
        return {k: v for k, v in r.items() if k not in ('func', 'boot_values')}

    extrapolation = {
        'd_H': serialize(dH_result),
        'd_s': serialize(ds_result),
        'kappa': serialize(kappa_result),
        'scales_used': [s['L'] for s in good],
        'N_values': N.tolist(),
        'connectivity': {str(s['L']): s['n_connected'] / s['n_total'] for s in scales},
    }

    out_path = 'data/results/scaling_4d_v3/extrapolation.json'
    with open(out_path, 'w') as f:
        json.dump(extrapolation, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Figures saved to {FIGDIR}/fig6_scaling_dH.pdf etc.")

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if dH_result:
        print(f"  d_H(∞) = {dH_result['O_inf']:.3f} ± {dH_result['O_inf_err']:.3f}  "
              f"[{dH_result['boot_ci_lo']:.3f}, {dH_result['boot_ci_hi']:.3f}]  "
              f"(model: {dH_result['model']})")
    if ds_result:
        print(f"  d_s(∞) = {ds_result['O_inf']:.3f} ± {ds_result['O_inf_err']:.3f}  "
              f"[{ds_result['boot_ci_lo']:.3f}, {ds_result['boot_ci_hi']:.3f}]  "
              f"(model: {ds_result['model']})")
    if kappa_result:
        print(f"  κ(∞)   = {kappa_result['O_inf']:.4f} ± {kappa_result['O_inf_err']:.4f}  "
              f"[{kappa_result['boot_ci_lo']:.4f}, {kappa_result['boot_ci_hi']:.4f}]  "
              f"(model: {kappa_result['model']})")


if __name__ == '__main__':
    main()
