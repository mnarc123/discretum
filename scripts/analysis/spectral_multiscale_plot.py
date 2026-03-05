#!/usr/bin/env python3
"""
DISCRETUM — Multiscale spectral dimension d_eff(t) plot.

Compares:
  1. Evolved graph (best 4D rule)
  2. Bare 4D lattice (baseline)
  3. Qualitative CDT band (Ambjørn et al., PRL 2005)

Output: paper/figures/fig9_spectral_multiscale.pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

FIGDIR = 'paper/figures'
os.makedirs(FIGDIR, exist_ok=True)
COL_WIDTH = 3.375


def smooth(y, w=15):
    """Moving-average smoothing."""
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode='valid')


def load_spectral(path):
    with open(path) as f:
        return json.load(f)


def main():
    evolved = load_spectral('data/results/scaling_4d_v3/spectral_evolved_L5.json')
    baseline = load_spectral('data/results/scaling_4d_v3/spectral_baseline_L5.json')

    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH * 2, 2.8))

    # ─── Panel (a): P(t) ───
    ax = axes[0]
    t_e = np.array(evolved['t'])
    P_e = np.array(evolved['P_t'])
    t_b = np.array(baseline['t'])
    P_b = np.array(baseline['P_t'])

    # Filter out zero P values for log plot
    mask_e = P_e > 0
    mask_b = P_b > 0

    ax.semilogy(t_e[mask_e], P_e[mask_e], 'b-', lw=0.8, alpha=0.8, label='Evolved')
    ax.semilogy(t_b[mask_b], P_b[mask_b], 'k--', lw=0.8, alpha=0.6, label='Bare lattice')
    ax.set_xlabel('Diffusion time $t$')
    ax.set_ylabel('$P(t)$')
    ax.set_title('(a) Return probability')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    # ─── Panel (b): d_eff(t) ───
    ax = axes[1]
    d_eff_e = np.array(evolved['d_eff'])
    d_eff_b = np.array(baseline['d_eff'])

    w = 15  # smoothing window
    if len(d_eff_e) > w and len(t_e) > w:
        d_smooth_e = smooth(d_eff_e, w)
        t_smooth_e = t_e[w // 2: w // 2 + len(d_smooth_e)]
        ax.plot(t_smooth_e, d_smooth_e, 'b-', lw=1.2, label='Evolved (smoothed)')

    if len(d_eff_b) > w and len(t_b) > w:
        d_smooth_b = smooth(d_eff_b, w)
        t_smooth_b = t_b[w // 2: w // 2 + len(d_smooth_b)]
        ax.plot(t_smooth_b, d_smooth_b, 'k--', lw=0.8, alpha=0.7, label='Bare lattice')

    # Raw (faint)
    if len(d_eff_e) > 0 and len(t_e) == len(d_eff_e):
        ax.plot(t_e, d_eff_e, 'b-', lw=0.2, alpha=0.15)

    # CDT band (qualitative, Ambjørn et al. 2005)
    t_cdt = np.logspace(0.3, 3.3, 200)
    ds_cdt_center = 1.8 + 2.2 * (1 - np.exp(-t_cdt / 80))
    ax.fill_between(t_cdt, ds_cdt_center - 0.4, ds_cdt_center + 0.4,
                    alpha=0.08, color='red', label='CDT (approx., Ambjorn+ 2005)')

    # Plateau region for evolved
    if evolved.get('has_plateau'):
        t_min = evolved['plateau_t_min']
        t_max = evolved['plateau_t_max']
        ax.axvspan(t_min, t_max, alpha=0.08, color='blue')
        ax.axhline(evolved['d_s'], color='blue', ls=':', lw=0.5, alpha=0.5)

    ax.axhline(4.0, color='gray', ls=':', lw=0.4, alpha=0.5, label='$d=4$')
    ax.axhline(2.0, color='gray', ls='-.', lw=0.4, alpha=0.5, label='$d=2$')

    # Annotations
    ax.annotate('Both curves: $d_{\\mathrm{eff}} < 1$\n(finite-size saturation)',
                xy=(30, 1.2), fontsize=7, color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    ax.annotate('CDT: $d_s \\approx 2 \\to 4$',
                xy=(200, 3.5), fontsize=7, color='darkred', alpha=0.8)

    ax.set_xlabel('Diffusion time $t$')
    ax.set_ylabel('$d_{\\mathrm{eff}}(t)$')
    ax.set_xscale('log')
    ax.set_ylim(-1, 8)
    ax.set_title('(b) Effective spectral dimension')
    ax.legend(loc='upper left', fontsize=6)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig9_spectral_multiscale.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig9_spectral_multiscale.png'))
    plt.close()

    print("Saved fig9_spectral_multiscale.pdf")

    # Print key findings
    print(f"\nEvolved graph (L=5, N={evolved['n_nodes']}):")
    print(f"  d_s(v2) = {evolved['d_s']:.3f} ± {evolved['d_s_error']:.3f}")
    print(f"  plateau: [{evolved['plateau_t_min']:.0f}, {evolved['plateau_t_max']:.0f}]")
    print(f"  global fit d_s = {evolved['d_s_global_fit']:.3f} (R²={evolved['global_fit_r2']:.3f})")

    print(f"\nBaseline lattice (L=5, N={baseline['n_nodes']}):")
    print(f"  d_s(v2) = {baseline['d_s']:.3f} ± {baseline['d_s_error']:.3f}")
    print(f"  plateau: [{baseline['plateau_t_min']:.0f}, {baseline['plateau_t_max']:.0f}]")
    print(f"  global fit d_s = {baseline['d_s_global_fit']:.3f} (R²={baseline['global_fit_r2']:.3f})")


if __name__ == '__main__':
    main()
