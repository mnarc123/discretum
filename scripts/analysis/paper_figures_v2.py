#!/usr/bin/env python3
"""
DISCRETUM — Paper figure generation (v2).

Adds 4D campaign results, curvature trends, and calibration baselines
to the existing Campaign 2 figures. Generates fig8-fig11.

Usage:
    python3 scripts/analysis/paper_figures_v2.py
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

OUTDIR = 'data/results/paper_figures'
os.makedirs(OUTDIR, exist_ok=True)

COL_WIDTH = 3.375
DBL_WIDTH = 7.0


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════

cal_3d = load_json('data/results/calibration_3d/baseline.json')
cal_4d = load_json('data/results/calibration_4d/baseline.json')
curv_trends = load_json('data/results/curvature_analysis/curvature_trends.json')
scaling_results = load_json('data/results/scaling_analysis/scaling_results.json')
ens_4d = load_json('data/results/4d_campaign/fast_ga_ensemble.json')
ens_4d_v3 = load_json('data/results/4d_v3_ga_ensemble.json')
diag_4d_v3 = load_json('data/results/4d_v3_ga_single_diag.json')
ens_4d_v3_cma = load_json('data/results/4d_v3_cmaes_ensemble.json')
diag_4d_v3_cma = load_json('data/results/4d_v3_cmaes_single_diag.json')

# Campaign 2 scaling
c2_scaling = {}
for s in [5, 8, 10, 12, 15]:
    d = load_json(f'data/results/campaign2/scaling/ga_size_{s}.json')
    if d:
        c2_scaling[s] = d


# ═══════════════════════════════════════════════════
# Fig 8: Calibration baselines — d_H vs N for 3D and 4D bare lattices
# ═══════════════════════════════════════════════════
print("Generating Fig 8: calibration baselines...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.8))

if cal_3d:
    N3 = [d['N'] for d in cal_3d]
    dH3 = [d['d_H'] for d in cal_3d]
    ax1.plot(N3, dH3, 'bo-', ms=5, label='Measured $d_H$')
    ax1.axhline(3.0, color='gray', ls='--', lw=0.5, label='$d=3$ (theoretical)')
    # Power law extrapolation
    from scipy.optimize import curve_fit
    def corr_scaling(N, d_inf, a, nu):
        return d_inf + a * np.power(N, -nu)
    try:
        popt, _ = curve_fit(corr_scaling, N3, dH3, p0=[3.0, -3, 0.3],
                            bounds=([0,-100,0.01],[10,100,2]))
        N_ext = np.linspace(min(N3), 20000, 200)
        ax1.plot(N_ext, corr_scaling(N_ext, *popt), 'r--', lw=0.8,
                 label=f'Fit: $d_H(\\infty) = {popt[0]:.2f}$')
    except:
        pass
    ax1.set_xlabel('$N$')
    ax1.set_ylabel('$d_H$')
    ax1.set_title('(a) Bare 3D lattice')
    ax1.legend(frameon=False, fontsize=7)

if cal_4d:
    N4 = [d['N'] for d in cal_4d]
    dH4 = [d['d_H'] for d in cal_4d]
    ax2.plot(N4, dH4, 'rs-', ms=5, label='Measured $d_H$')
    ax2.axhline(4.0, color='gray', ls='--', lw=0.5, label='$d=4$ (theoretical)')
    try:
        popt, _ = curve_fit(corr_scaling, N4, dH4, p0=[4.0, -5, 0.3],
                            bounds=([0,-100,0.01],[10,100,2]))
        N_ext = np.linspace(min(N4), 50000, 200)
        ax2.plot(N_ext, corr_scaling(N_ext, *popt), 'r--', lw=0.8,
                 label=f'Fit: $d_H(\\infty) = {popt[0]:.2f}$')
    except:
        pass
    ax2.set_xlabel('$N$')
    ax2.set_ylabel('$d_H$')
    ax2.set_title('(b) Bare 4D lattice')
    ax2.legend(frameon=False, fontsize=7)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig8_calibration.pdf')
fig.savefig(f'{OUTDIR}/fig8_calibration.png')
plt.close()


# ═══════════════════════════════════════════════════
# Fig 9: Curvature trends — κ vs N comparison
# ═══════════════════════════════════════════════════
print("Generating Fig 9: curvature trends...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.8))

colors = {'Bare 3D': 'blue', 'Bare 4D': 'red', 'Campaign 2': 'green'}

# (a) κ vs N (linear)
if cal_3d:
    N = [d['N'] for d in cal_3d]
    k = [d['mean_curvature'] for d in cal_3d]
    ax1.plot(N, k, 'bo-', ms=4, label='Bare 3D lattice')

if cal_4d:
    N = [d['N'] for d in cal_4d]
    k = [d['mean_curvature'] for d in cal_4d]
    ax1.plot(N, k, 'rs-', ms=4, label='Bare 4D lattice')

if c2_scaling:
    sides = sorted(c2_scaling.keys())
    N = [c2_scaling[s]['graph']['N_final'] for s in sides]
    k = [c2_scaling[s]['curvature']['mean'] for s in sides]
    ax1.plot(N, k, 'g^-', ms=4, label='Campaign 2 (evolved)')

ax1.axhline(0, color='gray', ls='--', lw=0.5)
ax1.set_xlabel('$N$')
ax1.set_ylabel('$\\langle \\kappa \\rangle$')
ax1.set_title('(a) Mean curvature vs $N$')
ax1.legend(frameon=False, fontsize=7)

# (b) log|κ| vs log N
if cal_3d:
    N = np.array([d['N'] for d in cal_3d])
    k = np.abs(np.array([d['mean_curvature'] for d in cal_3d]))
    m = k > 1e-10
    ax2.plot(np.log(N[m]), np.log(k[m]), 'bo-', ms=4, label='Bare 3D')
    c = np.polyfit(np.log(N[m]), np.log(k[m]), 1)
    ax2.plot(np.log(N[m]), np.polyval(c, np.log(N[m])), 'b--', lw=0.5,
             label=f'$\\alpha={c[0]:.2f}$')

if cal_4d:
    N = np.array([d['N'] for d in cal_4d])
    k = np.abs(np.array([d['mean_curvature'] for d in cal_4d]))
    m = k > 1e-10
    ax2.plot(np.log(N[m]), np.log(k[m]), 'rs-', ms=4, label='Bare 4D')
    c = np.polyfit(np.log(N[m]), np.log(k[m]), 1)
    ax2.plot(np.log(N[m]), np.polyval(c, np.log(N[m])), 'r--', lw=0.5,
             label=f'$\\alpha={c[0]:.2f}$')

if c2_scaling:
    sides = sorted(c2_scaling.keys())
    N = np.array([c2_scaling[s]['graph']['N_final'] for s in sides])
    k = np.abs(np.array([c2_scaling[s]['curvature']['mean'] for s in sides]))
    m = k > 1e-10
    if m.sum() >= 2:
        ax2.plot(np.log(N[m]), np.log(k[m]), 'g^-', ms=4, label='Campaign 2')
        c = np.polyfit(np.log(N[m]), np.log(k[m]), 1)
        ax2.plot(np.log(N[m]), np.polyval(c, np.log(N[m])), 'g--', lw=0.5,
                 label=f'$\\alpha={c[0]:.2f}$')

ax2.set_xlabel('$\\ln(N)$')
ax2.set_ylabel('$\\ln|\\langle \\kappa \\rangle|$')
ax2.set_title('(b) Power-law scaling of $|\\kappa|$')
ax2.legend(frameon=False, fontsize=6, ncol=2)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig9_curvature_trends.pdf')
fig.savefig(f'{OUTDIR}/fig9_curvature_trends.png')
plt.close()


# ═══════════════════════════════════════════════════
# Fig 10: 4D campaign ensemble results
# ═══════════════════════════════════════════════════
print("Generating Fig 10: 4D ensemble results...")

if ens_4d and 'runs' in ens_4d:
    fig, axes = plt.subplots(2, 2, figsize=(DBL_WIDTH, 4.5))

    runs = ens_4d['runs']
    valid = [r for r in runs if not r.get('aborted', False)]

    # (a) d_H distribution
    ax = axes[0, 0]
    dH_vals = [r['d_H'] for r in valid]
    ax.hist(dH_vals, bins=max(5, len(dH_vals)//2), color='steelblue',
            edgecolor='navy', lw=0.3)
    ax.axvline(np.mean(dH_vals), color='red', ls='--', lw=0.8,
               label=f'$\\langle d_H \\rangle = {np.mean(dH_vals):.3f}$')
    ax.axvline(4.0, color='gray', ls=':', lw=0.5, label='$d=4$ target')
    ax.set_xlabel('$d_H$')
    ax.set_ylabel('Count')
    ax.set_title('(a) Hausdorff dimension')
    ax.legend(frameon=False, fontsize=7)

    # (b) κ distribution
    ax = axes[0, 1]
    k_vals = [r['mean_curvature'] for r in valid]
    ax.hist(k_vals, bins=max(5, len(k_vals)//2), color='coral',
            edgecolor='darkred', lw=0.3)
    ax.axvline(np.mean(k_vals), color='blue', ls='--', lw=0.8,
               label=f'$\\langle \\kappa \\rangle = {np.mean(k_vals):.3f}$')
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('$\\langle \\kappa \\rangle$')
    ax.set_ylabel('Count')
    ax.set_title('(b) Mean curvature')
    ax.legend(frameon=False, fontsize=7)

    # (c) Fitness components bar chart
    ax = axes[1, 0]
    stat_keys = ['f_hausdorff', 'f_curvature', 'f_spectral', 'f_stability', 'f_regularity']
    labels = ['$F_{d_H}$', '$F_\\kappa$', '$F_{d_s}$', '$F_{stab}$', '$F_{reg}$']
    means = [ens_4d[k]['mean'] for k in stat_keys]
    errs = [ens_4d[k]['std_err'] for k in stat_keys]
    colors_bar = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71']
    bars = ax.barh(labels, means, xerr=errs, color=colors_bar,
                   edgecolor='black', linewidth=0.3, capsize=3)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Fitness contribution')
    ax.set_title(f'(c) Fitness breakdown (total = {ens_4d["fitness_total"]["mean"]:.3f})')

    # (d) N_final distribution
    ax = axes[1, 1]
    n_vals = [r['n_final'] for r in valid]
    ax.hist(n_vals, bins=max(5, len(n_vals)//2), color='lightgreen',
            edgecolor='darkgreen', lw=0.3)
    ax.axvline(np.mean(n_vals), color='red', ls='--', lw=0.8,
               label=f'$\\langle N \\rangle = {np.mean(n_vals):.0f}$')
    ax.set_xlabel('$N_{\\mathrm{final}}$')
    ax.set_ylabel('Count')
    ax.set_title('(d) Final graph size')
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig10_4d_ensemble.pdf')
    fig.savefig(f'{OUTDIR}/fig10_4d_ensemble.png')
    plt.close()
else:
    print("  Skipped (no 4D ensemble data)")


# ═══════════════════════════════════════════════════
# Fig 11: Comparison table (printed)
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("         COMPARISON TABLE — Bare vs Evolved Lattices")
print("="*70)

print(f"\n{'Dataset':<30} {'d_H':>8} {'⟨κ⟩':>10} {'σ(κ)':>8} {'⟨k⟩':>8}")
print("-"*66)

if cal_3d:
    d = cal_3d[-1]  # largest
    print(f"{'Bare 3D (L='+str(d['L'])+')':<30} {d['d_H']:>8.3f} {d['mean_curvature']:>+10.5f} {d['std_curvature']:>8.4f} {d['avg_degree']:>8.2f}")

if cal_4d:
    d = cal_4d[-1]
    print(f"{'Bare 4D (L='+str(d['L'])+')':<30} {d['d_H']:>8.3f} {d['mean_curvature']:>+10.5f} {d['std_curvature']:>8.4f} {d['avg_degree']:>8.2f}")

if c2_scaling:
    s = max(c2_scaling.keys())
    d = c2_scaling[s]
    print(f"{'Campaign 2 GA (L='+str(s)+')':<30} {d.get('geodesic',{}).get('hausdorff_dimension',0):>8.3f} {d['curvature']['mean']:>+10.5f} {d['curvature']['std']:>8.4f} {d['graph']['mean_degree']:>8.2f}")

if ens_4d:
    print(f"{'4D GA fast (ensemble)':<30} {ens_4d['d_H']['mean']:>8.3f} {ens_4d['mean_curvature']['mean']:>+10.5f} {ens_4d['std_curvature']['mean']:>8.4f} {ens_4d['mean_degree']['mean']:>8.2f}")

print("="*66)

# ═══════════════════════════════════════════════════
# Fig 12: v3 campaign comparison — single-run diagnostics
# ═══════════════════════════════════════════════════
print("Generating Fig 12: v3 campaign diagnostics...")

if diag_4d_v3:
    fig = plt.figure(figsize=(DBL_WIDTH, 5.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # (a) Curvature histogram
    ax = fig.add_subplot(gs[0, 0])
    curv = diag_4d_v3.get('curvature', {})
    if 'histogram' in curv:
        bins = curv['histogram']['bin_edges']
        counts = curv['histogram']['counts']
        centers = [(bins[i] + bins[i+1])/2 for i in range(len(counts))]
        ax.bar(centers, counts, width=(bins[1]-bins[0])*0.9, color='coral', edgecolor='darkred', lw=0.3)
    elif 'all_values' in curv:
        ax.hist(curv['all_values'], bins=20, color='coral', edgecolor='darkred', lw=0.3)
    ax.axvline(curv.get('mean', 0), color='blue', ls='--', lw=0.8,
               label=f'$\\langle \\kappa \\rangle = {curv.get("mean", 0):.3f}$')
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('$\\kappa$')
    ax.set_ylabel('Count')
    ax.set_title('(a) Edge curvature')
    ax.legend(frameon=False, fontsize=7)

    # (b) Degree distribution
    ax = fig.add_subplot(gs[0, 1])
    deg = diag_4d_v3.get('degree_distribution', {})
    if deg:
        degs = sorted(deg.keys(), key=int)
        ax.bar([int(d) for d in degs], [deg[d] for d in degs],
               color='steelblue', edgecolor='navy', lw=0.3)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        ax.set_title(f'(b) Degree dist ($\\langle k \\rangle = {diag_4d_v3.get("graph", {}).get("mean_degree", 0):.1f}$)')

    # (c) Volume growth N(r)
    ax = fig.add_subplot(gs[0, 2])
    geo = diag_4d_v3.get('geodesic', {})
    vg = geo.get('volume_growth', [])
    if vg:
        ax.plot(range(len(vg)), vg, 'ko-', ms=3)
        ax.set_xlabel('$r$')
        ax.set_ylabel('$N(r)$')
        ax.set_title(f'(c) Volume growth ($d_H = {geo.get("hausdorff_dimension", 0):.2f}$)')
        ax.grid(True, alpha=0.3)

    # (d) Metric tensor eigenvalues
    ax = fig.add_subplot(gs[1, 0])
    mt = diag_4d_v3.get('metric_tensor', {})
    evals = mt.get('eigenvalues', [])
    if evals:
        ax.bar(range(len(evals)), evals, color='#9b59b6', edgecolor='black', lw=0.3)
        ax.set_xlabel('Component')
        ax.set_ylabel('$\\lambda_i$')
        ax.set_title(f'(d) Metric tensor eigenvalues')
        ax.axhline(np.mean(evals), color='red', ls='--', lw=0.6,
                   label=f'mean={np.mean(evals):.3f}')
        ax.legend(frameon=False, fontsize=7)

    # (e) Fitness breakdown
    ax = fig.add_subplot(gs[1, 1])
    fit = diag_4d_v3.get('fitness', {})
    if fit:
        comps = ['F_ricci', 'F_dimension', 'F_connectivity', 'F_degree_reg', 'F_size']
        labels_f = ['$F_\\kappa$', '$F_{d_s}$', '$F_{conn}$', '$F_{deg}$', '$F_{size}$']
        vals = [fit.get(c, 0) for c in comps]
        cols = ['#f39c12', '#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        ax.barh(labels_f, vals, color=cols, edgecolor='black', lw=0.3)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_xlabel('Penalty')
        ax.set_title(f'(e) Fitness (total = {fit.get("F_total", 0):.3f})')

    # (f) Summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    graph = diag_4d_v3.get('graph', {})
    spectral = diag_4d_v3.get('spectral', {})
    summary = (
        f'v3 GA Best Rule\n'
        f'────────────────\n'
        f'N = {graph.get("N_final", "?")}, E = {graph.get("E_final", "?")}\n'
        f'd_H = {geo.get("hausdorff_dimension", 0):.3f}\n'
        f'd_s (v1) = {spectral.get("dimension", 0):.3f}\n'
        f'⟨κ⟩ = {curv.get("mean", 0):.4f}\n'
        f'σ(κ) = {curv.get("std", 0):.4f}\n'
        f'⟨k⟩ = {graph.get("mean_degree", 0):.2f}\n'
        f'Diameter = {geo.get("diameter", "?")}\n'
        f'R = {mt.get("scalar_curvature", 0):.4f}'
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.savefig(f'{OUTDIR}/fig12_v3_diagnostics.pdf')
    fig.savefig(f'{OUTDIR}/fig12_v3_diagnostics.png')
    plt.close()
else:
    print("  Skipped (no v3 diagnostic data)")


# ═══════════════════════════════════════════════════
# Updated comparison table
# ═══════════════════════════════════════════════════

if ens_4d_v3:
    print(f"{'4D v3 GA (ensemble)':<30} {ens_4d_v3['d_H']['mean']:>8.3f} {ens_4d_v3['mean_curvature']['mean']:>+10.5f} {ens_4d_v3['std_curvature']['mean']:>8.4f} {ens_4d_v3['mean_degree']['mean']:>8.2f}")
    print(f"  Connected: {ens_4d_v3['n_connected']}/{ens_4d_v3['n_total']}")

if ens_4d_v3_cma:
    print(f"{'4D v3 CMA-ES (ensemble)':<30} {ens_4d_v3_cma['d_H']['mean']:>8.3f} {ens_4d_v3_cma['mean_curvature']['mean']:>+10.5f} {ens_4d_v3_cma['std_curvature']['mean']:>8.4f} {ens_4d_v3_cma['mean_degree']['mean']:>8.2f}")
    print(f"  Connected: {ens_4d_v3_cma['n_connected']}/{ens_4d_v3_cma['n_total']}")

print("="*66)


# ═══════════════════════════════════════════════════
# Fig 13: GA vs CMA-ES comparison (v3 fitness)
# ═══════════════════════════════════════════════════
print("Generating Fig 13: GA vs CMA-ES comparison...")

if ens_4d_v3 and ens_4d_v3_cma:
    fig, axes = plt.subplots(2, 3, figsize=(DBL_WIDTH, 4.5))

    ga_runs = [r for r in ens_4d_v3.get('runs', []) if not r.get('aborted')]
    cma_runs = [r for r in ens_4d_v3_cma.get('runs', []) if not r.get('aborted')]

    # (a) d_H comparison
    ax = axes[0, 0]
    ga_dH = [r['d_H'] for r in ga_runs if r.get('is_connected', False)]
    cma_dH = [r['d_H'] for r in cma_runs if r.get('is_connected', False)]
    if ga_dH:
        ax.hist(ga_dH, bins=8, alpha=0.6, color='coral', label=f'GA (n={len(ga_dH)})', edgecolor='darkred', lw=0.3)
    if cma_dH:
        ax.hist(cma_dH, bins=8, alpha=0.6, color='steelblue', label=f'CMA-ES (n={len(cma_dH)})', edgecolor='navy', lw=0.3)
    ax.axvline(4.0, color='gray', ls=':', lw=0.5, label='$d=4$')
    ax.set_xlabel('$d_H$')
    ax.set_ylabel('Count')
    ax.set_title('(a) Hausdorff dimension')
    ax.legend(frameon=False, fontsize=6)

    # (b) d_s comparison
    ax = axes[0, 1]
    ga_ds = [r['d_s'] for r in ga_runs if r.get('is_connected', False)]
    cma_ds = [r['d_s'] for r in cma_runs if r.get('is_connected', False)]
    if ga_ds:
        ax.hist(ga_ds, bins=8, alpha=0.6, color='coral', label='GA', edgecolor='darkred', lw=0.3)
    if cma_ds:
        ax.hist(cma_ds, bins=8, alpha=0.6, color='steelblue', label='CMA-ES', edgecolor='navy', lw=0.3)
    ax.set_xlabel('$d_s$')
    ax.set_ylabel('Count')
    ax.set_title('(b) Spectral dimension (v2)')
    ax.legend(frameon=False, fontsize=6)

    # (c) Curvature comparison
    ax = axes[0, 2]
    ga_k = [r['mean_curvature'] for r in ga_runs if r.get('is_connected', False)]
    cma_k = [r['mean_curvature'] for r in cma_runs if r.get('is_connected', False)]
    if ga_k:
        ax.hist(ga_k, bins=8, alpha=0.6, color='coral', label='GA', edgecolor='darkred', lw=0.3)
    if cma_k:
        ax.hist(cma_k, bins=8, alpha=0.6, color='steelblue', label='CMA-ES', edgecolor='navy', lw=0.3)
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('$\\langle \\kappa \\rangle$')
    ax.set_ylabel('Count')
    ax.set_title('(c) Mean curvature')
    ax.legend(frameon=False, fontsize=6)

    # (d) Fitness component bar chart comparison
    ax = axes[1, 0]
    stat_keys = ['f_hausdorff', 'f_curvature', 'f_spectral', 'f_stability', 'f_regularity']
    labels = ['$F_{d_H}$', '$F_\\kappa$', '$F_{d_s}$', '$F_{stab}$', '$F_{reg}$']
    ga_means = [ens_4d_v3[k]['mean'] for k in stat_keys]
    cma_means = [ens_4d_v3_cma[k]['mean'] for k in stat_keys]
    x = np.arange(len(labels))
    w = 0.35
    ax.barh(x - w/2, ga_means, w, color='coral', label='GA', edgecolor='darkred', lw=0.3)
    ax.barh(x + w/2, cma_means, w, color='steelblue', label='CMA-ES', edgecolor='navy', lw=0.3)
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Penalty')
    ax.set_title('(d) Fitness components')
    ax.legend(frameon=False, fontsize=6)

    # (e) N_final comparison
    ax = axes[1, 1]
    ga_n = [r['n_final'] for r in ga_runs]
    cma_n = [r['n_final'] for r in cma_runs]
    if ga_n:
        ax.hist(ga_n, bins=8, alpha=0.6, color='coral', label='GA', edgecolor='darkred', lw=0.3)
    if cma_n:
        ax.hist(cma_n, bins=8, alpha=0.6, color='steelblue', label='CMA-ES', edgecolor='navy', lw=0.3)
    ax.axvline(625, color='gray', ls=':', lw=0.5, label='$N_0=625$')
    ax.set_xlabel('$N_{final}$')
    ax.set_ylabel('Count')
    ax.set_title('(e) Final graph size')
    ax.legend(frameon=False, fontsize=6)

    # (f) Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f'4D v3 Campaign Summary\n'
        f'\u2500'*22 + '\n'
        f'                GA     CMA-ES\n'
        f'fitness:    {ens_4d_v3["fitness_total"]["mean"]:>7.2f}  {ens_4d_v3_cma["fitness_total"]["mean"]:>7.2f}\n'
        f'd_H:        {ens_4d_v3["d_H"]["mean"]:>7.3f}  {ens_4d_v3_cma["d_H"]["mean"]:>7.3f}\n'
        f'd_s:        {ens_4d_v3["d_s"]["mean"]:>7.3f}  {ens_4d_v3_cma["d_s"]["mean"]:>7.3f}\n'
        f'kappa:      {ens_4d_v3["mean_curvature"]["mean"]:>+7.3f}  {ens_4d_v3_cma["mean_curvature"]["mean"]:>+7.3f}\n'
        f'connected:    {ens_4d_v3["n_connected"]}/{ens_4d_v3["n_total"]}     {ens_4d_v3_cma["n_connected"]}/{ens_4d_v3_cma["n_total"]}\n'
        f'\nWinner: CMA-ES'
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig13_ga_vs_cmaes.pdf')
    fig.savefig(f'{OUTDIR}/fig13_ga_vs_cmaes.png')
    plt.close()
else:
    print("  Skipped (need both GA and CMA-ES ensemble data)")

print(f"\nAll figures saved to {OUTDIR}/")
print("  fig8_calibration.pdf       — Bare lattice d_H calibration")
print("  fig9_curvature_trends.pdf  — κ vs N power-law scaling")
print("  fig10_4d_ensemble.pdf      — 4D campaign ensemble results")
print("  fig12_v3_diagnostics.pdf   — v3 4D campaign diagnostics")
