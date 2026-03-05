#!/usr/bin/env python3
"""
DISCRETUM — Paper figure generation script.

Generates publication-quality figures (Physical Review style) from
Campaign 2 diagnostic and finite-size scaling data.

Usage:
    python3 scripts/analysis/paper_figures.py

Output:
    data/results/paper_figures/fig{1..6}.pdf
"""

import json
import os
import re
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Physical Review style ──
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
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
})

OUTDIR = 'data/results/paper_figures'
os.makedirs(OUTDIR, exist_ok=True)

COL_WIDTH = 3.375   # inches (PRL single column)
DBL_WIDTH = 7.0     # inches (PRL double column)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════
# Load all data
# ═══════════════════════════════════════════════════

# Campaign 2 diagnostics
ga_diag = load_json('data/results/campaign2/ga_diag_full.json')
cmaes_diag = load_json('data/results/campaign2/cmaes_diag_full.json')

# Finite-size scaling
scaling = {}
for s in [5, 8, 10, 12, 15]:
    path = f'data/results/campaign2/scaling/ga_size_{s}.json'
    if os.path.exists(path):
        scaling[s] = load_json(path)

# GA convergence from log
ga_gens, ga_bests = [], []
log_path = 'data/campaigns/campaign2/ga_log.txt'
if os.path.exists(log_path):
    with open(log_path) as f:
        for line in f:
            m = re.search(r'GA gen (\d+)/\d+: best=([-\d.]+)', line)
            if m:
                ga_gens.append(int(m.group(1)))
                ga_bests.append(float(m.group(2)))

# CMA-ES convergence
cmaes_result_path = 'data/campaigns/campaign2/checkpoints_cmaes/cmaes_result.json'
cmaes_hist = []
if os.path.exists(cmaes_result_path):
    cmaes_hist = load_json(cmaes_result_path).get('fitness_history', [])


# ═══════════════════════════════════════════════════
# Fig 1: Convergence (GA vs CMA-ES)
# ═══════════════════════════════════════════════════
print("Generating Fig 1: convergence...")

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))
if ga_gens:
    ax.plot(ga_gens, ga_bests, 'b-', lw=0.5, label=f'GA (final = {ga_bests[-1]:.4f})')
if cmaes_hist:
    ax.plot(range(len(cmaes_hist)), cmaes_hist, 'r-', lw=0.8,
            label=f'CMA-ES (final = {cmaes_hist[-1]:.4f})')
ax.set_xlabel('Generation')
ax.set_ylabel('Best fitness')
ax.set_xlim(0, max(len(ga_gens), len(cmaes_hist)) if ga_gens or cmaes_hist else 100)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig1_convergence.pdf')
fig.savefig(f'{OUTDIR}/fig1_convergence.png')
plt.close()


# ═══════════════════════════════════════════════════
# Fig 2: Degree distribution + curvature distribution
# ═══════════════════════════════════════════════════
print("Generating Fig 2: distributions...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.5))

# (a) Degree distribution
dh = ga_diag['degree_histogram']
degs = sorted([int(k) for k in dh.keys()])
counts = [dh[str(d)] for d in degs]
N = ga_diag['graph']['N_final']
ax1.bar(degs, [c/N for c in counts], color='steelblue', edgecolor='navy', linewidth=0.3)
ax1.set_xlabel('Degree $k$')
ax1.set_ylabel('Fraction of nodes')
ax1.set_title('(a) Degree distribution')
mean_k = ga_diag['graph']['mean_degree']
ax1.axvline(mean_k, color='red', ls='--', lw=0.5, label=f'$\\langle k \\rangle = {mean_k:.1f}$')
ax1.legend(frameon=False)

# (b) Curvature distribution
curv_vals = ga_diag['curvature']['values']
ax2.hist(curv_vals, bins=30, density=True, color='coral', edgecolor='darkred', linewidth=0.3)
ax2.axvline(0, color='gray', ls='--', lw=0.5)
mean_curv = ga_diag['curvature']['mean']
ax2.axvline(mean_curv, color='blue', ls='-', lw=0.8,
            label=f'$\\langle \\kappa \\rangle = {mean_curv:.3f}$')
ax2.set_xlabel('Ollivier-Ricci curvature $\\kappa$')
ax2.set_ylabel('Density')
ax2.set_title('(b) Curvature distribution')
ax2.legend(frameon=False)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig2_distributions.pdf')
fig.savefig(f'{OUTDIR}/fig2_distributions.png')
plt.close()


# ═══════════════════════════════════════════════════
# Fig 3: Spectral dimension fit (log-log)
# ═══════════════════════════════════════════════════
print("Generating Fig 3: spectral dimension...")

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

log_t = np.array(ga_diag['spectral']['log_time'])
log_p = np.array(ga_diag['spectral']['log_prob'])

ax.plot(log_t, log_p, 'b.', ms=2, alpha=0.6, label='Data')

# Fit line (from d_s = -2 * slope)
d_s = ga_diag['spectral']['dimension']
# Simple fit on the middle points
n_pts = len(log_t)
i_start = int(n_pts * 0.05)
i_end = int(n_pts * 0.5)
if i_end > i_start + 2:
    coeffs = np.polyfit(log_t[i_start:i_end], log_p[i_start:i_end], 1)
    fit_x = np.linspace(log_t[i_start], log_t[i_end], 50)
    ax.plot(fit_x, np.polyval(coeffs, fit_x), 'r-', lw=1,
            label=f'Fit: $d_s = {d_s:.2f}$')

ax.set_xlabel('$\\ln(t)$')
ax.set_ylabel('$\\ln P(t)$')
ax.set_title('Spectral dimension fit')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig3_spectral_dim.pdf')
fig.savefig(f'{OUTDIR}/fig3_spectral_dim.png')
plt.close()


# ═══════════════════════════════════════════════════
# Fig 4: Volume growth + Hausdorff dimension
# ═══════════════════════════════════════════════════
print("Generating Fig 4: volume growth...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.5))

# (a) Volume growth for different sizes
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(scaling)))
for i, (s, d) in enumerate(sorted(scaling.items())):
    if 'geodesic' in d:
        vg = d['geodesic']['volume_growth']
        r = np.arange(len(vg))
        ax1.plot(r[1:], vg[1:], 'o-', color=colors[i], ms=3,
                 label=f'$L={s}$ ($N={d["graph"]["N_final"]}$)')

# Reference curves
r_ref = np.arange(1, 8)
ax1.plot(r_ref, r_ref**3, 'k--', lw=0.5, label='$r^3$')
ax1.plot(r_ref, 5*r_ref**2.5, 'k:', lw=0.5, label='$r^{2.5}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Geodesic radius $r$')
ax1.set_ylabel('$N(r)$')
ax1.set_title('(a) Volume growth')
ax1.legend(frameon=False, fontsize=6, ncol=2)

# (b) Finite-size scaling of d_H
sides_with_geo = sorted([s for s in scaling if 'geodesic' in scaling[s]])
N_geo = [scaling[s]['graph']['N_final'] for s in sides_with_geo]
dH_vals = [scaling[s]['geodesic']['hausdorff_dimension'] for s in sides_with_geo]

ax2.plot(N_geo, dH_vals, 'rs-', ms=5, label='$d_H$ (measured)')
ax2.axhline(3.0, color='gray', ls='--', lw=0.5, label='$d=3$ (target)')
ax2.set_xlabel('$N$ (nodes)')
ax2.set_ylabel('Hausdorff dimension $d_H$')
ax2.set_title('(b) Finite-size scaling')
ax2.legend(frameon=False)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig4_volume_growth.pdf')
fig.savefig(f'{OUTDIR}/fig4_volume_growth.png')
plt.close()


# ═══════════════════════════════════════════════════
# Fig 5: Finite-size scaling multi-panel
# ═══════════════════════════════════════════════════
print("Generating Fig 5: finite-size scaling...")

fig = plt.figure(figsize=(DBL_WIDTH, 5))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

sides_sorted = sorted(scaling.keys())
N_all = [scaling[s]['graph']['N_final'] for s in sides_sorted]

# (a) d_s vs N
ax = fig.add_subplot(gs[0, 0])
ds_vals = [scaling[s]['spectral']['dimension'] for s in sides_sorted]
ax.plot(N_all, ds_vals, 'bo-', ms=5)
ax.axhline(3.0, color='gray', ls='--', lw=0.5)
ax.set_xlabel('$N$')
ax.set_ylabel('$d_s$')
ax.set_title('(a) Spectral dimension')

# (b) d_H vs N
ax = fig.add_subplot(gs[0, 1])
sides_geo = [s for s in sides_sorted if 'geodesic' in scaling[s]]
N_geo = [scaling[s]['graph']['N_final'] for s in sides_geo]
dH = [scaling[s]['geodesic']['hausdorff_dimension'] for s in sides_geo]
ax.plot(N_geo, dH, 'rs-', ms=5)
ax.axhline(3.0, color='gray', ls='--', lw=0.5)
ax.set_xlabel('$N$')
ax.set_ylabel('$d_H$')
ax.set_title('(b) Hausdorff dimension')

# (c) <κ> vs N
ax = fig.add_subplot(gs[1, 0])
kappa_vals = [scaling[s]['curvature']['mean'] for s in sides_sorted]
ax.plot(N_all, kappa_vals, 'g^-', ms=5)
ax.axhline(0.0, color='gray', ls='--', lw=0.5)
ax.set_xlabel('$N$')
ax.set_ylabel('$\\langle \\kappa \\rangle$')
ax.set_title('(c) Mean Ollivier-Ricci curvature')

# (d) Scalar curvature vs N
ax = fig.add_subplot(gs[1, 1])
sides_metric = [s for s in sides_sorted if 'metric' in scaling[s]]
N_met = [scaling[s]['graph']['N_final'] for s in sides_metric]
R_vals = [scaling[s]['metric']['scalar_curvature'] for s in sides_metric]
ax.plot(N_met, R_vals, 'mD-', ms=5)
ax.axhline(0.0, color='gray', ls='--', lw=0.5)
ax.set_xlabel('$N$')
ax.set_ylabel('Scalar curvature $R$')
ax.set_title('(d) MDS scalar curvature')

fig.savefig(f'{OUTDIR}/fig5_scaling.pdf')
fig.savefig(f'{OUTDIR}/fig5_scaling.png')
plt.close()


# ═══════════════════════════════════════════════════
# Fig 6: Fitness component breakdown (GA vs CMA-ES)
# ═══════════════════════════════════════════════════
print("Generating Fig 6: fitness breakdown...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.5))

for ax, diag, title in [(ax1, ga_diag, 'GA'), (ax2, cmaes_diag, 'CMA-ES')]:
    f = diag['fitness']
    components = {
        '$F_{\\mathrm{Ricci}}$': f['ricci_term'],
        '$F_{\\mathrm{dim}}$': f['dimension_term'],
        '$F_{\\mathrm{conn}}$': f['connectivity_term'],
        '$F_{\\mathrm{deg}}$': f['degree_reg_term'],
        '$F_{\\mathrm{size}}$': f['size_term'],
    }
    names = list(components.keys())
    vals = list(components.values())
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax.barh(names, vals, color=colors_bar, edgecolor='black', linewidth=0.3)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Fitness contribution')
    ax.set_title(f'{title} (total = {f["total"]:.3f})')

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() - 0.01 if v < 0 else bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center',
                ha='right' if v < 0 else 'left', fontsize=7)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig6_fitness_breakdown.pdf')
fig.savefig(f'{OUTDIR}/fig6_fitness_breakdown.png')
plt.close()


# ═══════════════════════════════════════════════════
# Fig 7: Distance distribution
# ═══════════════════════════════════════════════════
print("Generating Fig 7: distance distribution...")

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

if 'geodesic' in ga_diag:
    dd = ga_diag['geodesic']['distance_distribution']
    d_vals = list(range(len(dd)))
    total_pairs = sum(dd)
    ax.bar(d_vals[1:], [x/total_pairs for x in dd[1:]], color='steelblue',
           edgecolor='navy', linewidth=0.3)
    ax.set_xlabel('Geodesic distance $d$')
    ax.set_ylabel('Fraction of node pairs')
    ax.set_title('Distance distribution (GA best rule, $N=1000$)')
    avg_d = ga_diag['geodesic']['avg_path_length']
    ax.axvline(avg_d, color='red', ls='--', lw=0.8,
               label=f'$\\langle d \\rangle = {avg_d:.2f}$')
    ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig7_distance_dist.pdf')
fig.savefig(f'{OUTDIR}/fig7_distance_dist.png')
plt.close()


# ═══════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("         SUMMARY TABLE — Campaign 2 Results")
print("="*60)
print(f"{'Metric':<25} {'GA':>12} {'CMA-ES':>12}")
print("-"*49)

metrics = [
    ('N_final', 'graph', 'N_final'),
    ('<k>', 'graph', 'mean_degree'),
    ('CV(k)', 'graph', 'cv_degree'),
    ('<κ>', 'curvature', 'mean'),
    ('d_s', 'spectral', 'dimension'),
]
for label, section, key in metrics:
    ga_val = ga_diag[section][key]
    cmaes_val = cmaes_diag[section][key]
    print(f"{label:<25} {ga_val:>12.4f} {cmaes_val:>12.4f}")

# Geodesic
if 'geodesic' in ga_diag and 'geodesic' in cmaes_diag:
    for label, key in [('Diameter', 'diameter'), ('<path>', 'avg_path_length'), ('d_H', 'hausdorff_dimension')]:
        print(f"{label:<25} {ga_diag['geodesic'][key]:>12.4f} {cmaes_diag['geodesic'][key]:>12.4f}")

# Metric
if 'metric' in ga_diag and 'metric' in cmaes_diag:
    print(f"{'Scalar curvature R':<25} {ga_diag['metric']['scalar_curvature']:>12.6f} {cmaes_diag['metric']['scalar_curvature']:>12.6f}")

print(f"{'F_total':<25} {ga_diag['fitness']['total']:>12.4f} {cmaes_diag['fitness']['total']:>12.4f}")
print("="*49)

# Finite-size scaling table
print("\n" + "="*70)
print("         FINITE-SIZE SCALING — GA Best Rule")
print("="*70)
print(f"{'L':>4} {'N':>6} {'<k>':>8} {'d_s':>8} {'d_H':>8} {'<κ>':>8} {'R':>8}")
print("-"*54)
for s in sorted(scaling.keys()):
    d = scaling[s]
    N = d['graph']['N_final']
    k = d['graph']['mean_degree']
    ds = d['spectral']['dimension']
    dH = d.get('geodesic', {}).get('hausdorff_dimension', float('nan'))
    kap = d['curvature']['mean']
    R = d.get('metric', {}).get('scalar_curvature', float('nan'))
    print(f"{s:>4} {N:>6} {k:>8.2f} {ds:>8.3f} {dH:>8.3f} {kap:>8.4f} {R:>8.4f}")
print("="*54)

print(f"\nAll figures saved to {OUTDIR}/")
print("  fig1_convergence.pdf    — GA vs CMA-ES convergence")
print("  fig2_distributions.pdf  — Degree + curvature distributions")
print("  fig3_spectral_dim.pdf   — Spectral dimension log-log fit")
print("  fig4_volume_growth.pdf  — Volume growth + d_H scaling")
print("  fig5_scaling.pdf        — 4-panel finite-size scaling")
print("  fig6_fitness_breakdown.pdf — Fitness component comparison")
print("  fig7_distance_dist.pdf  — Geodesic distance distribution")
