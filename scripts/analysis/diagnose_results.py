#!/usr/bin/env python3
"""
DISCRETUM — Diagnostic visualization for search results.

Reads JSON output from discretum_diagnose and generates publication-quality
figures (Physical Review style) showing:
  1. Node count evolution (size stability)
  2. Degree distribution
  3. Curvature distribution (if available)
  4. Spectral dimension fit (if available)
  5. Fitness component breakdown (bar chart)

Usage:
    python3 scripts/analysis/diagnose_results.py data/results/diagnostics/cmaes_diag.json
    python3 scripts/analysis/diagnose_results.py cmaes_diag.json ga_diag.json --compare
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Physical Review style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (3.375, 2.5),  # single-column PRL width
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
})


def load_diag(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_node_history(diag: dict, outdir: Path, label: str = ""):
    """Plot node and edge count over measurement steps."""
    node_hist = diag.get("stability", {}).get("node_history", [])
    edge_hist = diag.get("stability", {}).get("edge_history", [])
    if not node_hist:
        print("  [skip] No node_history in diagnostics")
        return

    fig, ax1 = plt.subplots()
    steps = np.arange(len(node_hist))
    ax1.plot(steps, node_hist, 'b-', label='Nodes')
    ax1.set_xlabel('Measurement step')
    ax1.set_ylabel('Node count', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_yscale('log')

    if edge_hist:
        ax2 = ax1.twinx()
        ax2.plot(steps, edge_hist, 'r--', alpha=0.7, label='Edges')
        ax2.set_ylabel('Edge count', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')

    N0 = diag.get("graph", {}).get("N_initial", None)
    if N0:
        ax1.axhline(N0, color='gray', ls=':', lw=0.5, label=f'$N_0={N0}$')

    ax1.legend(loc='upper left', frameon=False)
    title = f"Size evolution{' — ' + label if label else ''}"
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / f"node_history{'_' + label if label else ''}.png")
    plt.close(fig)
    print(f"  -> {outdir / f'node_history_{label}.png'}")


def plot_degree_distribution(diag: dict, outdir: Path, label: str = ""):
    """Plot degree histogram."""
    deg_hist = diag.get("degree_histogram", {})
    if not deg_hist:
        print("  [skip] No degree_histogram in diagnostics")
        return

    degrees = sorted([int(k) for k in deg_hist.keys()])
    counts = [deg_hist[str(d)] for d in degrees]
    total = sum(counts)
    fracs = [c / total for c in counts]

    fig, ax = plt.subplots()
    ax.bar(degrees, fracs, width=0.8, color='steelblue', edgecolor='navy', linewidth=0.3)
    ax.set_xlabel('Degree $k$')
    ax.set_ylabel('$P(k)$')
    mean_deg = diag.get("graph", {}).get("mean_degree", None)
    if mean_deg:
        ax.axvline(mean_deg, color='red', ls='--', lw=0.7, label=f'$\\langle k \\rangle={mean_deg:.1f}$')
        ax.legend(frameon=False)
    title = f"Degree distribution{' — ' + label if label else ''}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / f"degree_dist{'_' + label if label else ''}.png")
    plt.close(fig)
    print(f"  -> {outdir / f'degree_dist_{label}.png'}")


def plot_curvature_distribution(diag: dict, outdir: Path, label: str = ""):
    """Plot curvature histogram from per-edge values."""
    curv_values = diag.get("curvature", {}).get("values", [])
    if not curv_values:
        curv_hist = diag.get("curvature_histogram", [])
        if not curv_hist:
            print("  [skip] No curvature data in diagnostics")
            return
        # Use histogram bins
        fig, ax = plt.subplots()
        centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in curv_hist]
        widths = [b["bin_hi"] - b["bin_lo"] for b in curv_hist]
        counts = [b["count"] for b in curv_hist]
        ax.bar(centers, counts, width=widths[0] if widths else 0.01,
               color='darkorange', edgecolor='brown', linewidth=0.3)
        ax.set_xlabel(r'Ollivier-Ricci curvature $\kappa$')
        ax.set_ylabel('Count')
        title = f"Curvature distribution{' — ' + label if label else ''}"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outdir / f"curvature_dist{'_' + label if label else ''}.png")
        plt.close(fig)
        print(f"  -> {outdir / f'curvature_dist_{label}.png'}")
        return

    fig, ax = plt.subplots()
    ax.hist(curv_values, bins=30, color='darkorange', edgecolor='brown', linewidth=0.3, density=True)
    mean_c = diag.get("curvature", {}).get("mean", np.mean(curv_values))
    ax.axvline(mean_c, color='red', ls='--', lw=0.7,
               label=f'$\\langle\\kappa\\rangle={mean_c:.4f}$')
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel(r'Ollivier-Ricci curvature $\kappa$')
    ax.set_ylabel('Density')
    ax.legend(frameon=False)
    title = f"Curvature distribution{' — ' + label if label else ''}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / f"curvature_dist{'_' + label if label else ''}.png")
    plt.close(fig)
    print(f"  -> {outdir / f'curvature_dist_{label}.png'}")


def plot_spectral_fit(diag: dict, outdir: Path, label: str = ""):
    """Plot spectral dimension log-log fit."""
    spec = diag.get("spectral", {})
    log_t = spec.get("log_time", [])
    log_p = spec.get("log_prob", [])
    dim = spec.get("dimension", 0)
    if not log_t or dim == 0:
        print("  [skip] No spectral dimension data in diagnostics")
        return

    fig, ax = plt.subplots()
    ax.plot(log_t, log_p, 'ko', ms=2, label='Data')

    # Fit line
    log_t_arr = np.array(log_t)
    log_p_arr = np.array(log_p)
    mask = np.isfinite(log_t_arr) & np.isfinite(log_p_arr)
    if mask.sum() > 2:
        coeffs = np.polyfit(log_t_arr[mask], log_p_arr[mask], 1)
        fit_x = np.linspace(log_t_arr[mask].min(), log_t_arr[mask].max(), 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(fit_x, fit_y, 'r-', lw=0.8,
                label=f'$d_s = {-2*coeffs[0]:.2f}$')

    ax.set_xlabel(r'$\ln t$')
    ax.set_ylabel(r'$\ln P(t)$')
    ax.legend(frameon=False)
    title = f"Spectral dimension{' — ' + label if label else ''}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / f"spectral_dim{'_' + label if label else ''}.png")
    plt.close(fig)
    print(f"  -> {outdir / f'spectral_dim_{label}.png'}")


def plot_fitness_breakdown(diag: dict, outdir: Path, label: str = ""):
    """Bar chart of fitness components."""
    fit = diag.get("fitness", {})
    if not fit:
        print("  [skip] No fitness data in diagnostics")
        return

    components = {
        '$F_{\\mathrm{Ricci}}$': fit.get("ricci_term", 0),
        '$F_{\\mathrm{dim}}$': fit.get("dimension_term", 0),
        '$F_{\\mathrm{conn}}$': fit.get("connectivity_term", 0),
        '$F_{\\mathrm{deg}}$': fit.get("degree_reg_term", 0),
        '$F_{\\mathrm{size}}$': fit.get("size_term", 0),
    }

    names = list(components.keys())
    values = list(components.values())

    fig, ax = plt.subplots(figsize=(4, 2.5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Fitness contribution')
    ax.axvline(0, color='black', lw=0.5)

    # Annotate values
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        ha = 'left' if x_pos < 0 else 'right'
        offset = -0.02 if x_pos < 0 else 0.02
        if abs(val) > 0.01:
            ax.text(x_pos + offset * abs(min(values)), bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', ha=ha, fontsize=7)

    total = fit.get("total", sum(values))
    ax.set_title(f"Fitness breakdown ($F_{{\\mathrm{{total}}}}={total:.2f}$)"
                 + (f" — {label}" if label else ""))
    fig.tight_layout()
    fig.savefig(outdir / f"fitness_breakdown{'_' + label if label else ''}.png")
    plt.close(fig)
    print(f"  -> {outdir / f'fitness_breakdown_{label}.png'}")


def plot_comparison(diags: list, labels: list, outdir: Path):
    """Side-by-side comparison of multiple diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Panel 1: Node count history
    ax = axes[0]
    for diag, label in zip(diags, labels):
        hist = diag.get("stability", {}).get("node_history", [])
        if hist:
            ax.plot(hist, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('$N$')
    ax.set_yscale('log')
    ax.legend(frameon=False)
    ax.set_title('Size evolution')

    # Panel 2: Fitness breakdown comparison
    ax = axes[1]
    comp_names = ['ricci_term', 'dimension_term', 'connectivity_term', 'degree_reg_term', 'size_term']
    short_names = ['Ricci', 'Dim', 'Conn', 'Deg', 'Size']
    x = np.arange(len(comp_names))
    width = 0.8 / len(diags)
    for i, (diag, label) in enumerate(zip(diags, labels)):
        fit = diag.get("fitness", {})
        vals = [fit.get(c, 0) for c in comp_names]
        ax.bar(x + i * width, vals, width, label=label, alpha=0.8)
    ax.set_xticks(x + width * (len(diags) - 1) / 2)
    ax.set_xticklabels(short_names, rotation=30, ha='right')
    ax.set_ylabel('Fitness term')
    ax.legend(frameon=False)
    ax.set_title('Fitness components')

    # Panel 3: Key metrics table
    ax = axes[2]
    ax.axis('off')
    rows = []
    for diag, label in zip(diags, labels):
        g = diag.get("graph", {})
        s = diag.get("stability", {})
        f = diag.get("fitness", {})
        rows.append([
            label,
            f"{g.get('N_final', '?')}",
            f"{g.get('num_components', '?')}",
            f"{g.get('mean_degree', 0):.1f}",
            f"{s.get('var_over_mean_sq', 0):.3f}",
            f"{f.get('total', 0):.2f}",
        ])
    col_labels = ['', '$N_f$', '#comp', '$\\langle k\\rangle$', 'Stab', '$F_{tot}$']
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title('Summary')

    fig.tight_layout()
    fig.savefig(outdir / "comparison.png")
    plt.close(fig)
    print(f"  -> {outdir / 'comparison.png'}")


def process_single(path: str, outdir: Path, label: str = ""):
    """Generate all plots for a single diagnostic JSON."""
    print(f"\nProcessing: {path}")
    diag = load_diag(path)

    if not label:
        label = Path(path).stem.replace("_diag", "")

    plot_node_history(diag, outdir, label)
    plot_degree_distribution(diag, outdir, label)
    plot_curvature_distribution(diag, outdir, label)
    plot_spectral_fit(diag, outdir, label)
    plot_fitness_breakdown(diag, outdir, label)


def main():
    parser = argparse.ArgumentParser(description='DISCRETUM diagnostic visualization')
    parser.add_argument('files', nargs='+', help='Diagnostic JSON file(s)')
    parser.add_argument('--outdir', '-o', default='data/results/diagnostics/figures',
                        help='Output directory for figures')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison plot of multiple diagnostics')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for f in args.files:
        process_single(f, outdir)

    if args.compare and len(args.files) > 1:
        diags = [load_diag(f) for f in args.files]
        labels = [Path(f).stem.replace("_diag", "") for f in args.files]
        plot_comparison(diags, labels, outdir)

    print(f"\nAll figures saved to {outdir}/")


if __name__ == '__main__':
    main()
