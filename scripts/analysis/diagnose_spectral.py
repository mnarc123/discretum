#!/usr/bin/env python3
"""Spectral dimension diagnostic: visualise d_eff(t) and P(t) from baseline JSON.

Usage:
    python scripts/analysis/diagnose_spectral.py data/results/baseline_4d_L5.json
"""
import json, sys, os
import numpy as np

def load_data(path):
    with open(path) as f:
        return json.load(f)

def plot_spectral_diagnostics(data, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, printing text summary only")
        return

    sd = data["spectral_v2"]
    t = np.array(sd["time_pts"])
    P = np.array(sd["P_t"])
    d_eff = np.array(sd["d_eff_t"])

    os.makedirs(outdir, exist_ok=True)

    # Fig 1: log P(t) vs log t
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(np.log(t), np.log(P), 'b-', lw=0.8, alpha=0.7)
    if sd["has_plateau"]:
        mask = (t >= sd["plateau_t_min"]) & (t <= sd["plateau_t_max"])
        ax.plot(np.log(t[mask]), np.log(P[mask]), 'r-', lw=2, label=f'plateau d_s={sd["d_s"]:.2f}')
        # Show fit line
        slope = -sd["d_s"] / 2.0
        lt_mid = np.log(t[mask])
        intercept = np.log(P[mask]).mean() - slope * lt_mid.mean()
        ax.plot(lt_mid, slope * lt_mid + intercept, 'k--', lw=1, label=f'slope={slope:.3f}')
    ax.set_xlabel('log t')
    ax.set_ylabel('log P(t)')
    ax.set_title(f'Return probability (N={data["N"]})')
    ax.legend(fontsize=8)

    # Fig 2: d_eff(t) vs t
    ax = axes[1]
    ax.plot(t, d_eff, 'b-', lw=0.8, alpha=0.7)
    if sd["has_plateau"]:
        mask = (t >= sd["plateau_t_min"]) & (t <= sd["plateau_t_max"])
        ax.axhspan(sd["d_s"] - sd["d_s_error"], sd["d_s"] + sd["d_s_error"],
                    alpha=0.2, color='red', label=f'd_s={sd["d_s"]:.2f}±{sd["d_s_error"]:.2f}')
        ax.axhline(sd["d_s"], color='red', ls='--', lw=1)
        ax.axvline(sd["plateau_t_min"], color='gray', ls=':', lw=0.8)
        ax.axvline(sd["plateau_t_max"], color='gray', ls=':', lw=0.8)
    ax.set_xlabel('t')
    ax.set_ylabel('d_eff(t)')
    ax.set_title(f'd_eff(t) = -2 d(log P)/d(log t)')
    ax.set_ylim(-1, 8)
    ax.legend(fontsize=8)

    # Fig 3: d_eff(t) vs log t (scaling view)
    ax = axes[2]
    ax.plot(np.log(t), d_eff, 'b-', lw=0.8, alpha=0.7)
    target_dim = data.get("baseline", {}).get("d_H", 4.0)
    ax.axhline(target_dim, color='green', ls='--', lw=1, alpha=0.5,
               label=f'd_H={data["baseline"]["d_H"]:.2f}')
    if sd["has_plateau"]:
        ax.axhline(sd["d_s"], color='red', ls='--', lw=1, label=f'd_s={sd["d_s"]:.2f}')
    ax.set_xlabel('log t')
    ax.set_ylabel('d_eff(t)')
    ax.set_title('Scaling regime identification')
    ax.set_ylim(-1, 8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    figpath = os.path.join(outdir, 'spectral_diagnostic.png')
    plt.savefig(figpath, dpi=150)
    plt.close()
    print(f"Saved {figpath}")

def print_summary(data):
    bm = data.get("baseline", {})
    sd = data["spectral_v2"]
    print(f"\n{'='*50}")
    print(f"Spectral Dimension Diagnostic")
    print(f"{'='*50}")
    print(f"  Lattice: {data['lattice']} L={data['L']} (N={data['N']}, E={data['E']})")
    print(f"  Config: walkers={data['config']['walkers']}, steps={data['config']['steps']}")
    print(f"\n  Baseline metrics:")
    print(f"    d_H            = {bm['d_H']:.4f}")
    print(f"    d_s (v2)       = {sd['d_s']:.4f} ± {sd['d_s_error']:.4f}")
    print(f"    has_plateau    = {sd['has_plateau']}")
    print(f"    plateau range  = [{sd['plateau_t_min']:.0f}, {sd['plateau_t_max']:.0f}]")
    print(f"    global fit     = {sd['global_fit']:.4f} (R²={sd['global_fit_r2']:.4f})")
    print(f"    <kappa>        = {bm['mean_curvature']:.6f}")
    print(f"    sigma(kappa)   = {bm['std_curvature']:.6f}")
    print(f"    <degree>       = {bm['mean_degree']:.2f}")
    print(f"\n  P(t) data points = {len(sd['P_t'])}")
    if sd['has_plateau']:
        print(f"  ✓ Plateau found: d_s = {sd['d_s']:.3f} in t ∈ [{sd['plateau_t_min']:.0f}, {sd['plateau_t_max']:.0f}]")
    else:
        print(f"  ✗ No plateau found, using global fit d_s = {sd['global_fit']:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data = load_data(sys.argv[1])
    print_summary(data)

    outdir = os.path.join(os.path.dirname(sys.argv[1]), "figures")
    plot_spectral_diagnostics(data, outdir)
