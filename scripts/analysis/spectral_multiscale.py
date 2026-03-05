#!/usr/bin/env python3
"""Multiscale spectral dimension analysis: d_s(t) vs walk scale.

Loads diagnostics JSON outputs and plots d_eff(t) curves at different lattice
sizes to identify the scaling regime and finite-size crossover.

Usage:
    python scripts/analysis/spectral_multiscale.py data/results/baseline_4d_L5.json [more_files...]
    python scripts/analysis/spectral_multiscale.py --dir data/results/scaling/
"""
import json, sys, os, glob
import numpy as np

def load_spectral_data(path):
    with open(path) as f:
        d = json.load(f)
    sd = d.get("spectral_v2", d.get("spectral", {}))
    return {
        "path": path,
        "N": d.get("N", d.get("graph", {}).get("N_final", 0)),
        "L": d.get("L", 0),
        "lattice": d.get("lattice", "unknown"),
        "d_s": sd.get("d_s", sd.get("dimension", 0)),
        "d_s_error": sd.get("d_s_error", sd.get("fit_error", 0)),
        "has_plateau": sd.get("has_plateau", False),
        "plateau_t_min": sd.get("plateau_t_min", 0),
        "plateau_t_max": sd.get("plateau_t_max", 0),
        "time_pts": np.array(sd.get("time_pts", [])),
        "d_eff_t": np.array(sd.get("d_eff_t", [])),
        "P_t": np.array(sd.get("P_t", [])),
        "d_H": d.get("baseline", {}).get("d_H", d.get("geodesic", {}).get("hausdorff_dimension", 0)),
    }

def print_table(datasets):
    print(f"\n{'N':>6s} {'L':>3s} {'d_s':>8s} {'±':>7s} {'plateau':>8s} {'range':>12s} {'d_H':>6s}")
    print("-" * 55)
    for ds in sorted(datasets, key=lambda x: x["N"]):
        prange = f"[{ds['plateau_t_min']:.0f},{ds['plateau_t_max']:.0f}]" if ds["has_plateau"] else "none"
        print(f"{ds['N']:6d} {ds['L']:3d} {ds['d_s']:8.3f} {ds['d_s_error']:7.3f} "
              f"{'YES' if ds['has_plateau'] else 'NO':>8s} {prange:>12s} {ds['d_H']:6.3f}")

def plot_multiscale(datasets, outdir, target_dim=4.0):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.cm import viridis
    except ImportError:
        print("matplotlib not available")
        return

    os.makedirs(outdir, exist_ok=True)
    datasets = sorted(datasets, key=lambda x: x["N"])
    colors = viridis(np.linspace(0.2, 0.9, len(datasets)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: d_eff(t) vs t for all sizes
    ax = axes[0]
    for ds, c in zip(datasets, colors):
        t, de = ds["time_pts"], ds["d_eff_t"]
        if len(t) == 0:
            continue
        label = f"N={ds['N']}"
        ax.plot(t, de, color=c, lw=1.2, alpha=0.8, label=label)
        if ds["has_plateau"]:
            ax.axhline(ds["d_s"], color=c, ls='--', lw=0.6, alpha=0.5)
    ax.axhline(target_dim, color='red', ls=':', lw=1.5, alpha=0.5, label=f'target={target_dim}')
    ax.set_xlabel('t (walk steps)')
    ax.set_ylabel('d_eff(t)')
    ax.set_title('Effective spectral dimension')
    ax.set_ylim(-0.5, target_dim * 2)
    ax.legend(fontsize=7, ncol=2)

    # Panel 2: d_eff(t) vs log(t)
    ax = axes[1]
    for ds, c in zip(datasets, colors):
        t, de = ds["time_pts"], ds["d_eff_t"]
        if len(t) == 0:
            continue
        ax.plot(np.log(t), de, color=c, lw=1.2, alpha=0.8, label=f"N={ds['N']}")
    ax.axhline(target_dim, color='red', ls=':', lw=1.5, alpha=0.5)
    ax.set_xlabel('log(t)')
    ax.set_ylabel('d_eff(t)')
    ax.set_title('Scaling regime (log scale)')
    ax.set_ylim(-0.5, target_dim * 2)
    ax.legend(fontsize=7, ncol=2)

    # Panel 3: d_s vs N (convergence)
    ax = axes[2]
    Ns = [ds["N"] for ds in datasets if ds["d_s"] > 0]
    ds_vals = [ds["d_s"] for ds in datasets if ds["d_s"] > 0]
    ds_errs = [ds["d_s_error"] for ds in datasets if ds["d_s"] > 0]
    dH_vals = [ds["d_H"] for ds in datasets if ds["d_s"] > 0]
    ax.errorbar(Ns, ds_vals, yerr=ds_errs, fmt='o-', color='blue', label='d_s (v2)')
    ax.plot(Ns, dH_vals, 's--', color='green', alpha=0.7, label='d_H')
    ax.axhline(target_dim, color='red', ls=':', lw=1.5, alpha=0.5, label=f'target={target_dim}')
    ax.set_xlabel('N (graph size)')
    ax.set_ylabel('dimension')
    ax.set_title('Convergence with system size')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    plt.tight_layout()
    figpath = os.path.join(outdir, 'spectral_multiscale.png')
    plt.savefig(figpath, dpi=150)
    plt.close()
    print(f"Saved {figpath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    files = []
    for arg in sys.argv[1:]:
        if arg == "--dir":
            continue
        if os.path.isdir(arg):
            files.extend(sorted(glob.glob(os.path.join(arg, "*.json"))))
        else:
            files.append(arg)

    if not files:
        print("No JSON files found")
        sys.exit(1)

    datasets = []
    for path in files:
        try:
            ds = load_spectral_data(path)
            if ds["N"] > 0:
                datasets.append(ds)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    print_table(datasets)

    outdir = os.path.join(os.path.dirname(files[0]), "figures")
    target = 4.0
    if datasets and datasets[0]["lattice"].startswith("3"):
        target = 3.0
    plot_multiscale(datasets, outdir, target_dim=target)
