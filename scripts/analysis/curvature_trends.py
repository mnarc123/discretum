#!/usr/bin/env python3
"""
TASK 4.1: Curvature trend analysis — κ vs N with power-law and log fits.

Analyses:
  1. Bare lattice calibration: κ vs N for 3D and 4D lattices
  2. Campaign 2 evolved graphs: κ vs N from finite-size scaling
  3. Fit models: κ ~ N^α (power law) and κ ~ a·ln(N) + b (logarithmic)
  4. Extrapolate to continuum limit (N → ∞)
"""
import json
import numpy as np
from pathlib import Path
import sys

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

ROOT = Path(__file__).resolve().parent.parent.parent


def load_calibration(dim):
    """Load bare lattice calibration data."""
    cal_dir = ROOT / f"data/results/calibration_{dim}d"
    baseline = cal_dir / "baseline.json"
    if not baseline.exists():
        return None
    with open(baseline) as f:
        return json.load(f)


def load_campaign2_scaling():
    """Load Campaign 2 finite-size scaling data."""
    scaling_dir = ROOT / "data/results/campaign2/scaling"
    if not scaling_dir.exists():
        return None
    data = []
    for fp in sorted(scaling_dir.glob("ga_size_*.json")):
        with open(fp) as f:
            d = json.load(f)
        data.append({
            'L': d['config']['graph_side'],
            'N': d['graph']['N_final'],
            'mean_k': d['curvature']['mean'],
            'std_k': d['curvature']['std'],
            'd_s': d['spectral']['dimension'],
            'd_H': d.get('geodesic', {}).get('hausdorff_dimension', None),
            'mean_deg': d['graph']['mean_degree'],
            'cv_deg': d['graph']['cv_degree'],
        })
    return data


def fit_power_law(N_arr, k_arr):
    """Fit |κ| = a · N^α using log-log regression. Returns (a, α, R²)."""
    abs_k = np.abs(k_arr)
    mask = abs_k > 1e-10
    if mask.sum() < 2:
        return None, None, 0.0
    ln_N = np.log(N_arr[mask])
    ln_k = np.log(abs_k[mask])
    coeffs = np.polyfit(ln_N, ln_k, 1)
    alpha = coeffs[0]
    a = np.exp(coeffs[1])
    # R²
    pred = np.polyval(coeffs, ln_N)
    ss_res = np.sum((ln_k - pred)**2)
    ss_tot = np.sum((ln_k - ln_k.mean())**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, alpha, R2


def fit_logarithmic(N_arr, k_arr):
    """Fit κ = a·ln(N) + b. Returns (a, b, R²)."""
    if len(N_arr) < 2:
        return None, None, 0.0
    ln_N = np.log(N_arr)
    coeffs = np.polyfit(ln_N, k_arr, 1)
    a, b = coeffs
    pred = np.polyval(coeffs, ln_N)
    ss_res = np.sum((k_arr - pred)**2)
    ss_tot = np.sum((k_arr - k_arr.mean())**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, b, R2


def analyse_dataset(name, N_arr, k_arr, std_arr=None):
    """Analyse a single dataset: fit power law and logarithmic models."""
    N_arr = np.array(N_arr, dtype=float)
    k_arr = np.array(k_arr, dtype=float)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  {'N':>8}  {'<κ>':>10}  {'σ(κ)':>10}")
    print(f"  {'-'*32}")
    for i in range(len(N_arr)):
        s = f"  {N_arr[i]:>8.0f}  {k_arr[i]:>+10.6f}"
        if std_arr is not None:
            s += f"  {std_arr[i]:>10.6f}"
        print(s)

    # Power law fit on |κ|
    a_pw, alpha_pw, R2_pw = fit_power_law(N_arr, k_arr)
    if a_pw is not None:
        print(f"\n  Power law: |κ| = {a_pw:.4f} · N^({alpha_pw:+.4f})  (R² = {R2_pw:.4f})")
        if alpha_pw < -0.1:
            print(f"  → κ → 0 as N → ∞  (convergence rate ~ N^{alpha_pw:.2f})")
        elif alpha_pw > 0.1:
            print(f"  → |κ| DIVERGES as N → ∞  (⚠ possible issue)")
        else:
            print(f"  → κ roughly constant (weak N dependence)")

    # Logarithmic fit
    a_log, b_log, R2_log = fit_logarithmic(N_arr, k_arr)
    if a_log is not None:
        print(f"  Log fit:   κ = {a_log:+.6f}·ln(N) + {b_log:+.6f}  (R² = {R2_log:.4f})")
        if abs(a_log) < 1e-4:
            print(f"  → Flat (no log dependence)")
        elif a_log < 0:
            print(f"  → κ drifts negative with increasing N")
        else:
            print(f"  → κ drifts positive with increasing N")

    # Extrapolation
    print(f"\n  Extrapolation (power law):")
    for N_ext in [10000, 100000, 1e6]:
        if a_pw is not None:
            k_ext = a_pw * N_ext**alpha_pw
            sign = "-" if np.mean(k_arr) < 0 else "+"
            print(f"    N={N_ext:.0e}: |κ| ≈ {k_ext:.6f}")

    return {
        'name': name,
        'N': N_arr.tolist(),
        'mean_k': k_arr.tolist(),
        'power_law': {'a': a_pw, 'alpha': alpha_pw, 'R2': R2_pw},
        'log_fit': {'a': a_log, 'b': b_log, 'R2': R2_log},
    }


def main():
    out_dir = ROOT / "data/results/curvature_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. Bare 3D lattice calibration
    cal3 = load_calibration(3)
    if cal3:
        N_arr = [d['N'] for d in cal3]
        k_arr = [d['mean_curvature'] for d in cal3]
        s_arr = [d['std_curvature'] for d in cal3]
        r = analyse_dataset("Bare 3D Lattice (calibration)", N_arr, k_arr, s_arr)
        results.append(r)

    # 2. Bare 4D lattice calibration
    cal4 = load_calibration(4)
    if cal4:
        N_arr = [d['N'] for d in cal4]
        k_arr = [d['mean_curvature'] for d in cal4]
        s_arr = [d['std_curvature'] for d in cal4]
        r = analyse_dataset("Bare 4D Lattice (calibration)", N_arr, k_arr, s_arr)
        results.append(r)

    # 3. Campaign 2 evolved graphs (GA best rule)
    c2 = load_campaign2_scaling()
    if c2:
        N_arr = [d['N'] for d in c2]
        k_arr = [d['mean_k'] for d in c2]
        s_arr = [d['std_k'] for d in c2]
        r = analyse_dataset("Campaign 2 GA Best Rule (evolved)", N_arr, k_arr, s_arr)
        results.append(r)

    # Save combined results
    with open(out_dir / "curvature_trends.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'curvature_trends.json'}")

    # Generate plots if matplotlib available
    if HAS_PLT and results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: κ vs N
        ax = axes[0]
        for r in results:
            ax.plot(r['N'], r['mean_k'], 'o-', label=r['name'], markersize=5)
        ax.set_xlabel('N (nodes)')
        ax.set_ylabel('⟨κ⟩ (mean Ollivier-Ricci)')
        ax.set_title('Mean Curvature vs Graph Size')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: log|κ| vs log N
        ax = axes[1]
        for r in results:
            abs_k = np.abs(np.array(r['mean_k']))
            mask = abs_k > 1e-10
            N_plot = np.array(r['N'])[mask]
            k_plot = abs_k[mask]
            ax.plot(np.log(N_plot), np.log(k_plot), 'o-', label=r['name'], markersize=5)
            # Overlay power law fit
            pw = r['power_law']
            if pw['a'] is not None:
                N_fit = np.linspace(np.log(min(N_plot)), np.log(max(N_plot)), 50)
                k_fit = np.log(pw['a']) + pw['alpha'] * N_fit
                ax.plot(N_fit, k_fit, '--', alpha=0.5,
                        label=f"  α={pw['alpha']:.3f}, R²={pw['R2']:.3f}")
        ax.set_xlabel('ln(N)')
        ax.set_ylabel('ln|⟨κ⟩|')
        ax.set_title('Power-Law Fit: |κ| ~ N^α')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "curvature_trends.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}")
        plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
