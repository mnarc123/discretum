#!/usr/bin/env python3
"""
TASK 5.1-5.2: Finite-size scaling with error bars and extrapolation.

Given ensemble diagnostics at multiple system sizes L, fits:
  d_H(N) = d_H(∞) + a·N^(-1/d_H)     (leading correction to scaling)
  d_s(N) = d_s(∞) + b·N^(-1/d_s)
  κ(N)   = c·N^α                        (power law decay)

Produces tables and plots with error bars from ensemble data.
"""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

ROOT = Path(__file__).resolve().parent.parent.parent


def load_ensemble_data(data_dir):
    """Load ensemble JSON files from a directory. Returns list of dicts."""
    data_dir = Path(data_dir)
    results = []
    for fp in sorted(data_dir.glob("*.json")):
        with open(fp) as f:
            d = json.load(f)
        if 'd_H' in d and isinstance(d['d_H'], dict):
            results.append(d)
    return results


def load_single_run_scaling(scaling_dir):
    """Load Campaign 2 style single-run scaling data."""
    scaling_dir = Path(scaling_dir)
    data = []
    for fp in sorted(scaling_dir.glob("ga_size_*.json")):
        with open(fp) as f:
            d = json.load(f)
        entry = {
            'L': d['config']['graph_side'],
            'N': d['graph']['N_final'],
            'd_H': {'mean': d.get('geodesic', {}).get('hausdorff_dimension', 0), 'std_err': 0},
            'd_s': {'mean': d['spectral']['dimension'], 'std_err': d['spectral'].get('fit_error', 0)},
            'mean_curvature': {'mean': d['curvature']['mean'], 'std_err': 0},
            'std_curvature': {'mean': d['curvature']['std'], 'std_err': 0},
            'mean_degree': {'mean': d['graph']['mean_degree'], 'std_err': 0},
            'cv_degree': {'mean': d['graph']['cv_degree'], 'std_err': 0},
            'n_final': {'mean': d['graph']['N_final'], 'std_err': 0},
        }
        data.append(entry)
    return data


def load_calibration_as_scaling(dim):
    """Load bare lattice calibration as scaling data."""
    cal_file = ROOT / f"data/results/calibration_{dim}d/baseline.json"
    if not cal_file.exists():
        return []
    with open(cal_file) as f:
        cal = json.load(f)
    data = []
    for d in cal:
        entry = {
            'L': d['L'],
            'N': d['N'],
            'd_H': {'mean': d['d_H'], 'std_err': 0},
            'd_s': {'mean': d['d_s'], 'std_err': d.get('d_s_error', 0)},
            'mean_curvature': {'mean': d['mean_curvature'], 'std_err': 0},
            'std_curvature': {'mean': d['std_curvature'], 'std_err': 0},
            'mean_degree': {'mean': d['avg_degree'], 'std_err': 0},
        }
        data.append(entry)
    return data


# Bootstrap
def bootstrap_extrapolation(N_arr, d_arr, d_err_arr=None, n_boot=2000, label="d_H"):
    """Bootstrap confidence interval for d_inf via correction-to-scaling."""
    N = np.array(N_arr, dtype=float)
    d = np.array(d_arr, dtype=float)
    if len(N) < 3:
        return None

    rng = np.random.default_rng(42)
    d_inf_samples = []

    for _ in range(n_boot):
        # Resample with replacement + add Gaussian noise from errors
        idx = rng.choice(len(N), size=len(N), replace=True)
        d_boot = d[idx].copy()
        if d_err_arr is not None:
            errs = np.array(d_err_arr, dtype=float)
            d_boot += rng.normal(0, np.maximum(errs[idx], 1e-6))

        try:
            p0 = [max(d_boot) * 1.2, -1.0, 0.3]
            bounds = ([0, -100, 0.01], [10, 100, 2.0])
            popt, _ = curve_fit(correction_to_scaling, N[idx], d_boot,
                                p0=p0, bounds=bounds, maxfev=5000)
            d_inf_samples.append(popt[0])
        except Exception:
            pass

    if len(d_inf_samples) < 100:
        print(f"  {label} bootstrap: only {len(d_inf_samples)}/{n_boot} succeeded")
        return None

    d_inf_samples = np.array(d_inf_samples)
    median = np.median(d_inf_samples)
    ci_lo = np.percentile(d_inf_samples, 2.5)
    ci_hi = np.percentile(d_inf_samples, 97.5)
    print(f"  {label} bootstrap (n={len(d_inf_samples)}): median={median:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
    return {'median': median, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n_success': len(d_inf_samples)}


# Fitting models
def correction_to_scaling(N, d_inf, a, nu):
    """d(N) = d_inf + a * N^(-nu)"""
    return d_inf + a * np.power(N, -nu)


def power_law(N, c, alpha):
    """κ(N) = c * N^alpha"""
    return c * np.power(N, alpha)


def fit_dimension_extrapolation(N_arr, d_arr, d_err_arr=None, label="d_H"):
    """Fit correction-to-scaling model and extrapolate to N→∞."""
    N = np.array(N_arr, dtype=float)
    d = np.array(d_arr, dtype=float)

    if len(N) < 3:
        print(f"  {label}: too few points for extrapolation")
        return None

    # Initial guess: d_inf ~ max(d), a ~ -1, nu ~ 0.3
    p0 = [max(d) * 1.2, -1.0, 0.3]
    bounds = ([0, -100, 0.01], [10, 100, 2.0])

    try:
        sigma = np.array(d_err_arr) if d_err_arr is not None and any(e > 0 for e in d_err_arr) else None
        if sigma is not None:
            sigma = np.maximum(sigma, 1e-6)
        popt, pcov = curve_fit(correction_to_scaling, N, d, p0=p0, bounds=bounds,
                               sigma=sigma, absolute_sigma=True if sigma is not None else False,
                               maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        d_inf, a, nu = popt
        d_inf_err = perr[0]

        # R²
        pred = correction_to_scaling(N, *popt)
        ss_res = np.sum((d - pred)**2)
        ss_tot = np.sum((d - d.mean())**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"  {label}(∞) = {d_inf:.4f} ± {d_inf_err:.4f}")
        print(f"    Correction: a={a:.4f}, ν={nu:.4f}, R²={R2:.4f}")
        return {'d_inf': d_inf, 'd_inf_err': d_inf_err, 'a': a, 'nu': nu, 'R2': R2}
    except Exception as e:
        print(f"  {label}: fit failed ({e})")
        # Fallback: linear extrapolation in 1/N
        coeffs = np.polyfit(1.0 / N, d, 1)
        d_inf = coeffs[1]
        print(f"  {label}(∞) ≈ {d_inf:.4f} (linear in 1/N, less reliable)")
        return {'d_inf': d_inf, 'd_inf_err': None, 'method': 'linear_1_over_N'}


def fit_curvature_decay(N_arr, k_arr, k_err_arr=None):
    """Fit |κ| ~ c·N^α and extrapolate."""
    N = np.array(N_arr, dtype=float)
    k = np.array(k_arr, dtype=float)

    if len(N) < 2:
        return None

    # Log-log fit for |κ|
    abs_k = np.abs(k)
    mask = abs_k > 1e-10
    if mask.sum() < 2:
        return None

    ln_N = np.log(N[mask])
    ln_k = np.log(abs_k[mask])
    coeffs = np.polyfit(ln_N, ln_k, 1)
    alpha = coeffs[0]
    c = np.exp(coeffs[1])

    pred = np.polyval(coeffs, ln_N)
    ss_res = np.sum((ln_k - pred)**2)
    ss_tot = np.sum((ln_k - ln_k.mean())**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    sign = np.sign(np.mean(k))
    print(f"  |κ| = {c:.6f} · N^({alpha:+.4f})  (R²={R2:.4f})")
    if alpha < -0.1:
        for N_ext in [1e4, 1e5, 1e6]:
            k_ext = c * N_ext**alpha
            print(f"    N={N_ext:.0e}: |κ| ≈ {k_ext:.6f}")
        print(f"  → κ → 0 as N → ∞")
    elif alpha > 0.1:
        print(f"  ⚠ |κ| diverges! Rule does not produce flat geometry.")

    return {'c': c, 'alpha': alpha, 'R2': R2, 'sign': float(sign)}


def analyse_scaling(name, data, target_dim=None):
    """Full finite-size scaling analysis for a dataset."""
    print(f"\n{'='*65}")
    print(f"  Finite-Size Scaling: {name}")
    print(f"{'='*65}")

    N_arr = [d['N'] for d in data]
    dH_arr = [d['d_H']['mean'] for d in data]
    dH_err = [d['d_H'].get('std_err', 0) for d in data]
    ds_arr = [d['d_s']['mean'] for d in data]
    ds_err = [d['d_s'].get('std_err', 0) for d in data]
    k_arr = [d['mean_curvature']['mean'] for d in data]

    # Table
    print(f"\n  {'L':>4}  {'N':>6}  {'d_H':>8}  {'d_s':>8}  {'⟨κ⟩':>10}")
    print(f"  {'-'*42}")
    for d in data:
        L = d.get('L', '?')
        print(f"  {L:>4}  {d['N']:>6}  {d['d_H']['mean']:>8.3f}  {d['d_s']['mean']:>8.3f}  {d['mean_curvature']['mean']:>+10.5f}")

    result = {'name': name, 'data': data}

    # Extrapolate d_H
    print(f"\n── Hausdorff dimension extrapolation ──")
    result['d_H_fit'] = fit_dimension_extrapolation(N_arr, dH_arr, dH_err, "d_H")
    result['d_H_boot'] = bootstrap_extrapolation(N_arr, dH_arr, dH_err, label="d_H")

    # Extrapolate d_s
    print(f"\n── Spectral dimension extrapolation ──")
    result['d_s_fit'] = fit_dimension_extrapolation(N_arr, ds_arr, ds_err, "d_s")
    result['d_s_boot'] = bootstrap_extrapolation(N_arr, ds_arr, ds_err, label="d_s")

    # Curvature decay
    print(f"\n── Curvature scaling ──")
    result['curv_fit'] = fit_curvature_decay(N_arr, k_arr)

    if target_dim:
        print(f"\n── Target: d = {target_dim} ──")
        if result.get('d_H_fit') and result['d_H_fit'].get('d_inf'):
            dev = abs(result['d_H_fit']['d_inf'] - target_dim)
            print(f"  d_H(∞) deviation from target: {dev:.4f}")

    return result


def main():
    out_dir = ROOT / "data/results/scaling_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # 1. Bare 3D lattice
    cal3 = load_calibration_as_scaling(3)
    if cal3:
        r = analyse_scaling("Bare 3D Lattice", cal3, target_dim=3.0)
        all_results.append(r)

    # 2. Bare 4D lattice
    cal4 = load_calibration_as_scaling(4)
    if cal4:
        r = analyse_scaling("Bare 4D Lattice", cal4, target_dim=4.0)
        all_results.append(r)

    # 3. Campaign 2 evolved (single-run)
    c2_dir = ROOT / "data/results/campaign2/scaling"
    if c2_dir.exists():
        c2 = load_single_run_scaling(c2_dir)
        if c2:
            # Sort by N
            c2.sort(key=lambda x: x['N'])
            r = analyse_scaling("Campaign 2 GA Best (evolved, 3D)", c2, target_dim=3.0)
            all_results.append(r)

    # 4. 4D v3 campaign baseline multi-scale
    baseline_4d = ROOT / "data/results/baseline_4d_L5.json"
    if baseline_4d.exists():
        with open(baseline_4d) as f:
            bl = json.load(f)
        # Single point from baseline
        print(f"\n4D baseline L=5: d_H={bl['baseline']['d_H']:.3f}, d_s={bl['spectral_v2']['d_s']:.3f}")

    # Save
    with open(out_dir / "scaling_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'scaling_results.json'}")

    # Plots
    if HAS_PLT and all_results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for r in all_results:
            data = r['data']
            N = [d['N'] for d in data]
            dH = [d['d_H']['mean'] for d in data]
            ds = [d['d_s']['mean'] for d in data]
            k = [d['mean_curvature']['mean'] for d in data]
            dH_err = [d['d_H'].get('std_err', 0) for d in data]
            ds_err = [d['d_s'].get('std_err', 0) for d in data]

            axes[0].errorbar(N, dH, yerr=dH_err, fmt='o-', label=r['name'], markersize=4, capsize=3)
            axes[1].errorbar(N, ds, yerr=ds_err, fmt='o-', label=r['name'], markersize=4, capsize=3)
            axes[2].plot(N, k, 'o-', label=r['name'], markersize=4)

        axes[0].set_xlabel('N'); axes[0].set_ylabel('d_H')
        axes[0].set_title('Hausdorff Dimension vs N')
        axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('N'); axes[1].set_ylabel('d_s')
        axes[1].set_title('Spectral Dimension vs N')
        axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel('N'); axes[2].set_ylabel('⟨κ⟩')
        axes[2].set_title('Mean Curvature vs N')
        axes[2].axhline(y=0, color='k', ls='--', alpha=0.3)
        axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "scaling_analysis.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}")
        plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
