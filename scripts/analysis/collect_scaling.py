#!/usr/bin/env python3
"""
DISCRETUM — Collect finite-size scaling results from ensemble JSON files.

Produces:
  1. Console table with all observables
  2. LaTeX table for paper
  3. Summary JSON for downstream analysis (extrapolation.py)

Reports both full-ensemble and connected-only statistics.
"""
import json
import os
import sys
import numpy as np

OUTDIR = "data/results/scaling_4d_v3"
SCALES = [4, 5, 6, 7, 8]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def stats_from_connected_runs(data):
    """Compute stats from connected runs only."""
    runs = data.get('runs', [])
    connected = [r for r in runs if not r.get('aborted', False) and r.get('is_connected', False)]
    n_conn = len(connected)
    if n_conn == 0:
        return None

    def s(key):
        vals = np.array([r[key] for r in connected])
        return {
            'mean': float(np.mean(vals)),
            'std_err': float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            'std_dev': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'median': float(np.median(vals)),
            'n': len(vals),
        }

    return {
        'n_connected': n_conn,
        'n_total': len(runs),
        'd_H': s('d_H'),
        'd_s': s('d_s'),
        'mean_curvature': s('mean_curvature'),
        'std_curvature': s('std_curvature'),
        'cv_degree': s('cv_degree'),
        'mean_degree': s('mean_degree'),
        'n_final': s('n_final'),
    }


def main():
    print("=" * 90)
    print("  FINITE-SIZE SCALING — Best 4D Rule (CMA-ES v3)")
    print("=" * 90)

    # ─── Connected-only table ───
    print("\n── Connected Runs Only ──")
    header = f"{'L':>3} {'N':>6} {'conn':>6} {'d_H':>14} {'d_s':>14} {'⟨κ⟩':>16} {'⟨k⟩':>12} {'N_f':>14}"
    print(header)
    print("-" * 90)

    scaling_data = []

    for L in SCALES:
        N = L ** 4
        fpath = os.path.join(OUTDIR, f"evolved_L{L}.json")
        data = load_json(fpath)
        if data is None:
            print(f"{L:>3} {N:>6}  -- FILE NOT FOUND --")
            continue

        cs = stats_from_connected_runs(data)
        if cs is None:
            print(f"{L:>3} {N:>6}  {0:>3}/{data.get('n_total',20):>2}  -- NO CONNECTED RUNS --")
            continue

        c = cs
        print(f"{L:>3} {N:>6} {c['n_connected']:>3}/{c['n_total']:>2}"
              f"  {c['d_H']['mean']:>6.3f}±{c['d_H']['std_err']:>5.3f}"
              f"  {c['d_s']['mean']:>6.3f}±{c['d_s']['std_err']:>5.3f}"
              f"  {c['mean_curvature']['mean']:>+7.4f}±{c['mean_curvature']['std_err']:>6.4f}"
              f"  {c['mean_degree']['mean']:>6.2f}±{c['mean_degree']['std_err']:>4.2f}"
              f"  {c['n_final']['mean']:>7.0f}±{c['n_final']['std_err']:>5.0f}")

        scaling_data.append({
            'L': L, 'N': N,
            'n_connected': c['n_connected'],
            'n_total': c['n_total'],
            'd_H': c['d_H']['mean'], 'd_H_err': c['d_H']['std_err'],
            'd_s': c['d_s']['mean'], 'd_s_err': c['d_s']['std_err'],
            'kappa': c['mean_curvature']['mean'], 'kappa_err': c['mean_curvature']['std_err'],
            'std_kappa': c['std_curvature']['mean'], 'std_kappa_err': c['std_curvature']['std_err'],
            'mean_degree': c['mean_degree']['mean'], 'mean_degree_err': c['mean_degree']['std_err'],
            'cv_degree': c['cv_degree']['mean'], 'cv_degree_err': c['cv_degree']['std_err'],
            'n_final': c['n_final']['mean'], 'n_final_err': c['n_final']['std_err'],
        })

    print("-" * 90)

    # ─── Connectivity trend ───
    print("\n── Connectivity vs Scale ──")
    for sd in scaling_data:
        pct = 100 * sd['n_connected'] / sd['n_total']
        bar = "#" * int(pct / 5)
        print(f"  L={sd['L']}  N={sd['N']:>5}  {sd['n_connected']:>2}/{sd['n_total']:>2} ({pct:5.1f}%)  {bar}")

    # ─── LaTeX table ───
    print("\n\n% ─── LaTeX Table ───")
    print("\\begin{table}[t]")
    print("  \\centering")
    print("  \\caption{Finite-size scaling of the best 4D rule (CMA-ES v3).")
    print("    Connected runs only; $n_{\\rm conn}/n_{\\rm total}$ gives the")
    print("    fraction of 20-run ensemble members producing connected graphs.}")
    print("  \\label{tab:scaling4d}")
    print("  \\begin{tabular}{rrcrrrr}")
    print("    \\toprule")
    print("    $L$ & $N$ & conn & $d_H$ & $d_s$ & $\\langle\\kappa\\rangle$ & $\\langle k\\rangle$ \\\\")
    print("    \\midrule")
    for sd in scaling_data:
        print(f"    {sd['L']} & {sd['N']} & {sd['n_connected']}/{sd['n_total']} & "
              f"${sd['d_H']:.2f} \\pm {sd['d_H_err']:.2f}$ & "
              f"${sd['d_s']:.2f} \\pm {sd['d_s_err']:.2f}$ & "
              f"${sd['kappa']:+.3f} \\pm {sd['kappa_err']:.3f}$ & "
              f"${sd['mean_degree']:.1f} \\pm {sd['mean_degree_err']:.1f}$ \\\\")
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")

    # ─── Save summary JSON ───
    summary = {
        'description': 'Finite-size scaling of best 4D CMA-ES v3 rule (connected runs only)',
        'scales': scaling_data,
    }
    out_path = os.path.join(OUTDIR, "scaling_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == '__main__':
    main()
