#!/usr/bin/env python3
"""Rule-geometry correlation analysis.

Loads GA/CMA-ES checkpoint files and analyzes how rule parameters
correlate with geometric observables across the population.

Usage:
    python scripts/analysis/rule_geometry_correlation.py checkpoints/4d_v3_ga/ga_checkpoint.json
"""
import json, sys, os, math
import numpy as np

TOPO_NAMES = ["p_add", "p_rem", "p_rew", "p_spl", "p_mrg"]
STATE_NAMES = [f"s{i}{j}" for i in range(3) for j in range(3)]
ALL_NAMES = STATE_NAMES + TOPO_NAMES

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def load_ga_checkpoint(path):
    with open(path) as f:
        d = json.load(f)
    pop = np.array(d["population"])
    fitness = np.array(d["fitness"])
    gen = d["generation"]
    best_fitness = d["best_fitness"]
    best_params = np.array(d["best_params"])
    return pop, fitness, gen, best_fitness, best_params

def load_cmaes_checkpoint(path):
    with open(path) as f:
        d = json.load(f)
    best_params = np.array(d["best_params"])
    gen = d["generation"]
    best_fitness = d["best_fitness"]
    return None, np.array([best_fitness]), gen, best_fitness, best_params

def analyze_population(pop, fitness, gen, best_fitness, best_params):
    n_pop, n_dim = pop.shape
    print(f"\n{'='*60}")
    print(f"Rule-Geometry Correlation Analysis")
    print(f"{'='*60}")
    print(f"  Generation: {gen}")
    print(f"  Population: {n_pop}")
    print(f"  Best fitness: {best_fitness:.6f}")
    print(f"  Fitness range: [{fitness.min():.4f}, {fitness.max():.4f}]")
    print(f"  Fitness mean: {fitness.mean():.4f} ± {fitness.std():.4f}")

    # Best rule summary
    print(f"\n── Best Rule Parameters ──")
    for i, name in enumerate(ALL_NAMES):
        val = best_params[i]
        if i >= 9:
            prob = sigmoid(val)
            print(f"  {name:12s} = {prob:.4f}  (raw={val:.4f})")
        else:
            print(f"  {name:12s} = {val:.4f}")

    # Parameter statistics across population
    print(f"\n── Population Parameter Statistics ──")
    print(f"  {'Name':12s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}  {'Corr(fit)':>9s}")
    print(f"  {'-'*58}")

    correlations = {}
    for i, name in enumerate(ALL_NAMES):
        col = pop[:, i]
        if i >= 9:
            col = sigmoid(col)
        corr = np.corrcoef(col, fitness)[0, 1] if fitness.std() > 0 and col.std() > 0 else 0
        correlations[name] = corr
        print(f"  {name:12s} {col.mean():8.4f} {col.std():8.4f} {col.min():8.4f} {col.max():8.4f}  {corr:+9.4f}")

    # Top correlations
    print(f"\n── Strongest Correlations with Fitness ──")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr[:5]:
        direction = "HELPS" if corr > 0 else "HURTS"
        print(f"  {name:12s}: r = {corr:+.4f}  ({direction} fitness)")

    # Topology balance analysis
    print(f"\n── Topology Balance (Best Rule) ──")
    topo_probs = [sigmoid(best_params[9 + i]) for i in range(5)]
    total_topo = sum(topo_probs)
    for i, name in enumerate(TOPO_NAMES):
        frac = topo_probs[i] / total_topo * 100
        print(f"  {name}: {topo_probs[i]:.4f} ({frac:.1f}%)")
    growth_rate = topo_probs[0] + topo_probs[3] - topo_probs[1] - topo_probs[4]
    print(f"  Net growth tendency: {growth_rate:+.4f} (add+split - rem-merge)")

    # Elite analysis (top 20%)
    n_elite = max(1, n_pop // 5)
    elite_idx = np.argsort(fitness)[-n_elite:]
    print(f"\n── Elite Analysis (top {n_elite} individuals) ──")
    print(f"  Fitness range: [{fitness[elite_idx].min():.4f}, {fitness[elite_idx].max():.4f}]")
    for i, name in enumerate(TOPO_NAMES):
        col = sigmoid(pop[elite_idx, 9 + i])
        all_col = sigmoid(pop[:, 9 + i])
        print(f"  {name}: elite={col.mean():.4f}±{col.std():.4f}  pop={all_col.mean():.4f}±{all_col.std():.4f}")

    return correlations

def plot_correlations(pop, fitness, correlations, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Plot top 5 correlated params + fitness distribution
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for idx, (name, corr) in enumerate(sorted_corr[:5]):
        ax = axes.flat[idx]
        i = ALL_NAMES.index(name)
        col = sigmoid(pop[:, i]) if i >= 9 else pop[:, i]
        ax.scatter(col, fitness, s=8, alpha=0.4, c='steelblue')
        ax.set_xlabel(name)
        ax.set_ylabel('fitness')
        ax.set_title(f'{name} (r={corr:+.3f})')

    # Fitness distribution
    ax = axes.flat[5]
    ax.hist(fitness, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('fitness')
    ax.set_ylabel('count')
    ax.set_title('Fitness distribution')

    plt.tight_layout()
    figpath = os.path.join(outdir, 'rule_geometry_correlation.png')
    plt.savefig(figpath, dpi=150)
    plt.close()
    print(f"\nSaved {figpath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        d = json.load(f)

    if "population" in d:
        pop, fitness, gen, best_fitness, best_params = load_ga_checkpoint(path)
    elif "best_params" in d:
        pop, fitness, gen, best_fitness, best_params = load_cmaes_checkpoint(path)
    else:
        print("Unrecognized checkpoint format")
        sys.exit(1)

    if pop is not None and pop.shape[0] > 1:
        correlations = analyze_population(pop, fitness, gen, best_fitness, best_params)
        outdir = os.path.join(os.path.dirname(path), "figures")
        plot_correlations(pop, fitness, correlations, outdir)
    else:
        print(f"\nBest fitness: {best_fitness:.6f}")
        print(f"Best params: {best_params}")
