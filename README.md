# DISCRETUM — DISCrete Spacetime Research for Emergent Tensor Unification Model

## Overview

DISCRETUM searches for local update rules in cellular automata on dynamic graphs
whose continuum limit reproduces properties of spacetime geometry.
A parametric rule evolves both node states and graph topology, and optimisers
(CMA-ES, genetic algorithm) tune the 14 parameters to maximise a fitness
function measuring how closely the evolved graph matches target geometry.

## Scientific Goal

Find a rule **R\*** such that:
1. Hausdorff dimension d_H → d (target dimensionality)
2. Spectral dimension d_s → d (manifold-like diffusion)
3. Ollivier-Ricci curvature ⟨κ⟩ → 0 (flatness)
4. Connectivity and regularity are maintained at all scales

## Requirements

### Software
- **C++20** compiler (GCC 14+ tested)
- **CMake 3.24+**, **Ninja** (recommended)
- **Eigen3**, **Catch2 v3**, **fmt**, **spdlog**, **nlohmann-json**
- **ROCm/HIP 7.2+** (optional, for GPU acceleration on AMD GPUs)
- **Python 3** with numpy, scipy, matplotlib (for analysis scripts)

### Hardware (GPU optional)
- AMD RX 7900 XTX (gfx1100) or other ROCm-compatible GPU

## Building

```bash
# CPU only
cmake -B build -G Ninja
ninja -C build

# With GPU support (AMD ROCm)
cmake -B build -G Ninja \
  -DDISCRETUM_ENABLE_GPU=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0
ninja -C build
```

## Executables

| Binary | Description |
|--------|-------------|
| `tests` | Unit test suite (Catch2, 679 assertions, 39 test cases) |
| `discretum_validate` | Self-contained validation of all subsystems (7 checks) |
| `discretum_search` | Main search CLI (CMA-ES, GA, eval, analyze) |
| `discretum_diagnose` | Single-run and ensemble diagnostics with scaling support |
| `discretum_calibrate` | Bare lattice calibration tool (3D/4D) |
| `bench_evolution` | Evolution engine benchmarks |
| `bench_ollivier` | Ollivier-Ricci curvature benchmarks |
| `bench_gpu_vs_cpu` | GPU vs CPU performance comparison |

## Usage

### Run tests
```bash
./build/tests                  # 679 assertions, 39 test cases
```

### Validation suite
```bash
./build/discretum_validate     # 7 subsystem checks
```

### Search for rules
```bash
# CMA-ES optimiser
./build/discretum_search cmaes configs/search_4d_v3_cmaes.json

# Genetic algorithm
./build/discretum_search ga configs/search_4d_v3_ga.json
```

### Diagnose a rule
```bash
# Single-run diagnostics
./build/discretum_diagnose result.json --graph-type lattice_4d --graph-size 5 \
    --steps 300 --transient 50 --walkers 20000 --walk-steps 1250

# Ensemble (20 independent runs with error bars)
./build/discretum_diagnose result.json --graph-type lattice_4d --graph-size 5 \
    --ensemble 20 --fitness-v 3 --target-dim 4 --seed 42 \
    --output ensemble_result.json

# Bare lattice baseline (no evolution)
./build/discretum_diagnose result.json --graph-type lattice_4d --graph-size 5 \
    --no-evolution --ensemble 20 --output baseline.json

# Export spectral detail (P(t), d_eff(t) curves)
./build/discretum_diagnose result.json --graph-type lattice_4d --graph-size 5 \
    --steps 300 --transient 50 --walkers 50000 --walk-steps 2000 \
    --output-spectral-detail spectral_detail.json
```

### Analysis scripts
```bash
# Collect scaling results into table + LaTeX
python3 scripts/analysis/collect_scaling.py

# Extrapolate observables to N→∞ with bootstrap CI
python3 scripts/analysis/extrapolation.py

# Spectral multiscale plot with CDT comparison
python3 scripts/analysis/spectral_multiscale_plot.py

# Generate all paper figures
python3 scripts/analysis/paper_figures_final.py

# Parameter-geometry correlations
python3 scripts/analysis/rule_geometry_correlation.py checkpoints/4d_v3_ga/ga_checkpoint.json
```

### Full reproduction
```bash
# Quick (~10 min, uses existing checkpoints if available)
bash scripts/reproduce.sh --skip-search

# Full from scratch (~hours)
bash scripts/reproduce.sh --full
```

### Compile paper
```bash
cd paper && pdflatex discretum && bibtex discretum && pdflatex discretum && pdflatex discretum
```

## Project Structure

```
src/
├── core/              DynamicGraph (modified CSR), PCG32 PRNG
├── automaton/         ParametricRule (14 params), Evolution engine
├── geometry/          Ollivier-Ricci, spectral dimension, geodesic, metric tensor
├── gpu/               HIP kernels: Ollivier-Ricci, random walk, state evolution
├── search/            Fitness (v1-v3), CMA-ES, GA, ensemble framework
├── main_search.cpp    CLI for rule search
├── main_diagnose.cpp  Single-run and ensemble diagnostics
├── main_calibrate.cpp Bare lattice calibration
└── main_validate.cpp  Validation suite
tests/                 Catch2 unit tests (39 test cases)
benchmarks/            Performance benchmarks
scripts/
├── analysis/          Python analysis & figure generation
└── run/               Shell scripts for batch runs
paper/                 LaTeX paper (Physical Review D style)
├── discretum.tex      Main paper (5 pages)
├── references.bib     Bibliography
└── figures/           All paper figures (PDF + PNG)
data/results/
├── best_rule_4d_v3.json         Best CMA-ES rule (θ*)
└── scaling_4d_v3/               Finite-size scaling data
    ├── evolved_L{4..8}.json     Evolved ensemble results
    ├── baseline_L{4..8}.json    Bare lattice baselines
    ├── scaling_summary.json     Collected scaling table
    ├── extrapolation.json       Fit results with CI
    └── spectral_*.json          Spectral detail curves
```

## Architecture

### Parametric Rule (14 parameters)
- **9 state params**: 3×3 transition weight matrix θ_state
- **5 topo params**: sigmoid-transformed probabilities for edge add/remove/rewire, node split/merge

### Fitness Function (v3)
Six-component squared-penalty function:
- **Hausdorff dimension** |d_H − d*|² (weight 2.0)
- **Curvature** |⟨κ⟩|² (weight 1.5)
- **Spectral concordance** |d_s − d*|² (weight 1.0)
- **Stability** |ln(N/N₀)|² (weight 0.5)
- **Regularity** CV(k)² (weight 0.3)
- **Connectivity** 10 × n_components (disconnected penalty)
- **Density/degradation** guards against over-densification

### GPU Kernels (HIP/ROCm)
- **Ollivier-Ricci**: per-edge kernel with exact min-cost flow
- **Random walk**: per-walker kernel with PCG32 RNG for spectral dimension
- **State evolution**: per-node kernel for synchronous state updates

### Geometry Tools
- All-pairs shortest paths (BFS), diameter, average path length
- Volume growth and Hausdorff dimension (dual estimator: shell + local derivative)
- Spectral dimension with plateau-finding (v2) and global fit (v1)
- Classical MDS embedding, metric tensor from edge displacements

## Results

### 4D v3 Campaign (Definitive)

**Best CMA-ES rule** (fitness = −0.453, 20/20 connected at N=625):

| Observable | Value |
|-----------|-------|
| d_H (Hausdorff) | 3.88 ± 0.01 |
| d_s (spectral, v2) | 2.14 ± 0.14 |
| ⟨κ⟩ (Ollivier-Ricci) | −0.24 ± 0.002 |
| ⟨k⟩ | 9.85 |
| CV(k) | 0.27 |
| p_add / p_rm / p_rw / p_sp / p_mg | 0.085 / ~0 / ~0 / 0.040 / 0.214 |

### Finite-Size Scaling (4D, connected runs only)

| L | N | Connected | d_H | d_s | ⟨κ⟩ |
|---|------|-----------|--------------|--------------|------------------|
| 4 | 256 | 20/20 | 3.09 ± 0.01 | 0.95 ± 0.02 | −0.232 ± 0.002 |
| 5 | 625 | 18/20 | 4.26 ± 0.02 | 0.32 ± 0.00 | −0.291 ± 0.002 |
| 6 | 1296 | 4/20 | 4.39 ± 0.01 | 0.28 ± 0.03 | −0.318 ± 0.006 |
| 7 | 2401 | 5/20 | 4.45 ± 0.02 | 1.93 ± 1.65 | −0.326 ± 0.002 |
| 8 | 4096 | 1/20 | 4.65 ± 0.00 | 0.30 ± 0.00 | −0.352 ± 0.000 |

### Extrapolation to N → ∞

| Observable | Thermodynamic limit | 95% CI | Model |
|-----------|-------------------|--------|-------|
| d_H | 4.71 ± 0.04 | [4.69, 4.74] | Power-law |
| d_s | ≈ 0 | — | — |
| ⟨κ⟩ | −0.34 ± 0.006 | [−0.36, −0.33] | Power-law |

### Key Findings

1. **d_H overshoots**: The rule produces graphs with d_H > 4, driven by
   densification creating short-range connections.
2. **d_s ≪ d_H**: Spectral dimension remains anomalously low (~0.3–1.0),
   indicating strongly non-manifold diffusion. No CDT-like crossover is observed.
3. **Persistent negative curvature**: ⟨κ⟩ → −0.34, indicating hyperbolic geometry.
4. **Connectivity instability**: Graph fragmentation increases with N,
   from 100% connected at N=256 to 5% at N=4096.
5. **Absence of causal structure** is identified as the likely limiting factor
   preventing manifold-like spectral behaviour.

### Previous 3D Campaign (Campaign 2, GA, N₀=1000)

| L | N | d_H | ⟨κ⟩ |
|---|------|------|--------|
| 5 | 125 | 2.13 | +0.007 |
| 10 | 1000 | 2.56 | −0.097 |
| 15 | 3775 | **2.93** | −0.153 |

## Key Concepts

### Ollivier-Ricci Curvature
Discrete Ricci curvature via optimal transport (Wasserstein-1 distance)
between lazy random walk measures on adjacent nodes.

### Spectral Dimension
Effective dimension from random walk return probability: d_eff(t) = −2 d(ln P)/d(ln t).
Plateau-finding algorithm identifies stable scaling regime.

### Hausdorff Dimension
From volume growth N(r) ~ r^{d_H}, fitted over the scaling regime via BFS
from multiple random sources.

### Dynamic Graph
Modified CSR with O(1) amortised edge add/remove, node split/merge, and edge rewiring.

## License

MIT License — see LICENSE file for details.

[![DOI](https://zenodo.org/badge/1173414754.svg)](https://doi.org/10.5281/zenodo.18874775)

