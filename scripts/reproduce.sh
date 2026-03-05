#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# DISCRETUM — Full reproduction script
#
# Reproduces the key results from the paper:
#   1. Build the project
#   2. Run tests and validation
#   3. Run 3D GA search (Campaign 2 config)
#   4. Run 4D GA search (v2 fitness)
#   5. Calibrate bare lattice baselines
#   6. Diagnose best rules (single + ensemble)
#   7. Finite-size scaling
#   8. Curvature trend analysis
#   9. Generate paper figures
#
# Usage:
#   bash scripts/reproduce.sh [--full] [--4d-only] [--skip-search]
#
# Options:
#   --full         Run full searches (5000 gen 3D, 500 gen 4D, ~hours)
#   --4d-only      Skip 3D search, only run 4D campaign
#   --skip-search  Skip search, only run analysis on existing results
#
# Requirements:
#   - CMake >= 3.20, Ninja, GCC >= 13 (C++20)
#   - Eigen3, spdlog, fmt, nlohmann-json
#   - Python 3 with matplotlib, numpy, scipy
#   - Optional: ROCm 7.x for GPU acceleration
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

FULL=false
ONLY4D=false
SKIP_SEARCH=false
for arg in "$@"; do
    case "$arg" in
        --full) FULL=true ;;
        --4d-only) ONLY4D=true ;;
        --skip-search) SKIP_SEARCH=true ;;
    esac
done

echo "╔══════════════════════════════════════════════════════╗"
echo "║         DISCRETUM — Reproduction Script             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Mode: full=$FULL, 4d-only=$ONLY4D, skip-search=$SKIP_SEARCH"
echo ""

OUTDIR="data/results/reproduce"
mkdir -p "$OUTDIR/checkpoints_3d" "$OUTDIR/checkpoints_4d" \
         "$OUTDIR/scaling" "$OUTDIR/figures" "$OUTDIR/ensemble" \
         "$OUTDIR/calibration"

# ═══════════════════════════════════════════════════
# Step 1: Build
# ═══════════════════════════════════════════════════
echo "═══ Step 1: Building project ═══"
if [[ -f /opt/rocm-7.2.0/bin/hipcc ]]; then
    cmake -B build \
        -DDISCRETUM_ENABLE_GPU=ON \
        -DGPU_TARGETS=gfx1100 \
        -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0 \
        -G Ninja 2>&1 | tail -3
else
    echo "  (No ROCm found, building CPU-only)"
    cmake -B build -DDISCRETUM_ENABLE_GPU=OFF -G Ninja 2>&1 | tail -3
fi
ninja -C build 2>&1 | tail -3
echo "  Build complete."
echo ""

# ═══════════════════════════════════════════════════
# Step 2: Tests & Validation
# ═══════════════════════════════════════════════════
echo "═══ Step 2: Running tests ═══"
./build/tests 2>&1 | tail -3
echo ""

echo "═══ Step 3: Running validation ═══"
./build/discretum_validate 2>&1 | tail -5
echo ""

# ═══════════════════════════════════════════════════
# Step 4: Calibrate bare lattices
# ═══════════════════════════════════════════════════
echo "═══ Step 4: Bare lattice calibration ═══"
./build/discretum_calibrate lattice_3d 5 12 "$OUTDIR/calibration" 2>&1
echo ""
./build/discretum_calibrate lattice_4d 4 8 "$OUTDIR/calibration" 2>&1
echo ""

# ═══════════════════════════════════════════════════
# Step 5: Search campaigns
# ═══════════════════════════════════════════════════
if ! $SKIP_SEARCH; then
    # --- 3D Campaign (v1 fitness) ---
    if ! $ONLY4D; then
        if $FULL; then MAX_GEN_3D=5000; else MAX_GEN_3D=100; fi
        echo "═══ Step 5a: 3D GA search ($MAX_GEN_3D gen) ═══"

        cat > "$OUTDIR/config_3d_ga.json" << EOF
{
    "optimizer": "ga",
    "max_generations": $MAX_GEN_3D,
    "population_size": 50,
    "evo_steps": 200,
    "graph_size": 1000,
    "graph_type": "lattice_3d",
    "fitness_version": 1,
    "seed": 42,
    "target_dimension": 2.3,
    "target_curvature": 0.0,
    "spectral_walkers": 500,
    "spectral_steps": 50,
    "checkpoint_dir": "$OUTDIR/checkpoints_3d",
    "checkpoint_interval": 10
}
EOF
        ./build/discretum_search ga "$OUTDIR/config_3d_ga.json" 2>&1 | tail -5
        echo ""
    fi

    # --- 4D Campaign (v2 fitness) ---
    if $FULL; then MAX_GEN_4D=500; else MAX_GEN_4D=100; fi
    echo "═══ Step 5b: 4D GA search ($MAX_GEN_4D gen, v2 fitness) ═══"

    cat > "$OUTDIR/config_4d_ga.json" << EOF
{
    "optimizer": "ga",
    "max_generations": $MAX_GEN_4D,
    "population_size": 50,
    "evo_steps": 100,
    "graph_size": 256,
    "graph_type": "lattice_4d",
    "fitness_version": 2,
    "seed": 2024,
    "sigma0": 0.5,
    "target_dimension": 4.0,
    "w_hausdorff": 2.0,
    "w_curvature": 1.5,
    "w_spectral": 1.0,
    "w_connectivity": 10.0,
    "w_stability": 0.5,
    "w_regularity": 0.3,
    "curvature_fluctuation_penalty": 0.5,
    "curvature_samples": 100,
    "spectral_walkers": 500,
    "spectral_steps": 50,
    "hausdorff_sources": 15,
    "checkpoint_dir": "$OUTDIR/checkpoints_4d",
    "checkpoint_interval": 5
}
EOF
    ./build/discretum_search ga "$OUTDIR/config_4d_ga.json" 2>&1 | tail -5
    echo ""
fi

# ═══════════════════════════════════════════════════
# Step 6: Extract best results
# ═══════════════════════════════════════════════════
echo "═══ Step 6: Extracting best results ═══"

extract_best() {
    local ckpt="$1" result="$2"
    if [[ -f "$result" ]]; then
        python3 -c "import json; d=json.load(open('$result')); print(f'  Best fitness: {d[\"best_fitness\"]:.4f}')"
    elif [[ -f "$ckpt" ]]; then
        python3 -c "
import json
d = json.load(open('$ckpt'))
json.dump({'best_fitness': d['best_fitness'], 'best_params': d['best_params']}, open('$result', 'w'), indent=2)
print(f'  Best fitness: {d[\"best_fitness\"]:.4f}')
"
    else
        echo "  No results found at $ckpt"
    fi
}

if ! $ONLY4D; then
    echo "  3D GA:"
    extract_best "$OUTDIR/checkpoints_3d/ga_checkpoint.json" "$OUTDIR/checkpoints_3d/ga_result.json"
fi
echo "  4D GA:"
extract_best "$OUTDIR/checkpoints_4d/ga_checkpoint.json" "$OUTDIR/checkpoints_4d/ga_result.json"
echo ""

# ═══════════════════════════════════════════════════
# Step 7: Ensemble diagnostics (4D)
# ═══════════════════════════════════════════════════
RESULT_4D="$OUTDIR/checkpoints_4d/ga_result.json"
if [[ -f "$RESULT_4D" ]]; then
    echo "═══ Step 7: 4D Ensemble diagnostics (10 runs) ═══"
    ./build/discretum_diagnose "$RESULT_4D" \
        --ensemble 10 --graph-type lattice_4d --graph-size 4 \
        --steps 100 --walkers 500 --walk-steps 50 --target-dim 4.0 \
        --output "$OUTDIR/ensemble/4d_ga_ensemble.json" 2>&1
    echo ""
fi

# ═══════════════════════════════════════════════════
# Step 8: Finite-size scaling (3D, if available)
# ═══════════════════════════════════════════════════
RESULT_3D="$OUTDIR/checkpoints_3d/ga_result.json"
if [[ -f "$RESULT_3D" ]] && ! $ONLY4D; then
    echo "═══ Step 8: 3D Finite-size scaling ═══"
    for s in 5 8 10 12; do
        echo -n "  L=$s: "
        ./build/discretum_diagnose "$RESULT_3D" \
            --steps 200 --transient 50 --graph-size $s --target-dim 2.3 --quiet \
            --output "$OUTDIR/scaling/ga_3d_size_${s}.json" 2>&1 | \
            grep -E "Hausdorff" | head -1 || echo "(no geodesic data)"
    done
    echo ""
fi

# ═══════════════════════════════════════════════════
# Step 9: 4D v3 CMA-ES campaign (definitive)
# ═══════════════════════════════════════════════════
echo "═══ Step 9: 4D v3 CMA-ES campaign ═══"
RULE="data/results/best_rule_4d_v3.json"
SCALEDIR="data/results/scaling_4d_v3"
mkdir -p "$SCALEDIR"

if [[ ! -f "$RULE" ]]; then
    echo "  Running 4D v3 CMA-ES search..."
    ./build/discretum_search cmaes configs/search_4d_v3_cmaes.json 2>&1 | tail -5
    # Extract best rule
    python3 -c "
import json
d = json.load(open('data/checkpoints/4d_v3_cmaes/cmaes_result.json'))
out = {'description': 'Best 4D rule from CMA-ES v3', 'fitness': d['best_fitness'],
       'best_params': d['best_params']}
json.dump(out, open('$RULE', 'w'), indent=2)
print(f'  Best fitness: {d[\"best_fitness\"]:.4f}')
"
else
    echo "  Using existing rule: $RULE"
fi
echo ""

# ═══════════════════════════════════════════════════
# Step 10: Finite-size scaling (4D v3)
# ═══════════════════════════════════════════════════
echo "═══ Step 10: 4D v3 finite-size scaling ═══"
for L in 4 5 6 7 8; do
    N=$((L**4))
    WS=$((50*L*L))
    WK=20000; [[ $N -gt 3000 ]] && WK=10000

    # Evolved
    EFILE="$SCALEDIR/evolved_L${L}.json"
    if [[ ! -f "$EFILE" ]]; then
        echo "  Evolved L=$L (N=$N)..."
        ./build/discretum_diagnose "$RULE" \
            --graph-type lattice_4d --graph-size "$L" \
            --steps 300 --transient 50 --ensemble 20 \
            --fitness-v 3 --target-dim 4 \
            --walkers $WK --walk-steps $WS --seed 42 \
            --quiet --output "$EFILE" 2>&1 | grep "Completed"
    else
        echo "  [SKIP] $EFILE exists"
    fi

    # Baseline
    BFILE="$SCALEDIR/baseline_L${L}.json"
    if [[ ! -f "$BFILE" ]]; then
        echo "  Baseline L=$L (N=$N)..."
        ./build/discretum_diagnose "$RULE" \
            --graph-type lattice_4d --graph-size "$L" \
            --no-evolution --ensemble 20 \
            --fitness-v 3 --target-dim 4 \
            --walkers $WK --walk-steps $WS --seed 42 \
            --quiet --output "$BFILE" 2>&1 | grep "Completed"
    else
        echo "  [SKIP] $BFILE exists"
    fi
done
echo ""

# ═══════════════════════════════════════════════════
# Step 11: Spectral detail (L=5)
# ═══════════════════════════════════════════════════
echo "═══ Step 11: Spectral multiscale analysis (L=5) ═══"
if [[ ! -f "$SCALEDIR/spectral_evolved_L5.json" ]]; then
    ./build/discretum_diagnose "$RULE" \
        --graph-type lattice_4d --graph-size 5 \
        --steps 300 --transient 50 \
        --walkers 50000 --walk-steps 2000 --seed 42 --quiet \
        --output-spectral-detail "$SCALEDIR/spectral_evolved_L5.json" 2>&1 | grep "d_s"
fi
if [[ ! -f "$SCALEDIR/spectral_baseline_L5.json" ]]; then
    ./build/discretum_diagnose "$RULE" \
        --graph-type lattice_4d --graph-size 5 --no-evolution \
        --walkers 50000 --walk-steps 2000 --seed 42 --quiet \
        --output-spectral-detail "$SCALEDIR/spectral_baseline_L5.json" 2>&1 | grep "d_s"
fi
echo ""

# ═══════════════════════════════════════════════════
# Step 12: Analysis scripts
# ═══════════════════════════════════════════════════
echo "═══ Step 12: Running analysis scripts ═══"
python3 scripts/analysis/collect_scaling.py 2>&1 | head -20
echo ""
python3 scripts/analysis/extrapolation.py 2>&1 | tail -10
echo ""

# ═══════════════════════════════════════════════════
# Step 13: Generate figures
# ═══════════════════════════════════════════════════
echo "═══ Step 13: Generating paper figures ═══"
python3 scripts/analysis/paper_figures_final.py 2>&1
echo ""
python3 scripts/analysis/spectral_multiscale_plot.py 2>&1
echo ""

# ═══════════════════════════════════════════════════
# Step 14: Compile paper
# ═══════════════════════════════════════════════════
echo "═══ Step 14: Compiling paper ═══"
cd paper
pdflatex -interaction=nonstopmode discretum.tex > /dev/null 2>&1
bibtex discretum > /dev/null 2>&1
pdflatex -interaction=nonstopmode discretum.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode discretum.tex > /dev/null 2>&1
echo "  Paper compiled: paper/discretum.pdf"
cd ..
echo ""

echo "╔══════════════════════════════════════════════════════╗"
echo "║                    COMPLETE                         ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Rule:        data/results/best_rule_4d_v3.json     ║"
echo "║  Scaling:     data/results/scaling_4d_v3/           ║"
echo "║  Extrapol.:   data/results/scaling_4d_v3/extrapol.  ║"
echo "║  Figures:     paper/figures/                        ║"
echo "║  Paper:       paper/discretum.pdf                   ║"
echo "╚══════════════════════════════════════════════════════╝"
