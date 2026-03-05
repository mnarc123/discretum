#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════
# DISCRETUM — Finite-Size Scaling of Best 4D Rule (CMA-ES v3)
# ═══════════════════════════════════════════════════════════
#
# Runs 20-run ensembles at L=4,5,6,7,8 for both:
#   (a) Evolved graph (best rule applied)
#   (b) Bare lattice (baseline, no evolution)
#
# Usage:
#   bash scripts/run/run_scaling.sh          # all scales
#   bash scripts/run/run_scaling.sh 4 5      # specific scales only

RULE="data/results/best_rule_4d_v3.json"
OUTDIR="data/results/scaling_4d_v3"
BINARY="./build/discretum_diagnose"
ENSEMBLE=20
EVO_STEPS=300
TRANSIENT=50
SEED=42

mkdir -p "$OUTDIR"

if [ ! -f "$RULE" ]; then
    echo "ERROR: Rule file not found: $RULE"
    exit 1
fi

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found. Run: ninja -C build"
    exit 1
fi

# Scales to run (default: all; or pass as arguments)
if [ $# -gt 0 ]; then
    SCALES=("$@")
else
    SCALES=(4 5 6 7 8)
fi

echo "═══════════════════════════════════════════════════════"
echo "  DISCRETUM — Finite-Size Scaling"
echo "  Rule: $RULE"
echo "  Scales: ${SCALES[*]}"
echo "  Ensemble: $ENSEMBLE runs per scale"
echo "═══════════════════════════════════════════════════════"
echo ""

for L in "${SCALES[@]}"; do
    N=$((L**4))
    # Walk steps scale as ~50*L^2 to cover the graph diameter
    WALK_STEPS=$((50 * L * L))
    # Walkers: 20000 for small graphs, scale down for large ones to keep time reasonable
    if [ "$N" -le 1000 ]; then
        WALKERS=20000
    elif [ "$N" -le 3000 ]; then
        WALKERS=15000
    else
        WALKERS=10000
    fi
    # Hausdorff sources scale with graph size
    HAUSDORFF_SOURCES=30

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  L=$L  N=$N  walkers=$WALKERS  walk_steps=$WALK_STEPS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    OUTFILE="$OUTDIR/evolved_L${L}.json"
    if [ -f "$OUTFILE" ]; then
        echo "  [SKIP] $OUTFILE already exists"
    else
        echo "  [RUN]  Evolved ensemble L=$L ..."
        START=$(date +%s)
        "$BINARY" "$RULE" \
            --graph-type lattice_4d \
            --graph-size "$L" \
            --steps "$EVO_STEPS" \
            --transient "$TRANSIENT" \
            --ensemble "$ENSEMBLE" \
            --fitness-v 3 \
            --target-dim 4 \
            --walkers "$WALKERS" \
            --walk-steps "$WALK_STEPS" \
            --seed "$SEED" \
            --quiet \
            --output "$OUTFILE" \
            2>&1 | tail -20
        END=$(date +%s)
        echo "  [DONE] L=$L evolved in $((END - START))s"
    fi

    echo ""
done

echo "═══════════════════════════════════════════════════════"
echo "  All scaling runs complete."
echo "  Results in: $OUTDIR/"
echo "═══════════════════════════════════════════════════════"
