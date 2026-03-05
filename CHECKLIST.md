# DISCRETUM — Submission Checklist (Final Revision)

## Code Quality
- [x] All tests pass: `./build/tests` (679 assertions, 39 test cases)
- [x] Validation suite: `./build/discretum_validate` (7/7 pass)
- [x] Build succeeds with `-Wall -Wextra` (GCC 14, C++20)
- [x] GPU build succeeds (HIP/ROCm 7.2, gfx1100)
- [x] No memory leaks in critical paths (manual review)
- [x] `n_components` field added to ensemble JSON serialization

## Data Integrity
- [x] Best rule: `data/results/best_rule_4d_v3.json`
- [x] Evolved ensembles: `data/results/scaling_4d_v3/evolved_L{4..8}.json`
- [x] Baseline ensembles: `data/results/scaling_4d_v3/baseline_L{4..8}.json`
- [x] Scaling summary: `data/results/scaling_4d_v3/scaling_summary.json`
- [x] Extrapolation: `data/results/scaling_4d_v3/extrapolation.json`
- [x] Spectral detail: `data/results/scaling_4d_v3/spectral_{evolved,baseline}_L5.json`
- [x] Bare lattice d_s checks: `data/results/scaling_4d_v3/check_ds_bare_L{4..8}.json`
- [x] All ensemble runs seeded for reproducibility

## Analysis Scripts
- [x] `scripts/analysis/collect_scaling.py` — scaling table + LaTeX
- [x] `scripts/analysis/extrapolation.py` — N→∞ fits with bootstrap CI
- [x] `scripts/analysis/spectral_multiscale_plot.py` — annotated d_eff(t) with CDT
- [x] `scripts/analysis/paper_figures_final.py` — all paper figures (4-panel scaling)
- [x] `scripts/analysis/rule_geometry_correlation.py` — parameter-fitness analysis

## Paper Figures (paper/figures/)
- [x] fig1_convergence_4d — CMA-ES vs GA convergence
- [x] fig2_scaling_all — 4-panel: d_H, d_s, κ, connected fraction
- [x] fig3_rule_summary — topology balance + connectivity vs scale
- [x] fig4_baseline_dH — bare lattice Hausdorff dimension validation
- [x] fig5_ensemble_L5 — d_H and κ distributions at L=5
- [x] fig6_scaling_dH — d_H extrapolation with CI band
- [x] fig7_scaling_ds — d_s extrapolation
- [x] fig8_scaling_kappa — κ extrapolation
- [x] fig9_spectral_multiscale — annotated P(t) and d_eff(t) with CDT band
- [x] fig_connectivity — connectivity percentage vs N

## Paper (paper/discretum.tex) — Revised for CQG
- [x] Compiles: pdflatex + bibtex (7 pages, PRD/revtex4-2 format)
- [x] All references resolved (13 entries including Konopka2008)
- [x] Title: "Computational search for emergent spacetime geometry..."
- [x] Abstract: honest framing with three identified failures
- [x] Intro: EDT/CDT dichotomy context, threefold contributions
- [x] Validation: bare lattice table (Table I) with d_s caveat
- [x] Scaling: compact table (Table II) with connected fraction
- [x] Selection bias subsection (Sec. IV D) added
- [x] d_H = 4.71 interpreted as 18% overshoot from densification
- [x] κ = -0.34 interpreted as hyperbolic (not "close to flat")
- [x] d_s discussed as finite-size limited; no improvement from evolution
- [x] CDT comparison with volume caveat (N~10^5 vs N~4096)
- [x] Role of causality section (Sec. V C)
- [x] Quantum graphity comparison added
- [x] Limitations and future directions (6 items)
- [x] Code availability appendix (Appendix A)

## Cover Letter
- [x] `paper/cover_letter.tex` — compiles to 1 page PDF
- [x] Three contributions highlighted
- [x] Negative results framed constructively

## Reproduction
- [x] `scripts/reproduce.sh` updated (steps 1–14, including 4D v3 pipeline)
- [x] README.md updated with all results, usage, and structure
- [x] `--skip-search` mode works with existing checkpoints
